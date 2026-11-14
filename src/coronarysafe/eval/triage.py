"""Evaluate coronary failure rankings using only features available at inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .bootstrap import cluster_bootstrap, paired_bootstrap_difference
from .metrics import aurc, auroc, normalised_recall, oracle_recall_at_budget, recall_at_budget

GRAPH = [
    "g_n_components", "g_n_components_ge_10mm", "g_total_length_mm",
    "g_len_comp1_mm", "g_len_comp2_mm", "g_share_top2", "g_n_endpoints",
    "g_n_junctions", "g_max_branch_depth", "g_radius_violations", "g_mean_tortuosity",
    "g_max_radius_mm", "g_leak_fraction", "g_n_gaps",
]
UNCERTAINTY = [
    "u_mean_entropy_mask", "u_mean_entropy_skel2mm", "u_boundary_uncertain_frac",
]
RULES = {
    "leak_radius": "g_max_radius_mm",
    "components": "g_n_components_ge_10mm",
    "gaps": "g_n_gaps",
    "leak_fraction": "g_leak_fraction",
    "skeleton_entropy": "u_mean_entropy_skel2mm",
}
MODELS = {
    "graph": GRAPH,
    "uncertainty": UNCERTAINTY,
    "graph_and_uncertainty": GRAPH + UNCERTAINTY,
}
SEED = 14
BUDGET = 0.2


def load_table(path, *, require_uncertainty=True):
    df = pd.read_parquet(path)
    if "patient_group" in df and "scan_id" not in df:
        df = df.rename(columns={"patient_group": "scan_id"})
    required = {"scan_id", "method", "status", "critical", *GRAPH}
    if require_uncertainty:
        required.update(UNCERTAINTY)
    else:
        for column in UNCERTAINTY:
            if column not in df:
                df[column] = np.nan
    if missing := required.difference(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df.empty or not df.status.eq("ok").all():
        raise ValueError(f"{path}: empty table or failed reconstructions")
    for column in ("scan_id", "method"):
        df[column] = df[column].astype("string").str.strip()
        if df[column].isna().any() or df[column].eq("").any():
            raise ValueError(f"{path}: missing {column}")
    if df.duplicated(["scan_id", "method"]).any():
        raise ValueError(f"{path}: duplicate patient and segmenter rows")
    if df.critical.isna().any() or not pd.api.types.is_bool_dtype(df.critical):
        raise ValueError(f"{path}: critical must contain true/false labels")
    df[GRAPH + UNCERTAINTY] = df[GRAPH + UNCERTAINTY].replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["scan_id", "method"]).reset_index(drop=True)


def fit_score(train, test, features):
    if train.critical.nunique() != 2:
        raise ValueError("QA training requires both critical and noncritical reconstructions")
    if set(train.scan_id) & set(test.scan_id):
        raise ValueError("patient overlap between QA training and evaluation")
    available = train[features].notna().any()
    features = available[available].index.tolist()
    if not features or test[features].isna().all().any():
        raise ValueError("QA feature unavailable across an entire training or evaluation cohort")
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1000),
    )
    model.fit(train[features], train.critical)
    return model.predict_proba(test[features])[:, 1]


def out_of_fold(df, eval_method):
    """Keep a patient's reconstructions together across all segmentation methods."""
    target = df.loc[df.method == eval_method].reset_index(drop=True)
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {key: np.full(len(target), np.nan) for key in (*MODELS, "unseen_segmenter_graph")}
    for train_pos, test_pos in split.split(target, target.critical, target.scan_id):
        train = target.iloc[train_pos]
        test = target.iloc[test_pos]
        for name, features in MODELS.items():
            scores[name][test_pos] = fit_score(train, test, features)
        sources = df.loc[(df.method != eval_method) & ~df.scan_id.isin(test.scan_id)]
        if not sources.empty:
            scores["unseen_segmenter_graph"][test_pos] = fit_score(sources, test, GRAPH)
    if df.method.nunique() == 1:
        del scores["unseen_segmenter_graph"]
    return target, scores


def independent_scores(development, test, method):
    train = development.loc[development.method == method]
    scores = {name: fit_score(train, test, features) for name, features in MODELS.items()}
    sources = development.loc[development.method != method]
    if not sources.empty:
        scores["unseen_segmenter_graph"] = fit_score(sources, test, GRAPH)
    return scores


def rule_scores(df):
    return {
        name: df[column].fillna(df[column].min()).to_numpy()
        for name, column in RULES.items()
        if df[column].notna().any()
    }


def interval(result):
    return {
        key: float(result[key]) if np.isfinite(result[key]) else None
        for key in ("estimate", "lo", "hi")
    }


def evaluate_method(target, scores, n_boot, proposed_name, comparator):
    outcomes = target.critical.to_numpy(dtype=bool)
    patients = target.scan_id.to_numpy()
    if not outcomes.any() or outcomes.all() or int(BUDGET * len(target)) == 0:
        raise ValueError("evaluation requires both labels and room to review at least one scan")
    high_dice = target.dice.ge(0.9).to_numpy() if "dice" in target else None
    numbers = {}
    for name, risk in scores.items():
        def stat(rows):
            return normalised_recall(risk[rows], outcomes[rows], BUDGET)

        numbers[name] = {
            "recall": recall_at_budget(risk, outcomes, BUDGET),
            "normalised_recall": interval(cluster_bootstrap(patients, stat, n_boot, SEED)),
            "aurc": aurc(risk, outcomes),
            "auroc": auroc(risk, outcomes),
        }
        if high_dice is not None and (outcomes & high_dice).any():
            numbers[name]["high_dice_critical_recall_at_full_cohort_cutoff"] = recall_at_budget(
                risk, outcomes & high_dice, BUDGET,
            )
    proposed = scores[proposed_name]
    baseline = scores[comparator]

    def ranking(risk):
        return lambda rows: normalised_recall(risk[rows], outcomes[rows], BUDGET)

    delta = paired_bootstrap_difference(
        patients, ranking(proposed), ranking(baseline), n_boot, SEED,
    )
    result = {
        "reconstructions": len(target),
        "critical": int(outcomes.sum()),
        "oracle_recall": oracle_recall_at_budget(outcomes, BUDGET),
        "scores": numbers,
        "model_for_paired_comparison": proposed_name,
        "baseline_for_paired_comparison": comparator,
        "delta_normalised_recall": interval(delta),
    }
    if high_dice is not None:
        result["dice_at_least_0_9"] = int(high_dice.sum())
        result["critical_with_dice_at_least_0_9"] = int((outcomes & high_dice).sum())
        result["mean_dice"] = float(target.dice.mean())
    if "severe" in target:
        result["severe"] = int(target.severe.sum())
    if "centerline_fallback" in target:
        result["centerline_fallbacks"] = int(target.centerline_fallback.sum())
    return result


def strongest_rule(development):
    method_scores = []
    for _, target in development.groupby("method"):
        method_scores.append({
            name: normalised_recall(scores, target.critical, BUDGET)
            for name, scores in rule_scores(target).items()
        })
    averages = pd.DataFrame(method_scores).dropna(axis=1).mean()
    if averages.empty:
        raise ValueError("no shared baseline rule available on development segmenters")
    return averages.idxmax()


def evaluate(development, n_boot, test=None):
    results = {}
    if test is not None and set(development.scan_id) & set(test.scan_id):
        raise ValueError("development and independent test contain overlapping patients")
    cohort = development if test is None else test
    developed_methods = set(development.method)
    for method in cohort.method.unique():
        if method not in developed_methods:
            target = test.loc[test.method == method]
            rules = rule_scores(target)
            scores = {"unseen_segmenter_graph": fit_score(development, target, GRAPH)}
            results[method] = evaluate_method(
                target, rules | scores, n_boot, "unseen_segmenter_graph",
                strongest_rule(development),
            )
            continue
        target, scores = out_of_fold(development, method)
        rules = rule_scores(target)
        candidates = rules | {name: scores[name] for name in ("graph", "uncertainty")}
        comparator = max(
            candidates,
            key=lambda name: normalised_recall(candidates[name], target.critical, BUDGET),
        )
        if test is not None:
            target = test.loc[test.method == method]
            scores = independent_scores(development, target, method)
            rules = rule_scores(target)
        results[method] = evaluate_method(
            target, rules | scores, n_boot, "graph_and_uncertainty", comparator,
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path, help="v3 QA Parquet table")
    parser.add_argument("--independent-test", type=Path, help="disjoint patient cohort")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    development = load_table(args.development, require_uncertainty=not args.independent_test)
    test = (
        load_table(args.independent_test, require_uncertainty=False)
        if args.independent_test else None
    )
    results = {
        "mode": "exploratory_patient_grouped_cv" if test is None else "independent_evaluation",
        "seed": SEED,
        "review_budget": BUDGET,
        "bootstrap_resamples": args.bootstrap,
        "input_sha256": {
            "development": hashlib.sha256(args.development.read_bytes()).hexdigest(),
            "independent_test": hashlib.sha256(args.independent_test.read_bytes()).hexdigest()
            if args.independent_test else None,
        },
        "methods": evaluate(development, args.bootstrap, test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
