# CoronarySafe dated addenda — FROZEN 2026-11-15

Frozen 2026-11-15 (~17:40 IST), after the sealed primary test. S1 PASS (0.906), S2 FAIL (+12.5 pp,
CI [−0.3, +23.9]), S3 PASS (+0.41) are FIXED; nothing in this document changes any verdict, and no
addendum may be used to change one. The DRAFT sections of docs/ANALYSIS_PLAN_v1.md (lines 119–200,
2026-11-15) are SUPERSEDED by this document and kept only for provenance.

Approval record (session 2026-11-15, user Maadhav, explicit):
- D1: Addenda L and M frozen as in this document.
- D4: ONE canonical sealed read approved. Scope = Addendum M (m0–m6) + Addendum L TEST (480 units)
  + Addendum J + Addendum K evaluation. This is the single sealed-test computation; no further
  sealed-test computation is allowed without a new dated addendum + new explicit user approval.
- D3: Addendum K segmenter = CAS-Net (recorded here, before any perturbed-CT label is computed,
  as the K draft requires).
- D5: Addendum N (rule stability + feature ablation), val-only, approved.
- Paper policy: (a) reader study = in silico design + description section only; (b) LLM use
  disclosed, aligned with MIDL 2027 CT&V policy; (c) FFR-UNet provenance text = "checkpoint
  provided with the dataset release; no public primary source located" (verified searches found
  none) unless a citation is supplied.

Frozen invariants (all addenda): frozen model M1+M3+M4 (sha256 per Addendum H: frozen_model.json
56a666ef…, m4_model.pt baf841cf…, lr_convention.json 53ea24bf…), rule g_n_components_ge_10mm,
NR@20% (ties averaged), patient-block bootstrap 2000 (sealed S2 used seed 14 per pre-registration;
new addenda use seed 0), frozen v3 labeler + taxonomy_v3, data, splits.

## Addendum L (FROZEN): independent-lumen burden — feature-disjoint label-robustness

As drafted (ANALYSIS_PLAN_v1.md, L section), with ONE amendment:
- **Units, test: 480 = {cas_net, swin_unetr, ffr_unet} × 160** (draft said FFR-UNet only).
  Approved in D4. For all three segmenters the frozen scores were already computed in the
  Addendum H evaluation (S1 primary + pre-registered LOSO secondary, plan line 9); the canonical
  read READS those scores and never recomputes them. If any per-unit score file is missing, the
  score is computed once inside the canonical read from the same frozen code path (deterministic;
  output hash logged) — never by model re-fitting.
- Label, controls C1/C2, decision rule (i)/(ii)/(iii), exclusions, budget 0.2, B_alt5 sensitivity,
  and the construction disjointness argument: frozen as drafted. Probe-AUC (features→B_alt)
  framing is FORBIDDEN (score = f(features); self-referential).
- Val: Phase 1 = 160 dev-scope units (cas_net, swin_unetr; frozen scores exist); Phase 2 = 480
  units {ffr_unet, nnunet_f0..f4} × 80 via the valscore extension (CPU; m4_model.pt sha256-verified
  against frozen_model.json before first use). Val runs are free (no sealed data).

## Addendum M (FROZEN): S2 estimator diagnostics and exact power (sealed outputs only)

As drafted (m1–m5), extended as follows. All on the sealed FFR-UNet 160 test units, one canonical
read. FREEZE-ONE assumption to label in the paper: the pre-run derivation pinned the realized
2×2 discordance table to lattice A (see m0); every estimator below is applied first to the
REALIZED table, and grid rows B/C/D are reported only if the realized table matches that cell.

- **m0 (new): the realized 2×2 discordance table.** For critical patients, with S = top-20% flagged
  by the frozen model and R = top-20% flagged by the rule: a = |S∩R∩C|, b = |S-only criticals|,
  c = |R-only criticals|, Kc = |C|, d = b−c… (d = b + c is the discordant count; net D = b − c).
  imp = D/Kc. Expected (pre-run derivation): lattice A = (Kc=32, a=4, b=4, c=0), imp = +12.5 pp —
  the only lattice reproducing the posted +12.5 pp AND CI [−0.3, +23.9] (the paired-delta half-width
  12.1 pp = 1.96·√(160·s²)/32 is reproduced exactly only under A). Pre-specified grid cells in case
  the realized table differs: B = (Kc=32, a=3, b=5, c=1), C = (Kc=40, a=5, b=5, c=0),
  D = (Kc=24, a=3, b=3, c=0); all give imp = +12.5 pp. Also report |R∩C| vs its conditional
  hypergeometric expectation under S-uniform-false (P(a random 32-set captures ≤ 4 of the
  realized Kc criticals); under A that is 17.5%, E[imp|null] = +7.5 pp) — this is the
  realized-quota confound, reported as fact.
- **m1:** the identical frozen estimator at 20,000 resamples (Monte-Carlo stability of the −0.3
  boundary; seed 14).
- **m2:** BCa and studentized-t CIs of the same statistic.
- **m3 (extended): the rare-event exact family, all conditioned on the realized d = b + c.**
  (i) one-sided 97.5% lower bound and one-sided p of (model − rule); (ii) exact binomial
  (Clopper–Pearson / tail-inverted) CI and two-sided p at π0 = 1/2, plus mid-p p and mid-p CI;
  (iii) Wilson score CI on (b|d), transformed to the imp scale; (iv) posterior intervals under
  Jeffreys Beta(1/2,1/2) and uniform Beta(1,1); (v) randomization inference: E1 sharp null
  (S uniformly random 32-set, R fixed as realized) two-sided and one-sided-upper p, and E2
  (both S and R random) two-sided p; (vi) algebraic-identity check recording for both the
  vendor paired-delta and Wilson that the clearance condition on this lattice is D² ≳ c·M with
  c = 3.7546 (vendor) vs 3.8416 (Wilson) — they are the same test up to a 2.3% constant.
- **m4:** AURC difference (model − rule) and Spearman rank difference, each with paired patient
  bootstrap CI (as drafted).
- **m5 (extended): exact power and sample size (exhaustive lattice enumeration, no Monte-Carlo).**
  (i) null size of each m3 family under the E1 sharp null; (ii) clear probability of each m3
  family under the most favorable true effect consistent with the design (system truly targets the
  4 S-only criticals; E[imp] = 15.71 pp, above the 12.5 pp target); (iii) Kc needed for 80% power
  at 12.5 pp at discordant-capture 75/80/90/95% (conditional-binomial model, one-sided 97.5%);
  (iv) design-vs-sample SE: E1 null SD(imp) and E2 null SD(imp) vs the realized paired SE —
  documenting the paired-delta formula's under-coverage of the fully-random design variance;
  (v) the drafted nonparametric power curve as a function of budget (10/20/30/50%).
- **m6 (new; descriptive; future-proposal only):** conformal-risk estimand — the realized
  empirical critical-miss rate among the unflagged 80% at the frozen score threshold, and the
  hypergeometric rank-uniformity check (calibration against random ranks). Reported as an
  estimand candidate for a FUTURE pre-registration (conformal risk control lineage), never as a
  superiority test, and incapable of changing the verdict.
- **One-sidedness note (frozen fact):** the pre-registered bar ("two-sided 95% CI lower bound > 0")
  is the one-sided α = 0.025 test. A one-sided 95% reading (bound ≈ +12.5 − 1.645·SE ≈ +2.3 pp
  under A) is NOT admissible: the bar is fixed; only the estimator family may be widened.
- **Percentile-family clearance reporting rule:** resampling-percentile intervals (customer-
  multinomial or Kc-stratified) may show lo > 0 on this lattice; they are reported ONLY as
  secondary, with the proof that positivity is the lattice gate (c = 0 ∧ b ≥ 4: atom mass
  P(stat = 0) = 1.74% / 1.39% < 2.5% under A), their null size (0.11%), and their failure in
  grid cells B and D. They never enter verdict admissibility.
- **Unpaired diagnostics (reported, not verdict carriers):** Firth (bias-corrected) and plain MLE
  logistic on the 2×2, OR + CI on the imp scale; the paired design is the estimand, these only
  confirm direction.
- **Norms statement (frozen):** sharpening a CI within one sealed dataset does not meet the
  2022–2026 cross-site-validation norm for deployment claims; the paper states this explicitly.
- **Verdict rule:** S2 = FAIL as posted. M is disclosure only. No m* output changes it.
- **Citations (verified 2026-11-15):** DeLong et al. 1988, Biometrics 44(3):837–845,
  doi:10.2307/2531595; DiCiccio & Efron 1996, Ann. Statist. 24(4):1483–1512, doi:10.1214/ss/1032280214;
  arXiv:2208.02814; arXiv:2505.16027; arXiv:2512.12844; arXiv:2609.30714; Smart Health 41 (2026)
  100689, doi:10.1016/j.smhl.2026.100689. Unverified (bibliographic only, no URL): Firth 1993
  JRSS-B 55(2):261–280; Zou 2008 Stat. Med.; Efron 1987 JASA 82:171–185; Agresti & Coull 1998
  Am. Stat. 52:117–120; Lefly & Edwards 2004 Stat. Med. 23:1551–1562; Newcombe 1998 JRSS-A
  161:353–367. DO NOT CITE doi:10.1093/biomet/68.1.1 (obituary, not Firth).

## Addendum J (FROZEN): frozen-model replication on additional segmenters

Frozen verbatim as drafted (ANALYSIS_PLAN_v1.md, J section), including: Family A = nnU-Net
f0–f4 as ONE family; Family B EXCLUDED (stage-1 checkpoint absent, training split unverified —
decided before any outcome); the nnunet_mv disclosure (test result 0.600 vs 0.420 already seen →
a Family A pass is partly anticipated, weak evidence); Δ_J = mean D_k with patient-block
bootstrap (2000, seed 0); success bar = same as S2 (Δ_J ≥ +10 pp ∧ CI lo > 0); REML/HK only at
K ≥ 3. Uses existing test predictions + v3 labels (GPU cost zero).

## Addendum K (FROZEN): controlled acquisition-shift stress test

Frozen as drafted (K section), with:
- **Segmenter = CAS-Net** (per D3; recorded pre-outcome).
- Perturbations unchanged: noise (σ_add = 1.118·σ0, +50% noise SD in aorta-ROI sense),
  z-blur (σ = 0.6 mm), contrast (0.75×, aorta-median anchored); applied to raw CT, then
  segmentation; the frozen v3 labeler on ImageCAS-X GT (unperturbed); frozen model + rule scores
  on perturbed reconstructions; primary = per-perturbation paired NR@20 difference D_p with
  patient-block bootstrap (2000, seed 0); no pooling; K-eval runs INSIDE the canonical read.
- **Spec-check gate for pre-existing files (mandatory before any reuse):** the perturbed volumes
  in imagecas/volumes_shift/{noise,zblur,contrast} and preds/cas_net@noise (159/160) come from an
  earlier unrecorded chain. Before use: on ≥ 8 fixed random scans, verify (shape/affine identity
  vs the source CT) and, for noise, the aorta-ROI noise SD ratio perturbed/paired ≈ 1.12 ± 0.10.
  Gate PASS → reuse the 159 (regenerate only the missing scan); FAIL → regenerate all affected
  volumes + 480 inferences, keeping the old files as a LABELLED deviation. No selection uses
  these files. K-eval uses the 480 perturbed reconstructions (3 × 160).
- GPU budget: 480 CAS-Net inferences (prior estimate 2.7–8 GPU-h); single droplet-side chain
  script; flock gpu.lock; leave headroom for redline's vLLM process.

## Addendum N (FROZEN): rule stability + frozen-model feature ablation (val-only)

No sealed test; dev/val scope; CPU.
- **N1 (rule stability).** The pre-specified simple-rule candidates (n_components ≥ 10 mm, n_gaps,
  total-length z-score, max-radius z) + g_share_top2 (labelled EXPLORATORY): recompute each rule's
  val NR@20 and AUROC with its numeric threshold perturbed ×0.8 and ×1.2 (z-score rules: threshold
  shifted ±20%). No re-selection, no ranking change; report the 3-column table per rule.
- **N2 (feature ablation).** For each of the frozen model's 22 inputs, re-fit the identical L2
  logistic (C = 0.1, same median-imputation + z-score recipe on the 21 remaining features) on the
  160 dev units; report val NR@20, AURC, AUROC(critical) vs the frozen 22-input model. This is a
  post-hoc diagnostic on the FITTING set only (honest label: in-sample ablation), reported to
  explain which inputs carry the dev signal; it is not evidence about test.
- Outputs feed the paper's threat-register row D and one figure; they cannot change S1–S5.

## The ONE canonical sealed read (single batch)

- Location: /root/data/addenda/SEALED_READ_20261115/. Pre-launch MANIFEST written first:
  sha256 of the three frozen files; counts + directory listings of every sealed input consumed.
- Components, in one launch: M (m0–m6) + L-test (480 units: B_alt + all L statistics, per
  segmenter + pooled) + J (per-fold D_f, Δ_J + CI, disclosure table) + K-eval (3 × 160 CAS-Net
  perturbed units: v3 labels, frozen scores, D_p + CI).
- Launch only AFTER K GPU inference completes. One RUNLOG entry: UTC time, PIDs, output tree,
  manifest path. AFTER this batch: zero further sealed-test computation (discipline D4).
- Insurance: outputs are small (CSV/JSON/MD); rsync to laptop AND rely on the droplet's
  hf_backup.sh loop (private HF dataset vhmaadhav/coronarysafe-results) — verify both after
  collection.

## Compute plan and deadline

- Droplet credit/death: 2026-11-16 (credit expiry; ~14:00 UTC per account rules). ALL launches
  happen 2026-11-15; collection by ~2026-11-16 06:00 UTC.
- Wave 1 (immediately after this freeze; val/free): L Phase 1 (160) → L Phase 2 valscore (480,
  CPU M4) → N1 + N2 (val, minutes). One CPU chain, 8–16 workers.
- Wave 2 (immediately; GPU): K spec-check (CPU minutes) → perturbed-volume completion → 480
  CAS-Net inferences in ONE chain script under flock.
- Wave 3 (after Wave 2 finishes; CPU ~0.5–2 h): the canonical sealed read (single launch), then
  collection + rsync.
- Paper deadline: MIDL 2027 CT&V, abstract 2026-11-30 AoE, full 2026-12-04 AoE (venue research:
  docs/venue-research-MIDL2027.md).

## ERRATA (2026-11-15 ~13:20 UTC, coordinator review; before ANY L/M/N/J/K output was computed)

Approved by Maadhav in chat ("ok do whats required"). Nothing below changes any verdict, the frozen model, the rule, the labels or the split.

1. **Addendum M: m0 lattice derivation WITHDRAWN.** FFR-UNet test prevalence is 0.400, so P = 64 critical units, not 32. The budget is K = 32, so oracle recall = 0.5. NR 0.906 means C_model ≈ 29 and NR 0.781 means C_rule ≈ 25. Lattice A (Kc = 32, a = 4) is therefore impossible: under b − c = 4 the both-caught cell a is about 25. All quantities derived from that lattice are void: the grid cells B/C/D, the 17.5% hypergeometric figure, E[imp|null] = 7.5 pp, the 15.71 pp "favourable effect", the lattice-gate atom masses and the c-constant identity.
2. **Addendum M: reduced to a minimal, pre-declared set.** Everything not listed here is withdrawn, so the paper does not shop estimators on a failed criterion:
   - **m0:** the REALIZED 2×2 among the P critical units (a both caught, b model only, c rule only, n neither), with P, K, C_model, C_rule. No predicted table.
   - **m1:** the identical frozen estimator (paired patient-cluster percentile bootstrap) at 20,000 resamples, seed 14.
   - **m2:** ONE exact paired test: an exact McNemar (binomial on b given b + c, two-sided). Under a fixed budget, ΔNR = (b − c)/32.
   - **m4:** AURC and AUROC differences (model − rule), with the paired patient bootstrap (2000, seed 0).
   - **m5:** planning-only power. A patient-block resampling simulation at N ∈ {160, 240, 320, 480}, reporting the probability of meeting the S2 bar (≥ +10 pp AND CI lower > 0) under the observed discordance structure. Labelled optimistic, because the observed effect is used as the truth.
   - **Withdrawn:** BCa/t intervals, the m3 rare-event family (Wilson, Jeffreys, mid-p, randomization E1/E2), the percentile-family clearance rule, Firth/MLE logistic, and m6.
   - **Verdict:** S2 = FAIL. M is disclosure only.
3. **Addendum K: noise spec-check gate corrected.** σ_add = 1.118·σ0 gives a total noise SD of √(1 + 1.118²)·σ0 = 1.50·σ0. The gate is: aorta-ROI noise SD ratio, perturbed vs source = **1.50 ± 0.10** (was wrongly 1.12 ± 0.10). Blur gate: the volumes must have isotropic σ = 0.6 mm in physical units. The existing folder is named "zblur", which suggests z-only blur; if that is confirmed, those volumes FAIL and are regenerated. Contrast gate: within the aorta ROI, the median is preserved within ±5 HU and the IQR ratio is 0.75 ± 0.05.
4. **Addendum N:** stray text in N2 fixed (wording only).
5. **Deadline:** the credit EXPIRES on 2026-11-16 at an unknown time, possibly 00:00 UTC. All sealed outputs must be collected (rsync plus HF) by **2026-11-15 23:00 UTC**. Any K perturbation whose inference is not finished by 21:00 UTC is dropped and reported as not run. Perturbations are not partially reported.
6. **Paper policy (b), corrected.** The MIDL 2027 official policy prohibits LLM-generated ideas, study designs, analyses, results and substantive text. Disclosure does not make these permitted. The venue choice is pending Maadhav, for example contacting the MIDL program chairs or choosing a venue whose policy permits disclosed AI-assisted research. This is not an analysis change.
7. **Addendum L implementation pins (decided before any L output was computed; code review 2026-11-15):**
   - Dilation = Euclidean-distance ball in physical mm: a voxel is in the gate if its EDT distance to the label is ≤ PAD mm. No ceil to whole voxels.
   - Units with |orig_mj| = 0 are excluded and counted, the same as < 50.
   - Prediction and GT must match in shape AND affine (allclose, atol 0.01 mm). A unit that fails is an error and is counted.
   - Top-20% uses the frozen NR@20 convention: k = floor(0.2n), with ties at the cut-off weighted fractionally.
   - Clusters = patient ID. ImageCAS is one scan per patient; the code asserts this.
   - C2 residual regression is refitted inside every bootstrap resample.
   - Decision rule, complete: (i) ρ CI lower > 0 AND C2 CI lower > 0; (ii) ρ CI lower > 0 AND C2 CI lower ≤ 0; (iii) otherwise. That means a ρ CI straddling 0 falls into (iii), "transfer not shown", reported as a limitation.
   - The frozen-score clamp for m4_logit is p ∈ [1e-6, 1 − 1e-6], as in the frozen test_eval.py. All three frozen files are sha256-verified against Addendum H, and lr_convention comes ONLY from the frozen copy.
   - A missing feature is NOT silently imputed in valscore: the unit is flagged, and the count is reported. The frozen imputation applies only where the frozen pipeline itself imputes.
8. **Canonical read split into two one-shot reads (time pressure; decided 2026-11-15 ~13:15 UTC, before any sealed L/M/J/K output was computed).** Read 1 = Addendum M (reduced, Errata 2) + Addendum J. It runs as soon as its script passes a val dry run. Read 2 = Addendum L-test + Addendum K-eval. It runs after the K chain and the fixed L code. Each read is launched ONCE by the coordinator, with a manifest, and is never repeated. Nothing else changes.
9. **K contrast gate: criterion mis-specified, volumes kept (2026-11-15 ~13:35 UTC, before any K output).** The regenerated contrast volumes apply the FROZEN K transform exactly: HU' = m + 0.75·(HU − m), with m = the aorta-lumen median. The IQR ratio is 0.746–0.763 (PASS). The Errata-3 median check measured a different ROI (the central blood pool), whose median moves by 0.25 × (its median − m) by construction; it gave +10/−7/−9 HU on 3 of 8 scans. That is an error in the gate as written, not a spec deviation in the volumes. The contrast set is kept. The old E1 volumes FAILED all three gates; all K volumes are regenerated (volumes_shift_K, parameters in addenda/K/gen_params_*.json). The old files are untouched and unused.
10. **S2 seed corrected (2026-11-15 ~14:00 UTC, before Read 1 ran on test).** The sealed S2 was computed by m4/test_eval.py with eval.bootstrap.paired_bootstrap_difference, 2000 resamples, **seed 0** (test_eval.py: N_BOOT = 2000, SEED = 0). The statement "sealed S2 used seed 14" in this document is wrong. m1 therefore uses the same function with seed 0: a 2,000-resample check, which must reproduce the posted [−0.003, +0.239], and then 20,000 resamples.
