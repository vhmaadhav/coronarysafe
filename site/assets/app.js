(() => {
  const SEGS = { ffr_unet: "FFR-UNet", cas_net: "CAS-Net", swin_unetr: "Swin UNETR", nnunet_mv: "nnU-Net (vote)" };
  const C = { model: "#1d4ed8", rule: "#d97706", oracle: "#9aa0aa", crit: "#b91c1c", ok: "#9aa3b2", ink: "#14171f", line: "#e4e1da", muted: "#7a808c" };
  const NS = "http://www.w3.org/2000/svg";
  const tip = document.createElement("div"); tip.className = "tip"; document.body.appendChild(tip);
  const showTip = (e, html) => { tip.innerHTML = html; tip.style.opacity = 1; tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px"; };
  const hideTip = () => { tip.style.opacity = 0; };
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  };
  const text = (parent, x, y, s, attrs = {}) => { const t = el("text", { x, y, "font-size": 12, fill: C.muted, ...attrs }, parent); t.textContent = s; return t; };

  // caught critical units among the top-k by score, ties at the cut-off shared fractionally
  function caught(rows, key, k) {
    const s = rows.map(r => [r[key], r.critical]).sort((a, b) => b[0] - a[0]);
    if (k <= 0) return 0;
    if (k >= s.length) return s.reduce((a, r) => a + r[1], 0);
    const t = s[k - 1][0];
    let gt = 0, gtC = 0, eq = 0, eqC = 0;
    for (const [v, c] of s) { if (v > t) { gt++; gtC += c; } else if (v === t) { eq++; eqC += c; } }
    return gtC + (k - gt) / eq * eqC;
  }

  function switcher(host, options, current, onPick) {
    host.innerHTML = "";
    for (const [k, label] of options) {
      const b = document.createElement("button"); b.textContent = label; if (k === current) b.className = "on";
      b.onclick = () => { [...host.children].forEach(x => x.className = ""); b.className = "on"; onPick(k); };
      host.appendChild(b);
    }
  }

  fetch("data/units.json").then(r => r.json()).then(D => {
    const rows = D.rows.map(r => Object.fromEntries(D.cols.map((c, i) => [c, r[i]])));
    const by = m => rows.filter(r => r.method === m);

    // ---------- Dice plot ----------
    const dHost = document.getElementById("dice-plot");
    function drawDice(m) {
      const R = by(m);
      const k = Math.floor(0.2 * R.length + 1e-9);
      const sorted = [...R].sort((a, b) => b.score - a.score);
      const flagged = new Set(sorted.slice(0, k).map(r => r.scan));
      const W = 640, H = 330, L = 46, Rr = 16, T = 18, B = 44;
      const xmin = Math.min(0.6, Math.floor(Math.min(...R.map(r => r.dice)) * 20) / 20), xmax = 1.0;
      const X = v => L + (v - xmin) / (xmax - xmin) * (W - L - Rr);
      dHost.innerHTML = "";
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, dHost);
      const bands = [["Critical failure", 1, T + 40], ["Not critical", 0, T + 190]];
      for (let v = Math.ceil(xmin * 20) / 20; v <= xmax + 1e-9; v += 0.05) {
        el("line", { x1: X(v), x2: X(v), y1: T, y2: H - B, stroke: C.line }, svg);
        text(svg, X(v), H - B + 18, v.toFixed(2), { "text-anchor": "middle" });
      }
      el("rect", { x: X(0.9), y: T, width: X(1) - X(0.9), height: H - B - T, fill: "rgba(21,128,61,.06)" }, svg);
      text(svg, X(0.95), T + 14, "Dice ≥ 0.90 looks fine", { "text-anchor": "middle", fill: "#15803d", "font-weight": 600 });
      text(svg, (L + W - Rr) / 2, H - 6, "Dice overlap with the reference", { "text-anchor": "middle", fill: C.ink, "font-weight": 600 });
      let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      for (const [label, c, y0] of bands) {
        text(svg, L, y0 - 14, label, { fill: c ? C.crit : C.muted, "font-weight": 700 });
        for (const r of R.filter(r => r.critical === c)) {
          const y = y0 + 8 + rnd() * 92;
          const f = flagged.has(r.scan);
          const dot = el("circle", { cx: X(Math.max(r.dice, xmin)), cy: y, r: 5.2, fill: c ? C.crit : C.ok, "fill-opacity": c ? .85 : .55, stroke: f ? C.ink : "none", "stroke-width": f ? 1.6 : 0 }, svg);
          dot.addEventListener("mousemove", e => showTip(e, `Patient ${r.scan} · Dice ${r.dice.toFixed(3)}<br>${c ? "critical failure" : "not critical"} · ${f ? "flagged for review" : "not flagged"}`));
          dot.addEventListener("mouseleave", hideTip);
        }
      }
      const hi = R.filter(r => r.critical && r.dice >= 0.9).length;
      document.getElementById("dice-caption").textContent =
        `${SEGS[m]}: ${R.filter(r => r.critical).length} critical failures among 160 test patients, ${hi} of them at Dice ≥ 0.90. Dark outline = flagged by CoronarySafe at the 20% budget.`;
    }
    switcher(document.getElementById("dice-switch"), [["cas_net", "CAS-Net"], ["ffr_unet", "FFR-UNet"], ["swin_unetr", "Swin UNETR"]], "cas_net", drawDice);
    drawDice("cas_net");

    // ---------- Budget explorer ----------
    let bm = "ffr_unet";
    const bHost = document.getElementById("budget-plot");
    const slider = document.getElementById("budget-range");
    function drawBudget() {
      const R = by(bm), n = R.length, P = R.filter(r => r.critical).length;
      const b = +slider.value / 100, k = Math.floor(b * n + 1e-9);
      const cm = caught(R, "score", k), cr = caught(R, "rule", k), co = Math.min(P, k);
      document.getElementById("budget-val").textContent = slider.value + "%";
      document.getElementById("budget-k").textContent = k;
      document.getElementById("budget-kpis").innerHTML =
        `<div class="kpi model"><b>${fmt(cm)}</b><span>caught by CoronarySafe</span></div>` +
        `<div class="kpi rule"><b>${fmt(cr)}</b><span>caught by the simple rule</span></div>` +
        `<div class="kpi"><b>${co}</b><span>best possible (of ${P} failures)</span></div>`;
      const W = 640, H = 360, L = 46, Rr = 18, T = 16, B = 44;
      const X = v => L + v * (W - L - Rr), Y = v => H - B - v / P * (H - B - T);
      bHost.innerHTML = "";
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, bHost);
      for (let i = 0; i <= 5; i++) {
        const v = Math.round(P * i / 5);
        el("line", { x1: L, x2: W - Rr, y1: Y(v), y2: Y(v), stroke: C.line }, svg);
        text(svg, L - 8, Y(v) + 4, v, { "text-anchor": "end" });
      }
      for (let p = 0; p <= 100; p += 20) text(svg, X(p / 100), H - B + 18, p + "%", { "text-anchor": "middle" });
      text(svg, (L + W) / 2, H - 6, "Share of scans reviewed", { "text-anchor": "middle", fill: C.ink, "font-weight": 600 });
      const line = (fn, color, dash) => {
        let d = ""; for (let kk = 0; kk <= n; kk++) d += (kk ? "L" : "M") + X(kk / n).toFixed(1) + "," + Y(fn(kk)).toFixed(1);
        el("path", { d, fill: "none", stroke: color, "stroke-width": 2.4, "stroke-dasharray": dash || "" }, svg);
      };
      line(kk => Math.min(P, kk), C.oracle, "5 5");
      line(kk => P * kk / n, "#cfd3da", "2 4");
      line(kk => caught(R, "rule", kk), C.rule);
      line(kk => caught(R, "score", kk), C.model);
      el("line", { x1: X(b), x2: X(b), y1: T, y2: H - B, stroke: C.ink, "stroke-width": 1.2 }, svg);
      el("circle", { cx: X(b), cy: Y(cm), r: 5.5, fill: C.model }, svg);
      el("circle", { cx: X(b), cy: Y(cr), r: 5.5, fill: C.rule }, svg);
      const lg = [["CoronarySafe", C.model], ["Simple rule", C.rule], ["Best possible", C.oracle], ["Random", "#cfd3da"]];
      lg.forEach(([s, c], i) => { el("rect", { x: W - Rr - 128, y: H - B - 82 + i * 18, width: 12, height: 3, fill: c }, svg); text(svg, W - Rr - 110, H - B - 77 + i * 18, s, { fill: C.ink }); });
    }
    const fmt = v => Math.abs(v - Math.round(v)) < 0.05 ? Math.round(v) : v.toFixed(1);
    slider.addEventListener("input", drawBudget);
    switcher(document.getElementById("budget-switch"), [["ffr_unet", "FFR-UNet"], ["cas_net", "CAS-Net"], ["swin_unetr", "Swin UNETR"], ["nnunet_mv", "nnU-Net vote"]], bm, m => { bm = m; drawBudget(); });
    drawBudget();
  });

  // ---------- Forest plot ----------
  (function forest() {
    const F = [
      ["FFR-UNet (primary, S2)", .125, -.003, .239, "sealed"],
      ["CAS-Net", .102, -.034, .225, "sealed"],
      ["Swin UNETR", .163, .032, .318, "sealed"],
      ["5 nnU-Net models (mean)", .114, .030, .207, "replication"],
      ["Noisier CT (+50%)", .119, -.023, .275, "shift"],
      ["Lower contrast (×0.75)", .078, -.079, .195, "shift"],
      ["Blurred CT (0.6 mm)", .020, -.135, .131, "shift"],
    ];
    const host = document.getElementById("forest");
    const W = 640, rowH = 40, T = 26, B = 46, L = 190, Rr = 24, H = T + B + rowH * F.length;
    const xmin = -.16, xmax = .34, X = v => L + (v - xmin) / (xmax - xmin) * (W - L - Rr);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, host);
    for (let v = -.15; v <= .3001; v += .05) {
      el("line", { x1: X(v), x2: X(v), y1: T - 6, y2: H - B, stroke: Math.abs(v) < 1e-9 ? "#9aa0aa" : C.line }, svg);
      text(svg, X(v), H - B + 18, (v > 0 ? "+" : "") + Math.round(v * 100), { "text-anchor": "middle" });
    }
    el("line", { x1: X(.1), x2: X(.1), y1: T - 10, y2: H - B, stroke: C.crit, "stroke-dasharray": "5 4" }, svg);
    text(svg, X(.1), T - 14, "+10 pp bar", { "text-anchor": "middle", fill: C.crit, "font-weight": 600 });
    text(svg, (L + W) / 2, H - 8, "Extra critical failures caught vs the rule (percentage points)", { "text-anchor": "middle", fill: C.ink, "font-weight": 600 });
    const col = { sealed: C.model, replication: "#0f766e", shift: "#7c3aed" };
    F.forEach(([name, est, lo, hi, kind], i) => {
      const y = T + rowH * i + rowH / 2;
      if (i === 3 || i === 4) el("line", { x1: 0, x2: W - Rr, y1: y - rowH / 2, y2: y - rowH / 2, stroke: C.line, "stroke-dasharray": "2 3" }, svg);
      text(svg, L - 12, y + 4, name, { "text-anchor": "end", fill: C.ink, "font-weight": i === 0 ? 700 : 500, "font-size": 12.5 });
      el("line", { x1: X(lo), x2: X(hi), y1: y, y2: y, stroke: col[kind], "stroke-width": 2.4, "stroke-linecap": "round" }, svg);
      const d = el("circle", { cx: X(est), cy: y, r: 6, fill: col[kind] }, svg);
      d.addEventListener("mousemove", e => showTip(e, `${name}<br>${(est * 100).toFixed(1)} pp [${(lo * 100).toFixed(1)}, ${(hi * 100).toFixed(1)}]`));
      d.addEventListener("mouseleave", hideTip);
    });
  })();

  document.getElementById("copy-bib").addEventListener("click", e => {
    navigator.clipboard.writeText(document.getElementById("bib").textContent).then(() => { e.target.textContent = "Copied"; setTimeout(() => e.target.textContent = "Copy BibTeX", 1500); });
  });
})();
