"""Shared helpers for the six stage reports (stages/stageN_*/make_stageN.py).

Every number printed in a stage report is read here from a result file and written into a generated macro or table
(stages/<stage>/tables/*.tex); nothing is typed by hand. Intervals come only from the regenerated runs
(results/rerun_2026-09-28/, crossed seed x image bootstrap). Missing results render as \\na and are listed by the
build log.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "results" / "rerun_2026-09-28"
OLD = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
MISSING: list[str] = []

EXP_ALIGN = "p{3.1cm}p{2.1cm}p{2.4cm}p{3.2cm}p{3.3cm}"  # experiments tables (registration status)

# colours shared with preamble.tex
ROI_GREEN, ART_RED, MASK_BLUE, ERM_GREY = "#00a000", "#dc0000", "#1f77b4", "#6e6e6e"
ARM_COLOURS = {"erm": ERM_GREY, "mask": MASK_BLUE, "inpaint": "#9467bd", "balanced": "#2ca02c", "dfr": "#8c564b",
               "leace": "#e377c2", "groupdro": "#bcbd22", "umte": "#ff7f0e", "umte_balanced": "#17becf"}


# ------------------------------------------------------------------------------------------------ formatting
def f3(x) -> str:
    return "\\na" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def s3(x) -> str:
    return "\\na" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:+.3f}$"


def ci(e, lo, hi) -> str:
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (e, lo, hi)):
        return "\\na"
    return f"${e:+.3f}$ [${lo:+.3f}, {hi:+.3f}$]"


def ci_row(r, est="seed_delta_mean") -> str:
    if r is None:
        return "\\na"
    return ci(float(r[est]), float(r["ci95_lo"]), float(r["ci95_hi"]))


def verdict(r) -> str:
    if r is None:
        return "\\na"
    lo, hi = float(r["ci95_lo"]), float(r["ci95_hi"])
    return "excludes 0" if (lo > 0 or hi < 0) else "includes 0"


def tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


# ------------------------------------------------------------------------------------------------ readers
def _p(rel: str, root: Path = NEW) -> Path:
    return root / rel


def csv(rel: str, root: Path = NEW):
    f = _p(rel, root)
    if not f.exists():
        MISSING.append(str(f.relative_to(ROOT)))
        return None
    return pd.read_csv(f)


def js(rel: str, root: Path = NEW):
    f = _p(rel, root)
    if not f.exists():
        MISSING.append(str(f.relative_to(ROOT)))
        return None
    return json.loads(f.read_text())


def row(df, **kw):
    if df is None:
        return None
    q = df
    for k, v in kw.items():
        if k not in q.columns:
            return None
        q = q[np.isclose(q[k].astype(float), float(v))] if isinstance(v, float) else q[q[k].astype(str) == str(v)]
    return None if q.empty else q.iloc[0]


def syn_dir(cohort, backbone, artifact, phase, tag):
    return f"synthetic/{cohort}/{backbone}_{artifact}_{phase}_{tag}"


def syn_boot(d, method, ov, env="test_rev"):
    return row(csv(f"{d}/bootstrap_vs_erm.csv"), method_a=method, overlap=float(ov), env=env)


def syn_inter(d, method):
    return row(csv(f"{d}/location_interaction.csv"), method=method)


def syn_auc(d, method, ov, env):
    r = row(csv(f"{d}/summary_auc.csv"), method=method, overlap=float(ov), env=env)
    return None if r is None else float(r["auc"])


def trap_boot(d, arm, trap, env="test_rev"):
    return row(csv(f"{d}/bootstrap_vs_erm.csv"), arm=arm, trap=trap, env=env)


def per_seed_auc(d, arm, trap, env):
    m = csv(f"{d}/metrics_per_seed.csv")
    if m is None:
        return None
    q = m[(m.method == arm) & (m.trap == trap) & (m.env == env)]
    return None if q.empty else float(q.auc.mean())


# ------------------------------------------------------------------------------------------------ registrations
def reg(doc: str) -> str:
    """First commit of a registration document: '\\texttt{hash}, date time' (local commit time, not independent proof)."""
    import subprocess
    out = subprocess.run(["git", "-C", str(ROOT), "log", "--diff-filter=A", "--format=%h|%ad",
                          "--date=format:%Y-%m-%d %H:%M", "--", doc], capture_output=True, text=True).stdout.split("\n")
    out = [o for o in out if o.strip()]
    if not out:
        MISSING.append(doc)
        return "\\na"
    h, d = out[-1].split("|")
    return f"\\texttt{{{h}}}, {d}"


def head_commit() -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------------------------------------ writers
class Macros:
    """\\newcommand macros; names are letters only (digits spelled out)."""

    def __init__(self, prefix: str):
        self.prefix, self.items = prefix, {}

    @staticmethod
    def clean(name: str) -> str:
        words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
                 "8": "Eight", "9": "Nine"}
        return "".join(words.get(ch, ch) for ch in re.sub(r"[^A-Za-z0-9]", "", name))

    def add(self, name: str, value) -> str:
        n = self.prefix + self.clean(name)
        self.items[n] = "\\na" if value is None else str(value)
        return n

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("% generated by make_stage*.py from result files; do not edit\n" +
                        "\n".join(f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in self.items.items()) + "\n")


def table(path: Path, headers, rows, caption, label, align=None, size="\\small", note=None, resize=False, long=False):
    align = align or "l" + "c" * (len(headers) - 1)
    if long:  # page-breaking table for long ledgers
        head = " & ".join(headers) + " \\\\\\midrule"
        L = ["{" + size, f"\\begin{{longtable}}{{{align}}}", f"\\caption{{{caption}}}\\label{{{label}}}\\\\\\toprule",
             head, "\\endfirsthead", "\\toprule " + head, "\\endhead", "\\bottomrule", "\\endlastfoot"]
        L += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
        L += ["\\end{longtable}}"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(L) + "\n")
        return
    L = ["\\begin{table}[H]\\centering" + size, f"\\caption{{{caption}}}\\label{{{label}}}"]
    open_, close = ("\\resizebox{\\linewidth}{!}{", "}") if resize else ("", "")
    L += [open_ + f"\\begin{{tabular}}{{{align}}}\\toprule", " & ".join(headers) + " \\\\\\midrule"]
    L += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    L += ["\\bottomrule\\end{tabular}" + close]
    if note:
        L.append(f"\\par\\smallskip{{\\footnotesize {note}}}")
    L.append("\\end{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
                         "pdf.fonttype": 42})
    return plt


def report_missing(stage: str):
    if MISSING:
        print(f"[{stage}] missing result files ({len(set(MISSING))}):")
        for m in sorted(set(MISSING)):
            print("   ", m)


# ------------------------------------------------------------------------------------------------ estimator check
REMEDY_KEYS = r"(?:^|[|_])(?:u?mte|u?i2e|splice|jtt|protect|pbal|aug|cons|post|text|prevcal|lama|auto|insert)"


def estimator_check(prefixes, path: Path, label: str, caption: str, include_remedies: bool = False, extra=None,
                    changes_path: Path | None = None, changes_label: str = "", changes_caption: str = ""):
    """Per-seed (pilot) versus crossed bootstrap intervals for the analyses of one stage.

    Reads results/bootstrap_correction/sweeps_umte_old_vs_new.csv (both estimators on the same predictions), keeps the
    rows whose result file starts with one of `prefixes` (and, unless include_remedies, drops remedy arms), writes a
    summary table and returns summary numbers. Only counts, ratios and verdict changes are printed; no per-seed interval
    is reproduced."""
    f = ROOT / "results" / "bootstrap_correction" / "sweeps_umte_old_vs_new.csv"
    if not f.exists():
        MISSING.append(str(f.relative_to(ROOT)))
        return None
    d = pd.read_csv(f)
    keep = np.zeros(len(d), bool)
    for p in prefixes:
        keep |= d.file.str.startswith(p).to_numpy()
    d = d[keep].copy()
    if extra is not None:
        d = d[extra(d)]
    if not include_remedies:
        d = d[~d.key.str.contains(REMEDY_KEYS, regex=True)]
    d["group"] = d.file.str.split("/").str[:2].str.join("/").str.replace(".csv", "", regex=False).str.replace(".json", "", regex=False)
    rows = []
    for g, q in d.groupby("group", sort=True):
        rows.append([tex_escape(g).replace("/", "/\\allowbreak{}"), len(q), int((q.old_excludes_0 & ~q.new_excludes_0).sum()),
                     int((~q.old_excludes_0 & q.new_excludes_0).sum()), f"{q.width_ratio.median():.2f}"])
    rows.append(["\\textbf{all}", len(d), int((d.old_excludes_0 & ~d.new_excludes_0).sum()),
                 int((~d.old_excludes_0 & d.new_excludes_0).sum()), f"{d.width_ratio.median():.2f}"])
    table(path, ["Result group", "Intervals", "Excluded 0 only with the per-seed estimator", "Excluded 0 only with the crossed estimator",
                 "Median width ratio (crossed / per-seed)"], rows, caption, label,
          align="p{5.0cm}p{1.4cm}p{2.6cm}p{2.6cm}p{2.4cm}", size="\\scriptsize")
    if changes_path is not None:
        ch = d[d.verdict_changed].copy()
        crows = [[tex_escape(r.file.rsplit("/", 1)[0]).replace("/", "/\\allowbreak{}"),
                  tex_escape(r.key.replace("|", " / ")).replace("/", "/\\allowbreak{}"),
                  ci(r.new_est, r.new_lo, r.new_hi), "no longer excludes 0" if r.old_excludes_0 else "now excludes 0"]
                 for r in ch.itertuples()]
        table(changes_path, ["Result", "Contrast", "Crossed estimate [95\\% CI]", "Verdict vs per-seed estimator"],
              crows or [["none", "", "", ""]], changes_caption, changes_label, align="p{4.2cm}p{5.2cm}p{3.3cm}p{2.6cm}",
              size="\\scriptsize")
    return {"n": len(d), "lost": int((d.old_excludes_0 & ~d.new_excludes_0).sum()),
            "gained": int((~d.old_excludes_0 & d.new_excludes_0).sum()), "median": float(d.width_ratio.median()),
            "q25": float(d.width_ratio.quantile(.25)), "q75": float(d.width_ratio.quantile(.75)), "df": d}


# ------------------------------------------------------------------------------------------------ derived summaries
DERIVED = ROOT / "results" / "stage_derived"


def derived(name: str, df: "pd.DataFrame") -> "pd.DataFrame":
    """Write a summary computed by a stage generator from existing result files (means over seeds, point estimates of
    runs whose per-image predictions were not saved) to results/stage_derived/<name>.csv, so that every number a report
    prints can be traced to a file. No interval is ever written here."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DERIVED / f"{name}.csv", index=False)
    return df


def auc_means(d: str, root: Path = NEW, method_col: str = "method"):
    m = csv(f"{d}/metrics_per_seed.csv", root)
    if m is None:
        return None
    keys = [k for k in ("trap", method_col, "env") if k in m.columns]
    return m.groupby(keys).auc.mean().round(3).reset_index()
