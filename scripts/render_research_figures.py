"""Rebuild the research figures from frozen experiment reports (no model refit)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "research-figures"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

for font in [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]:
    if Path(font).exists():
        font_manager.fontManager.addfont(font)

INK = "#202020"
MUTED = "#575757"
RULE = "#b4b4b4"
BLUE = "#285b72"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.linewidth": .7,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": .7,
    "ytick.major.width": .7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "ranking-research-figures-v1",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

QUALITY_PATH = ROOT / "reports/ebnerd-top1-feedback/final_results.json"
SERVING_PATH = ROOT / "reports/ebnerd-serving/verification.json"
quality = json.loads(QUALITY_PATH.read_text())
serving = json.loads(SERVING_PATH.read_text())
metrics = quality["metrics"]
paired = quality["paired_user_bootstrap"][
    "d3600_feedback_ndcg-minus-d3600_exposure_ndcg"
]
BASE = "d3600_exposure_ndcg"
MODEL = "d3600_feedback_ndcg"
saved_data: dict = {"source_reports": {}, "figures": {}}
for path in [QUALITY_PATH, SERVING_PATH]:
    saved_data["source_reports"][str(path.relative_to(ROOT))] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    }


def rate(model: str, cohort: str) -> float:
    return metrics[model][cohort]["hit1"] * 100


def rule(fig, y, x0=.055, x1=.955, color=RULE, width=.65):
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                          color=color, lw=width, clip_on=False))


def label(fig, x, y, text, *, size=12, weight="normal", color=INK, **kwargs):
    return fig.text(x, y, text, fontsize=size, fontweight=weight, color=color,
                    va="center", **kwargs)


def panel(fig, x, y, letter, title):
    label(fig, x, y, letter, size=15, weight="bold")
    label(fig, x + .025, y, title, size=13, weight="bold")


def save(fig, name, title, description):
    fig.canvas.draw()
    # Text is retained as text in SVG; no screenshot or rasterized chart layers.
    fig.savefig(OUT / f"{name}.svg", metadata={"Title": title, "Description": description})
    fig.savefig(OUT / f"{name}.png", dpi=220, metadata={"Title": title, "Description": description})
    plt.close(fig)


def quality_figure():
    fig = plt.figure(figsize=(13, 10.3))
    label(fig, .055, .967, "News ranking with historical feedback", size=22,
          fontfamily="Times New Roman")
    label(fig, .055, .932,
          "EB-NeRD  |  38,920 held-out users  |  618,829 requests  |  25 May–1 June 2023",
          size=11.5, color=MUTED)
    rule(fig, .910, color=INK, width=.8)

    panel(fig, .055, .880, "a", "Paired comparison across evaluation cohorts")
    label(fig, .055, .831, "Evaluation cohort", size=11, weight="bold")
    label(fig, .338, .831, "Requests", size=11, weight="bold", ha="right")
    label(fig, .414, .831, "Baseline", size=11, weight="bold", ha="right")
    label(fig, .500, .831, "+ response", size=11, weight="bold", ha="right")
    label(fig, .458, .852, "Hit@1 (%)", size=10.5, color=MUTED, ha="center")
    label(fig, .676, .831, "Difference (pp)", size=11, weight="bold", ha="center")
    label(fig, .953, .831, "Estimate [95% CI]", size=11, weight="bold", ha="right")
    rule(fig, .812)

    cohorts = [
        ("overall", "All requests"),
        ("cold_target", "Model-cold target"),
        ("adapted_cold_target", "Cold + prior exposure"),
        ("zero_signal_target", "Zero-signal target"),
    ]
    forest = fig.add_axes([.555, .615, .235, .200])
    forest.set_xlim(-1, 6.3)
    forest.set_ylim(-.5, 3.5)
    forest.set_xticks([0, 2, 4, 6])
    forest.set_yticks([])
    forest.spines["left"].set_visible(False)
    forest.axvline(0, color=RULE, lw=.8, zorder=0)
    rows = []
    for i, (cohort, name) in enumerate(cohorts):
        y = 3-i
        fy = .615 + ((y+.5)/4)*.200
        b = rate(BASE, cohort)
        f = rate(MODEL, cohort)
        ci = paired[cohort]
        estimate, lo, hi = [ci[k]*100 for k in ("estimate", "lower", "upper")]
        assert abs((f-b)-estimate) < 1e-10
        assert lo <= estimate <= hi
        label(fig, .055, fy, name, size=12)
        label(fig, .338, fy, f"{ci['requests']:,}", size=11.5, ha="right")
        label(fig, .414, fy, f"{b:.2f}", size=12, ha="right")
        label(fig, .500, fy, f"{f:.2f}", size=12, ha="right")
        forest.errorbar(estimate, y, xerr=[[estimate-lo],[hi-estimate]], fmt="o",
                        color=BLUE, ecolor=BLUE, markersize=5,
                        capsize=4, capthick=.9, elinewidth=1.1, zorder=3)
        sign = lambda v: f"{v:+.2f}".replace("-", "−")
        label(fig, .953, fy, f"{sign(estimate)} [{sign(lo)}, {sign(hi)}]",
              size=11.2, ha="right")
        rows.append({"cohort":cohort, "requests":ci["requests"], "baseline_pct":b,
                     "response_pct":f, "difference_pp":estimate,
                     "ci95_lower_pp":lo, "ci95_upper_pp":hi})
    rule(fig, .615, x1=.515)
    rule(fig, .615, x0=.810)
    label(fig, .555, .570,
          "Paired user bootstrap; 2,000 resamples. Cohorts overlap.",
          size=10.3, color=MUTED)

    panel(fig, .055, .521, "b", "Performance by prior reference exposure")
    panel(fig, .552, .521, "c", "Sensitivity to feedback availability")
    axb = fig.add_axes([.097, .262, .354, .205])
    bins = ["zero_feedback_target", "cold_1to4_exposures", "cold_5to19_exposures", "cold_20plus_exposures"]
    bx = np.arange(4)
    by_base = [rate(BASE, g) for g in bins]
    by_model = [rate(MODEL, g) for g in bins]
    axb.plot(bx, by_base, "o--", color=INK, mfc="white", mew=1,
             lw=.85, ms=5.7, label="Baseline")
    axb.plot(bx, by_model, "s-", color=BLUE, mfc=BLUE, lw=1.2,
             ms=5.2, label="+ response")
    axb.set_xlim(-.22, 3.22)
    axb.set_ylim(0, 44)
    axb.set_yticks([0,10,20,30,40])
    axb.set_ylabel("Cold-target Hit@1 (%)", labelpad=8)
    axb.set_xticks(bx, ["0", "1–4", "5–19", "≥20"])
    axb.set_xlabel("Prior reference exposures", labelpad=8)
    axb.legend(loc="upper left", bbox_to_anchor=(-.015, 1.18), ncol=2,
               fontsize=10.8, handlelength=2.3, columnspacing=1.5)
    for x,y in zip(bx,by_model):
        axb.annotate(f"{y:.2f}",(x,y),xytext=(0,8),textcoords="offset points",
                     ha="center",fontsize=10.5)
    label(fig,.097,.202,"Requests: 98,112 / 2,635 / 3,741 / 487,819",size=10.3,color=MUTED)

    axc = fig.add_axes([.594, .262, .354, .205])
    delay_models = ["d3600_feedback_ndcg", "d86400_feedback_ndcg", "dfrozen_feedback_ndcg"]
    cy = [rate(m,"overall") for m in delay_models]
    cx = np.arange(3)
    axc.vlines(cx, 0, cy, color=RULE, lw=.8)
    axc.plot(cx,cy,linestyle="none",marker="s",ms=5.5,color=BLUE)
    for x,y in zip(cx,cy):
        axc.annotate(f"{y:.2f}",(x,y),xytext=(0,9),textcoords="offset points",
                     ha="center",fontsize=11)
    axc.set_xlim(-.45,2.45)
    axc.set_ylim(0,44)
    axc.set_yticks([0,10,20,30,40])
    axc.set_ylabel("Overall Hit@1 (%)", labelpad=8)
    axc.set_xticks(cx,["1 h delay","24 h delay","Frozen"])
    axc.set_xlabel("Reference-feedback condition",labelpad=8)
    label(fig,.594,.493,"Same trained model; input state changed",size=10.8,color=MUTED)
    label(fig,.594,.202,"Frozen: no reference updates during evaluation.",size=10.3,color=MUTED)

    rule(fig,.174)
    caption = [
        ("Figure 1.  Offline ranking evaluation without article metadata.", "bold"),
        ("Baseline uses exposure statistics and fixed user histories; + response adds past reference clicks. Hit@1 measures", "normal"),
        ("selection of an observed clicked candidate. Cold-target metrics require selecting a clicked model-cold candidate.", "normal"),
        ("Model-cold items are absent from training but may have later reference exposure. Zero-signal targets have no prior", "normal"),
        ("reference exposure, global click history or personal click history. Subsets in a and b may share requests.", "normal"),
        ("Feedback is assumed available after 1 h; receipt times are unavailable. CIs are conditional on this week and reference", "normal"),
        ("stream. Panel c is not retrained for each delay. These are same-period offline results, not online CTR or revenue lift.", "normal"),
    ]
    for i,(text,weight) in enumerate(caption):
        label(fig,.055,.149-i*.0193,text,size=10.8,weight=weight,color=INK if i==0 else MUTED)
    saved_data["figures"]["figure-1"] = {
        "a":rows,
        "b":[{"cohort":g,"requests":metrics[MODEL][g]["requests"],
               "baseline_pct":b,"response_pct":f} for g,b,f in zip(bins,by_base,by_model)],
        "c":[{"model_condition":m,"overall_pct":v} for m,v in zip(delay_models,cy)],
    }
    save(fig,"figure-1-feedback-ranking","News ranking with historical feedback",
         "Cohort effects with paired 95% bootstrap intervals, performance by prior exposure, and sensitivity to delayed feedback.")


def serving_figure():
    fig = plt.figure(figsize=(13,6.45))
    label(fig,.055,.950,"Serving validation of the earlier snapshot model",size=22,
          fontfamily="Times New Roman")
    label(fig,.055,.898,"Supplementary Figure S1  |  Local Mac benchmark  |  Separate model from Figure 1",
          size=11.5,color=MUTED)
    rule(fig,.861,color=INK,width=.8)
    panel(fig,.055,.817,"a","Observed response-time quantiles")
    panel(fig,.550,.817,"b","Request accounting")
    ax = fig.add_axes([.098,.322,.352,.398])
    reports = {r["target_rps"]:r for r in serving["load_reports"]}
    marks = ["p50","p95","p99","max"]
    for rps,dx,color,marker,fill,style in [(50,-.04,INK,"o","white","--"),(100,.04,BLUE,"s",BLUE,"-")]:
        values = [reports[rps]["latency_success_ms"][k] for k in marks]
        ax.plot(np.arange(4)+dx,values,color=color,marker=marker,mfc=fill,ms=5.2,
                linestyle=style,lw=1,label=f"{rps} requests/s")
    ax.axhline(50,color=MUTED,lw=.8,linestyle=(0,(4,3)))
    ax.text(3.27,51.3,"50 ms deadline",ha="right",fontsize=10.5)
    ax.set_xlim(-.25,3.3)
    ax.set_ylim(0,56)
    ax.set_yticks([0,10,20,30,40,50])
    ax.set_xticks(np.arange(4),["p50","p95","p99","max"])
    ax.set_xlabel("Empirical response-time quantile",labelpad=8)
    ax.set_ylabel("Response time (ms)",labelpad=8)
    ax.legend(loc="upper left",bbox_to_anchor=(-.015,1.17),ncol=2,fontsize=10.8,
              handlelength=2.2,columnspacing=1.2)
    label(fig,.550,.742,"Measurement",weight="bold",size=11)
    label(fig,.794,.742,"50 req/s",weight="bold",size=11,ha="right")
    label(fig,.949,.742,"100 req/s",weight="bold",size=11,ha="right")
    rule(fig,.716,x0=.550,x1=.950)
    table=[
        ("Duration (s)","duration_seconds"),
        ("Issued requests","issued"),
        ("Successful responses","successful"),
        ("Successful within 50 ms","on_time_successful"),
        ("Generator drops","generator_dropped"),
        ("Invalid responses","invalid_responses"),
    ]
    for i,(name,key) in enumerate(table):
        y=.674-i*.058
        label(fig,.550,y,name,size=11.5)
        for rps,x in [(50,.794),(100,.949)]:
            label(fig,x,y,f"{reports[rps][key]:,}",size=11.5,ha="right")
    rule(fig,.350,x0=.550,x1=.950)
    label(fig,.550,.312,"18 / 18 serving checks passed",size=11.5)
    label(fig,.550,.277,"201 offline/API comparisons; max score error = 0",size=10.5,color=MUTED)
    rule(fig,.223)
    notes=[
        ("Figure S1.  API measurements are independent of the latest offline ranking gains.","bold"),
        ("FastAPI /rank, Redis cache and fallback, bounded concurrency, version pinning, promotion/rollback and shadow", "normal"),
        ("isolation were implemented and checked. Server and load generator ran on the same Mac; these short runs are", "normal"),
        ("not a sustained production capacity claim. The latest response model still needs a live feedback adapter and its own", "normal"),
        ("HTTP load validation before promotion. Real receipt-time logging and an online A/B experiment remain outstanding.", "normal"),
    ]
    for i,(text,weight) in enumerate(notes):
        label(fig,.055,.184-i*.029,text,size=10.8,weight=weight,color=INK if i==0 else MUTED)
    saved_data["figures"]["figure-s1"] = {"load_reports":serving["load_reports"],
                                           "checks":serving["checks"],
                                           "parity_requests":serving["parity_requests"],
                                           "maximum_offline_http_score_error":serving["maximum_offline_http_score_error"]}
    save(fig,"figure-s1-serving","Serving validation of the earlier snapshot model",
         "Response-time quantiles and request accounting at 50 and 100 requests per second, for the previous serving model.")


if __name__ == "__main__":
    quality_figure()
    serving_figure()
    saved_data["design_references"] = [
        "https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/",
        "https://distill.pub/2020/communicating-with-interactive-articles/",
    ]
    (OUT / "figure-data.json").write_text(json.dumps(saved_data,indent=2)+"\n")
    print(f"Rendered 2 figures as PNG and editable SVG in {OUT}")
