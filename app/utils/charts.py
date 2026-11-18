"""Plotly chart builders. Conventions: one fixed colour per model, thin marks, recessive grid (from the Streamlit
theme), legend whenever there are two or more series, values reachable by hover and by a table under each chart."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .loaders import MODEL_ORDER
from .style import (METRIC_LABELS, MODEL_DASH, MODEL_SYMBOL, STATUS, diverging_scale, model_colors, muted, series,
                    sequential_scale)

FONT = dict(size=13)


def layout(fig, xtitle=None, ytitle=None, height=380, legend=True, xpct=False, ypct=False, xrange=None, yrange=None,
           hovermode="closest", legend_bottom=False):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36 if legend and not legend_bottom else 12, b=8), font=FONT,
        showlegend=legend, hovermode=hovermode, bargap=0.28, bargroupgap=0.06,
        legend=dict(orientation="h", x=0, xanchor="left", title_text="",
                    **(dict(y=-0.25, yanchor="top") if legend_bottom else dict(y=1.02, yanchor="bottom"))),
        hoverlabel=dict(font_size=13))
    fig.update_xaxes(title_text=xtitle, automargin=True, range=xrange, tickformat=".0%" if xpct else None)
    fig.update_yaxes(title_text=ytitle, automargin=True, range=yrange, tickformat=".0%" if ypct else None)
    return fig


# ---------------------------------------------------------------- default-rate bars (EDA)

def rate_bars(df, label="group", horizontal=True, overall=None, color=None, height=None, xtitle="Default rate"):
    color = color or series(1)
    text = [f"{r:.1%}" for r in df["default_rate"]]
    custom = np.stack([df["applicants"], df["defaults"]], axis=1)
    hover = "<b>%{" + ("y" if horizontal else "x") + "}</b><br>Default rate %{" + ("x" if horizontal else "y") + \
            ":.2%}<br>Applicants %{customdata[0]:,}<br>Defaults %{customdata[1]:,}<extra></extra>"
    if horizontal:
        bar = go.Bar(x=df["default_rate"], y=df[label].astype(str), orientation="h", marker_color=color, text=text,
                     textposition="outside", cliponaxis=False, customdata=custom, hovertemplate=hover,
                     marker_cornerradius=4)
    else:
        bar = go.Bar(x=df[label].astype(str), y=df["default_rate"], marker_color=color, text=text,
                     textposition="outside", cliponaxis=False, customdata=custom, hovertemplate=hover,
                     marker_cornerradius=4)
    fig = go.Figure(bar)
    top = df["default_rate"].max() * 1.22
    if overall is not None:
        line = dict(color=muted(), width=1, dash="dash")
        if horizontal:
            fig.add_vline(x=overall, line=line, annotation_text=f"overall {overall:.2%}", annotation_position="top")
        else:
            fig.add_hline(y=overall, line=line, annotation_text=f"overall {overall:.2%}",
                          annotation_position="top right")
    n = len(df)
    h = height or (max(260, 34 * n + 90) if horizontal else 380)
    if horizontal:
        layout(fig, xtitle=xtitle, ytitle=None, height=h, legend=False, xpct=True, xrange=[0, top])
        fig.update_yaxes(autorange="reversed")
    else:
        layout(fig, xtitle=None, ytitle=xtitle, height=h, legend=False, ypct=True, yrange=[0, top])
    if overall is not None:  # room for the reference-line label above the plot
        fig.update_layout(margin_t=34)
    return fig


def density_lines(hist, variable, xtitle):
    d = hist[hist["variable"] == variable]
    fig = go.Figure()
    for t, name, color in [(0, "Repaid (TARGET = 0)", series(0)), (1, "Default (TARGET = 1)", series(1))]:
        s = d[d["TARGET"] == t]
        mid = (s["bin_left"] + s["bin_right"]) / 2
        fig.add_trace(go.Scatter(x=mid, y=s["density"], mode="lines", name=name, line=dict(color=color, width=2),
                                 line_shape="hvh", customdata=s["count"],
                                 hovertemplate=f"{name}<br>{xtitle} %{{x:,.3g}}<br>density %{{y:.3g}}"
                                               "<br>applicants in bin %{customdata:,}<extra></extra>"))
    return layout(fig, xtitle=xtitle, ytitle="Density (share per unit)", height=340, hovermode="x unified")


def corr_bars(df):
    df = df.iloc[::-1]
    colors = [series(1) if v > 0 else series(0) for v in df["corr"]]
    fig = go.Figure(go.Bar(x=df["corr"], y=df["feature"], orientation="h", marker_color=colors,
                           marker_cornerradius=4, text=[f"{v:+.3f}" for v in df["corr"]], textposition="outside",
                           cliponaxis=False, hovertemplate="<b>%{y}</b><br>correlation %{x:+.4f}<extra></extra>"))
    lim = df["corr"].abs().max() * 1.35
    fig.add_vline(x=0, line=dict(color=muted(), width=1))
    return layout(fig, xtitle="Pearson correlation with TARGET", height=max(300, 30 * len(df) + 80), legend=False,
                  xrange=[-lim, lim])


# ---------------------------------------------------------------- model comparison

def model_bars(values, fmt=".4f", ytitle="", height=340, yrange=None, best=None):
    colors = model_colors()
    models = [m for m in MODEL_ORDER if m in values.index]
    fig = go.Figure(go.Bar(
        x=models, y=[values[m] for m in models], marker_color=[colors[m] for m in models], marker_cornerradius=4,
        text=[format(values[m], fmt) for m in models], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>" + ytitle + " %{y:" + fmt + "}<extra></extra>"))
    if best is not None:
        fig.add_annotation(x=best, y=values[best], text="best", showarrow=False, yshift=30, font=dict(size=12))
    lo = min(0, float(values.min()))
    return layout(fig, ytitle=ytitle, height=height, legend=False,
                  yrange=yrange or [lo, float(values.max()) * 1.18 if values.max() > 0 else 1])


def grouped_bars(table, metrics, height=420):
    """table: index = model, columns = metrics."""
    colors = model_colors()
    fig = go.Figure()
    for m in [m for m in MODEL_ORDER if m in table.index]:
        fig.add_trace(go.Bar(name=m, x=[METRIC_LABELS[k] for k in metrics], y=table.loc[m, metrics],
                             marker_color=colors[m], marker_cornerradius=3,
                             hovertemplate=f"<b>{m}</b><br>%{{x}} %{{y:.4f}}<extra></extra>"))
    return layout(fig, ytitle="Value on the holdout", height=height, yrange=[0, 1.05])


def metric_heatmap(values, norm, height=330):
    """values / norm: index = model, columns = metric labels; norm in [0, 1] with 1 = best in its column."""
    text = values.map(lambda v: f"{v:.3f}").to_numpy()
    fig = go.Figure(go.Heatmap(
        z=norm.to_numpy(), x=list(values.columns), y=list(values.index), text=text, texttemplate="%{text}",
        colorscale=sequential_scale(), zmin=0, zmax=1, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="1 = best<br>in column"), thickness=12),
        customdata=values.to_numpy(), hovertemplate="<b>%{y}</b><br>%{x}: %{customdata:.4f}"
                                                    "<br>relative position %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return layout(fig, height=height, legend=False)


def radar(norm):
    """norm: index = model, columns = metric labels, values in [0, 1] (1 = best)."""
    colors = model_colors()
    cats = list(norm.columns)
    fig = go.Figure()
    for m in [m for m in MODEL_ORDER if m in norm.index]:
        r = list(norm.loc[m]) + [norm.loc[m].iloc[0]]
        fig.add_trace(go.Scatterpolar(r=r, theta=cats + [cats[0]], name=m, line=dict(color=colors[m], width=2,
                                                                                         dash=MODEL_DASH[m]),
                                      marker=dict(size=8, symbol=MODEL_SYMBOL[m]), mode="lines+markers",
                                      hovertemplate=f"<b>{m}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0, 0.5, 1], showline=False)))
    return layout(fig, height=470)


def curves(points, curve, highlight=None, baseline=None, height=430):
    colors = model_colors()
    fig = go.Figure()
    for m in MODEL_ORDER:
        d = points[(points["model"] == m) & (points["curve"] == curve)]
        faded = highlight is not None and m != highlight
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["y"], mode="lines", name=m,
            line=dict(color=muted() if faded else colors[m], width=1.5 if faded else 2.5,
                      dash="solid" if highlight else MODEL_DASH[m]),
            opacity=0.55 if faded else 1,
            hovertemplate=f"<b>{m}</b><br>" + ("FPR %{x:.3f}<br>TPR %{y:.3f}" if curve == "ROC"
                                               else "Recall %{x:.3f}<br>Precision %{y:.3f}") + "<extra></extra>"))
    if curve == "ROC":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                 line=dict(color=muted(), width=1, dash="dot"), hoverinfo="skip"))
        return layout(fig, xtitle="False positive rate (good applicants flagged)",
                      ytitle="True positive rate (defaulters caught)", height=height, xrange=[0, 1], yrange=[0, 1])
    if baseline is not None:
        fig.add_hline(y=baseline, line=dict(color=muted(), width=1, dash="dot"),
                      annotation_text=f"random guess = default rate {baseline:.2%}", annotation_position="top right")
    return layout(fig, xtitle="Recall (share of defaulters caught)", ytitle="Precision (share of flagged who default)",
                  height=height, xrange=[0, 1], yrange=[0, 1])


def confusion(tn, fp, fn, tp, height=290, show_scale=False):
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    share = cm / cm.sum(axis=1, keepdims=True)
    total = cm.sum()
    labels = [["True negative<br>(good, approved)", "False positive<br>(good, rejected)"],
              ["False negative<br>(default, approved)", "True positive<br>(default, rejected)"]]
    text = [[f"<b>{cm[r, c]:,.0f}</b><br>{share[r, c]:.1%} of row<br>{cm[r, c] / total:.1%} of all"
             for c in range(2)] for r in range(2)]
    fig = go.Figure(go.Heatmap(
        z=share, x=["Approve", "Reject"], y=["Repaid", "Defaulted"],
        text=text, texttemplate="%{text}", colorscale=sequential_scale(), zmin=0, zmax=1, xgap=3, ygap=3,
        showscale=show_scale, customdata=labels,
        hovertemplate="%{customdata}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textfont=dict(size=13))
    return layout(fig, xtitle="Model decision", ytitle="Actual outcome", height=height, legend=False)


def threshold_lines(curves_df, metric, marker=None, extra_markers=None, height=380):
    colors = model_colors()
    fig = go.Figure()
    for m in MODEL_ORDER:
        d = curves_df[curves_df["model"] == m]
        fig.add_trace(go.Scatter(x=d["threshold"], y=d[metric], mode="lines", name=m,
                                 line=dict(color=colors[m], width=2, dash=MODEL_DASH[m]),
                                 hovertemplate=f"<b>{m}</b><br>threshold %{{x:.2f}}<br>{METRIC_LABELS[metric]} "
                                               "%{y:.4f}<extra></extra>"))
    if marker is not None:
        fig.add_vline(x=marker, line=dict(color=muted(), width=1.5),
                      annotation_text=f"slider {marker:.2f}", annotation_position="bottom right")
    for x, label in (extra_markers or []):
        fig.add_vline(x=x, line=dict(color=muted(), width=1, dash="dot"), annotation_text=label,
                      annotation_position="bottom right")
    return layout(fig, xtitle="Threshold (reject if score ≥ threshold)", ytitle=METRIC_LABELS[metric],
                  height=height, yrange=[0, 1.02], hovermode="x unified")


def lines(x, ys, xtitle, ytitle, height=320, marker_x=None, marker_label=None, yfmt=None):
    """Simple line chart; ys: list of (name, values, colour)."""
    fig = go.Figure()
    for name, y, color in ys:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name, line=dict(color=color, width=2),
                                 marker=dict(size=8), hovertemplate=f"{name}<br>{xtitle} %{{x}}<br>{ytitle} "
                                                                    f"%{{y:{yfmt or ',.4g'}}}<extra></extra>"))
    if marker_x is not None:
        fig.add_vline(x=marker_x, line=dict(color=muted(), width=1.5, dash="dash"), annotation_text=marker_label,
                      annotation_position="bottom right")
    return layout(fig, xtitle=xtitle, ytitle=ytitle, height=height, legend=len(ys) > 1)


def diverging_heatmap(z, x, y, height=360, colorbar_title="z-score"):
    lim = float(np.abs(z).max())
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=diverging_scale(), zmin=-lim, zmax=lim, xgap=2, ygap=2,
                               text=[[f"{v:+.2f}" for v in row] for row in z], texttemplate="%{text}",
                               colorbar=dict(title=dict(text=colorbar_title), thickness=12),
                               hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.2f} standard deviations<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return layout(fig, height=height, legend=False)


def gauge(score, threshold, height=280):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(valueformat=".3f", font=dict(size=42)),
        gauge=dict(axis=dict(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1], tickformat=".2f"),
                   bar=dict(color="#52514e", thickness=0.28),
                   steps=[dict(range=[0, threshold], color="rgba(12,163,12,0.22)"),
                          dict(range=[threshold, 1], color="rgba(208,59,59,0.22)")],
                   threshold=dict(line=dict(color=STATUS["critical"], width=3), thickness=0.9, value=threshold))))
    fig.update_layout(height=height, margin=dict(l=48, r=48, t=24, b=8), font=FONT)
    return fig


def waterfall(features, contributions, base, height=None):
    """Horizontal waterfall of SHAP contributions (log-odds): base value -> each feature -> final score."""
    labels = ["Average applicant (base)"] + list(features) + ["This applicant"]
    measure = ["absolute"] + ["relative"] * len(features) + ["total"]
    values = [base] + list(contributions) + [0]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=values, measure=measure,
        increasing=dict(marker=dict(color=STATUS["critical"])), decreasing=dict(marker=dict(color="#256abf")),
        totals=dict(marker=dict(color="#52514e")), connector=dict(line=dict(color=muted(), width=1)),
        text=[f"{base:+.2f}"] + [f"{v:+.3f}" for v in contributions] + [""], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return layout(fig, xtitle="Model output (log-odds of default; right = riskier)", height=height or 36 * len(labels) + 80,
                  legend=False)
