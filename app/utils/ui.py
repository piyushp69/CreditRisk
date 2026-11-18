"""Page building blocks: header, key-takeaway box, charts with a data-table twin, metric cards, diagrams."""
import streamlit as st

CSS = """
<style>
/* Comfortable reading width and spacing on large screens; Streamlit stacks columns on phones. */
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px;}
h1 {font-weight: 700; letter-spacing: -0.01em;}
h2, h3 {margin-top: 0.8rem;}
[data-testid="stMetric"] {padding: 0.65rem 0.85rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.page-kicker {text-transform: uppercase; letter-spacing: .08em; font-size: .78rem; opacity: .7; margin-bottom: -.4rem;}
.chart-title {font-weight: 600; font-size: 1.02rem; margin: .4rem 0 0 0;}
.chart-sub {opacity: .75; font-size: .9rem; margin: 0 0 .2rem 0;}
@media (max-width: 640px) {
  .block-container {padding-left: 1rem; padding-right: 1rem; padding-top: 1.2rem;}
  [data-testid="stMetricValue"] {font-size: 1.35rem;}
}
</style>
"""

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def inject_css():
    st.html(CSS)


def header(title, intro, kicker=None):
    if kicker:
        st.html(f'<div class="page-kicker">{kicker}</div>')
    st.title(title)
    st.markdown(intro)


def takeaway(text):
    st.info(text, icon=":material/lightbulb:", title="Key takeaway")


def chart(fig, title=None, subtitle=None, data=None, data_label="Show the data behind this chart", key=None,
          formats=None):
    """A chart with its title outside the plot (so it wraps on phones) and an optional table view."""
    if title:
        st.html(f'<p class="chart-title">{title}</p>' + (f'<p class="chart-sub">{subtitle}</p>' if subtitle else ""))
    st.plotly_chart(fig, theme="streamlit", config=PLOTLY_CONFIG, key=key)
    if data is not None:
        with st.expander(data_label, icon=":material/table:"):
            table(data, formats)


def table(df, formats=None, height="auto", hide_index=True):
    """Dataframe with optional per-column formats: {'col': '{:.2%}'} or callables."""
    if formats:
        st.dataframe(df.style.format(formats, na_rep="—"), hide_index=hide_index, height=height)
    else:
        st.dataframe(df, hide_index=hide_index, height=height)


def metrics_row(items, columns=None):
    """items: list of (label, value, help) or (label, value, help, delta)."""
    cols = st.columns(columns or len(items))
    for col, item in zip(cols, items):
        label, value, help_text = item[:3]
        delta = item[3] if len(item) > 3 else None
        with col:
            st.metric(label, value, delta=delta, help=help_text, border=True,
                      delta_color="off" if delta is not None and not str(delta).startswith(("+", "-")) else "normal")


def diagram(dot):
    st.graphviz_chart(dot, width="stretch")


def source_note(text):
    st.caption(text)


def flow_dot(steps, rankdir="LR", highlight=None):
    """A simple left-to-right flow diagram (Graphviz DOT) for pipelines and model sketches."""
    nodes = []
    for i, s in enumerate(steps):
        fill = "#2a78d6" if highlight is not None and i in highlight else "#f2f1ec"
        font = "white" if highlight is not None and i in highlight else "#0b0b0b"
        nodes.append(f'n{i} [label="{s}", fillcolor="{fill}", fontcolor="{font}"];')
    edges = " -> ".join(f"n{i}" for i in range(len(steps)))
    return ("digraph { rankdir=" + rankdir + '; bgcolor="transparent"; '
            'node [shape=box, style="rounded,filled", color="#c3c2b7", fontname="Helvetica", fontsize=11]; '
            'edge [color="#898781"]; ' + " ".join(nodes) + " " + edges + "; }")
