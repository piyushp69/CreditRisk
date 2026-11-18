"""Shared chart style, figure saving and the default-rate bar chart used across notebooks."""
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from .config import FIGURES_DIR, ROOT

# Validated categorical palette, always assigned in this order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REPAID, DEFAULT = SERIES[0], SERIES[1]  # TARGET = 0 / TARGET = 1 in every chart

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Single-hue sequential ramp (blue) for heatmaps.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.dpi": 100,
    })


def savefig(fig, name):
    """Save a figure as PNG (dpi=150) in reports/figures/ and return its path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path.relative_to(ROOT).as_posix()}")
    return path


# Annotation background so reference lines never run through label text.
_LABEL_BOX = dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none")


def plot_confusion(ax, tn, fp, fn, tp, title, xlabel):
    """2 x 2 confusion matrix: colour = share of the actual class; text = count, % of the row, % of all applicants."""
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(row_share, cmap=SEQ_BLUE, vmin=0, vmax=1)
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:,.0f}\n{row_share[r, c]:.1%} of row\n{v / cm.sum():.1%} of all", ha="center",
                va="center", fontsize=8.5, color="white" if row_share[r, c] > 0.5 else INK)
    ax.set_xticks([0, 1], ["Predicted repaid\n(approve)", "Predicted default\n(reject)"])
    ax.set_yticks([0, 1], ["Actual repaid", "Actual default"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Actual outcome")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(False)
    return ax


def plot_rate_bars(table, ax, title, xlabel, overall_rate=None, horizontal=True, color=None):
    """Bar chart of default rate per group, annotated with rate and applicant count.

    `table` is the output of src.eda.default_rate_table (index = group).
    """
    color = color or DEFAULT
    labels = [str(i) for i in table.index]
    rates = table["default_rate"].to_numpy()
    counts = table["applicants"].to_numpy()
    positions = range(len(table))
    if horizontal:
        ax.barh(positions, rates, color=color, height=0.7)
        ax.set_yticks(list(positions), labels)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Default rate (TARGET = 1)")
        ax.set_ylabel(xlabel)
        ax.grid(axis="y", visible=False)
        for y, (r, n) in enumerate(zip(rates, counts)):
            ax.annotate(f"{r:.1%}  (n={n:,})", (r, y), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK_2, bbox=_LABEL_BOX, zorder=3)
        ax.set_xlim(0, rates.max() * 1.45)
        if overall_rate is not None:
            ax.axvline(overall_rate, color=MUTED, linestyle="--", linewidth=1, zorder=2,
                       label=f"Overall default rate {overall_rate:.1%}")
            ax.legend(loc="lower right")
    else:
        ax.bar(positions, rates, color=color, width=0.7)
        ax.set_xticks(list(positions), labels)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("Default rate (TARGET = 1)")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
        for x, (r, n) in enumerate(zip(rates, counts)):
            ax.annotate(f"{r:.1%}\nn={n:,}", (x, r), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.5, color=INK_2, bbox=_LABEL_BOX, zorder=3)
        ax.set_ylim(0, rates.max() * 1.4)
        if overall_rate is not None:
            ax.axhline(overall_rate, color=MUTED, linestyle="--", linewidth=1, zorder=2,
                       label=f"Overall default rate {overall_rate:.1%}")
            ax.legend(loc="upper right")
    ax.set_title(title, loc="left")
    return ax
