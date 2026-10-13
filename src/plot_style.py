"""Shared matplotlib styling for the figures in docs/figures/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#f0efec"
SERIES = "#2a78d6"          # single-series colour
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order; first three validate all-pairs
NEGATIVE = "#e34948"        # diverging pole for declines
# Ordinal blue ramp, darkest = nearest to the fab (validated: monotone
# lightness, single hue, light end clears 2:1 on SURFACE).
DISTANCE_RAMP = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]


def figure(width: float = 8, height: float = 4.5):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    style_axes(ax)
    return fig, ax


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, text: str, subtitle: str | None = None) -> None:
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", color=TEXT, pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY, va="bottom")


def save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
