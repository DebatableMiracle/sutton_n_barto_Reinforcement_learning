import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

def draw_racetrack(track, layers=None, paths=None, ax=None, title=None, cell=0.4):
    """
    track  : (H, W) bool array, True = drivable, False = wall
    layers : {"name": (list_of_(x, y)_cells, color)}  drawn in order, later wins
    paths  : {"name": (list_of_(x, y)_cells, color)}  drawn as lines on top
    """
    h, w = track.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(w * cell + 1, h * cell + 1))

    # build the RGB image: track -> white, wall -> dark grey, then layers
    img = np.empty((h, w, 3))
    img[track] = to_rgb("white")
    img[~track] = to_rgb("#3a3a3a")
    for cells, color in (layers or {}).values():
        for x, y in cells:
            img[y, x] = to_rgb(color)
    ax.imshow(img, origin="upper", interpolation="nearest")

    # paths (e.g. trajectories), with markers at each visited cell
    for cells, color in (paths or {}).values():
        xs, ys = zip(*cells)
        ax.plot(xs, ys, "-o", color=color, ms=4, lw=1.5)

    # grid lines on cell EDGES, labels on cell CENTERS
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="#bbbbbb", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)  # y=0 at top, like the book figure

    # legend from layer + path names
    handles = [Patch(facecolor=c, edgecolor="k", label=n)
               for n, (_, c) in {**(layers or {}), **(paths or {})}.items()]
    if handles:
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.01, 1), fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def draw_windy_gridworld(wind, height, start, goal,
                         paths=None, ax=None, title=None, cell=0.8):
    """
    paths : {"name": (list_of_(x, y)_cells, color)}  drawn as arrows, in order
    """
    w = len(wind)
    if ax is None:
        _, ax = plt.subplots(figsize=(w * cell + 2, height * cell + 1.5))

    # background: windier columns are shaded bluer
    img = np.ones((height, w, 3))
    top = max(max(wind), 1)
    for x, s in enumerate(wind):
        t = 0.6 * s / top
        img[:, x] = (1 - t) * np.ones(3) + t * np.array(to_rgb("#6fa8dc"))
    img[start[1], start[0]] = to_rgb("#c97a7a")
    img[goal[1], goal[0]] = to_rgb("#7aaf8c")
    ax.imshow(img, origin="upper", interpolation="nearest")

    # faint upward arrows in every windy column, thicker = stronger
    for x, s in enumerate(wind):
        if s == 0:
            continue
        for y in range(height):
            ax.annotate("", xy=(x, y - 0.3), xytext=(x, y + 0.3),
                        arrowprops=dict(arrowstyle="-|>", color="#2a5d8f",
                                        alpha=0.3, lw=s))

    ax.text(*start, "S", ha="center", va="center", fontweight="bold")
    ax.text(*goal, "G", ha="center", va="center", fontweight="bold")

    # trajectories: one arrow per move (wind can make these long jumps)
    for cells, color in (paths or {}).values():
        for (x0, y0), (x1, y1) in zip(cells, cells[1:]):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                                        shrinkA=4, shrinkB=4))
        ax.plot(*zip(*cells), "o", color=color, ms=4)

    # grid on cell edges, wind strength under each column
    ax.set_xticks(range(w))
    ax.set_xticklabels([f"{x}\nwind {s}" for x, s in enumerate(wind)], fontsize=8)
    ax.set_yticks(range(height))
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#999999", lw=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    handles = [Patch(facecolor=c, edgecolor="k", label=n)
               for n, (_, c) in (paths or {}).items()]
    if handles:
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.01, 1), fontsize=8)
    if title:
        ax.set_title(title)
    return ax

def draw_greedy_policy_windygrid(Q, moves, wind, height, start, goal, paths=None,
                       show_values=True, ax=None, title=None):
    """One arrow per cell for argmax Q. Orange = best action is still untried (Q == 0)."""
    ax = draw_windy_gridworld(wind, height, start, goal, paths=paths, ax=ax, title=title)
    for x in range(len(wind)):
        for y in range(height):
            if (x, y) == tuple(goal):
                continue
            q = Q[x, y]
            if not q.any():                          # never visited: nothing learned here
                ax.plot(x, y, ".", color="gray", ms=4)
                continue
            dx, dy = moves[int(np.argmax(q))]
            color = "darkorange" if q.max() == 0 else "black"
            if (dx, dy) == (0, 0):                   # the 'stay' action
                ax.plot(x, y, "o", mfc="none", mec=color, ms=8)
            else:
                n = np.hypot(dx, dy)
                ax.annotate("", xy=(x + 0.32 * dx / n, y + 0.32 * dy / n),
                            xytext=(x - 0.32 * dx / n, y - 0.32 * dy / n),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
            if show_values:                          # state value = max_a Q, bottom-right corner
                ax.text(x + 0.44, y + 0.44, f"{q.max():.0f}", ha="right", va="bottom",
                        fontsize=6, color="#333333")
    return ax