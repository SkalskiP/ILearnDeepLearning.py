from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']
rcParams['font.size'] = 16

BG_COLOR = (240 / 255, 238 / 255, 223 / 255)


def lines(
    y_1: np.array,
    y_2: np.array,
    label_1: str,
    label_2: str,
    title: str,
    fig_size: Tuple[int, int],
    path: Optional[str] = None
) -> None:

    assert len(y_1) == len(y_2)
    x = np.arange(len(y_1))
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True, facecolor=BG_COLOR)

    # plot
    ax.plot(x, y_1, label=label_1, color='crimson', linewidth=2)
    ax.plot(x, y_2, label=label_2, linewidth=2)
    ax.set_title(title, loc='left', pad=20)

    # axis
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    ax.xaxis.set_ticks_position('none')
    ax.set_xlabel('epoch', fontsize=12)
    ax.xaxis.set_label_coords(0.97, 0.04)

    # grid
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)

    # legend
    fig.legend(bbox_to_anchor=(0.90, 0.15), loc="lower left", frameon=False,
               prop={'size': 14})

    ax.set_facecolor(BG_COLOR)

    if path:
        fig.savefig(path, facecolor=fig.get_facecolor(), dpi=100)
    plt.show()
