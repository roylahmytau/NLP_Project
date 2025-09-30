import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple

def plot_two_lines(
    x_values: Sequence[float],
    y1_values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    y1_legend: str,
    y2_values: Optional[Sequence[float]] = None,
    y2_legend: Optional[str] = None,
    y3_values: Optional[Sequence[float]] = None,
    y3_legend: Optional[str] = None,
    y4_values: Optional[Sequence[float]] = None,
    y4_legend: Optional[str] = None,
    style1: str = "-o",
    style2: str = "-s",
    style3: str = "-^",
    style4: str = "-d",
    grid: bool = True,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot one to four lines on the same axes with labels and legend.

    Returns (fig, ax).
    """
    if len(x_values) != len(y1_values):
        raise ValueError("x_values and y1_values must have the same length.")
    if y2_values is not None and len(x_values) != len(y2_values):
        raise ValueError("x_values and y2_values must have the same length when y2_values is provided.")
    if y3_values is not None and len(x_values) != len(y3_values):
        raise ValueError("x_values and y3_values must have the same length when y3_values is provided.")
    if y4_values is not None and len(x_values) != len(y4_values):
        raise ValueError("x_values and y4_values must have the same length when y4_values is provided.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_values, y1_values, style1, label=y1_legend)
    if y2_values is not None:
        second_label = y2_legend if y2_legend is not None else "Series 2"
        ax.plot(x_values, y2_values, style2, label=second_label)
    if y3_values is not None:
        third_label = y3_legend if y3_legend is not None else "Series 3"
        ax.plot(x_values, y3_values, style3, label=third_label)
    if y4_values is not None:
        fourth_label = y4_legend if y4_legend is not None else "Series 4"
        ax.plot(x_values, y4_values, style4, label=fourth_label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    # Enforce y-axis bounds 0 to 100
    ax.set_ylim(0, 100)

    # Show only provided x-values as ticks on the bottom axis
    ax.set_xticks(x_values)
    ax.tick_params(axis='x', labelbottom=True, labeltop=False)
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['top'].set_visible(False)
    ax.minorticks_off()

    if grid:
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


# Example usage
if __name__ == "__main__":
    # x = [2048, 32768, 131072]
    # y_a = [99, 53, 40]
    # y_b = [100, 73, 50]

    # plot_two_lines(
    #     x_values=x,
    #     y1_values=y_a,
    #     y2_values=y_b,
    #     title="LoRA QA Benchmark Accuracy by Token Size",
    #     x_label="Token Size",
    #     y_label="Accuracy (%)",
    #     y1_legend="RULER Exact Accuracy",
    #     y2_legend="RULER Partial Accuracy",
    #     save_path='plot.png',
    #     show=True,
    # )
    x = [0, 1, 5, 10, 20]
    y_a = [0, 0, 53, 94, 99]
    y_b = [36, 37, 66, 97, 100]
    y_c = [0, 0, 28, 44, 32]
    y_d = [36, 36, 43, 63, 49]

    plot_two_lines(
        x_values=x,
        y1_values=y_a,
        y2_values=y_b,
        y3_values=y_c,
        y4_values=y_d,
        title="LoRA QA 2K Accuracy by Number of Epochs",
        x_label="Epochs",
        y_label="Accuracy (%)",
        y1_legend="RULER Exact Accuracy",
        y2_legend="RULER Partial Accuracy",
        y3_legend="RULER Gen QA Exact Accuracy",
        y4_legend="RULER Gen QA Partial Accuracy",
        save_path='plot.png',
        show=True,
    )

    # x = [1e-4, 1e-3, 1e-2]
    # y_a = [0, 53, 1]
    # y_b = [0, 66, 15]

    # plot_two_lines(
    #     x_values=x,
    #     y1_values=y_a,
    #     y2_values=y_b,
    #     title="LoRA QA 2K Accuracy by Learning Rate",
    #     x_label="Learning Rate",
    #     y_label="Accuracy (%)",
    #     y1_legend="RULER Exact Accuracy",
    #     y2_legend="RULER Partial Accuracy",
    #     save_path='plot.png',
    #     show=True,
    # )

    # x = [1, 2, 4]
    # y_a = [91, 53, 5]
    # y_b = [97, 66, 15]

    # plot_two_lines(
    #     x_values=x,
    #     y1_values=y_a,
    #     y2_values=y_b,
    #     title="LoRA QA 2K Accuracy by Batch Size",
    #     x_label="Batch Size",
    #     y_label="Accuracy (%)",
    #     y1_legend="RULER Exact Accuracy",
    #     y2_legend="RULER Partial Accuracy",
    #     save_path='plot.png',
    #     show=True,
    # )

    # x = [5, 8, 11]
    # y_a = [55, 53, 48]
    # y_b = [64, 66, 64]

    # plot_two_lines(
    #     x_values=x,
    #     y1_values=y_a,
    #     y2_values=y_b,
    #     title="LoRA QA 2K Accuracy by Adapter Rank",
    #     x_label="Adapter Rank",
    #     y_label="Accuracy (%)",
    #     y1_legend="RULER Exact Accuracy",
    #     y2_legend="RULER Partial Accuracy",
    #     save_path='plot.png',
    #     show=True,
    # )

    # x = [5, 10, 15]
    # y_a = [57, 53, 7]
    # y_b = [70, 66, 23]

    # plot_two_lines(
    #     x_values=x,
    #     y1_values=y_a,
    #     y2_values=y_b,
    #     title="LoRA QA 2K Accuracy by Adapter Layers",
    #     x_label="Adapter Layers",
    #     y_label="Accuracy (%)",
    #     y1_legend="RULER Exact Accuracy",
    #     y2_legend="RULER Partial Accuracy",
    #     save_path='plot.png',
    #     show=True,
    # )

    # x = [2048, 32768, 131072]
    # y_a = [18, 0, 0]
    # y_b = [18, 3, 0]

    # plot_two_lines(
    #     x_values=x,
    #     y1_values=y_a,
    #     y2_values=y_b,
    #     title="LoRA Single Needle Accuracy by Token Size",
    #     x_label="Token Size",
    #     y_label="Accuracy (%)",
    #     y1_legend="RULER Exact Accuracy",
    #     y2_legend="RULER Partial Accuracy",
    #     save_path='plot.png',
    #     show=True,
    # )

    # x = [0, 1, 5, 10, 20]
    # y_a = [14.237, 13.748, 13.698, 20.043, 25.760]

    # plot_two_lines(
    #     x_values=x,
    #     y1_values=y_a,
    #     title="LoRA Model 2K QA Benchmark Perplexity by Epoch",
    #     x_label="Epoch",
    #     y_label="Perplexity",
    #     y1_legend="Perplexity",
    #     save_path='plot.png',
    #     show=True,
    # )

    # x = [2048, 32768, 131072]
    # y_a = [25.760, 18.364, 12.931]

    # plot_two_lines(
    #     x_values=x,
    #     y1_values=y_a,
    #     title="LoRA Model 2K QA Benchmark Perplexity by Token Size",
    #     x_label="Token Size",
    #     y_label="Perplexity",
    #     y1_legend="Perplexity",
    #     save_path='plot.png',
    #     show=True,
    # )

    # x = [0, 1, 5, 10, 20]
    # y_a = [0.69, 0.769, 0.428, 0.5, 0.661]

    # plot_two_lines(
    #     x_values=x,
    #     y1_values=y_a,
    #     title="LoRA Model 2K QA Benchmark GLUE Score by Epoch",
    #     x_label="Epoch",
    #     y_label="GLUE Score",
    #     y1_legend="GLUE Score",
    #     save_path='plot.png',
    #     show=True,
    # )

    # x = [8, 16, 32]
    # y_a = [0, 0, 3]
    # y_b = [13, 20, 17]

    # plot_two_lines(
    #     x_values=x,
    #     y1_values=y_a,
    #     y2_values=y_b,
    #     title="LoRA QA 32K Accuracy with 1 Adapter Layer by Adapter Rank",
    #     x_label="Adapter Rank",
    #     y_label="Accuracy (%)",
    #     y1_legend="RULER Exact Accuracy",
    #     y2_legend="RULER Partial Accuracy",
    #     save_path='plot.png',
    #     show=True,
    # )