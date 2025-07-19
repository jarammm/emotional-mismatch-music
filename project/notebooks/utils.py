import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns


def stacked_barplot(df, title="Stacked Bar Plot", xlabel="X", ylabel="Count",
                    cmap_name="tab20", percentage=True, show_values=True,
                    figsize=(10, 6), fontsize=9,
                    auto_height_by_legend=False, legend_item_height=0.25):
    num_categories = df.shape[1]
    cmap = cm.get_cmap(cmap_name, num_categories)
    colors = [cmap(i) for i in range(num_categories)]

    if auto_height_by_legend:
        estimated_legend_height = num_categories * legend_item_height
        base_height = figsize[1]
        new_height = max(base_height, estimated_legend_height)
        figsize = (figsize[0], new_height)

    fig, ax = plt.subplots(figsize=figsize)
    df_pct = df.div(df.sum(axis=1), axis=0)
    df.plot(kind='bar', stacked=True, ax=ax, color=colors)

    if show_values:
        for i, row in enumerate(df_pct.values):
            cumulative = 0
            for j, val in enumerate(row):
                count = df.iloc[i, j]
                if count == 0:
                    continue
                cumulative += val
                y = cumulative - val / 2
                label = f"{val * 100:.1f}%" if percentage else str(count)
                ax.text(i, y * df.sum(axis=1).iloc[i], label,
                        ha='center', va='center', fontsize=fontsize, color="black")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.index, rotation=0)

    ax.legend(title=df.columns.name or "genre",
              bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_kde(df, value_col, group_col, bw_adjust=0.2, palette='tab10'):
    plt.figure(figsize=(10, 6))

    sns.kdeplot(data=df, x=value_col, hue=group_col,
                fill=True, common_norm=False, palette=palette, alpha=0.4, bw_adjust=bw_adjust)

    plt.title(f"{value_col} KDE by {group_col}")
    plt.xlabel(value_col)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()


def plot_mfcc_boxplots(df, col_range, stat="mean", group_col="emotion_pair",
                           grid_shape=(3, 3), suptitle=None, base_width=5, base_height=4):
    num_plots = len(col_range)
    nrows, ncols = grid_shape
    if num_plots > nrows * ncols:
        raise ValueError("Grid shape is too small for the number of plots.")

    figsize = (base_width * ncols, base_height * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(suptitle or f"MFCC Feature Boxplots", fontsize=16)

    axes = axes.flatten()
    for i, num in enumerate(col_range):
        col_name = f"mfcc_{num}_{stat}"
        sns.boxplot(data=df, x=group_col, y=col_name, ax=axes[i])
        axes[i].set_title(col_name)
        axes[i].set_xlabel("label")
        axes[i].set_ylabel(col_name)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_single_boxplot(df, feature, group_col="emotion_pair", title=None, figsize=(6, 5)):
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=group_col, y=feature)
    plt.xlabel("label")
    plt.ylabel(feature)
    plt.title(title or f"{feature} by label")
    plt.tight_layout()
    plt.show()