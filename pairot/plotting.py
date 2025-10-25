from typing import Optional, Literal

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scanpy.plotting.palettes import default_102


def _plot_heatmap(
    data: pd.DataFrame,
    colormap: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    backend: Literal["plotly", "matplotlib"] = "plotly",
    **kwargs,
):
    if width is None:
        width = max(30 * data.shape[1], 500)
    if height is None:
        height = max(20 * data.shape[0] + 275, 500)

    if backend == "plotly":
        fig = px.imshow(data, text_auto=".2f", color_continuous_scale=colormap, **kwargs)
        fig.update_layout(autosize=False, width=width, height=height)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickangle=90)
        fig.update_yaxes(tickangle=0)
    elif backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            ax=ax,
            vmin=kwargs.pop("zmin", None),
            vmax=kwargs.pop("zmax", None),
            cmap=colormap,
            cbar=False,
            square=True,
            annot_kws={"size": 6},
        )
        ax.tick_params(axis='x', colors=(0.2, 0.2, 0.2, 1), length=0, labelsize=10)
        ax.tick_params(axis='y', colors=(0.2, 0.2, 0.2, 1), length=0, labelsize=10)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Backends supported: 'plotly', 'matplotlib'.")

    return fig


def plot_cluster_mapping(
    data: pd.DataFrame,
    width: Optional[int] = None,
    height: Optional[int] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    colormap: str = "Greens",
    sort_by_score: bool = True,
    show: bool = True,
    backend: Literal["plotly", "matplotlib"] = "plotly",
):
    if sort_by_score:
        data = data.loc[
            data.max(axis=1).sort_values(ascending=False).index.tolist(),
            data.max().sort_values(ascending=False).index.tolist(),
        ]
    fig = _plot_heatmap(data, colormap, width, height, zmin=zmin, zmax=zmax, backend=backend)
    if show:
        fig.show()
        return None
    else:
        return fig


def plot_cluster_distance(
    data: pd.DataFrame,
    width: Optional[int] = None,
    height: Optional[int] = None,
    show: bool = True,
    backend: Literal["plotly", "matplotlib"] = "plotly",
):
    fig = _plot_heatmap(data, "RdYlGn_r", width, height, zmin=0.0, zmax=2.0, backend=backend)
    if show:
        fig.show()
        return None
    else:
        return fig


def plot_sankey(
    cluster_mapping: pd.DataFrame,
    cluster_distance: pd.DataFrame,
    filter_threshold: float = 0.25,
    width: int = 1000,
    height: int = 1200,
    show: bool = True,
):
    def _filter_dict(unfiltered_dict, threshold=0.1):
        vals = pd.DataFrame(unfiltered_dict)
        vals_filtered = vals[vals.value >= threshold]
        return {col: vals_filtered[col].tolist() for col in vals_filtered.columns}

    norm = mpl.colors.Normalize(vmin=0.0, vmax=2.0)
    cmap = plt.get_cmap("RdYlGn_r")
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    nodes_query = cluster_mapping.index.tolist()
    nodes_ref = cluster_mapping.columns.tolist()
    nodes_combined = nodes_query + nodes_ref
    fig = go.Figure(
        data=[
            go.Sankey(
                valueformat=".2f",
                node=dict(
                    line=dict(color="black", width=1.0),
                    label=nodes_combined,
                    color=[
                        px.colors.qualitative.Plotly[
                            i % len(px.colors.qualitative.Plotly)
                        ]
                        for i in range(len(nodes_query + nodes_ref))
                    ],
                ),
                link=_filter_dict(
                    dict(
                        source=np.repeat(
                            np.arange(len(nodes_query)), len(nodes_ref)
                        ).tolist(),
                        target=np.tile(
                            np.arange(len(nodes_query), len(nodes_combined)),
                            len(nodes_query),
                        ).tolist(),
                        value=cluster_mapping.to_numpy().flatten().tolist(),
                        color=[
                            mpl.colors.rgb2hex(m.to_rgba(val))
                            for val in cluster_distance.to_numpy().flatten()
                        ],
                        label=[
                            f"distance: {val:.2f}"
                            for val in cluster_distance.to_numpy().flatten()
                        ],
                    ),
                    threshold=filter_threshold,
                ),
            )
        ]
    )
    fig.update_layout(autosize=False, width=width, height=height)
    if show:
        fig.show()
        return None
    else:
        return fig


def plot_scatterplot(adata, hue, title, width: int = 750, height: int = 600, cmap=None):
    assert "X_umap" in adata.obsm
    if cmap is None:
        cmap = {ct: color for ct, color in zip(adata.obs[hue].unique(), default_102)}
    cmap["None"] = "lightgray"

    fig = px.scatter(
        x=adata.obsm["X_umap"][:, 0],
        y=adata.obsm["X_umap"][:, 1],
        color=adata.obs[hue],
        title=title,
        width=width,
        height=height,
        color_discrete_map=cmap,
        labels={"x": "UMAP1", "y": "UMAP2"},
    )
    fig.update_traces(marker=dict(size=1))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        xaxis=dict(showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(showline=True, linecolor="black", linewidth=1, mirror=True),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            itemsizing="constant",
            title_text=None,
            tracegroupgap=6,
        ),
        legend_tracegroupgap=10,
    )

    return fig
