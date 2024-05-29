import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scanpy.plotting.palettes import default_102


def _plot_heatmap(data: pd.DataFrame, colormap: str, width: int, height: int):
    fig = px.imshow(data, text_auto=".2f", color_continuous_scale=colormap)
    fig.update_layout(autosize=False, width=width, height=height)
    return fig


def plot_cluster_mapping(
    data: pd.DataFrame, width: int = 1000, height: int = 1000, show: bool = True
):
    fig = _plot_heatmap(data, "Greens", width, height)
    if show:
        fig.show()
        return None
    else:
        return fig


def plot_cluster_distance(
    data: pd.DataFrame, width: int = 1000, height: int = 1000, show: bool = True
):
    fig = _plot_heatmap(data, "Greens_r", width, height)
    if show:
        fig.show()
        return None
    else:
        return fig


def plot_sankey(
    cluster_mapping: pd.DataFrame,
    cluster_distance: pd.DataFrame,
    filter_threshold: float = 0.1,
    width: int = 1000,
    height: int = 1200,
    show: bool = True,
):
    def _filter_dict(unfiltered_dict, threshold=0.1):
        vals = pd.DataFrame(unfiltered_dict)
        vals_filtered = vals[vals.value >= threshold]
        return {col: vals_filtered[col].tolist() for col in vals_filtered.columns}

    norm = mpl.colors.Normalize(vmin=0.0, vmax=2.0)
    cmap = mpl.cm.get_cmap("Greens_r")
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
