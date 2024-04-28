import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_heatmap(
    data: pd.DataFrame, width: int = 1000, height: int = 1000, show: bool = True
):
    fig = px.imshow(data, text_auto=".2f")
    fig.update_layout(autosize=False, width=width, height=height)
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
    cmap = cm.YlOrBr
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
