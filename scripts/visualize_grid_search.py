import os
import re
from os.path import exists, join
from typing import Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback, dash_table

from datasim.plotting import plot_cluster_mapping, plot_cluster_distance

DATA_DIR = "/Users/felix.fischer/similarityOT"


models = sorted(os.listdir(DATA_DIR))
datasets = sorted(list(set(["+".join(version.split("+")[:2]) for version in models])))


app = Dash(external_stylesheets=[dbc.themes.MATERIA])
app.layout = [
    html.H1("Cluster Similarity", style={"textAlign": "center"}),
    dbc.Card(
        [
            dbc.CardHeader("Dataset"),
            dbc.CardBody(dcc.Dropdown(datasets, datasets[0], id="dataset")),
            dbc.CardBody(
                dash_table.DataTable(
                    id="version_table",
                    page_size=10,
                    filter_action="native",
                    sort_action="native",
                    page_action="native",
                )
            ),
        ],
    ),
    dbc.Card(
        [
            dbc.CardHeader(children="Model Parameters"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(["Epsilon", dcc.Dropdown(id="epsilon")]),
                            dbc.Col(["Tau", dcc.Dropdown(id="tau")]),
                            dbc.Col(
                                [
                                    "# genes DE overlap (query)",
                                    dcc.Dropdown(id="N genes DE overlap (query)"),
                                ]
                            ),
                            dbc.Col(
                                [
                                    "# genes DE overlap (ref)",
                                    dcc.Dropdown(id="N genes DE overlap (ref)"),
                                ]
                            ),
                            dbc.Col(
                                [
                                    "Embedding",
                                    dcc.Dropdown(id="embedding"),
                                ],
                            ),
                            dbc.Col(
                                ["# top genes", dcc.Dropdown(id="n_top_genes")],
                            ),
                        ],
                    )
                ]
            ),
        ],
    ),
    dbc.Card(
        [
            dbc.CardHeader("Mapping + Distance Visualization"),
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                "Aggregation method",
                                dcc.Dropdown(
                                    [
                                        "mean",
                                        "jensen_shannon",
                                        "transported_mass",
                                    ],
                                    value="mean",
                                    id="aggregation",
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                "Transpose matrices",
                                dcc.Dropdown(["yes", "no"], value="no", id="transpose"),
                            ]
                        ),
                        dbc.Col(
                            [
                                "Sort rows",
                                dcc.Dropdown(["yes", "no"], value="no", id="sort-rows"),
                            ]
                        ),
                        dbc.Col(
                            [
                                "Sort columns",
                                dcc.Dropdown(["yes", "no"], value="no", id="sort-cols"),
                            ]
                        ),
                    ]
                ),
            ),
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(html.Div(id="mapping")),
                        dbc.Col(html.Div(id="distance")),
                    ]
                ),
            ),
        ]
    ),
    dbc.Card(
        [
            dbc.CardHeader("Connected Cell-type Clusters"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.P("Mass threshold"),
                                        dbc.Input(
                                            id="threshold_mass",
                                            type="number",
                                            value=0.25,
                                            min=0.0,
                                        ),
                                    ]
                                ),
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.P("Distance threshold"),
                                        dbc.Input(
                                            id="threshold_distance",
                                            type="number",
                                            value=1.0,
                                            min=0.0,
                                            max=2.0,
                                        ),
                                    ]
                                ),
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.P("Aggregation method"),
                                        dcc.Dropdown(
                                            [
                                                "mean",
                                                "jensen_shannon",
                                                "transported_mass",
                                            ],
                                            value="mean",
                                            id="aggregation_suggestions",
                                        ),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ]
            ),
            dbc.CardBody(
                [
                    dash_table.DataTable(
                        id="suggestion_table",
                        filter_action="native",
                        sort_action="native",
                        style_cell={"textAlign": "left"},
                    )
                ]
            ),
        ]
    ),
]


def select_options(value, param: str):
    selected_models = [model for model in models if model.startswith(value)]
    vals = []
    for model in selected_models:
        if model in selected_models:
            # match = re.search(param + r"=(-?\d+(\.\d+)?|[a-zA-Z_]+)", model)
            match = re.search(param + r"=(.*?)(\+|$)", model)
            if match:
                vals.append(match.group(1))
    try:
        vals = sorted(set(vals), reverse=True, key=lambda x: float(x))
    except (ValueError, TypeError):
        vals = sorted(set(vals), reverse=True)

    if any([param not in model for model in selected_models]):
        vals.append("None")

    return vals


def read_files(
    aggregation,
    dataset,
    n_top_genes,
    n_genes_de_overlap_query,
    n_genes_de_overlap_ref,
    tau,
    epsilon,
    embedding,
):
    version = dataset
    version += f"+n_top_genes={n_top_genes}"
    version += f"+n_genes_query_ova={n_genes_de_overlap_query}"
    version += f"+n_genes_ref_ova={n_genes_de_overlap_ref}"
    version += f"+tau={tau}"
    version += f"+epsilon={epsilon}"
    version += f"+embedding_layer={embedding}"

    mapping = join(DATA_DIR, version, f"mapping_{aggregation}.parquet")
    distance = join(DATA_DIR, version, "distance.parquet")
    if exists(mapping) and exists(distance):
        return pd.read_parquet(mapping), pd.read_parquet(distance)
    else:
        return None, None


def select_most_similar_clusters(
    cluster_mapping: pd.DataFrame,
    cluster_distance: pd.DataFrame,
    threshold_mass: float = 0.25,
    threshold_distance: float = 1.0,
) -> Dict[str, List[str]]:
    top_n_labels = {}
    labels_query, labels_ref = cluster_mapping.index.tolist(), cluster_mapping.columns
    for i, label in enumerate(labels_query):
        distance = cluster_distance.loc[label, :].to_numpy()
        mass = cluster_mapping.loc[label, :].to_numpy()
        top_n_labels[label] = [
            labels_ref[s]
            for s in np.argsort(-mass)
            if distance[s] <= threshold_distance and mass[s] >= threshold_mass
        ]

    return top_n_labels


@callback(Output("epsilon", "options"), Input("dataset", "value"))
def update_options(value):
    return select_options(value, "epsilon")


@callback(Output("tau", "options"), Input("dataset", "value"))
def update_options(value):
    return select_options(value, "tau")


@callback(Output("N genes DE overlap (query)", "options"), Input("dataset", "value"))
def update_options(value):
    return select_options(value, "n_genes_query_ova")


@callback(Output("N genes DE overlap (ref)", "options"), Input("dataset", "value"))
def update_options(value):
    return select_options(value, "n_genes_ref_ova")


@callback(Output("embedding", "options"), Input("dataset", "value"))
def update_options(value):
    return select_options(value, "embedding_layer")


@callback(Output("n_top_genes", "options"), Input("dataset", "value"))
def update_options(value):
    return select_options(value, "n_top_genes")


@callback(
    Output("version_table", "data"),
    Output("version_table", "columns"),
    Input("dataset", "value"),
)
def show_versions(value):
    return (
        [
            {"version": model.removeprefix(f"{value}+")}
            for model in models
            if model.startswith(value)
        ],
        [{"name": "Version", "id": "version"}],
    )


@callback(
    Output("mapping", "children"),
    Output("distance", "children"),
    Input("aggregation", "value"),
    Input("dataset", "value"),
    Input("epsilon", "value"),
    Input("tau", "value"),
    Input("N genes DE overlap (query)", "value"),
    Input("N genes DE overlap (ref)", "value"),
    Input("embedding", "value"),
    Input("n_top_genes", "value"),
    Input("transpose", "value"),
    Input("sort-rows", "value"),
    Input("sort-cols", "value"),
)
def heatmap_plots(
    aggregation,
    dataset,
    epsilon,
    tau,
    n_genes_de_overlap_query,
    n_genes_de_overlap_ref,
    embedding,
    n_top_genes,
    transpose,
    sort_rows,
    sort_cols,
):
    mapping, distance = read_files(
        aggregation,
        dataset,
        n_top_genes,
        n_genes_de_overlap_query,
        n_genes_de_overlap_ref,
        tau,
        epsilon,
        embedding,
    )
    if mapping is not None and distance is not None:
        if sort_rows == "yes":
            mapping = mapping.sort_index()
            distance = distance.sort_index()
        if sort_cols == "yes":
            mapping = mapping.sort_index(axis=1)
            distance = distance.sort_index(axis=1)
        if transpose == "yes":
            mapping = mapping.T
            distance = distance.T
        mapping_plot = plot_cluster_mapping(mapping, zmin=0.0, zmax=1.0, show=False)
        distance_plot = plot_cluster_distance(distance, show=False)
        for fig in [mapping_plot, distance_plot]:
            fig.update_layout(coloraxis_showscale=False)
        return dcc.Graph(figure=mapping_plot), dcc.Graph(figure=distance_plot)
    else:
        error_message = html.Div("No data found for selected version")
        return error_message, error_message


@callback(
    Output("suggestion_table", "data"),
    Output("suggestion_table", "columns"),
    Input("aggregation", "value"),
    Input("threshold_mass", "value"),
    Input("threshold_distance", "value"),
    Input("dataset", "value"),
    Input("epsilon", "value"),
    Input("tau", "value"),
    Input("N genes DE overlap (query)", "value"),
    Input("N genes DE overlap (ref)", "value"),
    Input("embedding", "value"),
    Input("n_top_genes", "value"),
)
def suggestions_table(
    aggregation,
    threshold_mass,
    threshold_distance,
    dataset,
    epsilon,
    tau,
    n_genes_de_overlap_query,
    n_genes_de_overlap_ref,
    embedding,
    n_top_genes,
):
    mapping, distance = read_files(
        aggregation,
        dataset,
        n_top_genes,
        n_genes_de_overlap_query,
        n_genes_de_overlap_ref,
        tau,
        epsilon,
        embedding,
    )
    table_columns = [
        {"name": "Cluster", "id": "Cluster"},
        {"name": "Suggestions", "id": "Suggestions"},
    ]
    if (
        mapping is not None
        and distance is not None
        and threshold_mass is not None
        and threshold_distance is not None
    ):
        suggestions = select_most_similar_clusters(
            mapping, distance, threshold_mass, threshold_distance
        )
        suggestions = pd.DataFrame(
            {
                "Cluster": list(suggestions.keys()),
                "Suggestions": [", ".join(elem) for elem in list(suggestions.values())],
            }
        )
        return suggestions.to_dict("records"), table_columns
    else:
        return [], table_columns


if __name__ == "__main__":
    app.run(debug=True)
