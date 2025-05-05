import os
import re
from os.path import exists, join
from typing import Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback, dash_table, State, ctx

from datasim.plotting import plot_cluster_mapping, plot_cluster_distance

DATA_DIR = "/Users/felix.fischer/similarityOT"


models = [elem for elem in sorted(os.listdir(DATA_DIR)) if elem != ".DS_Store"]
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
                                    "# genes OVA",
                                    dcc.Dropdown(id="n_genes_ova"),
                                ]
                            ),
                            dbc.Col(
                                [
                                    "# genes AVA",
                                    dcc.Dropdown(id="n_genes_ava"),
                                ]
                            ),
                            dbc.Col(
                                [
                                    "Overlap threshold AVA",
                                    dcc.Dropdown(id="overlap_threshold_ava"),
                                ],
                            ),
                            dbc.Col(
                                [
                                    "# genes overlap AVA",
                                    dcc.Dropdown(id="overlap_n_genes_ava"),
                                ],
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
                [
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
                                    dcc.Dropdown(
                                        ["yes", "no"], value="no", id="transpose"
                                    ),
                                ]
                            ),
                            dbc.Col(
                                [
                                    "Sorting",
                                    dcc.Dropdown(
                                        ["no", "alphabetical", "score"],
                                        value="score",
                                        id="sorting",
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(html.Div(id="mapping")),
                        dbc.Col(html.Div(id="distance")),
                    ]
                ),
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "Save Plots as SVG",
                                    color="primary",
                                    id="save_button",
                                    n_clicks=0,
                                )
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Save Plots as HTML",
                                    color="primary",
                                    id="save_button_html",
                                    n_clicks=0,
                                )
                            ),
                            dbc.Col(
                                dbc.Input(
                                    id="save_filename",
                                    type="text",
                                    placeholder="Enter output filename name here ...",
                                )
                            ),
                        ]
                    ),
                ]
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
    n_genes_ova,
    n_genes_ava,
    tau,
    epsilon,
    overlap_threshold_ava,
    overlap_n_genes_ava,
):
    version = dataset
    version += f"+n_genes_ova={n_genes_ova}"
    version += f"+n_genes_ava={n_genes_ava}"
    version += f"+tau={tau}"
    version += f"+epsilon={epsilon}"
    version += f"+overlap_threshold_ava={overlap_threshold_ava}"
    version += f"+overlap_n_genes_ava={overlap_n_genes_ava}"

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


@callback(Output("n_genes_ova", "options"), Input("dataset", "value"))
def update_options(value):
    return select_options(value, "n_genes_ova")


@callback(Output("n_genes_ava", "options"), Input("dataset", "value"))
def update_options(value):
    return select_options(value, "n_genes_ava")


@callback(Output("overlap_threshold_ava", "options"), Input("dataset", "value"))
def update_options(value):
    return select_options(value, "overlap_threshold_ava")


@callback(Output("overlap_n_genes_ava", "options"), Input("dataset", "value"))
def update_options(value):
    return select_options(value, "overlap_n_genes_ava")


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
    Input("n_genes_ova", "value"),
    Input("n_genes_ava", "value"),
    Input("overlap_threshold_ava", "value"),
    Input("overlap_n_genes_ava", "value"),
    Input("transpose", "value"),
    Input("sorting", "value"),
    Input("save_button", "n_clicks"),
    Input("save_button_html", "n_clicks"),
    State("save_filename", "value"),
)
def heatmap_plots(
    aggregation,
    dataset,
    epsilon,
    tau,
    n_genes_ova,
    n_genes_ava,
    overlap_threshold_ava,
    overlap_n_genes_ava,
    transpose,
    sorting,
    n_clicks,
    n_clicks_html,
    save_filename,
):
    mapping, distance = read_files(
        aggregation,
        dataset,
        n_genes_ova,
        n_genes_ava,
        tau,
        epsilon,
        overlap_threshold_ava,
        overlap_n_genes_ava,
    )
    if mapping is not None and distance is not None:
        match sorting:
            case "alphabetical":
                mapping = mapping.sort_index().sort_index(axis=1)
                distance = distance.sort_index().sort_index(axis=1)
            case "score":
                mapping = mapping.loc[
                    mapping.max(axis=1).sort_values(ascending=False).index.tolist(),
                    mapping.max().sort_values(ascending=False).index.tolist(),
                ]
                distance = distance.loc[
                    mapping.max(axis=1).sort_values(ascending=False).index.tolist(),
                    mapping.max().sort_values(ascending=False).index.tolist(),
                ]

        if transpose == "yes":
            mapping = mapping.T
            distance = distance.T

        mapping_plot = plot_cluster_mapping(
            mapping, zmin=0.0, zmax=1.0, show=False, sort_by_score=False
        )
        distance_plot = plot_cluster_distance(distance, show=False)
        for fig in [mapping_plot, distance_plot]:
            fig.update_layout(coloraxis_showscale=False)

        if "save_button" == ctx.triggered_id:
            os.makedirs("plots/svg", exist_ok=True)
            mapping_plot.write_image(f"plots/svg/{save_filename}_mapping.svg")
            distance_plot.write_image(f"plots/svg/{save_filename}_distance.svg")

        if "save_button_html" == ctx.triggered_id:
            os.makedirs("plots/html", exist_ok=True)
            mapping_plot.write_html(f"plots/html/{save_filename}_mapping.html")
            distance_plot.write_html(f"plots/html/{save_filename}_distance.html")

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
    Input("n_genes_ova", "value"),
    Input("n_genes_ava", "value"),
    Input("overlap_threshold_ava", "value"),
    Input("overlap_n_genes_ava", "value"),
)
def suggestions_table(
    aggregation,
    threshold_mass,
    threshold_distance,
    dataset,
    epsilon,
    tau,
    n_genes_ova,
    n_genes_ava,
    overlap_threshold_ava,
    overlap_n_genes_ava,
):
    mapping, distance = read_files(
        aggregation,
        dataset,
        n_genes_ova,
        n_genes_ava,
        tau,
        epsilon,
        overlap_threshold_ava,
        overlap_n_genes_ava,
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
