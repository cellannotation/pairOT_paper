import argparse
import ast
import json
import os
from os.path import join, isfile

from pairot.dataset_ot import DatasetMapping
from pairot.plotting import plot_cluster_mapping, plot_cluster_distance, plot_sankey
from utils import load_data_cached, none_or_str


def parse_de_overlap(n_genes):
    try:
        return int(n_genes)
    except ValueError:
        return [int(elem) for elem in ast.literal_eval(n_genes)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dataset", type=str)
    parser.add_argument("--ref_dataset", type=str)
    parser.add_argument("--version", type=str)
    parser.add_argument("--ct_col_query", type=str, default=None)
    parser.add_argument("--n_top_genes", type=int, default=750)
    parser.add_argument("--n_genes_ova", type=int, default=10)
    parser.add_argument("--n_genes_ava", type=int, default=3)
    parser.add_argument("--overlap_threshold_ava", type=float, default=0.3)
    parser.add_argument("--overlap_n_genes_ava", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--embedding_layer", type=none_or_str, default=None)
    parser.add_argument("--reuse_model", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    save_dir = join("/mnt/dssfs02/dataset-similarity/models/similarityOT", args.version)
    fig_dir = join("/mnt/dssfs02/dataset-similarity/figures/similarityOT", args.version)
    os.makedirs(save_dir, exist_ok=True), os.makedirs(fig_dir, exist_ok=True)
    # load data
    adata_query, adata_ref = load_data_cached(
        "/mnt/dssfs02/dataset-similarity/cache",
        "/mnt/dssfs02/dataset-similarity/preprocessed",
        args.query_dataset,
        args.ref_dataset,
        n_top_genes=args.n_top_genes,
        ct_col_query=args.ct_col_query,
    )
    # init and fit OT model
    dataset_map = DatasetMapping(adata_query, adata_ref)
    if not isfile(join(save_dir, f"ot-solution.pkl")) or not args.reuse_model:
        dataset_map.init_geom(
            n_genes_ova=args.n_genes_ova,
            n_genes_ava=args.n_genes_ava,
            overlap_threshold_ava=args.overlap_threshold_ava,
            overlap_n_genes_ava=args.overlap_n_genes_ava,
            embedding_layer=args.embedding_layer,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
            # logfc_threshold=0.0,  # uncomment for macrophage example
        )
        dataset_map.init_problem(tau_a=args.tau, tau_b=args.tau)
        dataset_map.solve()
        dataset_map.pickle_state(join(save_dir, f"ot-model-state.pkl"))
    else:
        print("Using cached OT model...")
        dataset_map.load_state(join(save_dir, f"ot-model-state.pkl"))

    # compute cluster mapping, cluster distances and suggestions for most similar clusters
    mappings = dataset_map.compute_cluster_mapping()
    distance = dataset_map.compute_cluster_distances()
    top_n_labels = DatasetMapping.select_most_similar_clusters(
        mappings["mean"], distance
    )
    print(top_n_labels)
    print(f"OT-cost: {dataset_map.ot_solution.reg_ot_cost}")
    # create and save plots
    for key, value in mappings.items():
        fig = plot_cluster_mapping(value, zmin=0.0, zmax=1.0, show=False)
        fig.write_html(join(fig_dir, f"mapping_{key}.html"))
    fig = plot_cluster_distance(distance, show=False)
    fig.write_html(join(fig_dir, f"distance.html"))
    fig = plot_sankey(mappings["mean"], distance, show=False)
    fig.write_html(join(fig_dir, f"sankey.html"))
    # save method output and used hyperparameters to disk
    for key, value in mappings.items():
        value.to_parquet(join(save_dir, f"mapping_{key}.parquet"))
    distance.to_parquet(join(save_dir, f"distance.parquet"))
    with open(join(save_dir, f"most-similar-clusters.json"), "w") as f:
        json.dump(top_n_labels, f, indent=4)
    with open(join(save_dir, f"hparams.json"), "w") as f:
        hparams = vars(args)
        hparams["reg_ot_cost"] = float(dataset_map.ot_solution.reg_ot_cost)
        json.dump(hparams, f, indent=4)
