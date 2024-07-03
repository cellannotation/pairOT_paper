import argparse
import json
import os
import warnings
from os.path import join, isfile

from datasim.dataset_ot import DatasetMapping
from datasim.plotting import plot_cluster_mapping, plot_cluster_distance, plot_sankey
from utils import load_data_cached

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str)
    parser.add_argument("--n_top_genes", type=int, default=3000)
    parser.add_argument("--n_genes_de_gene_overlap", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--reuse_model", action="store_true")
    parser.add_argument(
        "--query_dataset", type=str, default="7d7cabfd-1d1f-40af-96b7-26a0825a306d"
    )
    parser.add_argument(
        "--ref_dataset", type=str, default="ced320a1-29f3-47c1-a735-513c7084d508"
    )
    parser.add_argument("--ct_col_query", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_dir = join("/mnt/dssfs02/dataset-similarity/models/similarityOT", args.version)
    fig_dir = join("/mnt/dssfs02/dataset-similarity/figures/similarityOT", args.version)
    os.makedirs(save_dir, exist_ok=True), os.makedirs(fig_dir, exist_ok=True)
    # load data
    adata_query, adata_ref = load_data_cached(
        "/mnt/dssfs02/dataset-similarity/cache",
        "/mnt/dssfs02/dataset-similarity/preprocessed",
        args.query_dataset,
        args.ref_dataset,
        args.n_top_genes,
        args.ct_col_query,
    )
    # init and fit OT model
    dataset_map = DatasetMapping(adata_query, adata_ref)
    if not isfile(join(save_dir, f"ot-solution.pkl")) or not args.reuse_model:
        # ignore warnings here as scanpy.tl.rank_genes_groups throws a lot of warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset_map.init_geom(
                n_genes_de_gene_overlap=args.n_genes_de_gene_overlap,
                batch_size=args.batch_size,
                epsilon=args.epsilon,
            )
        dataset_map.init_problem(tau_a=args.tau, tau_b=args.tau)
        dataset_map.solve()
        # save state of fitted model
        dataset_map.pickle_state(join(save_dir, f"ot-model-state.pkl"))
    else:
        print("Using cached OT model...")
        dataset_map.load_state(join(save_dir, f"ot-model-state.pkl"))

    # compute cluster mapping, cluster distances and suggestions for most similar clusters
    mapping = dataset_map.compute_cluster_mapping()
    distance = dataset_map.compute_cluster_distances()
    top_n_labels = DatasetMapping.select_most_similar_clusters(mapping, distance)
    print(top_n_labels)
    print(f"OT-cost: {dataset_map.ot_solution.reg_ot_cost}")
    # create and save plots
    fig = plot_cluster_mapping(mapping, show=False)
    fig.write_html(join(fig_dir, f"mapping.html"))
    fig = plot_cluster_distance(distance, show=False)
    fig.write_html(join(fig_dir, f"distance.html"))
    fig = plot_sankey(mapping, distance, show=False)
    fig.write_html(join(fig_dir, f"sankey.html"))
    # save method output and used hyperparameters to disk
    mapping.to_parquet(join(save_dir, f"mapping.parquet"))
    distance.to_parquet(join(save_dir, f"distance.parquet"))
    with open(join(save_dir, f"most-similar-clusters.json"), "w") as f:
        json.dump(top_n_labels, f, indent=4)
    with open(join(save_dir, f"hparams.json"), "w") as f:
        hparams = vars(args)
        hparams["reg_ot_cost"] = float(dataset_map.ot_solution.reg_ot_cost)
        json.dump(hparams, f, indent=4)
