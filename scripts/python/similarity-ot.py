import argparse
import gc
import json
import os
import pickle
import warnings
from os.path import join, isfile

import scanpy as sc

from datasim.dataset_ot import DatasetMapping
from datasim.plotting import plot_cluster_mapping, plot_cluster_distance, plot_sankey

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str)
    parser.add_argument("--n_top_genes", type=int, default=1500)
    parser.add_argument("--n_genes_de_gene_overlap", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--reuse_model", action="store_true")
    parser.add_argument("--ct_col_query", type=str, default="ct2")
    return parser.parse_args()


DATA_PATH = "/mnt/dssfs02/dataset-similarity/preprocessed"
CACHE_DIR = "/mnt/dssfs02/dataset-similarity/cache"
SAVE_DIR = "/mnt/dssfs02/dataset-similarity/model-output/"
FIG_DIR = "/mnt/dssfs02/dataset-similarity/figures"

QUERY_DATASET = "7d7cabfd-1d1f-40af-96b7-26a0825a306d"
REF_DATASET = "ced320a1-29f3-47c1-a735-513c7084d508"


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # load data
    cache_file = join(
        CACHE_DIR,
        "+".join([QUERY_DATASET, REF_DATASET, f"{args.n_top_genes}HVGs"]) + ".pickle",
    )
    if os.path.isfile(cache_file):
        print("Using cached files...")
        with open(cache_file, "rb") as f:
            adata_query, adata_ref = pickle.load(f)
    else:
        # load and preprocess data
        adata_query, adata_ref = DatasetMapping.preprocess_adatas(
            sc.read_h5ad(join(DATA_PATH, f"{QUERY_DATASET}.h5ad")),
            sc.read_h5ad(join(DATA_PATH, f"{REF_DATASET}.h5ad")),
            n_top_genes=args.n_top_genes,
        )
        with open(cache_file, "wb") as f:
            pickle.dump((adata_query, adata_ref), f)

    adata_query.obs["cell_type_author"] = adata_query.obs[args.ct_col_query]
    gc.collect()

    # init and fit OT model
    dataset_map = DatasetMapping(adata_query, adata_ref)
    if (
        not isfile(join(SAVE_DIR, f"ot-solution_{args.version}.pkl"))
        or not args.reuse_model
    ):
        # ignore warnings here as scanpy.tl.rank_genes_groups throws a lot of warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset_map.init_geom(
                n_genes_de_gene_overlap=args.n_genes_de_gene_overlap,
                de_method="wilcoxon",
                p_value_threshold=0.01,
                batch_size=args.batch_size,
                epsilon=args.epsilon,
            )
        print(
            f"x size: {dataset_map.geom.x.shape[0] * dataset_map.geom.x.shape[1] * 4 / 1000**3} GB"
        )
        print(
            f"y size: {dataset_map.geom.y.shape[0] * dataset_map.geom.y.shape[1] * 4 / 1000**3} GB"
        )
        dataset_map.init_problem(
            marginals_distribution="balanced", tau_a=args.tau, tau_b=args.tau
        )
        dataset_map.solve()
        # save state of fitted model
        dataset_map.pickle_state(join(SAVE_DIR, f"ot-model-state{args.version}.pkl"))
    else:
        print("Using cached OT model...")
        dataset_map.load_state(join(SAVE_DIR, f"ot-model-state{args.version}.pkl"))

    print(dataset_map.ot_solution.reg_ot_cost)

    # compute cluster mapping and distances
    cluster_mapping = dataset_map.compute_cluster_mapping()
    cluster_distance = dataset_map.compute_cluster_distances(n_samples=25000)
    # create and save plots
    fig = plot_cluster_mapping(cluster_mapping, show=False)
    fig.write_html(join(FIG_DIR, f"cluster-mapping{args.version}.html"))
    fig = plot_cluster_distance(cluster_distance, show=False)
    fig.write_html(join(FIG_DIR, f"cluster-distance{args.version}.html"))
    fig = plot_sankey(cluster_mapping, cluster_distance, show=False)
    fig.write_html(join(FIG_DIR, f"sankey{args.version}.html"))
    # get suggestions for most similar clusters
    top_n_labels = DatasetMapping.select_most_similar_clusters(
        cluster_mapping,
        cluster_distance,
        threshold_mass=0.25,
        threshold_distance=1.0,
        n_top=None,
    )
    print(top_n_labels)
    with open(join(SAVE_DIR, f"most-similar-clusters{args.version}.json"), "w") as f:
        json.dump(top_n_labels, f, indent=4)
    # save cluster_mapping and cluster_distances
    cluster_mapping.to_parquet(join(SAVE_DIR, f"cluster_mapping{args.version}.parquet"))
    cluster_distance.to_parquet(
        join(SAVE_DIR, f"cluster_distance{args.version}.parquet")
    )
