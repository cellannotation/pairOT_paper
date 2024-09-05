import argparse
import pickle
from os.path import join

import scanpy as sc

from datasim.preprocessing import preprocess_adatas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dataset", type=str)
    parser.add_argument("--ref_dataset", type=str)
    parser.add_argument("--n_top_genes", type=int, default=750)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cache_dir = "/mnt/dssfs02/dataset-similarity/cache"
    data_path = "/mnt/dssfs02/dataset-similarity/preprocessed"

    n_top_genes = None if args.n_top_genes == -1 else args.n_top_genes

    cache_file = join(
        cache_dir,
        "+".join([args.query_dataset, args.ref_dataset, f"{n_top_genes}HVGs"])
        + ".pickle",
    )
    adata_query = sc.read_h5ad(join(data_path, f"{args.query_dataset}.h5ad"))
    adata_ref = sc.read_h5ad(join(data_path, f"{args.ref_dataset}.h5ad"))
    adata_query, adata_ref = preprocess_adatas(
        adata_query, adata_ref, n_top_genes=n_top_genes
    )
    with open(cache_file, "wb") as f:
        pickle.dump((adata_query, adata_ref), f)
