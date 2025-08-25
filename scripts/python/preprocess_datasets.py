import argparse
import pickle
from os.path import join

import scanpy as sc

from sconnect.preprocessing import preprocess_adatas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dataset", type=str)
    parser.add_argument("--ref_dataset", type=str)
    parser.add_argument("--n_top_genes", type=int, default=750)
    parser.add_argument("--query_ct_column", type=str, default="cell_type_author")
    parser.add_argument("--ref_ct_column", type=str, default="cell_type_author")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cache_dir = "/mnt/dssfs02/dataset-similarity/cache"
    data_path = "/mnt/dssfs02/dataset-similarity/preprocessed"

    n_top_genes = None if args.n_top_genes == -1 else args.n_top_genes
    cache_name = []
    if args.query_ct_column == "cell_type_author":
        cache_name.append(args.query_dataset)
    else:
        cache_name.append(f"{args.query_dataset}_{args.query_ct_column}")
    if args.ref_ct_column == "cell_type_author":
        cache_name.append(args.ref_dataset)
    else:
        cache_name.append(f"{args.ref_dataset}_{args.ref_ct_column}")
    cache_name.append(f"{n_top_genes}HVGs")

    cache_file = join(cache_dir, "+".join(cache_name) + ".pickle")
    adata_query = sc.read_h5ad(join(data_path, f"{args.query_dataset}.h5ad"))
    adata_ref = sc.read_h5ad(join(data_path, f"{args.ref_dataset}.h5ad"))
    adata_query, adata_ref = preprocess_adatas(
        adata_query,
        adata_ref,
        n_top_genes=n_top_genes,
        cell_type_column_adata1=args.query_ct_column,
        cell_type_column_adata2=args.ref_ct_column,
    )
    with open(cache_file, "wb") as f:
        pickle.dump((adata_query, adata_ref), f)
