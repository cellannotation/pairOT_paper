import argparse
import json
import os
from os.path import join

import anndata
import cellhint
import numpy as np
import scanpy as sc

from sconnect.plotting import plot_cluster_mapping
from utils import load_data_cached


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str)
    parser.add_argument("--n_top_genes", type=int, default=3000)
    parser.add_argument(
        "--use_pct", default=True, type=lambda x: x.lower() in ["true", "1", "1."]
    )
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
    print(args)
    save_dir = join("/mnt/dssfs02/dataset-similarity/models/CellHint", args.version)
    fig_dir = join("/mnt/dssfs02/dataset-similarity/figures/CellHint", args.version)
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
    adata = anndata.concat(
        [adata_query, adata_ref], label="dataset", keys=["query", "ref"]
    )
    # normalize and compute PCA for CellHint
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    # fit CellHint model
    alignment = cellhint.harmonize(
        adata,
        dataset="dataset",
        cell_type="cell_type_author",
        use_rep="X_pca",
        dataset_order=["query", "ref"],
        use_pct=args.use_pct,
    )
    # compute cluster mapping
    mapping = alignment.base_distance.to_meta()
    mapping = mapping.loc[
        mapping.index.str.startswith("query"), mapping.columns.str.startswith("ref")
    ]
    mapping.index = mapping.index.str.removeprefix("query:")
    mapping.columns = mapping.columns.str.removeprefix("ref:")
    mapping = -mapping + 1.0
    # compute suggestions for most similar clusters
    top_n_labels = {}
    for ct in np.unique(adata_query.obs["cell_type_author"]):
        top_hits = alignment.relation.query(f"`query` == '{ct}'")["ref"].tolist()
        if top_hits == ["UNRESOLVED"] or top_hits == ["NONE"]:
            top_hits = []
        top_n_labels[ct] = top_hits
    print(top_n_labels)
    # create and save plots
    fig = plot_cluster_mapping(mapping, show=False)
    fig.write_html(join(fig_dir, f"mapping.html"))
    # save method output to disk
    alignment.write(join(save_dir, f"alignment.pkl"))
    mapping.to_parquet(join(save_dir, f"mapping.parquet"))
    with open(join(save_dir, f"most-similar-clusters.json"), "w") as f:
        json.dump(top_n_labels, f, indent=4)
    with open(join(save_dir, f"hparams.json"), "w") as f:
        hparams = vars(args)
        json.dump(hparams, f, indent=4)
