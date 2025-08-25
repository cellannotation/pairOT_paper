import argparse
import json
import os
from os.path import join

import anndata
import pymn

from sconnect.plotting import plot_cluster_mapping
from utils import load_data_cached


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str)
    parser.add_argument("--n_top_genes", type=int, default=3000)
    parser.add_argument("--threshold", type=float, default=0.9)
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
    save_dir = join("/mnt/dssfs02/dataset-similarity/models/pyMN", args.version)
    fig_dir = join("/mnt/dssfs02/dataset-similarity/figures/pyMN", args.version)
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
    # MetaNeighbor only works on dense data
    adata.X = adata.X.toarray()
    # MetaNeighbor only does not work with categorical columns
    adata.obs["dataset"] = adata.obs["dataset"].astype(str)
    adata.obs["cell_type_author"] = adata.obs["cell_type_author"].astype(str)
    # fit MetaNeighbor model
    pymn.variableGenes(adata, study_col="dataset")
    pymn.MetaNeighborUS(
        adata, study_col="dataset", ct_col="cell_type_author", fast_version=True
    )
    # compute cluster mapping
    mapping = adata.uns["MetaNeighborUS"]
    mapping = mapping.loc[
        mapping.index.str.startswith("query"), mapping.columns.str.startswith("ref")
    ]
    mapping.index = mapping.index.str.removeprefix("query|")
    mapping.columns = mapping.columns.str.removeprefix("ref|")
    # compute suggestions for most similar clusters
    top_hits_mn = pymn.topHits(adata, threshold=args.threshold, save_uns=False)
    ind = top_hits_mn["Study_ID|Celltype_1"].str.startswith("ref|")
    (
        top_hits_mn.loc[ind, "Study_ID|Celltype_1"],
        top_hits_mn.loc[ind, "Study_ID|Celltype_2"],
    ) = (
        top_hits_mn.loc[ind, "Study_ID|Celltype_2"],
        top_hits_mn.loc[ind, "Study_ID|Celltype_1"],
    )
    top_hits_mn["Study_ID|Celltype_1"] = top_hits_mn[
        "Study_ID|Celltype_1"
    ].str.removeprefix("query|")
    top_hits_mn["Study_ID|Celltype_2"] = top_hits_mn[
        "Study_ID|Celltype_2"
    ].str.removeprefix("ref|")
    top_n_labels = (
        top_hits_mn[["Study_ID|Celltype_1", "Study_ID|Celltype_2"]]
        .groupby("Study_ID|Celltype_1")["Study_ID|Celltype_2"]
        .unique()
        .map(list)
        .to_dict()
    )
    print(top_n_labels)
    # create and save plots
    fig = plot_cluster_mapping(mapping, show=False)
    fig.write_html(join(fig_dir, f"mapping.html"))
    # save method output to disk
    mapping.to_parquet(join(save_dir, f"mapping.parquet"))
    with open(join(save_dir, f"most-similar-clusters.json"), "w") as f:
        json.dump(top_n_labels, f, indent=4)
    with open(join(save_dir, f"hparams.json"), "w") as f:
        hparams = vars(args)
        json.dump(hparams, f, indent=4)
