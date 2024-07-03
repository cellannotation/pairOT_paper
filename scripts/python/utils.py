import gc
import pickle
from os.path import isfile, join
from typing import Optional

import scanpy as sc

from datasim.dataset_ot import DatasetMapping


def load_data_cached(
    cache_dir: str,
    data_path: str,
    query_dataset: str,
    ref_dataset: str,
    n_top_genes: int,
    ct_col_query: Optional[str] = None,
):
    cache_file = join(
        cache_dir,
        "+".join([query_dataset, ref_dataset, f"{n_top_genes}HVGs"]) + ".pickle",
    )
    if isfile(cache_file):
        print("Using cached files...")
        with open(cache_file, "rb") as f:
            adata_query, adata_ref = pickle.load(f)
    else:
        # load and preprocess data
        adata_query, adata_ref = DatasetMapping.preprocess_adatas(
            sc.read_h5ad(join(data_path, f"{query_dataset}.h5ad")),
            sc.read_h5ad(join(data_path, f"{ref_dataset}.h5ad")),
            n_top_genes=n_top_genes,
        )
        with open(cache_file, "wb") as f:
            pickle.dump((adata_query, adata_ref), f)

    # for all the other datasets the default column is already "cell_type_author"
    if query_dataset == "7d7cabfd-1d1f-40af-96b7-26a0825a306d" and ct_col_query is None:
        adata_query.obs["cell_type_author"] = adata_query.obs["ct2"]
    if ct_col_query is not None:
        adata_query.obs["cell_type_author"] = adata_query.obs[ct_col_query]
    gc.collect()

    return adata_query, adata_ref
