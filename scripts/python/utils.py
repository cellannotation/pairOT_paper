import gc
import pickle
from os.path import isfile, join
from typing import Optional

import scanpy as sc

from datasim.dataset_ot import DatasetMapping


def none_or_str(value: str):
    if value.lower() == "none":
        return None
    else:
        return value


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
        print(f"Using cached file: {cache_file}")
        with open(cache_file, "rb") as f:
            adata_query, adata_ref = pickle.load(f)
    else:
        # load and preprocess data
        adata_query, adata_ref = DatasetMapping.preprocess_adatas(
            sc.read_h5ad(join(data_path, f"{query_dataset}.h5ad")),
            sc.read_h5ad(join(data_path, f"{ref_dataset}.h5ad")),
            n_top_genes=n_top_genes,
        )

    if ct_col_query is not None:
        adata_query.obs["cell_type_author"] = adata_query.obs[ct_col_query]
    gc.collect()

    return adata_query, adata_ref
