from typing import Dict, Tuple, Optional, List

import anndata
import jax
import jax.numpy as jnp
import numpy as np
import ott
import pandas as pd
import scanpy as sc
import tqdm
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.utils import tqdm_progress_fn
from scipy.stats import rankdata

from datasim.label_distance import de_gene_overlap_label_distance


def cosine_distance(x: jnp.ndarray, y: jnp.ndarray):
    """Cosine distance between vectors, denominator regularized with ridge."""
    x_norm = jnp.linalg.norm(x, axis=-1)
    y_norm = jnp.linalg.norm(y, axis=-1)
    cosine_similarity = jnp.vdot(x, y) / (x_norm * y_norm + 1e-8)
    return 0.5 * (1.0 - cosine_similarity)


@jax.tree_util.register_pytree_node_class
class CellCellTransportCost(ott.geometry.costs.CostFn):
    """Cost function to calculate the cell to cell transport cost."""

    def __init__(
        self,
        label_distance_matrix,
        lambda_feature: float,
        lambda_label: float,
    ):
        super().__init__()
        self.label_distance = label_distance_matrix
        self.lambda_feature = lambda_feature
        self.lambda_label = lambda_label

    def pairwise(self, x, y):
        x_, x_label = x[:-1], x[-1].astype(int)
        y_, y_label = y[:-1], y[-1].astype(int)

        return (
            self.lambda_feature * cosine_distance(x_, y_)
            + self.lambda_label * self.label_distance[x_label, y_label]
        )

    def tree_flatten(self):  # noqa: D102
        return [], {
            "label_distance_matrix": self.label_distance,
            "lambda_feature": self.lambda_feature,
            "lambda_label": self.lambda_label,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  # noqa: D102
        del children
        return cls(**aux_data)


class DatasetMapping:
    def __init__(self, adata1: anndata.AnnData, adata2: anndata.AnnData):
        self._validate_input(adata1, adata2)

        self.adata1 = adata1
        self.adata2 = adata2

        self.geom: Optional[pointcloud.PointCloud] = None
        self.ot_prob: Optional[linear_problem.LinearProblem] = None
        self.ot_solution: Optional[sinkhorn.SinkhornOutput] = None

    @staticmethod
    def _validate_input(adata1: anndata.AnnData, adata2: anndata.AnnData):
        assert "cell_type_author" in adata1.obs.columns
        assert "cell_type_author" in adata2.obs.columns
        for adata in [adata1, adata2]:
            assert adata.obs.cell_type_author.dtype == "category"
            assert np.allclose(
                np.unique(adata.obs.cell_type_author.cat.codes).astype("i8"),
                np.arange(len(adata.obs.cell_type_author.cat.categories), dtype="i8"),
            )

    @staticmethod
    def preprocess_adatas(
        adata1: anndata.AnnData,
        adata2: anndata.AnnData,
        n_top_genes: int = 2000,
    ) -> Tuple[anndata.AnnData, anndata.AnnData]:
        """
        Preprocess input data.

        This includes the following steps:
        1. Normalize total counts per cell via sc.pp.normalize_total
        2. Log1p-transform input data
        3. Select `n_top_genes` highly variable genes

        Parameters
        ----------
        adata1: anndata.AnnData
            Query data.
        adata2: anndata.AnnData
            Reference data.
        n_top_genes: int = 2000
            Number of highly variable genes to use.

        Returns
        -------
        Tuple[anndata.AnnData, anndata.AnnData]
        """
        sc.pp.normalize_total(adata1)
        sc.pp.log1p(adata1)
        sc.pp.normalize_total(adata2)
        sc.pp.log1p(adata2)
        adata = anndata.concat([adata1, adata2], label="dataset", keys=["ref", "query"])
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key="dataset")

        return (
            adata1[:, adata.var.highly_variable].copy(),
            adata2[:, adata.var.highly_variable].copy(),
        )

    def _compute_label_distances(self):
        """Compute distance between labels/clusters based on the overlap of differentially expressed genes."""
        label_distance = de_gene_overlap_label_distance(
            self.adata2,
            self.adata1,
            cell_type_column="cell_type_author",
            n_genes=50,
            method="t-test_overestim_var",
        )
        label_distance_ordered = np.zeros(label_distance.shape)
        for i, label1 in enumerate(self.adata1.obs["cell_type_author"].cat.categories):
            for j, label2 in enumerate(
                self.adata2.obs["cell_type_author"].cat.categories
            ):
                label_distance_ordered[i, j] = label_distance.loc[label1, label2]

        return jnp.array(label_distance_ordered.astype("f4"))

    def _preprocess_data(self, embedding_layer: str = None):
        adata1 = self.adata1
        adata2 = self.adata2
        if embedding_layer is None:
            x1 = adata1.X.toarray()
            x2 = adata2.X.toarray()
            # rank and zero center data that cosine similarity is equal to Spearman correlation
            x1 = rankdata(x1, axis=1)
            x2 = rankdata(x2, axis=1)
            x1 = x1 - x1.mean(axis=1, keepdims=True)
            x2 = x2 - x2.mean(axis=1, keepdims=True)
        else:
            x1 = adata1.obsm[embedding_layer]
            x2 = adata2.obsm[embedding_layer]
        y1 = adata1.obs.cell_type_author.cat.codes.to_numpy().astype("i8")
        y2 = adata2.obs.cell_type_author.cat.codes.to_numpy().astype("i8")
        x = np.hstack([x1.astype("f4"), y1.reshape((-1, 1)).astype("f4")])
        y = np.hstack([x2.astype("f4"), y2.reshape((-1, 1)).astype("f4")])

        return jnp.array(x), jnp.array(y)

    def init_geom(
        self,
        lambda_feature: float = 1.0,
        lambda_label: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the geometry of the optimal transport problem.
        This function calls the constructor of :class:`ott.geometry.pointcloud.PointCloud`.

        Parameters
        ----------
        lambda_feature: float = 1.0
            Weight for the distance in gene/feature space for the cell to cell transport cost.
        lambda_label: float = 1.0
            Weight for the distance in the label space for the cell to cell transport cost.
        kwargs: Dict[str, Any]
            Keyword arguments passed to :class:`ott.geometry.pointcloud.PointCloud`.
        """
        x, y = self._preprocess_data()
        label_distance = self._compute_label_distances()

        self.geom = pointcloud.PointCloud(
            x,
            y,
            cost_fn=CellCellTransportCost(
                label_distance, lambda_feature=lambda_feature, lambda_label=lambda_label
            ),
            **kwargs,
        )

    def init_problem(self, **kwargs):
        """
        Initialize the optimal transport problem.
        This function calls the constructor of :class:`ott.problem.linear.linear_problem.LinearProblem`.

        Parameters
        ----------
        kwargs: Dict[str, Any]
            Keyword arguments passed to :class:`ott.problem.linear.linear_problem.LinearProblem`.
        """
        assert self.geom is not None
        self.ot_prob = linear_problem.LinearProblem(self.geom, **kwargs)

    def solve(self, **kwargs):
        """
        Solve the underlying optimal transport problem.
        This function uses :class:`ott.solvers.linear.sinkhorn.Sinkhorn` to solve the optimal transport problem.

        Parameters
        ----------
        kwargs: Dict[str, Any]
            Keyword arguments passed to :class:`ott.solvers.linear.sinkhorn.Sinkhorn`.
        """
        assert self.geom is not None
        assert self.ot_prob is not None

        with tqdm.tqdm() as pbar:
            progress_fn = tqdm_progress_fn(pbar)
            solver = sinkhorn.Sinkhorn(progress_fn=progress_fn, **kwargs)
            self.ot_solution = jax.jit(solver)(self.ot_prob)

    def _get_label_vectors(self):
        assert self.geom is not None
        return tuple(
            np.array(arr[:, -1]).astype("i8") for arr in [self.geom.x, self.geom.y]
        )

    def compute_cluster_mapping(self, normalize: bool = True) -> pd.DataFrame:
        """
        Compute the mapping between cell-type clusters based on the aggregated transport matrix.
        Aggregation is done by cluster/cell-type.

        Parameters
        ----------
        normalize: bool = True
            Whether to normalize each row of the aggregated transport matrix to sum up to 1.

        Returns
        -------
        Returns :class:`pandas.DataFrame` containing the mapping between cell-type clusters.
        """
        assert self.geom is not None
        assert self.ot_prob is not None
        assert self.ot_solution is not None

        x_label_np, y_label_np = self._get_label_vectors()
        unique_labels_x, unique_labels_y = np.unique(x_label_np), np.unique(y_label_np)

        transport_map_agg = np.zeros((len(unique_labels_x), len(unique_labels_y)))
        for label_x in tqdm.tqdm(unique_labels_x):
            transported_mass = np.array(
                self.ot_solution.apply(
                    jnp.array(np.array(x_label_np == label_x).astype("f4"))
                )
            ).astype("f8")
            for label_y in unique_labels_y:
                transport_map_agg[label_x, label_y] = transported_mass[
                    y_label_np == label_y
                ].sum()

        if normalize:
            transport_map_agg /= transport_map_agg.sum(axis=1, keepdims=True)

        return pd.DataFrame(
            transport_map_agg,
            index=self.adata1.obs.cell_type_author.cat.categories,
            columns=self.adata2.obs.cell_type_author.cat.categories,
        )

    def compute_cluster_distances(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Compute the distance between cell-type clusters based on the optimal transport mappings.

        Parameters
        ----------
        n_samples: int = 10000
            The number of samples based on which the distance is calcuated.

        Returns
        -------
        Returns :class:`pandas.DataFrame` containing the distance between cell-type clusters.
        """
        assert self.geom is not None
        assert self.ot_prob is not None
        assert self.ot_solution is not None

        def select_idxs(label_arr: np.ndarray, label: int, n: int) -> jnp.ndarray:
            idxs_label = np.where(label_arr == label)[0]
            idxs_label = np.random.choice(
                idxs_label, min(n, len(idxs_label)), replace=False
            )
            return jnp.array(idxs_label)

        x_label_np, y_label_np = self._get_label_vectors()
        unique_labels_x, unique_labels_y = np.unique(x_label_np), np.unique(y_label_np)

        with tqdm.tqdm(total=len(unique_labels_x) * len(unique_labels_y)) as pbar:
            weighted_cost = np.zeros((len(unique_labels_x), len(unique_labels_y)))
            for label_x in unique_labels_x:
                for label_y in unique_labels_y:
                    idxs_label_x = select_idxs(x_label_np, label_x, n_samples)
                    idxs_label_y = select_idxs(y_label_np, label_y, n_samples)
                    geom_subset = self.geom.subset(idxs_label_x, idxs_label_y)
                    transport_matrix = geom_subset.transport_from_potentials(
                        self.ot_solution.f[idxs_label_x],
                        self.ot_solution.g[idxs_label_y],
                    )
                    transport_matrix /= transport_matrix.sum()
                    cost = self.geom.cost_fn.all_pairs(
                        self.geom.x[idxs_label_x, :], self.geom.y[idxs_label_y, :]
                    )
                    weighted_cost[label_x, label_y] = jnp.sum(cost * transport_matrix)
                    pbar.update(1)

        return pd.DataFrame(
            weighted_cost,
            index=self.adata1.obs.cell_type_author.cat.categories,
            columns=self.adata2.obs.cell_type_author.cat.categories,
        )

    @staticmethod
    def select_most_similar_clusters(
        cluster_mapping: pd.DataFrame, threshold: float = 0.1, n_top: int = 5
    ) -> Dict[str, List[str]]:
        """
        Select the `n_top` most similar cell-type clusters in the reference data for each cell-type cluster in the query
        data based on the aggregated transport matrix.

        Parameters
        ----------
        cluster_mapping: pd.DataFrame
            The cluster mapping matrix / aggregated transport matrix. The output of self.compute_cluster_mapping().
        threshold: float = 0.1
            The minimum value in the aggregated transport matrix for a cluster to be considered.
        n_top: int = 5
            Maximum number of most similar clusters to return.

        Returns
        -------
        Returns the most similar cell-type clusters in the reference dataset for each cell-type cluster in the query
        data as a :class:`Dict[str, List[str]]`
        """
        top_n_labels = {}
        for label_query in cluster_mapping.index:
            elems = (
                cluster_mapping.loc[label_query]
                .T.sort_values(ascending=False)
                .head(n_top)
            )
            top_n_labels[label_query] = elems[elems >= threshold].index.tolist()

        return top_n_labels
