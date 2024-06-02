from typing import Dict, Tuple, Optional, List, Literal

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
from scanpy.tools._rank_genes_groups import _Method
from scipy.sparse import issparse
from scipy.stats import rankdata
from sklearn.utils.class_weight import compute_class_weight

from datasim.label_distance import de_gene_overlap_label_distance
from datasim.utils import get_expressed_genes_intersection


def cosine_distance(x: jnp.ndarray, y: jnp.ndarray):
    """Cosine distance between vectors, denominator regularized with ridge."""
    x_norm = jnp.linalg.norm(x, axis=-1)
    y_norm = jnp.linalg.norm(y, axis=-1)
    cosine_similarity = jnp.vdot(x, y) / (x_norm * y_norm + 1e-8)
    return 0.5 * (1.0 - cosine_similarity)


def _compute_balanced_marginal(cell_types: pd.Series) -> np.ndarray:
    """Compute marginal balanced by cell cell_type."""
    assert isinstance(cell_types.dtype, pd.CategoricalDtype)

    classes = cell_types.cat.categories.to_numpy()
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=cell_types.to_numpy()
    )
    marginal = weights[cell_types.cat.codes.to_numpy()]

    return marginal / marginal.sum()


def _convert_to_dense_numpy(arr):
    if issparse(arr):
        return arr.toarray().astype("f4")
    else:
        return arr.astype("f4")


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
        Subset the input anndata.AnnData object to the top `n_top_genes` highly variable genes.

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
        # subset gene space to genes that are expressed in both datasets
        intersection_genes = get_expressed_genes_intersection(adata1, adata2)
        adata1 = adata1[:, intersection_genes]
        adata2 = adata2[:, intersection_genes]
        # subset to highly variable genes
        adata = anndata.concat([adata1, adata2], label="dataset", keys=["query", "ref"])
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key="dataset")
        highly_variable = adata.var["highly_variable"].to_numpy().copy()

        return adata1[:, highly_variable].copy(), adata2[:, highly_variable].copy()

    def _compute_label_distances(
        self,
        n_genes: int = 15,
        de_method: _Method = "wilcoxon",
        p_value_threshold: float = 0.01,
    ):
        """Compute distance between labels/clusters based on the overlap of differentially expressed genes."""
        # normalize data before doing DE tests
        adata1_test = self.adata1.copy()
        adata2_test = self.adata2.copy()
        sc.pp.normalize_total(adata1_test, target_sum=1e4)
        sc.pp.normalize_total(adata2_test, target_sum=1e4)
        sc.pp.log1p(adata1_test)
        sc.pp.log1p(adata2_test)

        label_distance = de_gene_overlap_label_distance(
            adata2_test,
            adata1_test,
            cell_type_column="cell_type_author",
            n_genes=n_genes,
            method=de_method,
            p_value_threshold=p_value_threshold,
        )
        label_distance_ordered = np.zeros(label_distance.shape)
        for i, label1 in enumerate(adata1_test.obs["cell_type_author"].cat.categories):
            for j, label2 in enumerate(
                adata2_test.obs["cell_type_author"].cat.categories
            ):
                label_distance_ordered[i, j] = label_distance.loc[label1, label2]

        return jnp.array(label_distance_ordered.astype("f4"))

    def _preprocess_data(self, embedding_layer: str = None):
        adata1 = self.adata1
        adata2 = self.adata2
        if embedding_layer is None:
            x1 = _convert_to_dense_numpy(adata1.X)
            x2 = _convert_to_dense_numpy(adata2.X)
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
        n_genes_de_gene_overlap: int = 15,
        de_method: _Method = "wilcoxon",
        p_value_threshold: float = 0.01,
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
        n_genes_de_gene_overlap: int = 15
            Number of genes used to calculate overlap of top differentially expressed genes.
        de_method: Literal["logreg", "t-test", "wilcoxon", "t-test_overestim_var"] = "wilcoxon"
            Method used to calculate differentially expressed genes.
            See sc.tl.rank_genes_groups for more details.
        p_value_threshold: float = 0.01
            Minimum p-value to consider a gene as differentially expressed.
        kwargs: Dict[str, Any]
            Keyword arguments passed to :class:`ott.geometry.pointcloud.PointCloud`.
        """
        x, y = self._preprocess_data()
        label_distance = self._compute_label_distances(
            n_genes=n_genes_de_gene_overlap,
            de_method=de_method,
            p_value_threshold=p_value_threshold,
        )

        self.geom = pointcloud.PointCloud(
            x,
            y,
            cost_fn=CellCellTransportCost(
                label_distance, lambda_feature=lambda_feature, lambda_label=lambda_label
            ),
            **kwargs,
        )

    def init_problem(
        self,
        marginals_distribution: Literal["uniform", "balanced"] = "balanced",
        **kwargs,
    ):
        """
        Initialize the optimal transport problem.
        This function calls the constructor of :class:`ott.problem.linear.linear_problem.LinearProblem`.

        Parameters
        ----------
        marginals_distribution: Literal["uniform", "balanced"] = "uniform"
            Whether the marginals should be uniform or balanced by cell-type frequency.
            Use "uniform" for uniform marginals.
            Use "balanced" for marginals balanced by cell-type frequency.
            This parameter is ignored, if marginals 'a' or 'b' are supplied via the **kwargs.
        kwargs: Dict[str, Any]
            Keyword arguments passed to :class:`ott.problem.linear.linear_problem.LinearProblem`.
        """
        assert self.geom is not None
        if marginals_distribution == "uniform":
            a, b = None, None
        elif marginals_distribution == "balanced":
            a = _compute_balanced_marginal(self.adata1.obs["cell_type_author"])
            b = _compute_balanced_marginal(self.adata2.obs["cell_type_author"])
        else:
            raise ValueError(
                f"marginals argument must be either 'uniform' or 'balanced'. You provided: {marginals_distribution}"
            )

        if ("a" not in kwargs) and ("b" not in kwargs):
            self.ot_prob = linear_problem.LinearProblem(self.geom, a=a, b=b, **kwargs)
        elif "a" in kwargs:
            self.ot_prob = linear_problem.LinearProblem(self.geom, b=b, **kwargs)
        elif "b" in kwargs:
            self.ot_prob = linear_problem.LinearProblem(self.geom, a=a, **kwargs)

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

    def compute_cluster_mapping(self, normalize: bool = False) -> pd.DataFrame:
        """
        Compute the mapping between cell-type clusters based on the aggregated transport matrix.
        Aggregation is done by cluster/cell-type.

        Parameters
        ----------
        normalize: bool = False
            Whether to normalize each row of the aggregated transport matrix to sum up to 1.

        Returns
        -------
        :class:`pandas.DataFrame` containing the mapping between cell-type clusters.
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

    def compute_cluster_distances(self, n_samples: int = 25000) -> pd.DataFrame:
        """
        Compute the distance between cell-type clusters based on the optimal transport mappings.

        Parameters
        ----------
        n_samples: int = 10000
            The number of samples based on which the distance is calculated.

        Returns
        -------
        :class:`pandas.DataFrame` containing the distance between cell-type clusters.
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

    def project_query_labels_onto_reference(self) -> np.ndarray:
        """
        Project the cell-type labels from the query dataset (adata1) onto the reference dataset (adata2).

        Returns
        -------
        :class:`numpy.ndarray`
        Cell-type labels of the query dataset (adata1) project onto the cells of the reference dataset (adata2).
        """
        assert self.geom is not None
        assert self.ot_prob is not None
        assert self.ot_solution is not None

        x_label_np, y_label_np = self._get_label_vectors()
        unique_labels_x, unique_labels_y = np.unique(x_label_np), np.unique(y_label_np)
        transport = {}
        for label_x in tqdm.tqdm(unique_labels_x):
            transport[label_x] = np.array(
                self.ot_solution.apply(
                    jnp.array(np.array(x_label_np == label_x).astype("f4"))
                )
            ).astype("f8")

        transport_np = np.array([transport[i] for i in sorted(transport.keys())])
        labels = self.adata1.obs.cell_type_author.cat.categories.to_numpy()

        return labels[np.argmax(transport_np, axis=0)]

    @staticmethod
    def select_most_similar_clusters(
        cluster_mapping: pd.DataFrame,
        cluster_distance: pd.DataFrame,
        threshold: float = 1.0,
        n_top: int = 3,
    ) -> Dict[str, List[str]]:
        """
        Select the `n_top` most similar cell-type clusters in the reference data for each cell-type cluster in the query
        data based on the aggregated transport matrix.

        Parameters
        ----------
        cluster_mapping: pd.DataFrame
            The cluster mapping matrix / aggregated transport matrix. The output of self.compute_cluster_mapping().
        cluster_distance: pd.DataFrame
            The cluster distance matrix. The output of self.compute_cluster_distances().
        threshold: float = 1.0
            The maximum distance between two cell-type clusters for a cluster to be suggested as a most similar
            cell-type cluster.
        n_top: int = 3
            Maximum number of most similar clusters to return.

        Returns
        -------
        Returns the most similar cell-type clusters in the reference dataset for each cell-type cluster in the query
        data as a :class:`Dict[str, List[str]]`
        """
        top_n_labels = {}
        most_similar_clusters = np.argsort(-cluster_mapping.to_numpy(), axis=1)[
            :, :n_top
        ]
        labels_query = cluster_mapping.index.tolist()
        labels_ref = cluster_mapping.columns

        for i, label in enumerate(labels_query):
            distances = cluster_distance.loc[label, :].to_numpy()
            top_n_labels[label] = [
                labels_ref[j]
                for j in most_similar_clusters[i, :]
                if distances[j] <= threshold
            ]

        return top_n_labels
