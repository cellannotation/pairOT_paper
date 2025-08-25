import os
from itertools import product

JOB_SCRIPT = r"""#!/bin/bash

#SBATCH -J similarityOT
#SBATCH --output=slurm_out/similarityOT.%j
#SBATCH --error=slurm_out/similarityOT.%j
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 1-00:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=12


CONTAINER_IMAGE="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer/enroot-images/datasim.sqsh"
CONTAINER_MOUNTS="/dss:/dss,/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer:/mnt/dssfs02"

SCRIPT="/dss/dsshome1/04/di93zer/git/dataset-similarity/scripts/python/run_similarityOT.py"
GIT_REPO="/dss/dsshome1/04/di93zer/git/dataset-similarity"

SCRIPT_ARGS="--query_dataset={query_dataset} "
SCRIPT_ARGS+="--ref_dataset={ref_dataset} "
SCRIPT_ARGS+="--version={version} "
SCRIPT_ARGS+="--n_top_genes={n_top_genes} "
SCRIPT_ARGS+="--n_genes_ova={n_genes_ova} "
SCRIPT_ARGS+="--n_genes_ava={n_genes_ava} "
SCRIPT_ARGS+="--batch_size={batch_size} "
SCRIPT_ARGS+="--tau={tau} "
SCRIPT_ARGS+="--epsilon={epsilon} "
SCRIPT_ARGS+="--embedding_layer={embedding_layer} "
SCRIPT_ARGS+="--overlap_threshold_ava={overlap_threshold_ava} "
SCRIPT_ARGS+="--overlap_n_genes_ava={overlap_n_genes_ava} "


srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
     --export=XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
     bash -c "pip install -e ${{GIT_REPO}} --no-deps && python -u ${{SCRIPT}} ${{SCRIPT_ARGS}}"
"""


SEARCH_SPACE = {
    # "query": {
    #     "values": ["7d7cabfd-1d1f-40af-96b7-26a0825a306d"],
    #     "include_in_version": True,
    # },
    # "ref": {
    #     "values": [
    #         "ced320a1-29f3-47c1-a735-513c7084d508_CAP_NKT_CD3E",
    #         "ced320a1-29f3-47c1-a735-513c7084d508_CAP",
    #         "ced320a1-29f3-47c1-a735-513c7084d508_CAP_CD4_naive",
    #     ],
    #     "include_in_version": True,
    # },
    #
    # "query": {
    #     "values": [
    #         "f6c50495-3361-40ed-a819-fb9644396ed9_updated",
    #     ],
    #     "include_in_version": True,
    # },
    # "ref": {
    #     "values": [
    #         "ced320a1-29f3-47c1-a735-513c7084d508_CAP",
    #         "ced320a1-29f3-47c1-a735-513c7084d508_CAP_CD4_naive",
    #     ],
    #     "include_in_version": True,
    # },
    #
    # "query": {
    #     "values": [
    #         "71f4bccf-53d4-4c12-9e80-e73bfb89e398",
    #     ],
    #     "include_in_version": True,
    # },
    # "ref": {
    #     "values": [
    #         "0f528c8a-a25c-4840-8fa3-d156fa11086f",
    #         "0f528c8a-a25c-4840-8fa3-d156fa11086f_Treg",
    #         "0f528c8a-a25c-4840-8fa3-d156fa11086f_aFIB",
    #     ],
    #     "include_in_version": True,
    # },
    #
    # "query": {
    #     "values": [
    #         "Liver_macrophages",
    #     ],
    #     "include_in_version": True,
    # },
    # "ref": {
    #     "values": [
    #         "Ileum_macrophages",  # Ileum
    #         "Colon_macrophages",  # Colon
    #         "Colon_macrophages_felix",  # Colon
    #     ],
    #     "include_in_version": True,
    # },
    #
    "query": {
        "values": ["arthritis_Felix", "colitis_Felix"],
        "include_in_version": True,
    },
    "ref": {
        "values": ["arthritis_Felix", "colitis_Felix"],
        "include_in_version": True,
    },
    "n_top_genes": {"values": [750], "include_in_version": False},
    "n_genes_ova": {
        "values": [10],  # 15 for macrophage example
        "include_in_version": True,
    },
    "n_genes_ava": {
        "values": [3],
        "include_in_version": True,
    },
    "batch_size": {"values": [4096], "include_in_version": False},
    "tau": {
        "values": [0.95, 0.975, 1.0],  # 0.95 for bone marrow and macrophage examples
        "include_in_version": True,
    },
    "epsilon": {
        "values": [0.05],
        "include_in_version": True,
    },
    "embedding_layer": {
        "values": [None],
        "include_in_version": False,
    },
    "overlap_threshold_ava": {
        "values": [0.3],
        "include_in_version": True,
    },
    "overlap_n_genes_ava": {
        "values": [10],
        "include_in_version": True,
    },
}


if __name__ == "__main__":
    for (
        query,
        ref,
        n_top_genes,
        n_genes_ova,
        n_genes_ava,
        batch_size,
        tau,
        epsilon,
        embedding_layer,
        overlap_threshold_ava,
        overlap_n_genes_ava,
    ) in product(
        SEARCH_SPACE["query"]["values"],
        SEARCH_SPACE["ref"]["values"],
        SEARCH_SPACE["n_top_genes"]["values"],
        SEARCH_SPACE["n_genes_ova"]["values"],
        SEARCH_SPACE["n_genes_ava"]["values"],
        SEARCH_SPACE["batch_size"]["values"],
        SEARCH_SPACE["tau"]["values"],
        SEARCH_SPACE["epsilon"]["values"],
        SEARCH_SPACE["embedding_layer"]["values"],
        SEARCH_SPACE["overlap_threshold_ava"]["values"],
        SEARCH_SPACE["overlap_n_genes_ava"]["values"],
    ):
        if query == ref:
            continue

        version = []
        for k, v in [
            ("query", query),
            ("ref", ref),
            ("n_top_genes", n_top_genes),
            ("n_genes_ova", n_genes_ova),
            ("n_genes_ava", n_genes_ava),
            ("batch_size", batch_size),
            ("tau", tau),
            ("epsilon", epsilon),
            ("embedding_layer", embedding_layer),
            ("overlap_threshold_ava", overlap_threshold_ava),
            ("overlap_n_genes_ava", overlap_n_genes_ava),
        ]:
            if SEARCH_SPACE[k]["include_in_version"]:
                version.append(v if k in ["query", "ref"] else f"{k}={v}")

        job_script = JOB_SCRIPT.format(
            version="+".join(version),
            query_dataset=query,
            ref_dataset=ref,
            n_top_genes=n_top_genes,
            n_genes_ova=n_genes_ova,
            n_genes_ava=n_genes_ava,
            batch_size=batch_size,
            tau=tau,
            epsilon=epsilon,
            embedding_layer=embedding_layer,
            overlap_threshold_ava=overlap_threshold_ava,
            overlap_n_genes_ava=overlap_n_genes_ava,
        )
        with open("job_script.sbatch", "w") as f:
            f.write(job_script)
        os.system("sbatch job_script.sbatch && sleep 0.5 && rm job_script.sbatch")
