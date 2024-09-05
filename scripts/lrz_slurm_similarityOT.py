import os
from itertools import product

JOB_SCRIPT = r"""#!/bin/bash

#SBATCH -J similarityOT
#SBATCH --output=slurm_out/similarityOT.%j
#SBATCH --error=slurm_out/similarityOT.%j
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 1-12:00:00
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
SCRIPT_ARGS+="--n_genes_query_ova={n_genes_query_ova} "
SCRIPT_ARGS+="--n_genes_ref_ova={n_genes_ref_ova} "
SCRIPT_ARGS+="--batch_size={batch_size} "
SCRIPT_ARGS+="--tau={tau} "
SCRIPT_ARGS+="--epsilon={epsilon} "
SCRIPT_ARGS+="--embedding_layer={embedding_layer} "

srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
     --export=XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
     bash -c "pip install -e ${{GIT_REPO}} --no-deps && python -u ${{SCRIPT}} ${{SCRIPT_ARGS}}"
"""


SEARCH_SPACE = {
    "query": {
        "values": ["7d7cabfd-1d1f-40af-96b7-26a0825a306d"],
        "include_in_version": True,
    },
    "ref": {
        "values": [
            "03f821b4-87be-4ff4-b65a-b5fc00061da7_Airway",
            "03f821b4-87be-4ff4-b65a-b5fc00061da7_PBMC",
            "4f889ffc-d4bc-4748-905b-8eb9db47a2ed",
            "b0cf0afa-ec40-4d65-b570-ed4ceacc6813",
            "b9fc3d70-5a72-4479-a046-c2cc1ab19efc",
            "ced320a1-29f3-47c1-a735-513c7084d508",
            "ddfad306-714d-4cc0-9985-d9072820c530",
            "eb735cc9-d0a7-48fa-b255-db726bf365af",
            "ed9185e3-5b82-40c7-9824-b2141590c7f0",
        ],
        "include_in_version": True,
    },
    "n_top_genes": {"values": [750], "include_in_version": True},
    "n_genes_query_ova": {
        "values": [10],
        "include_in_version": True,
    },
    "n_genes_ref_ova": {
        "values": [20],
        "include_in_version": True,
    },
    "batch_size": {"values": [4096], "include_in_version": False},
    "tau": {"values": [1.0], "include_in_version": True},
    "epsilon": {
        "values": [0.05, 0.1],
        "include_in_version": True,
    },
    "embedding_layer": {
        "values": [None],
        "include_in_version": True,
    },
    # "query": {
    #     "values": [
    #         # "0c8a364b-97b5-4cc8-a593-23c38c6f0ac5",
    #         "0f528c8a-a25c-4840-8fa3-d156fa11086f",
    #         # "2d40e6a7-f2fd-49ba-9db9-6b97e4c6dad5",
    #         # "2d40e6a7-f2fd-49ba-9db9-6b97e4c6dad5_Immune",
    #         # "48259aa8-f168-4bf5-b797-af8e88da6637_Immune",
    #         # "5c868b6f-62c5-4532-9d7f-a346ad4b50a7",
    #         # "71f4bccf-53d4-4c12-9e80-e73bfb89e398",
    #         "f6c50495-3361-40ed-a819-fb9644396ed9",
    #     ],
    #     "include_in_version": True,
    # },
    # "ref": {
    #     "values": ["ced320a1-29f3-47c1-a735-513c7084d508"],
    #     "include_in_version": True,
    # },
    # "n_top_genes": {"values": [750], "include_in_version": True},
    # "n_genes_query_ova": {
    #     "values": [10],
    #     "include_in_version": True,
    # },
    # "n_genes_ref_ova": {
    #     "values": [20],
    #     "include_in_version": True,
    # },
    # "batch_size": {"values": [4096], "include_in_version": False},
    # "tau": {"values": [1.0], "include_in_version": True},
    # "epsilon": {
    #     "values": [0.05, 0.1],
    #     "include_in_version": True,
    # },
    # "embedding_layer": {
    #     "values": ["X_scTab", "X_scimilarity"],
    #     "include_in_version": True,
    # },
    # "query": {
    #     "values": ["ced320a1-29f3-47c1-a735-513c7084d508"],
    #     "include_in_version": True,
    # },
    # "ref": {
    #     "values": ["f6c50495-3361-40ed-a819-fb9644396ed9"],
    #     "include_in_version": True,
    # },
    # "n_top_genes": {"values": [750], "include_in_version": True},
    # "n_genes_query_ova": {
    #     "values": [10],
    #     "include_in_version": True,
    # },
    # "n_genes_ref_ova": {
    #     "values": [[10, 15, 20, 25, 30, 35, 40]],
    #     "include_in_version": True,
    # },
    # "batch_size": {"values": [4096], "include_in_version": False},
    # "tau": {"values": [1.0], "include_in_version": True},
    # "epsilon": {
    #     "values": [0.05, 0.1],
    #     "include_in_version": True,
    # },
    # "embedding_layer": {"values": [None], "include_in_version": True},
}


if __name__ == "__main__":
    for (
        query,
        ref,
        n_top_genes,
        n_genes_query_ova,
        n_genes_ref_ova,
        batch_size,
        tau,
        epsilon,
        embedding_layer,
    ) in product(
        SEARCH_SPACE["query"]["values"],
        SEARCH_SPACE["ref"]["values"],
        SEARCH_SPACE["n_top_genes"]["values"],
        SEARCH_SPACE["n_genes_query_ova"]["values"],
        SEARCH_SPACE["n_genes_ref_ova"]["values"],
        SEARCH_SPACE["batch_size"]["values"],
        SEARCH_SPACE["tau"]["values"],
        SEARCH_SPACE["epsilon"]["values"],
        SEARCH_SPACE["embedding_layer"]["values"],
    ):
        n_genes_ref_ova = (
            n_genes_ref_ova if n_genes_ref_ova is int else f"'{n_genes_ref_ova}'"
        )

        version = []
        for k, v in [
            ("query", query),
            ("ref", ref),
            ("n_top_genes", n_top_genes),
            ("n_genes_query_ova", n_genes_query_ova),
            ("n_genes_ref_ova", n_genes_ref_ova),
            ("batch_size", batch_size),
            ("tau", tau),
            ("epsilon", epsilon),
            ("embedding_layer", embedding_layer),
        ]:
            if SEARCH_SPACE[k]["include_in_version"]:
                version.append(v if k in ["query", "ref"] else f"{k}={v}")

        job_script = JOB_SCRIPT.format(
            version="+".join(version),
            query_dataset=query,
            ref_dataset=ref,
            n_top_genes=n_top_genes,
            n_genes_query_ova=n_genes_query_ova,
            n_genes_ref_ova=n_genes_ref_ova,
            batch_size=batch_size,
            tau=tau,
            epsilon=epsilon,
            embedding_layer=embedding_layer,
        )
        with open("job_script.sbatch", "w") as f:
            f.write(job_script)
        os.system("sbatch job_script.sbatch && sleep 0.5 && rm job_script.sbatch")
