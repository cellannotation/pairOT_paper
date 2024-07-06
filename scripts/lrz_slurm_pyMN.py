import os
from itertools import product

JOB_SCRIPT = r"""#!/bin/bash

#SBATCH -J pyMN
#SBATCH --output=slurm_out/pyMN.%j
#SBATCH --error=slurm_out/pyMN.%j
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 1-00:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=12


CONTAINER_IMAGE="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer/enroot-images/datasim_extended.sqsh"
CONTAINER_MOUNTS="/dss:/dss,/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer:/mnt/dssfs02"

SCRIPT="/dss/dsshome1/04/di93zer/git/dataset-similarity/scripts/python/run_pyMN.py"
GIT_REPO="/dss/dsshome1/04/di93zer/git/dataset-similarity"

SCRIPT_ARGS="--version={version} "
SCRIPT_ARGS+="--n_top_genes={n_top_genes} "
SCRIPT_ARGS+="--threshold={threshold} "
SCRIPT_ARGS+="--query_dataset={query_dataset} "
SCRIPT_ARGS+="--ref_dataset={ref_dataset} "

srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
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
    "n_top_genes": {"values": [3000], "include_in_version": False},
    "threshold": {"values": [0.9], "include_in_version": False},
}


if __name__ == "__main__":
    for query, ref, n_top_genes, threshold in product(
        SEARCH_SPACE["query"]["values"],
        SEARCH_SPACE["ref"]["values"],
        SEARCH_SPACE["n_top_genes"]["values"],
        SEARCH_SPACE["threshold"]["values"],
    ):
        version = []
        for k, v in [
            ("query", query),
            ("ref", ref),
            ("n_top_genes", n_top_genes),
            ("threshold", threshold),
        ]:
            if SEARCH_SPACE[k]["include_in_version"]:
                version.append(v if k in ["query", "ref"] else f"{k}={v}")

        job_script = JOB_SCRIPT.format(
            version="+".join(version),
            query_dataset=query,
            ref_dataset=ref,
            n_top_genes=n_top_genes,
            threshold=threshold,
        )
        with open("job_script.sbatch", "w") as f:
            f.write(job_script)
        os.system("sbatch job_script.sbatch && sleep 0.5 && rm job_script.sbatch")
