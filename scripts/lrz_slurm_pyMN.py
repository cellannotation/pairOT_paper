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
        "values": [
            "0c8a364b-97b5-4cc8-a593-23c38c6f0ac5",
            "0f528c8a-a25c-4840-8fa3-d156fa11086f",
            "2d40e6a7-f2fd-49ba-9db9-6b97e4c6dad5",
            "2d40e6a7-f2fd-49ba-9db9-6b97e4c6dad5_Immune",
            "48259aa8-f168-4bf5-b797-af8e88da6637_Immune",
            "5c868b6f-62c5-4532-9d7f-a346ad4b50a7",
            "71f4bccf-53d4-4c12-9e80-e73bfb89e398",
            "f6c50495-3361-40ed-a819-fb9644396ed9",
        ],
        "include_in_version": True,
    },
    "ref": {
        "values": ["ced320a1-29f3-47c1-a735-513c7084d508"],
        "include_in_version": True,
    },
    "n_top_genes": {"values": [3000], "include_in_version": True},
    "threshold": {"values": [0.9], "include_in_version": True},
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
