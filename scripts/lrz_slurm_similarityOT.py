import os
from time import sleep

JOB_SCRIPT = r"""#!/bin/bash

#SBATCH -J similarityOT
#SBATCH --output=slurm_out/similarityOT.%j
#SBATCH --error=slurm_out/similarityOT.%j
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 3-00:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=12


CONTAINER_IMAGE="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer/enroot-images/datasim.sqsh"
CONTAINER_MOUNTS="/dss:/dss,/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer:/mnt/dssfs02"

SCRIPT="/dss/dsshome1/04/di93zer/git/dataset-similarity/scripts/python/run_similarityOT.py"
GIT_REPO="/dss/dsshome1/04/di93zer/git/dataset-similarity"

SCRIPT_ARGS="--version={version} "
SCRIPT_ARGS+="--n_top_genes={n_top_genes} "
SCRIPT_ARGS+="--n_genes_de_gene_overlap={n_genes_de_gene_overlap} "
SCRIPT_ARGS+="--batch_size={batch_size} "
SCRIPT_ARGS+="--tau={tau} "
SCRIPT_ARGS+="--epsilon={epsilon} "
SCRIPT_ARGS+="--query_dataset={query_dataset} "
SCRIPT_ARGS+="--ref_dataset={ref_dataset} "

srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
     bash -c "pip install -e ${{GIT_REPO}} --no-deps && python -u ${{SCRIPT}} ${{SCRIPT_ARGS}}"
"""


SEARCH_SPACE = {
    "query": ["7d7cabfd-1d1f-40af-96b7-26a0825a306d"],
    "ref": [
        "03f821b4-87be-4ff4-b65a-b5fc00061da7_Airway",
        "03f821b4-87be-4ff4-b65a-b5fc00061da7_PBMC",
        "4f889ffc-d4bc-4748-905b-8eb9db47a2ed",
        "7d7cabfd-1d1f-40af-96b7-26a0825a306d",
        "b0cf0afa-ec40-4d65-b570-ed4ceacc6813",
        "b9fc3d70-5a72-4479-a046-c2cc1ab19efc",
        "ced320a1-29f3-47c1-a735-513c7084d508",
        "ddfad306-714d-4cc0-9985-d9072820c530",
        "eb735cc9-d0a7-48fa-b255-db726bf365af",
        "ed9185e3-5b82-40c7-9824-b2141590c7f0",
    ],
    "n_top_genes": [3000],
    "n_genes_de_gene_overlap": [10],
    "batch_size": [1024],
    "tau": [1.0],
    "epsilon": [0.05],
}


if __name__ == "__main__":
    for query in SEARCH_SPACE["query"]:
        for ref in SEARCH_SPACE["ref"]:
            for n_top_genes in SEARCH_SPACE["n_top_genes"]:
                for n_genes_de_gene_overlap in SEARCH_SPACE["n_genes_de_gene_overlap"]:
                    for batch_size in SEARCH_SPACE["batch_size"]:
                        for tau in SEARCH_SPACE["tau"]:
                            for epsilon in SEARCH_SPACE["epsilon"]:
                                version = f"{query}+{ref}"
                                for k, v in [
                                    ("n_top_genes", n_top_genes),
                                    (
                                        "n_genes_de_gene_overlap",
                                        n_genes_de_gene_overlap,
                                    ),
                                    ("batch_size", batch_size),
                                    ("tau", tau),
                                    ("epsilon", epsilon),
                                ]:
                                    if len(SEARCH_SPACE[k]) > 1:
                                        version += f"+{k}={v}"
                                job_script = JOB_SCRIPT.format(
                                    version=version,
                                    query_dataset=query,
                                    ref_dataset=ref,
                                    n_top_genes=n_top_genes,
                                    n_genes_de_gene_overlap=n_genes_de_gene_overlap,
                                    batch_size=batch_size,
                                    tau=tau,
                                    epsilon=epsilon,
                                )
                                with open("job_script.sbatch", "w") as f:
                                    f.write(job_script)
                                os.system("sbatch job_script.sbatch")
                                sleep(0.5)
                                os.system("rm job_script.sbatch")
