import itertools
import os

JOB_SCRIPT = r"""#!/bin/bash

#SBATCH -J preprocess
#SBATCH --output=slurm_out/preprocess.%j
#SBATCH --error=slurm_out/preprocess.%j
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 2-00:00:00
#SBATCH --mem=250GB
#SBATCH --cpus-per-task=12


CONTAINER_IMAGE="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer/enroot-images/pairOT.sqsh"
CONTAINER_MOUNTS="/dss:/dss,/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer:/mnt/dssfs02"

SCRIPT="/dss/dsshome1/04/di93zer/git/dataset-similarity/scripts/python/preprocess_datasets.py"
GIT_REPO="/dss/dsshome1/04/di93zer/git/dataset-similarity"

SCRIPT_ARGS+="--n_top_genes={n_top_genes} "
SCRIPT_ARGS+="--query_dataset={query_dataset} "
SCRIPT_ARGS+="--ref_dataset={ref_dataset} "
SCRIPT_ARGS+="--query_ct_column={query_ct_column} "
SCRIPT_ARGS+="--ref_ct_column={ref_ct_column} "

srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
     bash -c "pip install -e ${{GIT_REPO}} --no-deps && python -u ${{SCRIPT}} ${{SCRIPT_ARGS}}"
"""


SEARCH_SPACE = {
    # "query": [("7d7cabfd-1d1f-40af-96b7-26a0825a306d", "cell_type_author")],
    # "ref": [
    #     (
    #         "ced320a1-29f3-47c1-a735-513c7084d508_CAP",
    #         "cell_type_author",
    #     ),
    #     (
    #         "ced320a1-29f3-47c1-a735-513c7084d508_CAP_NKT_CD3E",
    #         "cell_type_author",
    #     ),
    #     (
    #         "ced320a1-29f3-47c1-a735-513c7084d508_CAP_CD4_naive",
    #         "cell_type_author",
    #     ),
    # ],
    # "query": [("f6c50495-3361-40ed-a819-fb9644396ed9_updated", "cell_type_author")],
    # "ref": [
    #     (
    #         "ced320a1-29f3-47c1-a735-513c7084d508_CAP",
    #         "cell_type_author",
    #     ),
    #     (
    #         "ced320a1-29f3-47c1-a735-513c7084d508_CAP_CD4_naive",
    #         "cell_type_author",
    #     ),
    # ],
    # "query": [
    #     ("71f4bccf-53d4-4c12-9e80-e73bfb89e398", "cell_type_author"),
    # ],
    # "ref": [
    #     (
    #         "0f528c8a-a25c-4840-8fa3-d156fa11086f",
    #         "cell_type_author",
    #     ),
    #     (
    #         "0f528c8a-a25c-4840-8fa3-d156fa11086f_Treg",
    #         "cell_type_author",
    #     ),
    #     (
    #         "0f528c8a-a25c-4840-8fa3-d156fa11086f_aFIB",
    #         "cell_type_author",
    #     ),
    # ],
    # "query": [
    #     (
    #         "Liver_macrophages",
    #         "cell_type_author",
    #     ),
    # ],
    # "ref": [
    #     (
    #         "Ileum_macrophages",
    #         "cell_type_author",
    #     ),  # ileum
    #     (
    #         "Colon_macrophages",
    #         "cell_type_author",
    #     ),  # colon
    #     (
    #         "Colon_macrophages_felix",
    #         "cell_type_author",
    #     ),  # colon
    # ],
    "query": [
        ("arthritis_Felix", "cell_type_author"),
        ("colitis_Felix", "cell_type_author"),
    ],
    "ref": [
        ("arthritis_Felix", "cell_type_author"),
        ("colitis_Felix", "cell_type_author"),
    ],
    "n_top_genes": [750],
}


if __name__ == "__main__":
    for (
        (query, query_ct_col),
        (ref, ref_ct_col),
        n_top_genes,
    ) in itertools.product(
        SEARCH_SPACE["query"],
        SEARCH_SPACE["ref"],
        SEARCH_SPACE["n_top_genes"],
    ):
        if query == ref:
            continue
        job_script = JOB_SCRIPT.format(
            n_top_genes=n_top_genes,
            query_dataset=query,
            ref_dataset=ref,
            query_ct_column=query_ct_col,
            ref_ct_column=ref_ct_col,
        )
        with open("job_script.sbatch", "w") as f:
            f.write(job_script)
        os.system("sbatch job_script.sbatch && sleep 0.5 && rm job_script.sbatch")
