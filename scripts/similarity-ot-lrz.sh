#!/bin/bash

#SBATCH -J similarity-OT
#SBATCH --output=slurm_out/similarity-OT.%j
#SBATCH --error=slurm_out/similarity-OT.%j
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 3-00:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=12


CONTAINER_IMAGE="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer/enroot-images/datasim.sqsh"
CONTAINER_MOUNTS="/dss:/dss,/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer:/mnt/dssfs02"

SCRIPT="/dss/dsshome1/04/di93zer/git/dataset-similarity/scripts/python/similarity-ot.py"
GIT_REPO="/dss/dsshome1/04/di93zer/git/dataset-similarity"

SCRIPT_ARGS="--version=_CoarseLabels_tau=1.00 --tau=1.0 --batch_size=1024 --ct_col_query=ct1"

srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
     bash -c "pip install -e ${GIT_REPO} --no-deps && python -u ${SCRIPT} ${SCRIPT_ARGS}"
