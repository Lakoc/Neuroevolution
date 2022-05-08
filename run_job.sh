#!/bin/bash
#PBS -N Neuroevolution
#PBS -q gpu
#PBS -l select=1:ngpus=1:cl_adan=True:gpu_cap=cuda75:mem=6gb:scratch_local=1gb
#PBS -l walltime=12:00:00
#PBS -J 1-40


# Clean up after exit
trap 'clean_scratch' EXIT

DATADIR=/storage/brno2/home/lakoc/Neuroevolution
config=$(<$DATADIR/configs_random/config"${PBS_ARRAY_INDEX}".txt)


echo "$PBS_JOBID is running on node $(hostname -f) in a scratch directory $SCRATCHDIR: $(date +"%T")"
echo "Config: $config"
cp -r "$DATADIR/main.py" "$SCRATCHDIR" || {
  echo >&2 "Couldnt copy main."
  exit 3
}

cp -r "$DATADIR/src" "$SCRATCHDIR" || {
  echo >&2 "Couldnt copy src."
  exit 3
}

cp -r "$DATADIR/requirements.txt" "$SCRATCHDIR" || {
  echo >&2 "Couldnt copy requirements."
  exit 3
}

module add conda-modules-py37
conda create -n Neuroevolution python=3.9.7
conda activate Neuroevolution
echo "ENV created. $(date +"%T") Installing requirements ..."
pip install -r "$SCRATCHDIR/requirements.txt"
echo "All packages installed. $(date +"%T")"

cd "$SCRATCHDIR" || exit 1
mkdir "$SCRATCHDIR/results"
mkdir "$SCRATCHDIR/results/best"
mkdir "$SCRATCHDIR/results/models"
mkdir "$SCRATCHDIR/results/macs"
mkdir "$SCRATCHDIR/results/params"
mkdir "$SCRATCHDIR/results/fitness"

echo "All ready. Starting evolution: $(date +"%T")"
python main.py $config

echo "Training done. Copying back to FE: $(date +"%T")"
# Copy data back to FE
cp -r "$SCRATCHDIR/results" "$DATADIR/experiments/$PBS_ARRAY_INDEX" || {
  echo >&2 "Couldnt copy results to datadir."
  exit 3
}
