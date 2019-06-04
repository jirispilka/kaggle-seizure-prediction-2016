#!/bin/bash

#PBS -N python-3-SLCV-lr
#PBS -l nodes=1:ppn=4
#PBS -l walltime=6h
#PBS -l mem=4g
#PBS -l scratch=4g
#PBS -j oe
#PBS -m be

# let's initialize modules
. /packages/run/modules-2.0/init/sh

trap "clean_scratch" TERM EXIT

#!/bin/bash
export PYTHONUSERBASE=/storage/praha1/home/spilkjir/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python2.7/site-packages:$PYTHONPATH

# let's set the necessary variables
DATADIR="$PBS_O_WORKDIR"

# let's copy the input data
cp -r $DATADIR/python/ "$SCRATCHDIR/"
cp -r $DATADIR/features/ "$SCRATCHDIR/"

# let's change the working directory
cd "$SCRATCHDIR/python"

python spp_02_SLCV_gridsearch.py 1
cp *.res "$DATADIR/python"
python spp_02_SLCV_gridsearch.py 2
cp *.res "$DATADIR/python"
python spp_02_SLCV_gridsearch.py 3
cp *.res "$DATADIR/python"
