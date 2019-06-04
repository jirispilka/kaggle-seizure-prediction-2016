#!/bin/bash

#PBS -N python-5-all-p-SLCV-2-fold
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12h
#PBS -l mem=1g
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

python spp_02_SLCV_all_subjects.py
cp *.res "$DATADIR/python"
