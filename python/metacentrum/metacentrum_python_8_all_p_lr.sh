#!/bin/bash

#PBS -N python-8-all-p-SLCV-LR
#PBS -l nodes=1:ppn=8#excl
#PBS -l walltime=1d
#PBS -l mem=8g
#PBS -l scratch=8g
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

python spp_02_SLCV_all_subjects.py 2 50
cp *.res "$DATADIR/python"
python spp_02_SLCV_all_subjects.py 4 30
cp *.res "$DATADIR/python"
python spp_02_SLCV_all_subjects.py 5 20
cp *.res "$DATADIR/python"
python spp_02_SLCV_all_subjects.py 6 20
cp *.res "$DATADIR/python"
python spp_02_SLCV_all_subjects.py 10 10
cp *.res "$DATADIR/python"

