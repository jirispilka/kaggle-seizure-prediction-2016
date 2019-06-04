#!/bin/bash

#PBS -N spec1-3
#PBS -l nodes=1:ppn=9
#PBS -l walltime=12h
#PBS -l mem=8g
#PBS -l scratch=8g
#PBS -l matlab=1
#PBS -j oe
#PBS -m e

# let's initialize modules
. /packages/run/modules-2.0/init/sh

trap "clean_scratch" TERM EXIT

# let's add the maple module
module add matlab

# let's set the necessary variables
DATADIR="$PBS_O_WORKDIR"

# let's copy the input data
cp -r $DATADIR/matlab/ "$SCRATCHDIR/"
# cp -r $DATADIR/data_desc/ "$SCRATCHDIR/"
# cp -r $DATADIR/features/ "$SCRATCHDIR/"
# cp -r $DATADIR/toolboxes/ "$SCRATCHDIR/"
# cp -r $DATADIR/utils/ "$SCRATCHDIR/"

# let's change the working directory
cd "$SCRATCHDIR/matlab"

# let's perform the computation
matlab -nosplash -nodisplay -nodesktop <spm_01_compute_features.m >results-spec-p1-3.log

# let's copy-out the result
cp *.mat $DATADIR
cp *.log $DATADIR
