#!/bin/bash

BASENJIDIR=$CONDA_PREFIX
PATH=$(pwd)/basenji/bin:$BASENJIDIR/bin:$PATH
PYTHONPATH=$BASENJIDIR/bin:$PYTHONPATH
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export BASENJIDIR
export PATH
export PYTHOPATH
export LD_LIBRARY_PATH


echo "Checks if all variables are added:"

echo $PATH
echo $PYTHONPATH
echo $LD_LIBRARY_PATH

echo "Done"


