#! /bin/bash
#COBALT -o stdout.log -e stderr.log -A CSC249ADCD08 --attrs=mcdram=cache
# 1: Batch size

# Example submission: qsub -n 8 -t 60 -q debug-cache-quad train-theta.sh --learning-rate 1e-4 --batch-size 8192 single-task 117232

# Load up the Python environment
module load miniconda-3.6/conda-4.5.4
source activate jcesr_datascience
export PYTHONPATH=""  # Do not look back at original datascience path, I cloned it
export HDF5_USE_FILE_LOCKING=FALSE

# Set up the parallelism
export OMP_NUM_THREADS=128
export KMP_BLOCKTIME=0
export KMP_AFFINITY="granularity=fine,compact,1,0"
export MPICH_GNI_FORK_MODE=FULLCOPY
export MPICH_MAX_THREAD_SAFETY=multiple

# Invoke the model
aprun -n $COBALT_JOBSIZE -N 1 -d $OMP_NUM_THREADS -j 2 --cc depth python train_models.py $@ >> train.log
