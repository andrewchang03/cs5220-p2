export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

salloc -n 1 -N 1 --qos interactive -t 01:00:00 -C cpu -c 32 -J sph -A m4776