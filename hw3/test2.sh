mpiexec=/usr/lib64/openmpi/bin/mpiexec
$mpiexec -n $1 -hostfile hostfile --map-by node $2 $3
