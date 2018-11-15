mpiexec=/usr/lib64/openmpi/bin/mpiexec
if [[ $1 -gt 4 ]]
then
$mpiexec -n $1 -hostfile hostfile --map-by node $2 $3
else
$mpiexec -n $1 $2 $3
fi
