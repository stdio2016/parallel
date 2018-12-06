./cuda_wave $1 $2 > test.txt
./serial_wave $1 $2 > good.txt
./compare test.txt good.txt
