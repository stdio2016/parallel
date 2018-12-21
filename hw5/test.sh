make hw
tests="300M 1200M 2400M"
for i in $tests
do
	rm input
	mkfifo input
	echo $i
	cat $i > input &
	time ./histogram
	diff yyyyyy.out $i.out > /dev/null
	v=$?
	if [ $v = 0 ]
	then
		echo yes
	else
		echo no
	fi
done
rm input
