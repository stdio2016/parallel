if [ "$1" = "" ]
then
	echo 0487
	exit 87
fi
more_files=""
mv histogram.cpp h1stogram.cpp
sed s/yyyyyy/$1/ h1stogram.cpp > histogram.cpp
zip $1.zip histogram.cpp $more_files
rm histogram.cpp
mv h1stogram.cpp histogram.cpp
