more_files=""
mv histogram.cpp h1stogram.cpp
sed s/xxxxxx/$1/ h1stogram.cpp > histogram.cpp
zip $1.zip histogram.cpp $more_files
rm histogram.cpp
mv h1stogram.cpp histogram.cpp
