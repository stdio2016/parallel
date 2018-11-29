pushd "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"
nvcc cuda_wave.cu -Xcompiler "/source-charset:utf-8" -o cuda_wave
