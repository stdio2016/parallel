SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG
PROGRAMNAME=cg
DATASIZE=MEDIUMN

include make.common

OBJS = cg.o \
       ${COMMON}/${RAND}.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/wtime.o

${PROGRAM}: config ${OBJS}
	mkdir -p ./bin
	${CLINK} ${CLINKFLAGS} $(o)${PROGRAM} ${OBJS} ${C_LIB}

OBJS_CUDA = cg_cuda.cu ${COMMON}/${RAND}.c ${COMMON}/c_timers.c ${COMMON}/wtime.c
CUDAFLAGS = -g -O3

bin/cg_cuda: $(OBJS_CUDA)
	mkdir -p ./bin
	${NVCC} $(C_INC) ${CUDAFLAGS} $(OBJS_CUDA) -D${DATASIZE} -o bin/cg_cuda

.c.o:
	${CCOMPILE} $< -D${DATASIZE} $(o)$(basename $<).o

cg.o:		cg.c  globals.h

clean:
	- rm -f *.o *~
	rm -f ${COMMON}/*.o
	rm -f bin/*
