LIBS = -lm
MPICC = /usr/lib64/openmpi/bin/mpicc
OBJS = integrate prime

all: $(OBJS)

integrate: integrate.c
	$(MPICC) $+ -o $@ $(CFLAGS) $(LIBS)

prime: prime.c
	$(MPICC) $+ -o $@ $(CFLAGS) $(LIBS)

.PHONY: clean
clean:
	-rm $(OBJS)
