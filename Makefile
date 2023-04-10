CC = g++
CFLAGS = -Wall -std=c++23 -fopenmp

all: main

serial: main.o eccsd.o
	$(CC) $(CFLAGS) -D DEBUG_SERIAL -mavx -o ./bin/eccsd.o -c eccsd.cc
	$(CC) $(CFLAGS) -o ./bin/ccsd ./bin/main.o ./bin/eccsd.o

omp: main.o
	$(CC) $(CFLAGS) -D DEBUG_OMP -mavx -o ./bin/eccsd.o -c eccsd.cc
	$(CC) $(CFLAGS)  -o ./bin/ccsd ./bin/main.o ./bin/eccsd.o

main: main.o eccsd.o
	$(CC) $(CFLAGS) -o ./bin/ccsd ./bin/main.o ./bin/eccsd.o

main.o: main.cc eccsd.h
	$(CC) $(CFLAGS) -o ./bin/main.o -c main.cc

eccsd.o: eccsd.cc eccsd.h
	$(CC) $(CFLAGS) -mavx -o ./bin/eccsd.o -c eccsd.cc

.PHONY: clean
clean:
	rm -f ./bin/*.o
