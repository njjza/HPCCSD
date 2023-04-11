CC = g++
CFLAGS = -Wall -O2 -g -std=c++23 -fopenmp

all: main

omp: main.o
	$(CC) $(CFLAGS) -D OMP -mavx -o ./bin/eccsd.o -c eccsd.cc
	$(CC) $(CFLAGS)  -o ./bin/ccsd ./bin/main.o ./bin/eccsd.o

main: main.o eccsd.o
	$(CC) $(CFLAGS) -o ./bin/ccsd ./bin/main.o ./bin/eccsd.o

main.o: main.cc eccsd.hpp
	$(CC) $(CFLAGS) -o ./bin/main.o -c main.cc

eccsd.o: eccsd.cc eccsd.hpp
	$(CC) $(CFLAGS) -mavx -o ./bin/eccsd.o -c eccsd.cc

.PHONY: clean
clean:
	rm -f ./bin/*.o
