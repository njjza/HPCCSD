CC = g++
CFLAGS = -Wall -g -std=c++23

all: main

main: main.o eccsd.o
	$(CC) $(CFLAGS) -o ./bin/ccsd ./bin/main.o ./bin/eccsd.o

main.o: main.cc eccsd.h
	$(CC) $(CFLAGS) -o ./bin/main.o -c main.cc

eccsd.o: eccsd.cc eccsd.h
	$(CC) $(CFLAGS) -o ./bin/eccsd.o -c eccsd.cc

.PHONY: clean
clean:
	rm -f ./bin/*.o
