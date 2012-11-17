CC=gcc
CLAGS=-fopenmp -g -O2
BIN_NAME=mandel

default: mandel

mandel: mandel.c
	$(CC) $(CFLAGS) -o $(BIN_NAME) mandel.c

clean:
	rm $(BIN_NAME)
