CC=gcc
CFLAGS=-std=c99 -fopenmp -g -O2
LFLAGS=-lrt
BIN_NAME=mandel

default: mandel

mandel: mandel.c timer.o
	$(CC) $(CFLAGS) -o $(BIN_NAME) mandel.c timer.o $(LFLAGS)

timer.o: timer.c
	$(CC) $(CFLAGS) -c -o timer.o timer.c

run: mandel
	./$(BIN_NAME) > $(BIN_NAME).out

image: run
	convert -depth 8 -size 501x400 gray:$(BIN_NAME).out $(BIN_NAME).jpg

show: image
	display $(BIN_NAME).jpg

clean:
	rm $(BIN_NAME) *.o
