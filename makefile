CC=gcc
CFLAGS=-Wall -Wextra -std=c99 -Wno-unused-function 
LDFLAGS=-lm

all: main xor 

main: main.o network.o matrix.o
	$(CC) $^ -o $@ $(LDFLAGS)

xor: xor.o network.o matrix.o
	$(CC) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o main xor

.PHONY: all clean