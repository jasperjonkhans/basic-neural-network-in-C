CC=gcc
CFLAGS=-Wall -Wextra -std=c99 
LDFLAGS=-lm

# Default target - build both executables
all: main xor

# Main executable
main: main.o network.o matrix.o
	$(CC) $^ -o $@ $(LDFLAGS)

# XOR executable  
xor: xor.o network.o matrix.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Generic rule for object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o main xor

# Phony targets
.PHONY: all clean