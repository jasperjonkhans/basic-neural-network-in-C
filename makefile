CC=gcc
CFLAGS=-Wall -Wextra -std=c99
LDFLAGS=-lm

main: main.o network.o matrix.o
	$(CC) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o main