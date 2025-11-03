BOARD_DIM ?= 11

.PHONY: all run clean

all: hex

hex: hex.c
	gcc -DBOARD_DIM=$(BOARD_DIM) -o hex hex.c

run: hex
	./hex

clean:
	rm -f hex
