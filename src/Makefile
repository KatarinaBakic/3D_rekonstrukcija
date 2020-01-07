CC=g++
CFLAGS = -std=c++11 -I /usr/include/eigen3/ -Wall -Werror -g
OBJS = main.o 

all: geom

geom: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o geom 

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm $(OBJS) geom
