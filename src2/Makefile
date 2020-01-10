CC=g++
CFLAGS = -g -std=c++11 -I /usr/include/eigen3/ -Wall -Werror 
LDLIBS  = -lglut -lGLU -lGL -lm 
OBJS = main.o 

all: geom

geom: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o geom $(LDLIBS)

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm $(OBJS) geom
