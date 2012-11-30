CC=nvcc

CFLAGS= 

LIBS=

all: sort

sort: sort.o qsort.o bsort.o pprefix.o
	${CC} ${LIBS} -o sort sort.o qsort.o bsort.o pprefix.o

sort.o: sort.cu sarray.h qsort.h pprefix.h bsort.h
	${CC} ${CFLAGS} -c sort.cu
	
qsort.o:qsort.cu qsort.h pprefix.h
	${CC} ${CFLAGS} -c qsort.cu

bsort.o: bsort.cu bsort.h pprefix.h
	${CC} ${CFLAGS} -c bsort.cu

pprefix.o: pprefix.cu pprefix.h
	${CC} ${CFLAGS} -c pprefix.cu

clean:
	rm *.o sort
