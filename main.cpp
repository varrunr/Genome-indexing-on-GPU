#include <stdio.h>
#include <cutil.h>
#include <cuda.h>
#include "MatrixMult.h"

int main (int argc, char * const argv[]) {
    // insert code here...
	printf("Matrix Multiplication Calculation...\n");
	runCUDA();
	printf("Done !!\n");
    return 0;
}
