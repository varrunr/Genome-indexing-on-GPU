 #include <stdio.h>
 #include <assert.h>
 #include <stdlib.h>
 #include <time.h>

 int MAX_ROWS = 100;
 int MAX_COLS = 100;
 
 // kernel that executes on the CUDA device
void matrixMult(float *A, float *B, float *C, int m, int n){
	float value=0.0;
	int i,j,k;	
	for(i = 0;i <m;i++)
		for(j = 0; j<n;j++)	
			for (k = 0; k<n; k++) {
				value = value + (A[i*m+k] * B[k*n+j]);
				C[i*m+j] = value;
			}
}


void generate1DMatrix(float *x){

	int i,j,m,n,index;
	m = MAX_ROWS;
	n = MAX_COLS;
	 
	for(i=0;i<m;i++)
	 {
		for(j=0;j<n;j++)
		 {
			index = i*m + j;
			x[index]=(rand()%100);
		 }
	  }
 
}


int main()
{
    int count = 0;
    while(count<20){
	    int i,j,index;
	    clock_t start, end;
	    double runTime;
	
	    unsigned int mem_size = sizeof(float)*MAX_ROWS*MAX_COLS;
	    float *A = (float*) malloc(mem_size);
	    float *B = (float*) malloc(mem_size);
	    float *C = (float*) malloc(mem_size);
	    generate1DMatrix(A);
	    generate1DMatrix(B);
	
	    start = clock();
	    matrixMult(A,B,C,MAX_ROWS,MAX_COLS);
	    end = clock();
	    runTime = (end - start) / (double) CLOCKS_PER_SEC ;
	
    	printf("%d\t%f\n",MAX_ROWS,runTime);
	
	    free(A); free(B); free(C);
        MAX_ROWS += 100;
        MAX_COLS += 100;
        count++;
    }
	return 0;
}
