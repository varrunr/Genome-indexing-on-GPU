 #include <stdio.h>
 #include <cuda.h>
 #include <assert.h>
 #include <cutil.h>
 #include <stdlib.h>
 #include "MatrixMult.h"
 #include <time.h>

 int MAX_ROWS = 100;
 int MAX_COLS = 100;
 
 
 // kernel that executes on the CUDA device
__global__ void matrixMult(float *A, float *B, float *C, int m, int n){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	float value=0.0;
	int index = j*m+i, k;
	if(i < m && j < n){
		for (k = 0; k<n; k++) {
			value = value + (A[j*m+k] * B[k*n+i]);
			}
		C[index] = value;
	}
}

void printMatrix(float *Mat, int n){
	int i,j;
	for (i = 0; i< n; i++){
		for(j = 0; j< n; j++){
			printf("%f\n",Mat[j*n+i]);
		}
	}
}

void matrixMultNormal(float *A, float *B, float *C1, int m, int n){
	int i,j,k;	
	for(i = 0;i <n;i++)
		for(j = 0; j<n;j++)	
			C1[i*n+j] = 0;
	for(i = 0;i <n;i++)
		for(j = 0; j<n;j++)	
			for (k = 0; k<n; k++) {
				C1[i*n+j] = C1[i*n+j] + (A[i*n+k] * B[k*n+j]);				
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
			x[index]=(rand()%3);
		 }
	  }
}


// main routine to execute on host

void runCUDA(){
 int count = 0;
 while(count<20){
	int i,j,index;
	clock_t start, end;
	double runTime;
	
	unsigned int mem_size = sizeof(float)*MAX_ROWS*MAX_COLS;
	float *A = (float*) malloc(mem_size);
	float *B = (float*) malloc(mem_size);
	float *C = (float*) malloc(mem_size);
	float *C1 = (float*) malloc(mem_size);
	
	generate1DMatrix(A);
	generate1DMatrix(B);
	
	// allocate device memory
	float *dA, *dB, *dC;
	cudaMalloc((void **)&dA, mem_size);
	cudaMalloc((void **)&dB, mem_size);
	cudaMalloc((void **)&dC, mem_size);
	
	// copy to device from host
	
	cudaMemcpy(dA,A, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dB,B, mem_size, cudaMemcpyHostToDevice);
	
	// declare size
	dim3 block(16,16);
	dim3 grid(ceil(float(MAX_ROWS)/float(block.x)),ceil(float(MAX_COLS)/float(block.y)));

	start = clock();
	
	// invoke kernel...
	matrixMult<<<grid,block>>>(dA,dB,dC,MAX_ROWS,MAX_COLS);
    cudaThreadSynchronize();
	cudaMemcpy(C,dC, mem_size, cudaMemcpyDeviceToHost);
	
//	matrixMultNormal(A,B,C1,MAX_ROWS,MAX_COLS);

	end = clock();
	runTime = (end - start);

	printf("%d %d\n",MAX_ROWS, (int)runTime);
	
	free(A); free(B); free(C);	
	cudaFree(dA); cudaFree(dB); cudaFree(dC);

	MAX_ROWS += 100;
	MAX_COLS += 100;
	count++;
 }
}

