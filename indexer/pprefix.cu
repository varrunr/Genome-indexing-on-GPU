#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>

#define CUDA_SAFE_CALL(x) x
using namespace std;

void block_calculation(dim3* blockGridRows, dim3* threadBlockRows, int* block_size, int prefixesCount){
	///////////////////////////////////////////////////////////////////
	// Get Device Properties
	cudaDeviceProp prop;
	int count, max_threads_per_block;
	int maxBlockGridWidth;
	cudaGetDeviceCount(&count);
	cudaGetDeviceProperties(&prop, 0);
	max_threads_per_block = prop.maxThreadsPerBlock;
	maxBlockGridWidth = prop.maxGridSize[0];
	///////////////////////////////////////////////////////////////////

	*block_size = max_threads_per_block;
	int blockGridWidth = prefixesCount/(*block_size) + 1;
	int blockGridHeight = 1;
	if(blockGridWidth > maxBlockGridWidth)
	{
		blockGridWidth = maxBlockGridWidth;		
		blockGridHeight = (prefixesCount/(maxBlockGridWidth * (*block_size))) + 1;
	}

	(*blockGridRows).x = blockGridWidth;
	(*blockGridRows).y = blockGridHeight;
	(*threadBlockRows).x = (*block_size);
	(*threadBlockRows).y = 1;
}
__global__ void block_scan_write_up( int *g_idata, int block_offset, int block_size, int start, int end, int n)
{
    int tid = (int) (threadIdx.x + blockDim.x * threadIdx.y + \
							( blockIdx.x * blockDim.x * blockDim.y ) \
							+ ( blockIdx.y * blockDim.x * blockDim.y * gridDim.x)); 

    int prev_block_offset = block_offset/block_size;

    if((((tid+1)*prev_block_offset) - 1) >=start && (((tid+1)*prev_block_offset) - 1)<=end)
    {
	        
	    int prev_block_index = (((tid+1)*prev_block_offset) - 1);
	    
	    int x = prev_block_index/block_offset;

	    if((prev_block_index+1) % block_offset != 0 && x > 0)
	    {
	    	g_idata[prev_block_index] += g_idata[(x)*block_offset - 1];    	
	    }    	
    }
}

__global__ void block_scan( int *g_idata, int block_offset, int block_size, int start, int end, int n) 
{ 
    extern __shared__ int temp[]; // allocated on invocation 

    int tid = (int) (threadIdx.x + blockDim.x * threadIdx.y + \
							( blockIdx.x * blockDim.x * blockDim.y ) \
							+ ( blockIdx.y * blockDim.x * blockDim.y * gridDim.x)); 

    int thid = threadIdx.x; 
	int pout = 0, pin = 1; 
	int n1 = block_size;
	temp[thid] = 0;
	temp[n1+thid] = 0;

    // load input into shared memory.  
    // This is exclusive scan, so shift right by one and set first elt to 0 
	if( (((tid+1)*block_offset) - 1) >= start && (((tid+1)*block_offset) - 1) <= end)
	{
	    temp[pout*n1 + thid] = g_idata[((tid+1)*block_offset) - 1];  
	    temp[pin*n1 + thid] = g_idata[((tid+1)*block_offset) - 1]; 
	}
	else
	{
	    temp[pout*n1 + thid] = 0; 
	    temp[pin*n1 + thid] = 0; 
	}

    __syncthreads(); 

    for (int offset = 1; offset < n1; offset *= 2) 
    { 
        pout = 1 - pout; // swap double buffer indices 
        pin  = 1 - pin; 

        if (thid >= offset) 
            temp[pout*n1+thid] = temp[pin*n1+thid] + temp[pin*n1+thid - offset]; 
   		else 
  			temp[pout*n1+thid] = temp[pin*n1+thid]; 

        __syncthreads(); 
    }

	if( (((tid+1)*block_offset) - 1) >= start && (((tid+1)*block_offset) - 1) <= end)
		g_idata[((tid+1)*block_offset) - 1] = temp[pout*n1+thid];
}

void prefixCompute( int *gpu_prefix_arr, int prefixesCount)
{
	dim3 blockGridRows, threadBlockRows;
	int block_size;
	block_calculation(&blockGridRows, &threadBlockRows, &block_size, prefixesCount);

	int sharedMemory = 2*block_size*sizeof(int);

	int block_offset = 0;
	int l = log2((float)prefixesCount)/log2((float)block_size);

	for(int i=0; i<=l; i++)
	{
		block_offset = pow((float)block_size, i);
		block_scan <<<  blockGridRows, 
                        threadBlockRows, 
                        sharedMemory >>> (  gpu_prefix_arr, block_offset, 
                                            block_size, 0, prefixesCount - 1, 
                                            prefixesCount);
		cudaThreadSynchronize();
	}
	for(int i=l; i > 0; i--)
	{
		block_offset = pow((float)block_size, i);	
		block_scan_write_up <<< blockGridRows, 
                                threadBlockRows, 
                                sharedMemory>>>(gpu_prefix_arr, block_offset, block_size, 
                                                0, prefixesCount - 1, prefixesCount);
		cudaThreadSynchronize();
	}

}
