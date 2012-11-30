#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "qsort.h"
#include "pprefix.h"

#define CUDA_SAFE_CALL(x) x

using namespace std;

int *start_ind_arr;
int *end_ind_arr;
int *start_ind_arr_copy;
int *end_ind_arr_copy;
int *sh_gpu_suf_arr;
int *sh_gpu_suf_arr_copy;
int *sh_gpu_aux_arr;
int *sh_gpu_aux_arr_copy;

dim3 blockGridRows;
dim3 threadBlockRows;
int block_size;

void alloc_device_pointers(unsigned long int suff_size);

void quick_sort_genome(int *device_arr, unsigned long int suff_size, char *gpu_genome);

__device__  int g_pivotIndex;

__device__ int strcmp_cuda(char *source, char *dest)
{
	int i;
	for (i = 0; source[i] == dest[i]; i++)
		if (source[i] == '\0')
			return 0;
	return source[i] - dest[i];
}

__device__ int get_pivot_index(int start, int end)
{
    /*
     * TODO : Select the Pivot Index using either some operations using
     * property of DNA string or Sampling or any form of randomization
     * which will split the recursive call between 1/4 to 3/4 for 
     * efficient running of Quick Sort. For now selecting random/mid-point value
     * as pivot.
     *
     */

    return start + (end - start)/2;
}

__global__ void quickSortGPU(   int *sh_gpu_suf_arr, int *sh_gpu_suf_arr_copy, int *sh_gpu_aux_arr, 
                                int *start_ind_arr, int *end_ind_arr, int *start_ind_arr_copy, 
                                int *end_ind_arr_copy, char* sh_gpu_genome,  int block_size, int suff_size)
{
    extern __shared__ int shared[];
    int *lt_pivot = &shared[0];

    int tid = (int) (threadIdx.x + blockDim.x * threadIdx.y + \
                            ( blockIdx.x * blockDim.x * blockDim.y ) \
                            + ( blockIdx.y * blockDim.x * blockDim.y * gridDim.x));

    int pivotIndex;
    if(tid < suff_size)
    {
        start_ind_arr[tid] = start_ind_arr_copy[tid];
        end_ind_arr[tid] = end_ind_arr_copy[tid];
        sh_gpu_suf_arr_copy[tid] = sh_gpu_suf_arr[tid];

        if(start_ind_arr[tid] != -1)
        {
            lt_pivot[threadIdx.x] = 0;
            pivotIndex = get_pivot_index(start_ind_arr[tid], end_ind_arr[tid]);

            if(tid != pivotIndex)
            {
                if(strcmp_cuda(sh_gpu_genome + sh_gpu_suf_arr[tid] ,sh_gpu_genome + sh_gpu_suf_arr[pivotIndex] ) < 0)
                {
                    lt_pivot[threadIdx.x] = 1;
                }
                else
                {
                    lt_pivot[threadIdx.x] = 0;
                }
            }   
        }
        else
        {
            lt_pivot[threadIdx.x] = 0;
        }       
        __syncthreads(); // Barrier - all threads must finish comparing suffixes 

        sh_gpu_aux_arr[tid] = lt_pivot[threadIdx.x];
    }
}

__device__ int terminate_t = 1;

__global__ void write_pivot(int *sh_gpu_suf_arr, int *sh_gpu_suf_arr_copy, int *sh_gpu_aux_arr, 
                            int *start_ind_arr, int *end_ind_arr, int *start_ind_arr_copy, 
                            int *end_ind_arr_copy, int n)
{
    int tid = (int) (threadIdx.x + blockDim.x * threadIdx.y + \
                            ( blockIdx.x * blockDim.x * blockDim.y ) \
                            + ( blockIdx.y * blockDim.x * blockDim.y * gridDim.x));

    int pivotIndex, pivotIndexPosition;
    pivotIndex = get_pivot_index(start_ind_arr[tid], end_ind_arr[tid]);
    if(tid == 0)
    {
        terminate_t = 0;
    }
    if(tid < n)
    {

        pivotIndexPosition = start_ind_arr[tid] + sh_gpu_aux_arr[end_ind_arr[tid]];
        if(start_ind_arr[tid] != -1)
        {
            if(tid == pivotIndex)
            {
                sh_gpu_suf_arr[pivotIndexPosition] = sh_gpu_suf_arr_copy[pivotIndex];
                start_ind_arr_copy[pivotIndexPosition] = -1;
                end_ind_arr_copy[pivotIndexPosition] = -1;
            }
            else if(tid == start_ind_arr[tid])
            {
                if(sh_gpu_aux_arr[tid] == 0)
                {
                    sh_gpu_suf_arr[pivotIndexPosition + tid - start_ind_arr[tid] - sh_gpu_aux_arr[tid] + 1] = sh_gpu_suf_arr_copy[tid];
                    start_ind_arr_copy[pivotIndexPosition + tid - start_ind_arr[tid] - sh_gpu_aux_arr[tid] + 1] = pivotIndexPosition + 1;
                }
                else
                {
                    sh_gpu_suf_arr[start_ind_arr[tid] + sh_gpu_aux_arr[tid] - 1] = sh_gpu_suf_arr_copy[tid];
                    end_ind_arr_copy[start_ind_arr[tid] + sh_gpu_aux_arr[tid] - 1] = pivotIndexPosition - 1;
                }
            }
            else
            {
                if(sh_gpu_aux_arr[tid-1] == sh_gpu_aux_arr[tid])      // if equal it means current value of gpu suf arr is greater than pivot
                {
                    if(tid > pivotIndex)
                    {
                        sh_gpu_suf_arr[pivotIndexPosition + tid - start_ind_arr[tid] - sh_gpu_aux_arr[tid]] = sh_gpu_suf_arr_copy[tid];
                        start_ind_arr_copy[pivotIndexPosition + tid - start_ind_arr[tid] - sh_gpu_aux_arr[tid]] = pivotIndexPosition + 1;                  
                    }
                    else
                    {
                        sh_gpu_suf_arr[pivotIndexPosition + tid - start_ind_arr[tid] - sh_gpu_aux_arr[tid] + 1] = sh_gpu_suf_arr_copy[tid];
                        start_ind_arr_copy[pivotIndexPosition + tid - start_ind_arr[tid] - sh_gpu_aux_arr[tid] + 1] = pivotIndexPosition + 1;
                    }
                }
                else
                {
                    sh_gpu_suf_arr[start_ind_arr[tid] + sh_gpu_aux_arr[tid] - 1] = sh_gpu_suf_arr_copy[tid];
                    end_ind_arr_copy[start_ind_arr[tid] + sh_gpu_aux_arr[tid] - 1] = pivotIndexPosition - 1;
                }
            }
            terminate_t = 1;
        }
        else
        {
            sh_gpu_suf_arr[tid] = sh_gpu_suf_arr_copy[tid];
        }
    }
}

__global__ void adjust_prefix_sum(int *sh_gpu_aux_arr, int *sh_gpu_aux_arr_copy, int *start_ind_arr, int *end_ind_arr, int n)
{
    int tid = (int) (threadIdx.x + blockDim.x * threadIdx.y + \
                            ( blockIdx.x * blockDim.x * blockDim.y ) \
                            + ( blockIdx.y * blockDim.x * blockDim.y * gridDim.x));                 // This gives every thread a unique ID.

    if(tid < n)
    {
        if(start_ind_arr[tid] != 0)
        {
                sh_gpu_aux_arr_copy[tid] = sh_gpu_aux_arr[tid] - sh_gpu_aux_arr[start_ind_arr[tid] - 1];
        }
        else
        {
            sh_gpu_aux_arr_copy[tid] = sh_gpu_aux_arr[tid];
        }
    }
}

__global__ void init_kernel(int *start_ind_arr, int *end_ind_arr, int *start_ind_arr_copy, int *end_ind_arr_copy, int n)
{

    unsigned long int tid = (unsigned long int) (threadIdx.x + blockDim.x * threadIdx.y + \
                            ( blockIdx.x * blockDim.x * blockDim.y ) \
                            + ( blockIdx.y * blockDim.x * blockDim.y * gridDim.x));                 // This gives every thread a unique ID.

    if(tid < n)
    {
        start_ind_arr[tid] = 0;
        start_ind_arr_copy[tid] = 0;
        end_ind_arr[tid] = n-1;
        end_ind_arr_copy[tid] = n-1;
    }

}

__global__ void init_suffix_array(int *device_arr, int n)
{

    unsigned long int tid = (unsigned long int) (threadIdx.x + blockDim.x * threadIdx.y + \
                            ( blockIdx.x * blockDim.x * blockDim.y ) \
                            + ( blockIdx.y * blockDim.x * blockDim.y * gridDim.x));                 // This gives every thread a unique ID.

    if(tid < n)
    {
        device_arr[tid] = tid;
    }

}

void quick_sort_genome(int *device_arr, long unsigned int suff_size, char *gpu_genome)
{

    sh_gpu_suf_arr = device_arr;
    
    init_kernel <<<blockGridRows, threadBlockRows>>> (start_ind_arr, end_ind_arr, start_ind_arr_copy, end_ind_arr_copy, suff_size);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Kernel Execution
    int end_loop = 0;
    while(true)
    {   
        int total_shared_memory = block_size*(sizeof(int));
        quickSortGPU <<<blockGridRows, 
                        threadBlockRows, 
                        total_shared_memory>>> (sh_gpu_suf_arr, sh_gpu_suf_arr_copy, 
                                                sh_gpu_aux_arr, start_ind_arr, 
                                                end_ind_arr, start_ind_arr_copy, 
                                                end_ind_arr_copy, gpu_genome, 
                                                block_size, suff_size);

        //CUT_CHECK_ERROR("Quick sort Kernel execution failed\n");
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        
        prefixCompute(sh_gpu_aux_arr, blockGridRows, threadBlockRows, block_size, 0, suff_size-1, suff_size);
    
        adjust_prefix_sum <<<   blockGridRows, 
                                threadBlockRows >>> (   sh_gpu_aux_arr, 
                                                        sh_gpu_aux_arr_copy, 
                                                        start_ind_arr, 
                                                        end_ind_arr, 
                                                        suff_size   );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        write_pivot <<< blockGridRows, 
                        threadBlockRows >>> (   sh_gpu_suf_arr, 
                                                sh_gpu_suf_arr_copy, 
                                                sh_gpu_aux_arr_copy, 
                                                start_ind_arr, 
                                                end_ind_arr, 
                                                start_ind_arr_copy, 
                                                end_ind_arr_copy, 
                                                suff_size   );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        cudaMemcpyFromSymbol(&end_loop, terminate_t, sizeof(end_loop), 0, cudaMemcpyDeviceToHost);

//        printf("Execute More:- %d \n", end_loop);
        if(end_loop == 0)
        {
            break;
        }
    }   

}

void set_quickSort_kernel(int suff_size)
{
    // Get Device Properties
    cudaDeviceProp prop;
    int count; 
//    int MAX_THREADS_PER_BLOCK;
    int maxBlockGridWidth, maxBlockGridHeight, maxBlockGridDepth;
    cudaGetDeviceCount(&count);
    cudaGetDeviceProperties(&prop, 0);
//    MAX_THREADS_PER_BLOCK = prop.maxThreadsPerBlock;
    maxBlockGridWidth = prop.maxGridSize[0];
    maxBlockGridHeight = prop.maxGridSize[1];
    maxBlockGridDepth = prop.maxGridSize[2];

    // Block and grid dimensions
    block_size = 512; // MAX_THREADS_PER_BLOCK;

    if((unsigned long int)suff_size > (int)(maxBlockGridWidth * maxBlockGridHeight * maxBlockGridDepth * block_size))
    {
        cout << "Suffix Array Length out of Device block Size" << endl;
        exit(1);        
    }

    int blockGridWidth = suff_size/block_size + 1;
    int blockGridHeight = 1;
    if(blockGridWidth > maxBlockGridWidth)
    {
        blockGridWidth = maxBlockGridWidth;     
        blockGridHeight = (suff_size/(maxBlockGridWidth * block_size)) + 1;
    }

    blockGridRows.x = blockGridWidth; 
    blockGridRows.y = blockGridHeight;
    threadBlockRows.x = block_size;
    threadBlockRows.y = 1;
}

void alloc_device_pointers(unsigned long int suff_size)
{
    // Allocating memory    
    CUDA_SAFE_CALL( cudaMalloc( (void **)&sh_gpu_suf_arr_copy, sizeof(int) * suff_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void **)&start_ind_arr_copy, sizeof(int) * suff_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void **)&end_ind_arr_copy, sizeof(int) * suff_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void **)&start_ind_arr, sizeof(int) * suff_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void **)&end_ind_arr, sizeof(int) * suff_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void **)&sh_gpu_aux_arr, sizeof(int) * suff_size) );
    CUDA_SAFE_CALL( cudaMalloc( (void **)&sh_gpu_aux_arr_copy, sizeof(int) * suff_size) );

}

void free_gpu_memory(){
    cudaFree(sh_gpu_suf_arr_copy);
    cudaFree(sh_gpu_aux_arr);
    cudaFree(sh_gpu_aux_arr_copy);
    cudaFree(start_ind_arr);
    cudaFree(end_ind_arr);
    cudaFree(start_ind_arr_copy);
    cudaFree(end_ind_arr_copy);
}


static int max_bucket_size = -1;

void quick_sort_bucket(int *device_arr, char *gpu_genome, int bucket_size, int bucket_number, bool last_bucket){

    if(bucket_number == 0){
        alloc_device_pointers(bucket_size);
        max_bucket_size = bucket_size;        
    } else if(max_bucket_size < bucket_size){
        free_gpu_memory();
        alloc_device_pointers(bucket_size);
        max_bucket_size = bucket_size;
    }

    quick_sort_genome(device_arr, bucket_size, gpu_genome);

    if(last_bucket){
        free_gpu_memory();
    }

}
