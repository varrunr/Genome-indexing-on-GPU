#include "bsort.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "pprefix.h"

#define NTHREADS 1024
#define NALPHA 4
#define OFFSET(x,y) (info->n_threads*x + y)

using namespace std;

int     *cpu_bucket_ct;
int     *gpu_bucket_ct;
int     *cpu_bucket_ct_final;

__device__ int get_index(int c)
{
    if(c==(int)'A' || c==(int)'a')return 0;
    if(c==(int)'C' || c==(int)'c')return 1;
    if(c==(int)'G' || c==(int)'g')return 2;
    if(c==(int)'T' || c==(int)'t')return 3;
    return 0;
}

__device__ int get_bucket_no(char *perm, int b_size)
{
    int num = 1,i=0;
    // num = NALPHA ^ b_size
    for(i=0;i<b_size;i++)
        num *= NALPHA;
    int fin=0;
    for(i=0;i<b_size;i++)
    {
        num = num/4;
        fin += num*get_index((int)perm[i]);
    }
    return fin;
}


__global__ void countBuckets( int suff_size , int b_size , int s_seg, int n_threads,
                            char* gpu_genome , int * gpu_suf_arr , 
                            int *gpu_aux_arr , int *gpu_bucket_ct)
{
    // This gives every thread a unique ID.
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int start = tid * s_seg ;
    int end = start + s_seg;
    int b = 0;
    int i = 0;
    for( i = start; i < end ; i++ )
    {
        if(i < suff_size)
        {
            b =  get_bucket_no( gpu_genome + gpu_suf_arr[i] , b_size);
            gpu_bucket_ct[ b * n_threads + tid ] += 1;
        }
    }
    __syncthreads();

}
__global__ void BsortWriteBackOne(  int suff_size , int b_size , int s_seg, 
                                    int n_threads, int base, int bucket,
                                    char* gpu_genome , int *gpu_suf_arr , 
                                    int *gpu_aux_arr , int *gpu_bucket_ct)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int start = tid * s_seg ;
    int end = start + s_seg;
    int b = 0;
    int i = 0;
    int loc,offset;
    for( i = start; i < end ; i++ )
    {
        if(i < suff_size)
        {
            b =  get_bucket_no( gpu_genome + gpu_suf_arr[i] , b_size);
            if(b == bucket)
            {
                offset = b * n_threads + tid;
                loc = gpu_bucket_ct[offset] - base;
                gpu_aux_arr[loc-1] = gpu_suf_arr[i];
                gpu_bucket_ct[offset] -= 1;
            }
        }
    }

    __syncthreads();  
}

__global__ void BsortWriteBackAll( int suff_size , int b_size , int s_seg, 
                                char* gpu_genome , int * gpu_suf_arr , 
                                int *gpu_aux_arr , int *gpu_bucket_ct)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int start = tid * s_seg ;
    int end = start + s_seg;
    int b = 0;
    int i = 0;
    int loc,offset;
    for( i = start; i < end ; i++ )
    {
        if(i < suff_size)
        {
            b =  get_bucket_no( gpu_genome + gpu_suf_arr[i] , b_size);
        
            offset = b * NTHREADS + tid;
        
            loc = gpu_bucket_ct[offset];
            gpu_aux_arr[loc-1] = gpu_suf_arr[i];
            gpu_bucket_ct[offset] = loc - 1;
        }
    }
    __syncthreads();  
}

void alloc2d(int **arr , int rows , int cols)
{
    *arr = (int*) malloc( rows * cols * sizeof(int) );
}

void alloc2d_gpu(int **arr , int rows , int cols)
{ 
    cudaMalloc( (void **) arr, sizeof(int) * rows * cols) ;
}


void copy2gpu( int *frm , int *dest , int size)
{
   cudaMemcpy( dest , frm , sizeof(int) * size , cudaMemcpyHostToDevice);
}

void init2d(int *arr , int rows , int cols , int val)
{
    for(int i=0; i < rows ; i++)
    {
        for(int j=0; j < cols ; j++)
        {
            arr[ i*cols + j ] = val;
        }
    }
}

void print2d(int *arr , int rows , int cols)
{
    for(int i=0; i< rows ; i++)
    {
        for( int j = 0 ; j < cols ; j++ )
        {
            printf("%5d ", arr[i*cols + j]);
        }
        printf("\n");
    }
}

int getMaxCount(int *arr , int rows , int cols)
{
    int maxCount = 0, count = 0;
    for(int i=0; i< rows ; i++)
    {
        if(i == 0) count = arr[cols - 1];
        else count = (arr[i*cols + cols - 1] - arr[ (i-1)*cols + cols - 1 ]);
        cpu_bucket_ct_final[i] = count;
        maxCount = count > maxCount ? count : maxCount;
    }
    return maxCount;
}

void doPrefixSum(struct bsort_info *info)
{
    cudaDeviceProp prop;
    int count; 
    int maxBlockGridWidth;
    //int maxBlockGridHeight, maxBlockGridDepth;
    
    cudaGetDeviceCount(&count);
    cudaGetDeviceProperties(&prop, 0);
    
    maxBlockGridWidth = prop.maxGridSize[0];
    //maxBlockGridHeight = prop.maxGridSize[1];
    //maxBlockGridDepth = prop.maxGridSize[2];
    
    int block_size = 512;
    
    int blockGridWidth = (info->n_threads * info->n_buckets)/block_size + 1;
    int blockGridHeight = 1;
    
    if(blockGridWidth > maxBlockGridWidth)
    {
        blockGridWidth = maxBlockGridWidth;     
        blockGridHeight = ((info->n_threads * info->n_buckets)/(maxBlockGridWidth * block_size)) + 1;
    }

    dim3 threadBlockRows, blockGridRows;
    
    blockGridRows.x = blockGridWidth; 
    blockGridRows.y = blockGridHeight;
    

    threadBlockRows.x = PPREFIX_BLK_SZ;//block_size;
    threadBlockRows.y = 1;
    
    prefixCompute(  gpu_bucket_ct, blockGridRows, threadBlockRows, 
                    block_size , 0 , 
                    ( (info->n_threads * info->n_buckets) - 1 ) , 
                    info->n_threads * info->n_buckets);
}

int* findMaxBucketCount( struct bsort_info *info, char *gpu_genome, int *gpu_suf_arr, int *gpu_aux_arr)
{
    /* Block and grid dimensions */
    int blkGridWidth = info->n_threads/info->threads_per_blk + 1;
    int blkGridHeight = 1;
    
    dim3 blkGridRows(blkGridWidth, blkGridHeight);
    dim3 thdBlkRows(info->threads_per_blk, 1);
	
    /* Setting values */
   
    
    alloc2d_gpu (&gpu_bucket_ct, info->n_buckets, info->n_threads);
    alloc2d     (&cpu_bucket_ct, info->n_buckets, info->n_threads);
    alloc2d     (&cpu_bucket_ct_final, 1, info->n_buckets);
    init2d      (cpu_bucket_ct, info->n_buckets, info->n_threads, 0);
    init2d      (cpu_bucket_ct_final, 1, info->n_buckets, 0);
 
    /* Copying values to device */
    
    copy2gpu(cpu_bucket_ct, gpu_bucket_ct, info->n_threads * info->n_buckets );
    
    countBuckets<<< blkGridRows, thdBlkRows >>> (info->suff_size , info->prefix_len , 
                                                info->s_seg , info->n_threads,
                                                gpu_genome , gpu_suf_arr, 
                                                gpu_aux_arr , gpu_bucket_ct);

    
    /* Parallel prefix block and grid dimensions */                
     

#ifdef _DEBUG_                    
    cudaMemcpy( cpu_bucket_ct , gpu_bucket_ct , 
                (info->n_buckets * info->n_threads * sizeof(int)), 
                cudaMemcpyDeviceToHost);
    
    print2d(cpu_bucket_ct, info->n_buckets, info->n_threads);
#endif    
    doPrefixSum(info);
    
    cudaMemcpy( cpu_bucket_ct , gpu_bucket_ct , 
                (info->n_buckets * info->n_threads * sizeof(int)), 
                cudaMemcpyDeviceToHost);
#ifdef _DEBUG_
    print2d(cpu_bucket_ct, info->n_buckets, info->n_threads);
#endif    
    info->max_bucket_sz = getMaxCount(cpu_bucket_ct, info->n_buckets, info->n_threads);
    
    return gpu_bucket_ct;
}

int loadBucket( struct bsort_info *info, char *gpu_genome, int *gpu_suf_arr, 
                int *gpu_aux_arr, int bucket_no)
{
    int blkGridWidth = info->n_threads/info->threads_per_blk + 1;
    int blkGridHeight = 1;
    
    dim3 blkGridRows(blkGridWidth, blkGridHeight);
    dim3 thdBlkRows(info->threads_per_blk, 1);
    
    int base = cpu_bucket_ct[ info->n_threads * bucket_no];
    
    BsortWriteBackOne<<< blkGridRows, thdBlkRows >>> (  info->suff_size , 
                                                        info->prefix_len , 
                                                        info->s_seg, 
                                                        info->n_threads, 
                                                        base, bucket_no, 
                                                        gpu_genome , gpu_suf_arr , 
                                                        gpu_aux_arr , gpu_bucket_ct);
    return cpu_bucket_ct_final[bucket_no];
    
}
    
void init_bsort(struct bsort_info *info, int prefix_len, int n_threads, int suff_size, int tpb)
{
    info->prefix_len = prefix_len;
    info->n_threads = n_threads;
    info->suff_size = suff_size;
    info->n_buckets = (int) pow( (double) NALPHA , (double) info->prefix_len);
    info->s_seg =  (int) ceil((double)suff_size / (double)n_threads);
    info->max_bucket_sz = 0;
    info->threads_per_blk = tpb;
#ifdef _DEBUG_
    cout<<" No of buckets: "<< info->n_buckets << endl;
    cout<<" s_seg: " << info->s_seg << endl;
    cout<<" threads_per_blk: " << tpb << endl; 
#endif
}

