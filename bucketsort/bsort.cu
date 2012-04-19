#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>

using namespace std;
#include "sarray.h"
#include "bsort.h"


void test(int,int);
__device__  int g_pivotIndex;

__device__ int strcmp_cuda(char *source, char *dest)
{
	int i;
	for (i = 0; source[i] == dest[i]; i++)
		if (source[i] == '\0')
			return 0;
	return source[i] - dest[i];
}

__device__ int get_index(int c)
{
    if(c==(int)'A')return 0;
    if(c==(int)'C')return 1;
    if(c==(int)'G')return 2;
    return 3;
}

__device__ int get_bucket_no(int *perm, int b_size)
{
    int num = 1,i=0;
    for(i=0;i<b_size;i++)
        num *= NALPHA;
    int fin=0;
    for(i=0;i<b_size;i++)
    {
        num = num/4;
        fin += num*get_index(perm[i]);
    }
    return fin+1;
}

__global__ void bucketSort( int suff_size , int b_size , int s_seg, 
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
        b =  (int) gpu_genome[ gpu_suf_arr[i] ];
        
        if(b == 65) b = 1;
        if(b == 67) b = 1;
        if(b == 71) b = 2;
        if(b == 84) b = 3;

        gpu_bucket_ct[ b * 1024 + tid ] += 1;
        //gpu_bucket_ct[ tid ] += 1;
    }
    __syncthreads();

}

void myfunc( int suff_size )
{
    // Read genome from disk
    setup( suff_size , "genome" );
    // Block and grid dimensions
    int blkGridWidth = n_thds/thd_per_blk;
    int blkGridHeight = 1;
    //int blockGridWidth = suff_size/threads_per_blk + 1;
    dim3 blkGridRows(blkGridWidth, blkGridHeight);
    dim3 thdBlkRows(thd_per_blk, 1);

	// Allocating memory
    alloc_arr( suff_size , n_thds );
	
    // Setting values
    init_arr( suff_size , n_thds );
    
    // Copying values to device
    copy2dev( suff_size , n_thds );
    
    int n_buck = 4;


    alloc2d     ( &cpu_bucket_ct , n_buck , n_thds );
    alloc2d_gpu ( &gpu_bucket_ct , n_buck , n_thds );
    init2d      ( cpu_bucket_ct , n_buck , n_thds , 0 );
    copy2gpu    ( cpu_bucket_ct , gpu_bucket_ct , n_buck * n_thds );
    /* 
       cudaMemcpyToSymbol(  "g_pivotIndex",&pivotIndex, 
                            sizeof(int), size_t(0),
                            cudaMemcpyHostToDevice);
    
    int d_per_thread = suff_size/nthreads;
    create_buckets(b_size);
    */

    int b_size = 1;
    int s_seg = suff_size / n_thds;
  
    bucketSort<<< blkGridRows, thdBlkRows >>>(  suff_size , b_size , s_seg ,
                                                gpu_genome , gpu_suf_arr, 
                                                gpu_aux_arr , gpu_bucket_ct);
	
    cudaMemcpy( cpu_bucket_ct , gpu_bucket_ct , 
                n_buck * n_thds , cudaMemcpyDeviceToHost);
    
    for( int i = 0; i < n_buck; i++)
    {
        int sum = 0;
        for(int j = 0 ; j < n_thds ; j++ )
        {
            sum += cpu_bucket_ct[ i * n_thds + j];
        }
        printf(" Bucket %d : %d\n", i , sum );
    }
    /*
    // cudaThreadSynchronize();
    // Copy the data back to the host
    cudaMemcpy(cpu_final_arr, gpu_aux_arr, 
                sizeof(int ) * suff_size, 
                cudaMemcpyDeviceToHost);
    // Final results
    
    for(int i=0;i<(suff_size/THREADS_PER_BLOCK+1);i++)
    	printf("%d ",cpu_final_arr[i]);
    printf("\n");
    */
}

int main( int argc, char** argv) 
{
    myfunc( atoi(argv[1]) );	
    //test(10,100);
    return 0;
}

void copy2gpu( int *frm , int *dest , int size)
{
   cudaMemcpy( dest , frm , sizeof(int) * size , cudaMemcpyHostToDevice);
}

void print2d(int *arr , int rows , int cols)
{
    for(int i=0; i< rows ; i++)
    {
        for( int j = 0 ; j < cols ; j++ )
        {
            printf("%d", arr[i*cols + j]);
        }
        printf("\n");
    }
}
void alloc2d(int **arr , int rows , int cols)
{
    *arr = (int*) malloc( rows * cols * sizeof(int) );
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

void alloc2d_gpu(int **arr , int rows , int cols)
{ 
    cudaMalloc( (void **) arr, sizeof(int) * rows * cols) ;
}


void copy2dev( int suff_size , int nthreads )
{
    // Copy the data to the device
    cudaMemcpy( gpu_suf_arr, cpu_suf_arr, sizeof(int) * suff_size, 
                cudaMemcpyHostToDevice);
    
    cudaMemcpy( gpu_genome, cpu_genome, sizeof(char) * suff_size, 
                cudaMemcpyHostToDevice);
    
    cudaMemcpy( gpu_aux_arr, cpu_final_arr, sizeof(int) * suff_size, 
                cudaMemcpyHostToDevice);
}

void init_arr(int suff_size , int nthreads)
{
    for(int i=0;i<suff_size;i++)
    {
		cpu_suf_arr[i] = (int)i;
		//printf("%d - %s\n", cpu_suf_arr[i], cpu_genome+i);
    }
	for(int i=0;i<suff_size;i++)
	{
		cpu_final_arr[i] = 0;
	} 
}

void alloc_arr( int suff_size , int nthreads )
{
    cpu_suf_arr = (int*) malloc( sizeof(int) * suff_size);  
    cpu_final_arr = (int*) malloc( sizeof(int) * suff_size);

    cudaMalloc( (void **)&gpu_suf_arr, sizeof(int) * suff_size) ;
    cudaMalloc( (void **)&gpu_aux_arr, sizeof(int) * suff_size) ;
    cudaMalloc( (void **)&gpu_genome, sizeof(char) * suff_size) ;    
}

int setup( int num , char* filename ) 
{
    cpu_genome = (char *) malloc(sizeof(char)*(num+1));
    read_genome2(filename, cpu_genome, num);
    return (strlen(cpu_genome));        
}

void read_genome2( char *filename , char *buffer , int num )
{
    FILE *fh;
    fh = fopen(filename, "r");
    fread(buffer, 1, num, fh);
    buffer[num] = '\0';
    fclose(fh);
}

void copy( string temp , char *loc )
{
    for( int i=0; i<temp.size(); i++ )
    {   
        loc[i] = temp[i];
    }
}

void gen_perms( uint32 depth , string path )
{
    if( path.size() == depth )
    {
        buckets.push_back(path);
        return;
    }
    for(int i=0;i<NALPHA;i++)
    {
        string temp = path;
        temp.append(1,alpha[i]);
        gen_perms(depth,temp);
    }
}

void create_buckets(int bucket_size)
{
    gen_perms(bucket_size,"");
}
