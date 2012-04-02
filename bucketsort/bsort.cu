#include "bsort.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include<algorithm>

using namespace std;

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

__global__ void bucketSort( int suff_size,int d_size,int b_size,
                            char* gpu_genome, int * gpu_suf_arr, 
                            int *gpu_aux_arr, int **gpu_bucket_ct)
{
    
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;    // This gives every thread a unique ID.
    int base = tid * d_size;
    int end = base + d_size;
    int buck_no = 0;
    for(int i=base;i<end;i++)
    {
        buck_no = get_bucket_no(gpu_suf_arr + gpu_suf_arr[i],b_size);
        gpu_bucket_ct[buck_no][tid] += 1;;
    }
    
    //	__syncthreads(); // Barrier - all threads must finish comparing suffixes 
    
}

void myfunc(int suff_size,int b_size)
{
	// Read genome from disk
	setup(suff_size,"genome");
	// Block and grid dimensions
	int threads_per_blk = 1;
    int nthreads = 32;
	int blockGridWidth = nthreads/threads_per_blk;
    //int blockGridWidth = suff_size/threads_per_blk + 1;
	int blockGridHeight = 1;
	dim3 blockGridRows(blockGridWidth, blockGridHeight);
	dim3 threadBlockRows(threads_per_blk, 1);
	
	// Allocating memory
    alloc_arr(suff_size, b_size, nthreads);
	// Setting values
    init_arr(suff_size, b_size, nthreads);
    // Copying values to device
	copy2dev(suff_size, b_size, nthreads);
	//cudaMemcpyToSymbol("g_pivotIndex",&pivotIndex, sizeof(int), size_t(0),cudaMemcpyHostToDevice);
	int d_per_thread = suff_size/nthreads;
    create_buckets(b_size);    
    bucketSort<<<blockGridRows, threadBlockRows>>>( suff_size, d_per_thread, b_size,gpu_genome, gpu_suf_arr, gpu_aux_arr, gpu_bucket_ct);
	//CUT_CHECK_ERROR("Quick sort execution  execution failed\n");
    //cudaThreadSynchronize();

	// Copy the data back to the host
    cudaMemcpy(cpu_final_arr, gpu_aux_arr, 
                sizeof(int ) * suff_size, 
                cudaMemcpyDeviceToHost);
     // Final results
    /*
	for(int i=0;i<(suff_size/THREADS_PER_BLOCK+1);i++)
		printf("%d ",cpu_final_arr[i]);
	printf("\n");
    */
}

int main( int argc, char** argv) 
{
    //print_perm(bs);
    //cout<<get_perm_no(chk,bs)<<endl;
    int b_size = 3;
    myfunc(atoi(argv[1]),b_size);	
    return 0;
}

void copy2dev(int suff_size, int b_size, int nthreads)
{
	// Copy the data to the device
    cudaMemcpy( gpu_suf_arr, cpu_suf_arr, sizeof(int) * suff_size, 
                cudaMemcpyHostToDevice);
    
    cudaMemcpy( gpu_genome, cpu_genome, sizeof(char) * suff_size, 
                cudaMemcpyHostToDevice);
    
    cudaMemcpy( gpu_aux_arr, cpu_final_arr, sizeof(int) * suff_size, 
                cudaMemcpyHostToDevice);
    for(int i=0;i<b_size;i++)
    {
        cudaMemcpy( gpu_bucket_ct + i, cpu_bucket_ct + i, sizeof(int) * nthreads, 
                cudaMemcpyHostToDevice);
    }

}
void init_arr(int suff_size,int b_size,int nthreads)
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
    for(int i=0;i<pow(NALPHA,b_size);i++)
    {
        for(int j=0;j<nthreads;j++)
        {
            cpu_bucket_ct[i][j] = 0;
        }
    }
}
void alloc_arr(int suff_size,int b_size,int nthreads)
{
	cpu_suf_arr = (int*) malloc( sizeof(int) * suff_size);
	cpu_final_arr = (int*) malloc( sizeof(int) * suff_size);
    cpu_bucket_ct = (int **) malloc( sizeof(int*) * b_size);
    
    cudaMalloc( (void ***)&gpu_bucket_ct, sizeof(int*) * b_size);
    cudaMalloc( (void **)&gpu_suf_arr, sizeof(int) * suff_size) ;
    cudaMalloc( (void **)&gpu_aux_arr, sizeof(int) * suff_size) ;
    cudaMalloc( (void **)&gpu_genome, sizeof(char) * suff_size) ;
    for(int i=0;i<b_size;i++)
    {
        cpu_bucket_ct[i] = (int*) malloc( sizeof(int) * nthreads);
        cudaMalloc( (void **)(gpu_bucket_ct+i), sizeof(int) * nthreads );
    }
}

int setup(int num, char* filename)
{
    cpu_genome = (char *) malloc(sizeof(char)*(num+1));
    read_genome2(filename, cpu_genome, num);
    return (strlen(cpu_genome));        
}

void read_genome2(char *filename, char *buffer, int num){
    FILE *fh;
    fh = fopen(filename, "r");
    fread(buffer, 1, num, fh);
    buffer[num] = '\0';
    fclose(fh);
}
void copy(string temp, char *loc)
{
    for(int i=0;i<temp.size();i++)
    {   
        loc[i] = temp[i];
    }
}

void gen_perms(uint32 depth,string path)
{
    if(path.size() == depth)
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

