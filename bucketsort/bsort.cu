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
#include "qsort.h"

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
    if(c==(int)'T')return 3;
    return 0;
}

__device__ int get_bucket_no(char *perm, int b_size)
{
    int num = 1,i=0;
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
        b =  get_index( (int) gpu_genome[ gpu_suf_arr[i] ] );
        gpu_bucket_ct[ b * 1024 + tid ] += 1;
    }
    __syncthreads();

}

__global__ void bucketSort2( int suff_size , int b_size , int s_seg, 
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
        b =  get_bucket_no( gpu_genome + gpu_suf_arr[i] , b_size);
        gpu_bucket_ct[ b * NTHREADS + tid ] += 1;
    }
    __syncthreads();

}

__global__ void BsortWriteBack( int suff_size , int b_size , int s_seg, 
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
        b =  get_bucket_no( gpu_genome + gpu_suf_arr[i] , b_size);
        offset = b * NTHREADS + tid;
        loc = gpu_bucket_ct[offset];
        gpu_aux_arr[loc-1] = gpu_suf_arr[i];
        gpu_bucket_ct[offset] = loc - 1;
    }
    __syncthreads();  
}

__global__ void block_scan_write_up( int *g_idata, int block_offset, int block_size, int start, int end, int n)
{
    int tid = (unsigned long int) (threadIdx.x + blockDim.x * threadIdx.y + \
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

    int tid = (unsigned long int) (threadIdx.x + blockDim.x * threadIdx.y + \
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

void prefixCompute( int *gpu_prefix_arr, dim3 blockGridRows, dim3 threadBlockRows, 
                    int block_size, int start, int end, int prefixesCount){
	int sharedMemory = 2*block_size*sizeof(int);
	
	int block_offset = 0;
	int l = log2((float)prefixesCount)/log2((float)block_size);

	for(int i=0; i<=l; i++)
	{
		block_offset = pow((float)block_size, i);
		block_scan <<<  blockGridRows, 
                        threadBlockRows, 
                        sharedMemory >>> (  gpu_prefix_arr, block_offset, 
                                            block_size, start, end, 
                                            prefixesCount);
		cudaThreadSynchronize();
	}
	for(int i=l; i > 0; i--)
	{
		block_offset = pow((float)block_size, i);	
		block_scan_write_up <<< blockGridRows, 
                                threadBlockRows, 
                                sharedMemory>>>(gpu_prefix_arr, block_offset, block_size, 
                                                start, end, prefixesCount);
		cudaThreadSynchronize();
	}

}

void init_b(int prefix_len)
{
    b_size = prefix_len;
    n_buck = (int) pow( NALPHA , b_size);
    n_el = n_thds * n_buck;
    fin_bucket_ct = (int *) malloc( sizeof(int) * n_buck );
}

void do_bsort( int suff_size , int prefix_len)
{
    /* Read genome from disk */
    setup( suff_size , "genome" );
    /* Block and grid dimensions */
    int blkGridWidth = n_thds/thd_per_blk;
    int blkGridHeight = 1;
    dim3 blkGridRows(blkGridWidth, blkGridHeight);
    dim3 thdBlkRows(thd_per_blk, 1);
    
	/* Allocating memory */
    alloc_arr( suff_size , n_thds );
	
    /* Setting values */
    init_arr( suff_size , n_thds );
    init_b(prefix_len);

    /* Copying values to device */
    copy2dev( suff_size , n_thds );

    /* Initialize bucket count arrays */
    alloc2d     ( &cpu_bucket_ct , n_buck , n_thds );
    alloc2d_gpu ( &gpu_bucket_ct , n_buck , n_thds );
    init2d      ( cpu_bucket_ct , n_buck , n_thds , 0 );
    copy2gpu    ( cpu_bucket_ct , gpu_bucket_ct , n_buck * n_thds );
    
    int s_seg = suff_size / n_thds;
    
    bucketSort2<<< blkGridRows, thdBlkRows >>> (suff_size , b_size , s_seg ,
                                                gpu_genome , gpu_suf_arr, 
                                                gpu_aux_arr , gpu_bucket_ct);
    
    /* Parallel prefix block and grid dimensions */
    int PP_blkGridWidth = n_el/thd_per_blk + 1;
    dim3 PP_blkGridRows(PP_blkGridWidth, blkGridHeight);
    dim3 PP_thdBlkRows(thd_per_blk, 1);
    
    prefixCompute(  gpu_bucket_ct, PP_blkGridRows, PP_thdBlkRows, 
                    thd_per_blk , 0 , n_el - 1, n_el);
   
    /*
    for(int i = 0; i < n_buck; i++)
    { 
        prefixCompute(  gpu_bucket_ct, PP_blkGridRows, 
                        PP_thdBlkRows, thd_per_blk , 
                        i*n_thds , (i+1)*n_thds -1, n_el);
    }
    */
    
    /* Copy back bucket count and store bucket sizes */
    cudaMemcpy( cpu_bucket_ct , gpu_bucket_ct , 
                n_buck * n_thds * sizeof(int), 
                cudaMemcpyDeviceToHost);

    int offset = 0;;
    for( int i = 0; i < n_buck ; i++)
    {
        offset = (i + 1) * n_thds - 1;
        fin_bucket_ct[i] = cpu_bucket_ct[offset];
        //printf(" Bucket %d : %d\n", i , cpu_bucket_ct[offset] );
    }     

    BsortWriteBack<<< blkGridRows, thdBlkRows >>> ( suff_size , b_size , s_seg, 
                                                    gpu_genome , gpu_suf_arr , 
                                                    gpu_aux_arr , gpu_bucket_ct);

    /* Copy the data back to the host */
    cudaMemcpy( cpu_final_arr, gpu_aux_arr, 
                sizeof(int ) * suff_size, 
                cudaMemcpyDeviceToHost);
        
    /* Final results */
    int debug = 0 , bstart = 0;
    if(debug){
        for( int i = 0; i < n_buck; i++)
        {
            if(i>0) bstart = fin_bucket_ct[i-1];
            printf("Bucket %d :",i);
            for(int j = bstart ; j < fin_bucket_ct[i] ; j++ )
            {
                 printf("%d,",cpu_final_arr[j]);
            }
            printf("\n");
        }
    }  
}

int main( int argc, char** argv) 
{
    int suff_size = atoi(argv[1]);
    int prefix_len = 2;
    /* Distribute suffixes into buckets */
    do_bsort( suff_size , prefix_len );
    /* Do Quick Sort on buckets */
    do_preproc_qsort(suff_size);
    
    return 0;
}

void do_preproc_qsort(int suff_size)
{
    start = (int *) malloc( sizeof(int) * suff_size);
    end   = (int *) malloc( sizeof(int) * suff_size);
    
    int bstart = 0;
    for( int i = 0; i < n_buck; i++)
    {
        if(i>0) bstart = fin_bucket_ct[i-1];
        for(int j = bstart ; j < fin_bucket_ct[i] ; j++ )
        {
            start[ j ] = bstart;
            end[ j ] = fin_bucket_ct[i] - 1;
        }
    }
    int debug = 1;
    if(debug){
        cout<<"Start"<<endl;
        for(int i=0; i < suff_size ; i++)
        {
            cout<<start[i]<<" ";
        }cout<<endl;
    
        cout<<"End"<<endl;
        for(int i=0; i < suff_size ; i++)
        {
            cout<<end[i]<<" ";
        }cout<<endl;
    }
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
