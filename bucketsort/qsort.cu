#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <cutil_inline.h>
#include "sarray.h"
#include "bsort.h"
#include "qsort.h"

using namespace std;

__device__  int g_pivotIndex;

void print(int *array, int n)
{
    for(int i=0; i < n; i++)
        cout << array[i] << endl;
}

void print_gene_array(int *array, int n)
{
    for(int i=0;i<n;i++){
        printf("%d - %s\n", array[i], cpu_genome+array[i]);
    }
}

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
    if(c==(int)'A' || c==(int)'a')return 0;
    if(c==(int)'C' || c==(int)'c')return 1;
    if(c==(int)'G' || c==(int)'g')return 2;
    if(c==(int)'T' || c==(int)'t')return 3;
    else return 0;
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

void prefixCompute( int *gpu_prefix_arr, dim3 blockGridRows, dim3 threadBlockRows, 
                    int block_size, int start, int end, int prefixesCount)
{
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
    n_buck = (int) pow( (double)NALPHA , (double)b_size);
    n_el = n_thds * n_buck;
    fin_bucket_ct = (int *) malloc( sizeof(int) * n_buck );
}

void do_bsort( int suff_size , int prefix_len)
{
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
      
    /* Final results */
    int debug = 0 , bstart = 0;
    if(debug){
        for( int i = 0; i < n_buck; i++)
        {
            if(i>0) bstart = fin_bucket_ct[i-1];
            printf("Bucket %d : %d",i, fin_bucket_ct[i]);
            for(int j = bstart ; j < fin_bucket_ct[i] ; j++ )
            {
                 printf("%d,",cpu_final_arr[j]);
            }
            printf("\n");
        }
    }  
}

void do_preproc_qsort(int suff_size)
{
    start = (int *) malloc( sizeof(int) * suff_size);
    end   = (int *) malloc( sizeof(int) * suff_size);

    int bstart = 0;
    for( int i = 0; i < n_buck; i++)
    {

        if(i>0)
        {
          bstart = fin_bucket_ct[i-1];
        } 
          
        for(int j = bstart ; j < fin_bucket_ct[i] ; j++ )
        {
            start[ j ] = bstart;
            end[ j ] = fin_bucket_ct[i] - 1;
        }
    }

    int debug = 0;
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

void read_genome2(char *filename, char *buffer, int num)
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

void quick_sort_genome(int *device_arr, long unsigned int suff_size)
{

    sh_gpu_suf_arr = device_arr;
    
    init_kernel <<<blockGridRows, threadBlockRows>>> (start_ind_arr, end_ind_arr, start_ind_arr_copy, end_ind_arr_copy, suff_size);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Kernel Execution
    int end_loop = 0;
    while(true)
    {   
        int total_shared_memory = block_size*(sizeof(int));
        quickSortGPU<<<blockGridRows, threadBlockRows, total_shared_memory>>>(sh_gpu_suf_arr, sh_gpu_suf_arr_copy, sh_gpu_aux_arr, start_ind_arr, end_ind_arr, start_ind_arr_copy, end_ind_arr_copy, gpu_genome, block_size, suff_size);
        CUT_CHECK_ERROR("Quick sort Kernel execution failed\n");
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        
        prefixCompute(sh_gpu_aux_arr, blockGridRows, threadBlockRows, block_size, 0, suff_size-1, suff_size);
    
        adjust_prefix_sum <<< blockGridRows, threadBlockRows >>> (sh_gpu_aux_arr, sh_gpu_aux_arr_copy, start_ind_arr, end_ind_arr, suff_size);
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
            
        write_pivot <<< blockGridRows, threadBlockRows >>> (sh_gpu_suf_arr, sh_gpu_suf_arr_copy, sh_gpu_aux_arr_copy, start_ind_arr, end_ind_arr, start_ind_arr_copy, end_ind_arr_copy, suff_size);
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

void set_kernel_memory(unsigned long int suff_size)
{

    // Setting values   
    CUDA_SAFE_CALL( cudaMemcpy(start_ind_arr, start, sizeof(int) * suff_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(start_ind_arr_copy, start, sizeof(int) * suff_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(end_ind_arr, end, sizeof(int) * suff_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(end_ind_arr_copy, end, sizeof(int) * suff_size, cudaMemcpyHostToDevice) );

}

void free_memory()
{
    free(cpu_genome);
    free(start);
    free(end);
    free(cpu_suf_arr);
    free(cpu_final_arr);
    free(cpu_bucket_ct);
    cudaFree(gpu_genome);
    cudaFree(gpu_aux_arr);
    cudaFree(gpu_suf_arr);
    cudaFree(gpu_bucket_ct);
    cudaFree(sh_gpu_suf_arr_copy);
    cudaFree(sh_gpu_aux_arr);
    cudaFree(sh_gpu_aux_arr_copy);
    cudaFree(start_ind_arr);
    cudaFree(end_ind_arr);
    cudaFree(start_ind_arr_copy);
    cudaFree(end_ind_arr_copy);
}

void sort_buckets(int *gpu_aux_arr, int suff_size)
{
    int size = 260000, before = 0;

    for( int i = 0; i < n_buck; i++)
    {
        if(i == 0)
        {
            size = fin_bucket_ct[i];
            before = 0;  
        } 
        else 
        {
            size = fin_bucket_ct[i] - fin_bucket_ct[i-1];
            before = fin_bucket_ct[i-1];
        }
//        printf("Bucket %d: %d\n", i, size); 

//        CUDA_SAFE_CALL( cudaMemcpy(cpu_final_arr, gpu_aux_arr+before, sizeof(int) * size, cudaMemcpyDeviceToHost) );
//        cout << "\n\n\nSuffix Array Before Sorting Genome: " << endl;
//        print_gene_array(cpu_final_arr, size);

        quick_sort_genome(gpu_aux_arr+before, size);        

//        CUDA_SAFE_CALL( cudaMemcpy(cpu_final_arr, gpu_aux_arr+before, sizeof(int) * size, cudaMemcpyDeviceToHost) );
//        cout << "\n\n\n\nSuffix Array After Sorting Genome: " << endl;
//        print_gene_array(cpu_final_arr, size);
//        if(i==1) break;
    }    

    // Final results
    CUDA_SAFE_CALL( cudaMemcpy(cpu_final_arr, gpu_aux_arr, sizeof(int) * suff_size, cudaMemcpyDeviceToHost) );
    cout << "Suffix Array for Genome: " << endl;
    print_gene_array(cpu_final_arr, suff_size);

}

int main( int argc, char** argv) 
{
    int suff_size = atoi(argv[1]);
    int prefix_len = 3;
    
    /* Read genome from disk */
    setup(suff_size, argv[2] );

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    /* Distribute suffixes into buckets */
    do_bsort( suff_size, prefix_len );

    /* Do Quick Sort on buckets */
    do_preproc_qsort(suff_size);

    alloc_device_pointers(suff_size);
    set_kernel_memory(suff_size);
    set_quickSort_kernel(suff_size);

    sort_buckets(gpu_aux_arr, suff_size);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );

    printf("%d %f\n", suff_size, elapsedTime * (0.001));
    free_memory();
    return 0;
}
