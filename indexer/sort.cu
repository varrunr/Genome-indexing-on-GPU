#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "sarray.h"
#include "qsort.h"
#include "pprefix.h"
#include "bsort.h"

#define BSORT_PAD 20

#define CUDA_SAFE_CALL(x) x

using namespace std;

void print(int *array, int n)
{
    for(int i=0; i < n; i++)
        cout << array[i] << endl;
}

void print_gene_array(int *array, int n)
{
    for(int i = 0;i < n;i++){
        printf("%d - %s\n", array[i], cpu_genome+array[i]);
        //printf("%d\n", array[i]);
    }
}

void print_part_gene_array(int *array, int n)
{
    for(int i = 0;i < 100;i++){
        printf("%d - %.10s\n", array[i], cpu_genome+array[i]);
        //printf("%d\n", array[i]);
    }
}

int setup( int num , char* filename ) 
{
    cpu_genome = (char *) malloc(sizeof(char)*(num + BSORT_PAD + 1));
    read_genome2(filename, cpu_genome, num);
    int i;
    
    for(i = num; i < num + BSORT_PAD + 1; i++)
    {
        cpu_genome[i] = 0;
    }
    
    return (num);        
}

void read_genome2(char *filename, char *buffer, int num)
{
    FILE *fh;
    fh = fopen(filename, "r");
    fread(buffer, 1, num, fh);
    fclose(fh);
}

void init_arr(int suff_size)
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

void free_memory()
{
    free(cpu_genome);
    free(cpu_suf_arr);
    free(cpu_final_arr);
    cudaFree(gpu_genome);
    cudaFree(gpu_aux_arr);
    cudaFree(gpu_suf_arr);
}

void copy_to_memory( int suff_size)
{
    // Copy the data to the device
    cudaMemcpy( gpu_suf_arr, cpu_suf_arr, sizeof(int) * suff_size, 
                cudaMemcpyHostToDevice);
    
    cudaMemcpy( gpu_genome, cpu_genome, sizeof(char) * (suff_size + BSORT_PAD + 1), 
                cudaMemcpyHostToDevice);
}

int main( int argc, char** argv) 
{
    int suff_size = atoi(argv[1]);
    
    /* Initialize bucket sort */
    struct bsort_info *b_info = (struct bsort_info *) malloc( sizeof(struct bsort_info) );
    
    init_bsort( b_info, 
                3  /* prefix_len */, 
                1024 /* n_threads */, 
                suff_size, 
                512 /* threads_per_block */);
    
    /* Read genome from disk */
    setup(suff_size, argv[2]);
 
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );
    
    /* Initialize base arrays */
    cpu_final_arr = (int*) malloc( sizeof(int) * suff_size);
    cpu_suf_arr = (int*) malloc( sizeof(int) * suff_size);
    
    CUDA_SAFE_CALL( cudaMalloc( (void **) &gpu_suf_arr, sizeof(int) * suff_size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &gpu_genome, sizeof(char) * (suff_size + 1 + BSORT_PAD) ) );

    init_arr(suff_size);
    copy_to_memory(suff_size);
    
    /* Initialize the quick sort kernel */
//    set_quickSort_kernel(suff_size);

    /* Find the max no of buckets */
    findMaxBucketCount(b_info, gpu_genome, gpu_suf_arr, gpu_aux_arr);
    //cout<<"max_bucket_sz: "<< b_info->max_bucket_sz<<endl;
    CUDA_SAFE_CALL( cudaMalloc( (void **) &gpu_aux_arr, sizeof(int) * b_info->max_bucket_sz) );
    
    /* Sort one bucket at a time using quick sort */
    int cur_suff = 0;
    int n_buck;
    
    for(int i = 0; i < b_info->n_buckets; i++)
    {

        n_buck = loadBucket( b_info, gpu_genome, gpu_suf_arr, gpu_aux_arr, i);
        
        cout<<"=== Bucket "<< i << ": " << n_buck <<" elements : "<< cur_suff <<" Position ===="<<endl;
        
        /*
            Do Quick sort here on gpu_aux_arr[0...n_buck]
         */

        if(n_buck > 0)
        {
            cudaMemcpy( cpu_suf_arr + cur_suff , gpu_aux_arr , 
                        (n_buck * sizeof(int)), 
                        cudaMemcpyDeviceToHost);
            
            if(i == 54){
//                cout<<"=== Bucket "<< i << ": " << n_buck <<" elements : "<< cur_suff <<" Position ===="<<endl;
                cout << "TCG Bucket Before Qsort and max size:" << b_info->max_bucket_sz << endl;
                print_part_gene_array( cpu_suf_arr+cur_suff, n_buck);
            }

            if(i == b_info->n_buckets - 1)
            {
                quick_sort_bucket(gpu_aux_arr, gpu_genome,  n_buck, i, true, b_info->max_bucket_sz);
            } 
            else 
            {
                quick_sort_bucket(gpu_aux_arr, gpu_genome,  n_buck, i, false, b_info->max_bucket_sz);
            }
            
            /*
                Copy back to suffix array 
             */
            
            cudaMemcpy( cpu_suf_arr + cur_suff , gpu_aux_arr , 
                        (n_buck * sizeof(int)), 
                        cudaMemcpyDeviceToHost);
            
            if(i == 54){
                cout << "\n\nTCG Bucket After Qsort: " << endl;
                print_part_gene_array( cpu_suf_arr+cur_suff, n_buck);
            }
            /*for(int j = cur_suff; j < cur_suff + n_buck; j++)
            {
                cout<<cpu_suf_arr[j]<<" : " << (cpu_genome + cpu_suf_arr[j])<<endl;
            }*/
            cur_suff += n_buck;
        }
    }
    
//    CUDA_SAFE_CALL( cudaMemcpy(cpu_final_arr, gpu_suf_arr, sizeof(int) * suff_size, cudaMemcpyDeviceToHost) );
    
#ifdef _PRINT_
    //cout << "Suffix Array for Genome: " << endl;
    //print_gene_array(cpu_suf_arr, suff_size);
#endif

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );
#ifdef _PRINT_
    printf("%d %f\n", suff_size, elapsedTime * (0.001));
#endif    
    free_memory();

    return 0;
}
