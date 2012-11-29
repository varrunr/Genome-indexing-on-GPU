#ifndef __QSORT_H__
#define __QSORT_H__

int* start;
int* end;

dim3 blockGridRows;
dim3 threadBlockRows;
int block_size;

int *start_ind_arr;
int *end_ind_arr;
int *start_ind_arr_copy;
int *end_ind_arr_copy;
int *sh_gpu_suf_arr = gpu_aux_arr;
int *sh_gpu_suf_arr_copy;
int *sh_gpu_aux_arr;
int *sh_gpu_aux_arr_copy;

void set_quickSort_kernel(unsigned long int suff_size);
void alloc_device_pointers(unsigned long int suff_size);
void free_memory();
void quick_sort_genome(int *device_arr, unsigned long int suff_size);
void quick_sort_bucket(int *device_arr, int bucket_size, int bucket_number, bool last_bucket);
void print(int *array, int n);
void print_gene_array(int *array, int n);
int setup( int num , char* filename );
void read_genome2(char *filename, char *buffer, int num);

#endif
