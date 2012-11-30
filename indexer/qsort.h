#ifndef __QSORT_H__
#define __QSORT_H__


void set_quickSort_kernel(int suff_size);

void quick_sort_bucket(int *device_arr, char *gpu_genome, int bucket_size, int bucket_number, bool last_bucket);

#endif
