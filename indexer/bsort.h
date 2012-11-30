#ifndef __BSORT_H__
#define __BSORT_H__

struct bsort_info
{
    int prefix_len;
    int n_buckets;
    int n_threads;
    int s_seg;
    int suff_size;
    int max_bucket_sz;
    int threads_per_blk;
};


void countBuckets(  struct bsort_info *info, char *gpu_genome, int *gpu_suf_arr,
                    int *gpu_aux_arr, int *gpu_bucket_ct);

int* findMaxBucketCount( struct bsort_info *info, char *gpu_genome, int *gpu_suf_arr, int *gpu_aux_arr);
                    
void init_bsort(struct bsort_info *info, int prefix_len, int n_threads, int suff_size, int tpb);

int loadBucket( struct bsort_info *info, char *gpu_genome, int *gpu_suf_arr, int *gpu_aux_arr, int bucket_no);

#endif
