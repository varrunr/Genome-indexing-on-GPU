#ifndef __BSORT_H__
#define __BSORT_H__
#define NTHREADS 1024

const int   thd_per_blk =   32;
const int   n_thds      =   1024;

std::vector< std::string > buckets;
int     nbuckets;
int     *cpu_bucket_ct;
int     *gpu_bucket_ct;
int     *fin_bucket_ct;
int     b_size;
int     n_buck;
int     n_el;
void    gen_perms       ( uint32 , std::string );
void    create_buckets  ( int ); 

void    do_bsort        ( int , int);

void    alloc2d         (int ** , int , int );
void    init2d          (int * , int , int , int );
void    alloc2d_gpu     (int ** , int , int );
void    print2d         (int * , int , int );
void    copy2gpu        (int * , int * , int );

#endif
