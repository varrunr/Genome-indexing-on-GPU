#ifndef __BSORT_H__
#include<iostream>
#include<string>
#include<vector>
using namespace std;
#define MAX_DATA_SIZE 262144
#define THREADS_PER_BLOCK 256
#define ALPHA_SIZE 4
#define NALPHA 4
#define uint32 unsigned int

    //constants
    const char alpha[4]={'A','C','G','T'};
    //variables
    int nbuckets;
    int num_p;
    vector<string> buckets;
    int *cpu_suf_arr;
    int *cpu_final_arr;
    int **cpu_bucket_ct;
	int *gpu_suf_arr;
    int *gpu_aux_arr;
	char *gpu_genome;

    int **gpu_bucket_ct;
    // functions
    void myfunc(int);
    char *cpu_genome;
    void read_genome2(char*, char*, int);
    int setup(int, char*);
    void alloc_arr(int,int,int);
    void init_arr(int,int,int);
    void copy2dev(int,int,int);
    // Bucket related
    void gen_perms(uint32 depth,string path);
    void create_buckets(int bucket_size); 
 
#endif
