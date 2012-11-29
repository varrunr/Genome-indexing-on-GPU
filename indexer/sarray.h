#ifndef __SARRAY_H__
#define __SARRAY_H__


#define MAX_DATA_SIZE 262144
#define THREADS_PER_BLOCK 256
#define ALPHA_SIZE 4
#define NALPHA 4
#define uint32 unsigned int

    /* Constants */
    const char  alpha[4]    =   {'A','C','G','T'};

    /* Variables */

    int     *cpu_suf_arr;
    int     *cpu_final_arr;
    int     *gpu_suf_arr;
    int     *gpu_aux_arr;
    char    *gpu_genome;
    char    *cpu_genome;

    /* Functions */

    void    read_genome2    ( char* , char* , int );
    int     setup           (int, char*);
    void    alloc_arr       (int,int);
    void    init_arr        (int,int);
    void    copy2dev        (int,int);
    
    
#endif
