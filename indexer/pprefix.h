#ifndef __PPREFIX_H__
#define __PPREFIX_H__

#define PPREFIX_BLK_SZ 512

void prefixCompute( int *gpu_prefix_arr, dim3 blockGridRows, dim3 threadBlockRows, 
                    int block_size, int start, int end, int prefixesCount);

#endif
