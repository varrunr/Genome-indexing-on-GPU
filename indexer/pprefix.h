#ifndef __PPREFIX_H__
#define __PPREFIX_H__

void prefixCompute( int *gpu_prefix_arr, dim3 blockGridRows, dim3 threadBlockRows, 
                    int block_size, int start, int end, int prefixesCount);
#endif
