Genome Indexing on the GPU
--------------------------
Construction of suffix arrays on the GPU using CUDA.

<last edit: 05/14/12>

Parallel quick sort
-------------------
```bucketsort/qsort.cu ``` - Parallel bucket sort + Parallel quick sort of suffix array.
 
Compile using 

``` nvcc -I <NVIDIA_SDK_PATH>/C/common/inc qsort.cu -o qsort```

Run as 

``` ./qsort <suffix_size, multiple of 1024> <genome_file> ```

e.g. ``` ./qsort 10240 genome ```


```bucketsort/bsort.cu``` - Parallel Bucket sort based on common prefix.

Thrust sort
-----------

```sort.cu``` - Program to sort the suffixes of a string on the GPU using the thrust libraries and CUDA. 

DC3 algorithm
--------------
```dc3.c``` -  [Linear time suffix array consruction on the CPU](http://www.cs.helsinki.fi/u/tpkarkka/publications/jacm05-revised.pdf "Title")

Quick sort CPU
--------------
```quick_sort.c``` - A program to sort suffixes of a string using quick sort on CPU.

Time Complexity: O(n^2*log(n))

Authors
-------

Varrun Ramani <vramani@cs.stonybrook.edu>

Shishir Horane <shorane@cs.stonybrook.edu>

Ashish Bagate <abagate@cs.stonybrook.edu>