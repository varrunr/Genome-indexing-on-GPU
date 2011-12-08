#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>

// Device String Comparator Function
template <typename T>
struct greater_functor {

	thrust::device_ptr<unsigned long> gene;
	unsigned long i,j;

	greater_functor(thrust::device_ptr<unsigned long> _gene) : gene(_gene) {}
	
	__device__
	int operator()( T x, T y){
		for(i = x,j=y;; j++,i++)
		{
			if(gene[i] == (unsigned long)'\0')
				return(0<1);
			if (gene[j]== (unsigned long)'\0') 
				return(1<0); 
			if(gene[i] != gene[j])
				return(gene[i] < gene[j]);
		}
	}
};

// Allocate space on device and copy genome onto it
// call Thrust::stable_sort function with our custom comparator.
void sort_Suffixes(unsigned long* gene, thrust::device_vector<unsigned long>& A ,unsigned long N){
	unsigned long *dgene;
	cudaMalloc((void **) &dgene, (N+1) * sizeof(unsigned long));
	cudaMemcpy(dgene,gene, (N+1) * sizeof(unsigned long), cudaMemcpyHostToDevice);
	thrust::device_ptr<unsigned long> dev_ptr(dgene);
	thrust::stable_sort(A.begin(),A.end(),greater_functor<unsigned long>(dev_ptr));
	cudaFree(dgene);
}


void print_suffix_list(thrust::device_vector<unsigned long>& list, unsigned long len, char* genome){
    int i=0;
    for(i=0; i<len; i++){
        printf("%ld: %s\n", (unsigned long)list[i], genome+(unsigned long)list[i]);
    }
}

void read_genome2(char *filename, char *buffer, unsigned long num){
    FILE *fh;
    fh = fopen(filename, "r");
    fread(buffer, 1, num, fh);
    buffer[num] = '\0';
    fclose(fh);
}

 unsigned long setup(unsigned long num, char* filename, char** genome){
	*genome = (char *) malloc((num+1)*sizeof(char));
    read_genome2(filename, *genome, num);
    return (strlen(*genome));	
 }
 
int main(int argc, char* argv[])
{
	if(argc < 5){
		printf("Less Arguments!! \n");
		return 0;
	}
	unsigned long count = atol(argv[1]);
	unsigned long increaseSize = atol(argv[2]);
	unsigned long maxSize = atol(argv[3]);
	
	while(count <= maxSize){
		char * genome;
		unsigned long N = setup(count,argv[4], &genome);
		unsigned long i = 0;
		unsigned long * genome2;
		genome2 =(unsigned long *)malloc(N*sizeof(unsigned long));
		for(i=0;i<N;i++){
			genome2[i] = (unsigned long)genome[i];
		}
		free(genome);
		thrust::device_vector<unsigned long> A(count);
		thrust::sequence(A.begin(),A.end());

		clock_t start, end;
		double runTime;
		start = clock();

		try
		{
			sort_Suffixes(genome2, A,N);
	    }
	    catch(thrust::system_error e)			// Terminate Gracefully
	    {
	      std::cerr << "Error inside sort: " << e.what() << std::endl;
	    }
		end = clock();
		runTime = (end - start) / (double) CLOCKS_PER_SEC ;

		printf("%ld %f\n",count, runTime);
		count = count + increaseSize;
	}
}
