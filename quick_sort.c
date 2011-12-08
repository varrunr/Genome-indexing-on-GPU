#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
int DEBUG = 1;
char *genome;

void read_genome2(char *filename, char *buffer, int num){

    FILE *fh;
    fh = fopen(filename, "r");
    fread(buffer, 1, num, fh);
    buffer[num] = '\0';
    fclose(fh);
}

int * get_suffix_list(int len){

    int *suffix_list = (int *) malloc(len*sizeof(int));
    int i;

    for(i=0; i<len; i++){
        suffix_list[i] = i;         
    }
    return suffix_list;
}

void quicksort(int* x, int first, int last){
    int pivot,j,temp,i;

     if(first<last){
         pivot=first;
         i=first;
         j=last;

         while(i<j){
             while(strcmp(genome+x[i], genome+x[pivot]) < 1  && i<last)
                 i++;
             while(strcmp(genome+x[j], genome+x[pivot]) >= 1)
                 j--;
             if(i<j){
                 temp=x[i];
                  x[i]=x[j];
                  x[j]=temp;
             }   
         }   

         temp=x[pivot];
         x[pivot]=x[j];
         x[j]=temp;
         quicksort(x,first,j-1);
         quicksort(x,j+1,last);

    }   
}
  
void print_suffix_list(int *list, int len){
    int i=0;
    for(i=0; i<len; i++){
        printf("%d: %s\n", list[i], genome+list[i]);
    }
}

int main(int argc, char *argv[]){
	clock_t start, end;
	double runTime;


    if(argc != 2){
        printf("Usage: ./build_suffix_array <num of bases to read>\n");
        exit(-1);
    }
    
    int num = atoi(argv[1]);
    char *filename = "genome";

	start = clock();
    genome = (char *) malloc((num+1)*sizeof(char));
    read_genome2(filename, genome, num);

    int genome_len = strlen(genome);

    int *suffix_list = get_suffix_list(strlen(genome));
    quicksort(suffix_list, 0, genome_len-1);
//   print_suffix_list(suffix_list, genome_len);
    
	end = clock();
	free(genome);

	runTime = (end - start) / (double) CLOCKS_PER_SEC ;
	printf("%d %f\n", num, runTime);
}


