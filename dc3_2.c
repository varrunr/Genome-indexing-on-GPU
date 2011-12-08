#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>

#define MAX_ALPHA 26

#define GetI() (SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2)

void print_arr(int*, int, int);
void suffixArray(int* , int*, int , int ) ;
void radixPass(int* , int* , int* , int , int ); 
void merge_suffixes(int *, int *, int *, int *, int *, int, int, int, int);

char *cc;

inline int leq(int a1, int a2, int b1, int b2) {
        return(a1 < b1 || (a1 == b1 && a2 <= b2));
}

inline int leq2(int a1, int a2, int a3, int b1, int b2, int b3) {
        return(a1 < b1 || (a1 == b1 && leq(a2,a3, b2,b3)));
}

void read_genome(char *filename, char *buffer, int num){
    FILE *fh;
    fh = fopen(filename, "r");
    fread(buffer, 1, num, fh);
    buffer[num] = '\0';
    fclose(fh);
}


int to_i(char c)
{
    return (int)c - 61;
}
void print_suffix(char *cc, int i)
{
    int j=0;
    printf("%d: ",i);
    for(j=i;j<strlen(cc);j++)
        printf("%c",cc[j]);
    printf("\n");
}
int main(int argc, char* argv[])
{
	clock_t start, end;
	double runTime;
	char *filename = "genome";
	int num = atoi(argv[1]);;
	char *genome;
	int n = 0,i = 0;
    int *inp;
    int *SA;
    genome = (char *) malloc((num+1)*sizeof(char)); 
	start = clock();

    read_genome(filename, genome, num);
	cc = genome;
    n = strlen(cc);

	inp = (int *)malloc( (n+3)*sizeof(int) );
    SA  = (int *)malloc( (n+3)*sizeof(int) );
   
    for(i=0;i<n;i++)
        inp[i] = to_i(cc[i]);
    inp[i]=0;inp[i+1]=0;inp[i+2]=0;

    for(i=0;i<n+3;i++)
        SA[i] = 0;
    
    suffixArray(inp,SA,n,MAX_ALPHA);
    
    free(genome);
	end = clock();
	runTime = (end - start) / (double) CLOCKS_PER_SEC ;
	printf("%d %f\n", num, runTime);
    return 0;
}

void suffixArray(int* s, int* SA, int n, int K) {
    int n0=(n+2)/3, n1=(n+1)/3, n2=n/3, n02=n0+n2; 
    int i=0,j=0;
    int *s12 = (int *)malloc((n02 + 3)*sizeof(int));   
    int *SA12 =(int *)malloc((n02 + 3)*sizeof(int)); 
    int *s0 = (int *)malloc(n0*sizeof(int));
    int *SA0 = (int *)malloc(n0*sizeof(int));
    
    for(i=0;i<n02+3;i++)
    {
        SA12[i]=0;
        s12[i]=0;
    }
    SA12[n02]=SA12[n02+1]=SA12[n02+2]=0;

    s12[n02]= s12[n02+1]= s12[n02+2]=0; 
    // generate positions of mod 1 and mod  2 suffixes
    // the "+(n0-n1)" adds a dummy mod 1 suffix if n%3 == 1
  for (i=0, j=0;  i < n+(n0-n1);  i++) 
    if (i%3 != 0) s12[j++] = i;

    radixPass(s12 , SA12, s+2, n02, K);

    radixPass(SA12, s12 , s+1, n02, K);  

    radixPass(s12 , SA12, s  , n02, K);

    ///////////////////////////////

    // find lexicographic names of triples
    int max_rank = set_suffix_rank(s,s12,SA12,n02,n0);

    // if max_rank is less than the size of s12, we have a repeat. repeat dc3.
	// else generate the suffix array of s12 directly
	if(max_rank < n02){
	//	printf("Going Again !! \n");
		suffixArray(s12,SA12,n02,max_rank);
		for(i = 0;  i < n02;  i++)
			s12[SA12[i]] = i + 1;
	}else{
	    for(i = 0;  i < n02;  i++)
	    	SA12[s12[i] - 1] = i;
	}

	// stably sort the mod 0 suffixes from SA12 by their first character
	for (i=0, j=0;  i < n02;  i++)
		if (SA12[i] < n0) s0[j++] = 3*SA12[i];
	radixPass(s0, SA0, s, n0, K);

	// merge sorted SA0 suffixes and sorted SA12 suffixes
	merge_suffixes(SA0, SA12, SA, s, s12, n0, n1, n02, n);

	//printf("End of suffix array !!\n");
}

void merge_suffixes(int * SA0, int * SA12, int * SA, int * s, int * s12, int n0, int n1, int n02, int n){
	int p,t,k,i,j;
	  for (p=0,  t=n0-n1,  k=0;  k < n;  k++) {
	    int i = GetI(); // pos of current offset 12 suffix
	    int j = SA0[p]; // pos of current offset 0  suffix
	    if (SA12[t] < n0 ? leq(s[i], s12[SA12[t] + n0], s[j], s12[j/3]) : leq2(s[i],s[i+1],s12[SA12[t]-n0+1],s[j],s[j+1],s12[j/3+n0]))
	    { // suffix from SA12 is smaller
	      SA[k] = i;  t++;
	      if (t == n02) { // done --- only SA0 suffixes left
	        for (k++;  p < n0;  p++, k++) SA[k] = SA0[p];
	      }
	    } else {
	      SA[k] = j;  p++;
	      if (p == n0)  { // done --- only SA12 suffixes left
	        for (k++;  t < n02;  t++, k++) SA[k] = GetI();
	      }
	    }
	  }
}

int set_suffix_rank(int *orig_str, int *set_rank_arr, int *sorted_suff_arr, int n02, int n0){
    int name = 0, c0 = -1, c1 = -1, c2 = -1,i;
    for (i = 0;  i < n02;  i++) {
      if (orig_str[sorted_suff_arr[i]] != c0 || orig_str[sorted_suff_arr[i]+1] != c1 || orig_str[sorted_suff_arr[i]+2] != c2) {
        name++;
        c0 = orig_str[sorted_suff_arr[i]];
        c1 = orig_str[sorted_suff_arr[i]+1];
        c2 = orig_str[sorted_suff_arr[i]+2];
      }
      if (sorted_suff_arr[i] % 3 == 1) {
    	  set_rank_arr[sorted_suff_arr[i]/3] = name;
      } // left half
      else{
    	  set_rank_arr[sorted_suff_arr[i]/3 + n0] = name;
      } // right half
    }
    return name;
}

void radixPass(int* to_be_sorted, int* sorted_suf_arr, int* orig_str, int n, int K) 
{ // count occurrences
  int *count = malloc((K + 1)*sizeof(int)); // counter array
  int i=0,t=0,sum=0;
  for (i = 0;  i <= K;  i++) count[i] = 0;
  // reset counters
  for (i = 0;  i < n;  i++){
	  count[orig_str[to_be_sorted[i]]]++;
    }    
// count occurences
  for (i = 0, sum = 0;  i <= K;  i++) { // exclusive prefix sums
      t = count[i];  count[i] = sum;  sum += t;
  }
  for (i = 0;  i < n;  i++)
	  sorted_suf_arr[count[orig_str[to_be_sorted[i]]]++] = to_be_sorted[i];      // sort
}

void print_arr(int *arr, int no_arr,int dimension)
{
    int i = 0;
    for(i=0; i<no_arr;i++){
        printf("%d ", arr[i]);
        if( (i+1)%dimension == 0)
            printf(", ");
    }
    printf("\n");
}

