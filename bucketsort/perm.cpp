#include<iostream>
#include<vector>
#define NALPHA 4
#define uint32 unsigned int
using namespace std;

char alpha[] = {'A','C','T','G'};
vector<string> buckets;

void gen_perms(uint32 depth,string path)
{
    if(path.size() == depth)
    {
        buckets.push_back(path);
        return;
    }
    for(int i=0;i<NALPHA;i++)
    {
        string temp = path;
        temp.append(1,alpha[i]);
        gen_perms(depth,temp);
    }
}

void create_buckets(int bucket_size)
{
    gen_perms(bucket_size,"");
}

int main()
{
    int N;
    cin>>N;
    create_buckets(N);
    cout<<buckets.size();
    return 0;
}
