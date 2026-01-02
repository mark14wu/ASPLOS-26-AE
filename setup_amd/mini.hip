#include <cstdlib>
#include <hip/hip_runtime.h>

__global__ void
set1(int *p)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    p[i] = 1;
}

int
main(int argc, char **argv)
{
    int m = std::atoi(argv[1]);
    int n1 = std::atoi(argv[2]);
    int n2 = std::atoi(argv[3]);
    int c = std::atoi(argv[4]);
    int *dp;
    hipMalloc(&dp, m*sizeof(int));
    hipLaunchKernelGGL(set1, dim3(n1), dim3(n2), 0, 0, dp);
    int *hp = (int*)malloc(c * sizeof(int));
    hipMemcpy(hp, dp, m*sizeof(int), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    hipFree(dp);
    free(hp);
    std::puts("Done.");
    return 0;
}

