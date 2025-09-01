/*
Aim: In the version with error checking, use your CHECK macro to catch and report the cudaErrorIllegalAddress.
 
My thinking:
The requirement is to build a minimal CUDA program that intentionally triggers an illegal memory access (cudaErrorIllegalAddress) and uses a custom error checking macro to capture and report the error.  
To do this I will:

1. Define a `CHECK` macro that wraps a function `gpuAssert`.  `gpuAssert` will examine the CUDA error code, print a detailed message if the code is not `cudaSuccess`, and optionally abort execution.  
2. Allocate a small array on the device.  
3. Launch a kernel that writes past the bounds of this array.  A single thread will write to an index that is out of range, which should cause `cudaErrorIllegalAddress`.  
4. Immediately after launching the kernel I will call `CHECK(cudaGetLastError());` – this forces the runtime to report any launch-time errors, including the illegal address.  
5. I will also call `CHECK(cudaDeviceSynchronize());` so that any deferred errors (like the illegal address that can surface at sync time) are also caught.  

The program will print the error message produced by `gpuAssert` and then exit gracefully.  The code is a complete `.cu` file that can be compiled with `nvcc`. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API calls and kernel launch errors */
#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/* Error handling function */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Kernel that deliberately accesses memory out of bounds */
__global__ void illegalAccessKernel(int *arr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* The array is only 5 elements long, but we write to index 10 */
    arr[10] = idx;  /* This will trigger cudaErrorIllegalAddress */
}

int main(void)
{
    const int N = 5;
    int *devArray = nullptr;

    /* Allocate device memory */
    CHECK(cudaMalloc((void**)&devArray, N * sizeof(int)));

    /* Launch kernel with 1 block and 1 thread */
    illegalAccessKernel<<<1, 1>>>(devArray);

    /* Check for launch errors */
    CHECK(cudaGetLastError());

    /* Synchronize to catch any deferred errors */
    CHECK(cudaDeviceSynchronize());

    /* Clean up */
    CHECK(cudaFree(devArray));

    printf("Program completed without detecting an error (this line should not be reached).\n");
    return 0;
}
