/*
Aim: 
Can you call a `__global__` function from a `__device__` function? (No, unless using Dynamic Parallelism). Can you call a `__device__` function from a `__global__` function? (Yes). Explain this in a comment.

Thinking:
The CUDA programming model distinguishes between host code, device code, and kernels. 
A `__global__` function is a kernel that must be launched from host (or via dynamic parallelism). 
Inside device code (`__device__` functions, device kernels), you cannot launch a kernel directly because that would require a launch configuration, which is only allowed from host or via dynamic parallelism. Hence, calling a `__global__` function from a `__device__` function is illegal unless dynamic parallelism is enabled (CUDA 5.0+). 
Conversely, a `__global__` kernel can freely call any `__device__` function; device functions are ordinary functions that run on the GPU and are available to all device code. The compiler will generate the appropriate calls. 

The following code demonstrates a simple kernel (`kernel`) that calls a device function (`devFunc`). It also contains commented-out code that would attempt to call a kernel from a device function, illustrating why that is disallowed. 
The main function allocates memory, launches the kernel, copies results back, and prints them. 
*/

#include <stdio.h>

__device__ void devFunc(int *d, int val)
{
    // Simple device function that writes a value to memory
    *d = val;
}

/*
// This code would NOT compile because a __device__ function cannot launch a __global__ kernel.
// The following is for illustration only and is commented out.

// __device__ void attemptToLaunchKernel(int *d)
// {
//     // Attempt to launch kernel from device code (illegal)
//     kernel<<<1,1>>>(d); // This line would produce a compilation error
// }
*/

__global__ void kernel(int *d)
{
    // Each thread writes its index into the array using the device function
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    devFunc(d + idx, idx);
}

int main(void)
{
    const int N = 10;
    int *h = new int[N];
    int *d;

    cudaMalloc((void**)&d, N * sizeof(int));

    // Launch kernel: one block of N threads
    kernel<<<1, N>>>(d);
    cudaDeviceSynchronize(); // Ensure kernel completion

    cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h[i]);
    }
    printf("\n");

    cudaFree(d);
    delete[] h;

    return 0;
}
