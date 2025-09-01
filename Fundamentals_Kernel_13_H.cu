/*
In the host code, use `cudaMemset` to initialize the input array `a` to a specific value (e.g., 1) on the device before launching the kernel.

My approach to writing this CUDA program is as follows:
1. I need a minimal, self‑contained example that shows the use of `cudaMemset` on device memory.  
2. I will allocate an array of floats on the device, set all of its bytes to a particular value using `cudaMemset`, and then launch a simple kernel that reads from this array, does something trivial (like scaling each element by 2.0f), and writes the result to another array on the device.  
3. After the kernel finishes, I will copy the result back to host memory and print it out so that the user can see the effect of the initial memset.  
4. I will include proper error checking after each CUDA API call to make the code robust and easier to debug.  
5. The code will be written in plain C/C++ CUDA syntax, compiled with `nvcc`, and will reside entirely in a single `.cu` file as requested.  
6. The kernel will be simple: it will read from `a`, multiply by 2.0f, and store into `b`.  
7. The host will create an array `h_a` on the CPU, but it will never be used to initialize the device memory; instead, the device memory will be initialized directly with `cudaMemset`.  
8. The `cudaMemset` call will set each byte of the device array to the value `1`, which means that every float element will have the bit pattern `0x01010101`.  This is an intentional demonstration of how `cudaMemset` operates at the byte level, not as a “set to 1.0” operation.  After the kernel runs, the output will therefore be `2.0 * 0x01010101` interpreted as a float.  
9. To keep the example clear, I will size the array to 10 elements, but this can be adjusted easily.  
10. Finally, I will print the host result array to illustrate the final values.  

With this plan, the resulting code will satisfy the requirement of using `cudaMemset` to initialize device memory before launching a kernel, and it will be straightforward for users to compile and run with `nvcc`. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that doubles each element of array a and writes to array b
__global__ void doubleArray(const float *a, float *b, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        b[idx] = 2.0f * a[idx];
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void)
{
    const int N = 10;                // Number of elements
    const size_t size = N * sizeof(float);

    // Allocate host memory (unused for initializing device memory)
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);

    // Allocate device memory
    float *d_a = NULL;
    float *d_b = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));

    // Initialize device array d_a with the byte value 1 using cudaMemset
    // This sets each byte of the float array to 0x01, resulting in bit pattern 0x01010101
    CUDA_CHECK(cudaMemset(d_a, 1, size));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    doubleArray<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Result of doubling the memset-initialized array:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("h_b[%d] = %f\n", i, h_b[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
