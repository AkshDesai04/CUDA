/*
Use a struct as the type for your __constant__ variable. Copy a host struct to it and access its members in the kernel.
The goal of this program is to demonstrate how to store a custom C struct in the CUDA constant memory,
copy data from the host to that constant memory, and then access those values from a kernel.
We define a struct MyData that contains an integer, a float, and a character array.
The host creates an instance of this struct, populates its fields, and uses cudaMemcpyToSymbol
to copy it into the constant memory symbol d_constData.
In the kernel, we read the struct members directly from constant memory and use them to compute
values for an output array.
The program also demonstrates basic error checking and uses device-side printf to verify that the
kernel sees the correct constant data. This example is simple but illustrates the mechanism
for using structs in constant memory.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

struct MyData {
    int    val1;
    float  val2;
    char   name[32];
};

// Constant memory variable of type MyData
__constant__ MyData d_constData;

// Kernel that uses the constant memory struct
__global__ void computeKernel(int *d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Simple computation using the constant data
        int result = d_constData.val1 + (int)(d_constData.val2 * idx);
        d_out[idx] = result;
    }

    // Device-side printf to confirm constant data values
    if (idx == 0) {
        printf("Kernel sees constant data: val1=%d, val2=%.2f, name=%s\n",
               d_constData.val1, d_constData.val2, d_constData.name);
    }
}

int main(void)
{
    const int N = 10;
    size_t size = N * sizeof(int);

    // Allocate host memory
    int h_out[N];

    // Allocate device memory
    int *d_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_out, size));

    // Prepare host struct and copy it to constant memory
    MyData h_data;
    h_data.val1 = 42;
    h_data.val2 = 3.14f;
    snprintf(h_data.name, sizeof(h_data.name), "example");

    CHECK_CUDA(cudaMemcpyToSymbol(d_constData, &h_data, sizeof(h_data)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, N);
    CHECK_CUDA(cudaGetLastError());

    // Wait for GPU to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Kernel output:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
