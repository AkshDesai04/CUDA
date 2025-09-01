```cpp
/*
Aim of the program: Implement using `__half` precision numbers if your GPU supports it.

Thinking:
- The task is to write a CUDA C program that uses half-precision (__half) arithmetic
  only if the device supports it. GPUs with compute capability 5.3 and above
  support half precision in device code. For older GPUs we should gracefully
  exit with a message.
- We'll demonstrate half precision by performing a simple element-wise vector
  addition. We will allocate two input vectors, add them on the GPU using a
  kernel that operates on __half, and then copy the result back to the host
  for verification.
- Host code will use float arrays for convenience and convert to __half
  using the CUDA runtime intrinsic __float2half. When copying from host to
  device we can cast the float array to __half* because the binary layout
  of __half matches a 16-bit unsigned short. For safety we will explicitly
  cast each element to __half before copying.
- The kernel will be very small: it reads two __half inputs, adds them with
  __hadd, and writes the result to the output __half array. We'll use
  a standard block/grid launch with 256 threads per block.
- After the kernel, the result __half array is copied back to the host
  and converted back to float with __half2float for printing.
- Throughout the program we check for errors after each CUDA API call
  and print a helpful message if something goes wrong.
- The program uses only the CUDA runtime API, so it compiles with a
  simple `nvcc myprogram.cu -arch=sm_XX` command.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Number of elements in the vectors
#define N 1024

// CUDA error checking macro
#define CUDA_CHECK(err)                                           \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
                cudaGetErrorString(err), (int)err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                       \
    }

// Kernel that adds two __half vectors
__global__ void add_half(const __half *a, const __half *b, __half *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

int main() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Check if device supports half precision
    if (!(prop.major > 5 || (prop.major == 5 && prop.minor >= 3))) {
        fprintf(stderr, "Device %d (Compute %d.%d) does not support __half in device code.\n",
                device, prop.major, prop.minor);
        return EXIT_FAILURE;
    }
    printf("Using device %d: %s (Compute %d.%d)\n", device, prop.name, prop.major, prop.minor);

    // Allocate host memory
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    __half *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc((void**)&d_c, N * sizeof(__half)));

    // Convert host float arrays to __half and copy to device
    __half *h_a_half = (__half*)malloc(N * sizeof(__half));
    __half *h_b_half = (__half*)malloc(N * sizeof(__half));
    if (!h_a_half || !h_b_half) {
        fprintf(stderr, "Failed to allocate temporary half memory.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_a_half[i] = __float2half(h_a[i]);
        h_b_half[i] = __float2half(h_b[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_a, h_a_half, N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_half, N * sizeof(__half), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_half<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host and convert to float
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(__half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        h_c[i] = __half2float(((__half*)h_c)[i]);  // reinterpret as __half then convert
    }

    // Print a few results for verification
    printf("First 10 results of a + b (half precision):\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_a_half);
    free(h_b_half);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return EXIT_SUCCESS;
}
```