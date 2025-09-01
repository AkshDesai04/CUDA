```cpp
/* kernel.cu
Aim: Write the kernel definition in a separate .cu file and the main function in another. Try to compile and link them together with nvcc.

Thinking:
The assignment requires splitting a CUDA program into two .cu files: one containing only the kernel definition and the other containing the host main function.  
To make the compiler/linker happy we must ensure that the main file can see the kernel declaration.  
The easiest way is to forward‑declare the kernel in main.cu using `extern __global__ void`.  
We’ll implement a simple vector addition kernel, `vectorAdd`, that adds two float vectors element‑wise.

Kernel file (`kernel.cu`) will:
- Include `<cuda_runtime.h>` and `<cstdio>` for error handling.
- Define a constexpr for `BLOCK_SIZE`.
- Provide the kernel implementation.
- Optionally print a small message from the kernel for demonstration (though usually kernel prints are avoided).

Main file (`main.cu`) will:
- Include `<cuda_runtime.h>` and `<cstdio>`.
- Allocate host and device memory for two input vectors and one output vector.
- Initialize the inputs.
- Copy inputs to device.
- Launch the kernel with a grid that covers all elements.
- Copy the result back to host.
- Verify the result and print a success message.
- Clean up all allocated memory.
- Return 0 on success.

The program can be compiled and linked with:
```
nvcc -o vecAdd main.cu kernel.cu
```
and run as:
```
./vecAdd
```
which should print “Vector addition successful.” and the resulting vector. The code is fully self‑contained and uses only CUDA runtime APIs. */

#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 256;

// Simple vector addition kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}
```

```cpp
// main.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Forward declaration of the kernel defined in kernel.cu
extern __global__ void vectorAdd(const float*, const float*, float*, int);

int main()
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            ok = false;
            printf("Mismatch at index %d: %f != %f\n", i, h_C[i], h_A[i] + h_B[i]);
            break;
        }
    }

    if (ok)
        printf("Vector addition successful.\n");
    else
        printf("Vector addition failed.\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
```