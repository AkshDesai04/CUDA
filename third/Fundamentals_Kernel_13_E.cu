```cpp
/* 
Aim of the program: Use `float` arrays instead of `int` arrays.

Thinking process:
1. The requirement explicitly states that we should use float arrays instead of int arrays. 
2. The simplest CUDA example that demonstrates array usage is vector addition. 
3. I will implement a basic vector addition kernel that operates on float arrays. 
4. The host code will allocate float arrays, initialize them with some values, copy them to the device, launch the kernel, copy the result back, and verify the output.
5. For clarity and debugging, I will print the first few elements of the result.
6. I will include basic error checking for CUDA API calls.
7. The program will be self‑contained and compile as a .cu file using nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel for adding two float vectors
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i * 0.5f;
        h_b[i] = (float)i * 1.5f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, size));
    CHECK_CUDA(cudaMalloc((void **)&d_b, size));
    CHECK_CUDA(cudaMalloc((void **)&d_c, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    int correct = 1;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            printf("Mismatch at index %d: GPU %f, CPU %f\n", i, h_c[i], expected);
            correct = 0;
            break;
        }
    }
    if (correct) {
        printf("Vector addition successful. First 10 results:\n");
        for (int i = 0; i < 10; ++i) {
            printf("h_c[%d] = %f\n", i, h_c[i]);
        }
    } else {
        printf("Vector addition failed.\n");
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```