```c
/*
Aim of the program:
Write a program that finds the GPU with the most available free memory at runtime (using `cudaMemGetInfo`) and selects it to run a computation.

Thinking:
1. We need to enumerate all CUDA devices using `cudaGetDeviceCount`.
2. For each device, we query free memory with `cudaMemGetInfo(&free, &total)`.
3. Track the device with the maximum free memory.
4. Set that device as the current device with `cudaSetDevice`.
5. Perform a simple computation (vector addition) to demonstrate usage.
6. Include error checking macros for clean code.
7. Use a basic kernel that adds two arrays.
8. All code is within a single .cu file, starting with a multiline comment containing the aim and this reasoning.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API return values */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

/* Simple vector addition kernel */
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    size_t maxFree = 0;
    int selectedDevice = 0;

    /* Find device with most free memory */
    for (int dev = 0; dev < deviceCount; ++dev) {
        size_t freeMem, totalMem;
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        printf("Device %d: free = %zu MB, total = %zu MB\n", dev,
               freeMem / (1024 * 1024), totalMem / (1024 * 1024));
        if (freeMem > maxFree) {
            maxFree = freeMem;
            selectedDevice = dev;
        }
    }

    printf("Selected Device %d with %zu MB free memory.\n",
           selectedDevice, maxFree / (1024 * 1024));

    /* Set the selected device */
    CUDA_CHECK(cudaSetDevice(selectedDevice));

    /* Prepare data for vector addition */
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Device memory allocation */
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy data to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Simple verification */
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            ok = false;
            printf("Mismatch at index %d: %f != %f\n", i, h_C[i], h_A[i] + h_B[i]);
            break;
        }
    }
    printf("Vector addition %s.\n", ok ? "succeeded" : "failed");

    /* Cleanup */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```