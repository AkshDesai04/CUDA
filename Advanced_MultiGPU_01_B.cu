```cuda
/*
 * Aim of the program:
 * Write a program that finds the GPU with the most available free memory at runtime (using cudaMemGetInfo) 
 * and selects it to run a computation.
 *
 * Thinking and design notes:
 * 1. The program will query the number of CUDA-capable devices using cudaGetDeviceCount().
 * 2. For each device, it will:
 *    a. Query total and free memory using cudaMemGetInfo().
 *    b. Keep track of the device with the maximum free memory.
 * 3. Once the best device is identified, the program will set it as the active device using cudaSetDevice().
 * 4. As a demonstration of running a computation, the program will perform a simple vector addition kernel
 *    on the selected device. The kernel will add two vectors of floats and store the result in a third vector.
 * 5. Error handling: All CUDA API calls will be checked for errors. If any error occurs, a message is printed
 *    and the program exits.
 * 6. The program will allocate host memory for two input vectors and one output vector, transfer them to the
 *    device, launch the kernel, copy the result back, and verify correctness by comparing with a CPU result.
 * 7. Clean up: free device and host memory before exiting.
 * 8. The code is written as a single .cu file and can be compiled with nvcc:
 *        nvcc -o max_mem_gpu max_mem_gpu.cu
 * 9. The program is self-contained, uses only standard CUDA runtime API, and does not require external libraries.
 * 10. Edge cases:
 *     - If no CUDA devices are present, the program reports this and exits.
 *     - If multiple devices have the same free memory, the first one encountered is chosen.
 * 11. Performance note: The kernel is intentionally simple; the focus is on device selection logic.
 * 12. Future extensions could involve profiling or dynamic memory allocation based on available free memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Simple error checking macro */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel: vector addition */
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

/* Main function */
int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    int bestDevice = 0;
    size_t maxFreeMem = 0;
    size_t freeMem, totalMem;

    /* Find device with most free memory */
    for (int dev = 0; dev < deviceCount; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        printf("Device %d: free memory = %zu MB, total memory = %zu MB\n",
               dev, freeMem / (1024 * 1024), totalMem / (1024 * 1024));
        if (freeMem > maxFreeMem) {
            maxFreeMem = freeMem;
            bestDevice = dev;
        }
    }

    printf("\nSelecting device %d with %zu MB free memory.\n", bestDevice, maxFreeMem / (1024 * 1024));
    CUDA_CHECK(cudaSetDevice(bestDevice));

    /* Problem size */
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    if (!h_a || !h_b || !h_c || !h_ref) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input vectors */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify result on CPU */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        h_ref[i] = h_a[i] + h_b[i];
        if (fabs(h_c[i] - h_ref[i]) > 1e-5f) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at %d: GPU %f != CPU %f\n", i, h_c[i], h_ref[i]);
            }
            ++errors;
        }
    }

    if (errors == 0) {
        printf("Result verified: all values match.\n");
    } else {
        printf("Result verification failed: %d mismatches.\n", errors);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);

    return EXIT_SUCCESS;
}
```