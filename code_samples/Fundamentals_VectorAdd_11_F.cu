/*
 * Aim of the program:
 * Can you still get out-of-bounds errors with a grid-stride loop if N is not a multiple of anything? (No, the loop condition i < N handles it perfectly).
 *
 * Thought process:
 * 1. The main point is to show that a grid-stride loop automatically handles cases where the total number of elements N is not a multiple of the total number of threads (blockDim.x * gridDim.x). 
 * 2. In the kernel, each thread starts at its global index and then steps through the array by the total stride. The loop condition i < N ensures that once i goes beyond the array bounds, the loop terminates for that thread, thus preventing out-of-bounds memory accesses.
 * 3. To illustrate this, we allocate an array of N elements on the host, copy it to the device, launch a kernel that squares each element using a grid-stride loop, and copy the result back.
 * 4. We choose N that is not a multiple of the total number of threads (e.g., N = 1000, blockDim = 256, gridDim = 2 -> stride = 512, so 1000 % 512 != 0). 
 * 5. We also add a small diagnostic: after the kernel we verify that each element has been squared correctly. If an out-of-bounds error occurred, the program would likely crash or produce wrong results. The absence of errors confirms the loop condition works.
 * 6. The program uses basic CUDA API calls, error checking, and prints a success message if all elements are correct.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that squares each element of the array using a grid‑stride loop
__global__ void squareArray(float *d_data, int N)
{
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Total stride (number of threads in the grid)
    int stride = blockDim.x * gridDim.x;

    // Grid‑stride loop with bounds check
    for (int i = idx; i < N; i += stride)
    {
        d_data[i] = d_data[i] * d_data[i];
    }
}

// Simple CUDA error checker
#define CHECK_CUDA(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

int main()
{
    const int N = 1000;                 // Size of the array (not a multiple of stride)
    const int threadsPerBlock = 256;    // Typical block size
    const int blocks = 2;               // Total threads = 512, stride = 512

    // Allocate host memory
    float *h_data = (float*)malloc(N * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with some values
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)i * 0.5f;  // arbitrary values
    }

    // Allocate device memory
    float *d_data;
    CHECK_CUDA(cudaMalloc((void**)&d_data, N * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    squareArray<<<blocks, threadsPerBlock>>>(d_data, N);
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = (i * 0.5f) * (i * 0.5f);
        if (fabs(h_data[i] - expected) > 1e-5f) {
            if (errors < 10) { // print first few errors
                printf("Mismatch at index %d: got %f, expected %f\n",
                       i, h_data[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("All %d elements processed correctly. No out-of-bounds errors detected.\n", N);
    } else {
        printf("Total mismatches: %d out of %d elements.\n", errors, N);
    }

    // Clean up
    free(h_data);
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
