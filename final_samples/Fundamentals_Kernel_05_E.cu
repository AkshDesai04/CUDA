```cuda
/*
   Write the global ID to an output array at the corresponding index: `output[id] = id;`. Verify the result on the host.

   Thinking process:
   1. The goal is to demonstrate how each CUDA thread can write its global thread ID into a position in a device array,
      and then check that this operation was performed correctly by inspecting the results on the host.
   2. For this we need:
      - A kernel that calculates the global ID of the thread and writes it into the output array at that index.
      - A host routine that:
        * Allocates a device array large enough to hold the results.
        * Launches the kernel with enough blocks/threads to cover all indices.
        * Copies the results back to host memory.
        * Iterates over the results verifying that each entry equals its index.
   3. We'll use a simple 1D grid and block configuration. The total number of threads will be at least the size of the array.
      Threads with an ID beyond the array size will simply return.
   4. Error handling: define a macro `CUDA_CHECK` that checks the return value of CUDA runtime calls and aborts with an error message if necessary.
   5. For demonstration purposes we use a fixed array size (e.g., 1 << 20 = 1,048,576). This can easily be changed.
   6. After copying back to host, we loop over the array and print a message if any element is incorrect; otherwise we print a success message.
   7. The kernel is trivial: `int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < N) output[id] = id;`.

   This program is self‑contained and can be compiled with `nvcc`:
       nvcc -o global_id_writer global_id_writer.cu
   and then run it. It should print "Verification passed." on successful execution.
*/

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Define the size of the output array
const int N = 1 << 20; // 1,048,576 elements

__global__ void write_global_id(int *output, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        output[id] = id;
    }
}

int main(void) {
    int *d_output = NULL;
    int *h_output = NULL;

    size_t bytes = N * sizeof(int);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_output, bytes));

    // Allocate host memory
    h_output = (int *)malloc(bytes);
    if (h_output == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Determine grid and block dimensions
    const int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    write_global_id<<<blocksPerGrid, threadsPerBlock>>>(d_output, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Verify results on host
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_output[i] != i) {
            fprintf(stderr, "Verification failed at index %d: expected %d, got %d\n",
                    i, i, h_output[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Verification passed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_output));
    free(h_output);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```