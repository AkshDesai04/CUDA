/*
Demonstrate its use by checking a successful kernel launch.

The aim of this program is to showcase how to verify that a CUDA kernel
has launched correctly.  The core idea is to perform a simple
computation on the GPU, capture any launch or execution errors, and
report success only when the kernel executes without error.  The
program follows these steps:

1.  Define a small CUDA kernel that writes its global thread index into
    an output array.  This kernel is intentionally trivial so that we
    can focus on error handling.

2.  In the host code, allocate a host array and a device array of a
    fixed size.  Copy the host data to the device (the data is not
    actually used by the kernel but the copy demonstrates a normal
    workflow).

3.  Launch the kernel with a configuration that covers the entire
    array.  Immediately after the launch we call
    `cudaGetLastError()` to detect any launch‑time problems
    (e.g. illegal configuration).  After that we call
    `cudaDeviceSynchronize()` to catch any runtime errors that
    happen during execution.

4.  If both error checks return `cudaSuccess`, we copy the results
    back to the host and print a success message along with the first
    few results.  If any error occurs, the program reports it and
    exits.

5.  A small helper macro `CHECK_CUDA` is defined to simplify error
    checking for CUDA API calls.

The program is self‑contained and can be compiled with a standard CUDA
toolkit.  It demonstrates the essential pattern for robust kernel
execution: launch → launch‑error check → sync → runtime‑error check.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro to check CUDA API calls */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Simple kernel that writes the thread index into an array */
__global__ void writeIndices(int *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = idx;
    }
}

int main(void) {
    const int N = 256;               // Number of elements
    const int threadsPerBlock = 64;  // Threads per block
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Host memory allocation */
    int *h_out = (int*)malloc(N * sizeof(int));
    if (h_out == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Device memory allocation */
    int *d_out = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_out, N * sizeof(int)));

    /* Optional: initialize device memory to zero */
    CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(int)));

    /* Launch kernel */
    writeIndices<<<blocksPerGrid, threadsPerBlock>>>(d_out, N);

    /* Check for launch errors */
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        CHECK_CUDA(cudaFree(d_out));
        free(h_out);
        return EXIT_FAILURE;
    }

    /* Synchronize and check for runtime errors */
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(syncErr));
        CHECK_CUDA(cudaFree(d_out));
        free(h_out);
        return EXIT_FAILURE;
    }

    /* Copy results back to host */
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Clean up device memory */
    CHECK_CUDA(cudaFree(d_out));

    /* Display a few results to confirm correctness */
    printf("Kernel launch successful! First 10 results:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    /* Free host memory */
    free(h_out);

    return EXIT_SUCCESS;
}
