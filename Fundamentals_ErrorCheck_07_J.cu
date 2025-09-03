/*
Aim: Add detailed comments to your two code snippets explaining exactly why one error is synchronous and the other is asynchronous.

Thinking:
- The CUDA runtime API generally reports errors synchronously for functions that block until the operation completes, e.g. cudaMemcpy, cudaMalloc, cudaMemset. These calls return an error code immediately because they are executed on the host and the host thread waits until the operation finishes. Thus if a problem is detected (invalid pointer, size mismatch, memory out‑of‑range, etc.), the error is returned right away.

- Kernel launches are asynchronous by design. The CUDA driver queues the kernel for execution on the device and returns control to the host immediately. Errors that occur during kernel execution (e.g. accessing memory out of bounds, illegal instruction) cannot be detected until the kernel actually runs on the device. The CUDA runtime therefore defers error reporting until a synchronization point such as cudaDeviceSynchronize(), cudaStreamSynchronize(), or a subsequent API call that forces the host to wait for the kernel to finish (e.g. cudaMemcpy from device to host). Hence a kernel launch error is asynchronous.

- In the following code I provide two snippets. The first demonstrates a synchronous error with cudaMemcpy by intentionally passing an invalid host pointer. The error is returned immediately. The second snippet launches a kernel that writes out of bounds, causing an illegal memory access. The error is not reported until we call cudaDeviceSynchronize(). The comments explain why each error behaves as it does.
*/

#include <cuda_runtime.h>
#include <stdio.h>

/* --------------------------------------------------------------------- */
/* 1. Synchronous error example: cudaMemcpy with an invalid host pointer */
/* --------------------------------------------------------------------- */
__global__ void dummyKernel(int *d_out, int val) {
    int idx = threadIdx.x;
    d_out[idx] = val;
}

int main() {
    const int N = 10;
    int *d_out = nullptr;
    int *h_out = (int *)malloc(N * sizeof(int));

    /* Allocate device memory */
    cudaError_t err = cudaMalloc((void **)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Launch kernel to initialize d_out */
    dummyKernel<<<1, N>>>(d_out, 42);
    /* We deliberately do not synchronize here; kernel launch is async */

    /* --------------------------------------------------------------- */
    /* Synchronous error: pass an invalid host pointer to cudaMemcpy */
    /* --------------------------------------------------------------- */
    int *invalidHostPtr = nullptr;  // Intentionally set to NULL
    err = cudaMemcpy(invalidHostPtr, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    /* Explanation:
     * cudaMemcpy is a host blocking call. It copies data from device to host
     * and waits for the transfer to complete before returning. Because the
     * host pointer is NULL, the runtime detects the invalid pointer before
     * initiating the transfer and returns cudaErrorInvalidValue immediately.
     * This is a *synchronous* error – the caller sees the error at the point
     * of the function call. */
    if (err != cudaSuccess) {
        printf("Synchronous error detected: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaMemcpy succeeded (unexpected).\n");
    }

    /* Clean up device memory */
    cudaFree(d_out);
    free(h_out);

    /* --------------------------------------------------------------- */
    /* 2. Asynchronous error example: kernel launch with out-of-bounds write */
    /* --------------------------------------------------------------- */
    // Reallocate memory for new test
    h_out = (int *)malloc(N * sizeof(int));
    err = cudaMalloc((void **)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Kernel launch: each thread writes to d_out[threadIdx.x] */
    /* But we intentionally request more threads than allocated memory to trigger out-of-bounds. */
    const int threads = N + 5;  // 5 threads will write past the end of d_out
    dummyKernel<<<1, threads>>>(d_out, 99);

    /* Explanation:
     * Kernel launches are asynchronous. The launch call enqueues the kernel
     * and returns immediately without waiting for the kernel to finish. The
     * device will execute the kernel, and because we have 5 extra threads
     * writing past the allocated array, they will access illegal memory.
     * The CUDA runtime cannot detect this fault during the launch, so it
     * defers error reporting until a synchronization point. */
    err = cudaGetLastError();  // Checks errors that happened during launch only (none yet)
    if (err != cudaSuccess) {
        printf("cudaGetLastError after launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaGetLastError after launch: No error (as expected).\n");
    }

    /* The actual error will surface when we synchronize. */
    err = cudaDeviceSynchronize();
    /* Explanation:
     * cudaDeviceSynchronize blocks the host until all preceding device work
     * (including the kernel) has completed. At this point, the device reports
     * the illegal memory access error. Therefore the error is *asynchronous*:
     * it is only observed after synchronization, not at the launch point. */
    if (err != cudaSuccess) {
        printf("Asynchronous error detected after cudaDeviceSynchronize: %s\n",
               cudaGetErrorString(err));
    } else {
        printf("cudaDeviceSynchronize succeeded (unexpected).\n");
    }

    /* Clean up */
    cudaFree(d_out);
    free(h_out);

    return 0;
}
