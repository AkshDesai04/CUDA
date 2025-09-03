```cuda
/*
In the version without error checking, verify the output array and show that the results are wrong or nonsensical.

Thinking:
- The goal is to demonstrate that if we skip CUDA error checking after memory allocation, kernel launch, and memcpy, the program can silently produce incorrect results.
- We will allocate a device array with an incorrect size (smaller than the number of threads we launch).
- The kernel will write to this array using a global index that goes beyond the allocated space.
- Because we skip cudaGetLastError() and cudaDeviceSynchronize() checks, the program will not abort or report the out‑of‑bounds write.
- We then copy back more data than was actually allocated, causing undefined behavior in the host array.
- Finally, we print the host array to show nonsensical or corrupted values, confirming the lack of error checking leads to wrong results.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that writes twice the thread index to the output array
__global__ void doubleIndexKernel(int* out, int n) {
    int idx = threadIdx.x;
    // Intentionally writing even if idx >= n to trigger out‑of‑bounds
    out[idx] = idx * 2;
}

int main() {
    const int N = 20;                 // Desired number of elements
    const int BLOCK_SIZE = 20;        // Launch 20 threads
    int h_out[N];                     // Host array to hold results

    int *d_out;
    // Intentionally allocate memory for only 10 ints, not 20
    cudaMalloc((void**)&d_out, 10 * sizeof(int)); // Wrong size

    // Launch kernel without checking for errors
    doubleIndexKernel<<<1, BLOCK_SIZE>>>(d_out, N);
    // No cudaGetLastError() or cudaDeviceSynchronize() call here

    // Copy back 20 ints from device to host, even though only 10 were allocated
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    // No error checking on cudaMemcpy

    // Print the output array to show nonsensical values
    printf("Output array after kernel launch without error checking:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    // Clean up (device memory still freed even if it was incorrectly allocated)
    cudaFree(d_out);

    return 0;
}
```