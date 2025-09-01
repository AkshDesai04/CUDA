```c
/*
Aim of the program: Copy only a portion of `d_a` to `d_b` using pointer arithmetic.

Thinking:
1. Allocate two device arrays `d_a` and `d_b` of the same size `N`.
2. Fill `d_a` on the host with known values and copy it to the device.
3. Launch a kernel that copies a contiguous segment of `d_a` to the beginning of `d_b`.
   - The segment is specified by a starting offset `offset` and a length `len`.
   - Inside the kernel we use pointer arithmetic: `const int *src = d_a + offset;`.
   - Each thread copies one element: `d_b[threadIdx.x] = src[threadIdx.x];`.
4. Copy `d_b` back to the host and print its contents to verify that only the
   specified portion has been copied and the rest remains unchanged.
5. Include error checking after each CUDA API call.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10           // Size of the arrays
#define SEGMENT_LEN 4  // Number of elements to copy
#define OFFSET 2       // Starting offset in d_a

// Kernel that copies a segment of d_a to the beginning of d_b
__global__ void copy_segment(const int *d_a, int *d_b, int offset, int len)
{
    int idx = threadIdx.x;
    if (idx < len) {
        // Pointer arithmetic to obtain the source pointer at the offset
        const int *src = d_a + offset;
        d_b[idx] = src[idx];
    }
}

int main(void)
{
    int h_a[N], h_b[N];
    int *d_a = NULL, *d_b = NULL;
    size_t size = N * sizeof(int);

    // Initialize host array h_a with values 0..N-1
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = -1; // sentinel value to detect unchanged elements
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_a failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_b failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to d_a failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        return 1;
    }

    // Launch kernel: one block, SEGMENT_LEN threads
    copy_segment<<<1, SEGMENT_LEN>>>(d_a, d_b, OFFSET, SEGMENT_LEN);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to h_b failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        return 1;
    }

    // Print host arrays to verify
    printf("Original array h_a: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_a[i]);
    printf("\n");

    printf("Copied array h_b:   ");
    for (int i = 0; i < N; ++i) printf("%d ", h_b[i]);
    printf("\n");

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
```