```cuda
/*
  Allocate a 2D array on the host (e.g., M rows, N cols) and represent it as a 1D array in row‑major order.
  Allocate a corresponding 2D array on the device using cudaMallocPitch.
  Copy the host array to the device using cudaMemcpy2D.
  Explain in comments the purpose of the 'pitch' value.

  Thinking:
  1. We need a host array that is logically 2D but stored in a single 1D buffer.
     The simplest layout is row‑major: element (r,c) is stored at index r*N + c.
  2. On the device we will use cudaMallocPitch, which allocates a buffer that
     guarantees each row starts on a memory address that is a multiple of a
     "pitch" value (the number of bytes between the start of successive rows).
     This alignment can improve memory coalescing and is required for some
     hardware constraints.
  3. The pitch returned by cudaMallocPitch is usually larger than the exact
     row size (N*sizeof(T)).  For example, a row of 5 floats (20 bytes) might
     be padded to 32 bytes.  We must use this pitch when copying or accessing
     the array on the device.
  4. To copy from the host 1D array to the pitched device memory we use
     cudaMemcpy2D, which takes the source and destination pitch values.
  5. We also include a small kernel that prints the device array to verify the
     copy succeeded.  The kernel accesses the device array using the pitch.
  6. Finally we copy the data back to the host and print it to confirm correctness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define M 5   // number of rows
#define N 7   // number of columns

// Simple error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that prints the device array (for demonstration; not efficient)
__global__ void printDeviceArray(const float* d_arr, size_t pitch, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Compute pointer to the element using pitch
        const float* rowPtr = (const float*)((const char*)d_arr + row * pitch);
        printf("%6.1f ", rowPtr[col]);
        if (col == cols - 1) {
            printf("\n");
        }
    }
}

int main(void) {
    // Host array in row-major order
    float h_arr[M * N];
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            h_arr[r * N + c] = (float)(r * N + c);
        }
    }

    // Device pitched memory
    float* d_arr;
    size_t pitch; // in bytes

    // Allocate pitched memory on device
    CUDA_CHECK(cudaMallocPitch(&d_arr, &pitch, N * sizeof(float), M));
    // pitch is the number of bytes between successive rows.
    // It may be larger than N*sizeof(float) due to alignment requirements.
    // Using the pitch ensures that accesses are aligned and can improve performance.

    // Copy host array to device (2D copy)
    CUDA_CHECK(cudaMemcpy2D(d_arr, pitch,
                            h_arr, N * sizeof(float),
                            N * sizeof(float), M,
                            cudaMemcpyHostToDevice));

    // Verify by copying back to host and printing
    float h_arr_back[M * N];
    CUDA_CHECK(cudaMemcpy2D(h_arr_back, N * sizeof(float),
                            d_arr, pitch,
                            N * sizeof(float), M,
                            cudaMemcpyDeviceToHost));

    printf("Host array after copy back from device:\n");
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            printf("%6.1f ", h_arr_back[r * N + c]);
        }
        printf("\n");
    }

    // Optional: launch kernel to print device array directly
    dim3 blockSize(8, 8);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    printf("\nDevice array printed by kernel:\n");
    printDeviceArray<<<gridSize, blockSize>>>(d_arr, pitch, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
```