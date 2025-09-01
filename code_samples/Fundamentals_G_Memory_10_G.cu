/*
Free the pitched memory using `cudaFree`.

The aim of this program is to demonstrate how to correctly free memory that was allocated using cudaMallocPitch.
In CUDA, cudaMallocPitch is used to allocate 2D arrays with proper alignment for better coalesced memory access. The returned pointer is a linear address that represents the start of the pitched allocation. Even though the allocation is logically 2D, the deallocation must be done with cudaFree, not a special pitched-free function. This program will:

1. Allocate a 2D array on the device using cudaMallocPitch.
2. Copy data from a host 2D array to the device.
3. Launch a simple kernel that performs an element-wise operation (e.g., adds 1.0f to each element).
4. Copy the results back to the host.
5. Free the pitched memory with cudaFree.
6. Perform error checking after each CUDA API call to ensure robustness.

The code includes a small helper macro CUDA_CHECK for error handling, a simple kernel, and a main function that follows the sequence above. By the end of execution, the pitched memory will be correctly freed, demonstrating the proper usage of cudaFree for pitched allocations.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Simple kernel that adds 1.0f to each element of a 2D array
__global__ void add_one(float *d_data, size_t pitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (x < width && y < height) {
        // pitch is in bytes, convert to float*
        float *row = (float *)((char *)d_data + y * pitch);
        row[x] += 1.0f;
    }
}

int main(void) {
    const int width  = 5; // number of columns
    const int height = 4; // number of rows

    // Host allocation and initialization
    float h_host[height][width];
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_host[y][x] = static_cast<float>(y * width + x);

    // Device allocation with pitch
    float *d_pitchPtr = nullptr;
    size_t pitch = 0; // pitch in bytes
    CUDA_CHECK(cudaMallocPitch((void **)&d_pitchPtr, &pitch, width * sizeof(float), height));

    // Copy host 2D array to device pitched memory
    CUDA_CHECK(cudaMemcpy2D(d_pitchPtr, pitch,
                            h_host, width * sizeof(float),
                            width * sizeof(float), height,
                            cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    add_one<<<gridDim, blockDim>>>(d_pitchPtr, pitch, width, height);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy2D(h_host, width * sizeof(float),
                            d_pitchPtr, pitch,
                            width * sizeof(float), height,
                            cudaMemcpyDeviceToHost));

    // Print results
    printf("Result after adding 1.0f:\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%6.2f ", h_host[y][x]);
        }
        printf("\n");
    }

    // Free the pitched memory
    CUDA_CHECK(cudaFree(d_pitchPtr));

    // Optional: Reset device (not strictly necessary)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
