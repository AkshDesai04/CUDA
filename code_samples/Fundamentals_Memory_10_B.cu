/*
After modifying the 2D array on the device with a kernel, copy it back to the host using `cudaMemcpy2D` and verify the results.

Thinking:
- We'll allocate a 2D array on the host with dimensions MxN.
- Use cudaMallocPitch on the device to get a pitched allocation, which aligns rows to power-of-two multiples for performance.
- Copy the host array to the device using cudaMemcpy2D, providing source/target pitches.
- Launch a simple kernel that increments every element by 1.
- Copy the modified data back to the host with another cudaMemcpy2D.
- Verify the result by printing a few elements and checking against expected values.
- Include error checking macros for clarity and robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void incrementKernel(int* d_data, size_t pitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (x < width && y < height) {
        // pitch is in bytes, so cast to char* for offset calculation
        int* row = (int*)((char*)d_data + y * pitch);
        row[x] += 1;
    }
}

int main(void) {
    const int width  = 8;  // number of columns
    const int height = 6;  // number of rows
    const size_t elemSize = sizeof(int);

    // Host allocation with regular contiguous memory
    int h_data[height][width];
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            h_data[i][j] = i * width + j; // initialize with unique values

    // Device pitched allocation
    int* d_data = NULL;
    size_t pitch = 0;
    CHECK_CUDA(cudaMallocPitch((void**)&d_data, &pitch, width * elemSize, height));

    // Copy host to device
    CHECK_CUDA(cudaMemcpy2D(d_data, pitch,
                            h_data, width * elemSize,
                            width * elemSize, height,
                            cudaMemcpyHostToDevice));

    // Kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width  + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    incrementKernel<<<numBlocks, threadsPerBlock>>>(d_data, pitch, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy device back to host
    int h_result[height][width];
    CHECK_CUDA(cudaMemcpy2D(h_result, width * elemSize,
                            d_data, pitch,
                            width * elemSize, height,
                            cudaMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int expected = h_data[i][j] + 1;
            if (h_result[i][j] != expected) {
                printf("Mismatch at (%d,%d): expected %d, got %d\n",
                       i, j, expected, h_result[i][j]);
                ++errors;
            }
        }
    }

    if (errors == 0) {
        printf("All values verified correctly.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
