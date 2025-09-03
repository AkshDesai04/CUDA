```c
/* 
Launch with a 4x4 grid of 8x8 blocks. Each thread should write its global flattened ID to a 2D output matrix.
Thinking:
- We want to launch a kernel with grid dimensions (4,4) and block dimensions (8,8).
- Each thread will compute its global linear ID based on its block and thread indices.
- The output will be a 2D matrix of size (gridDim.x * blockDim.x) by (gridDim.y * blockDim.y),
  i.e., 32x32 in this case.
- We'll allocate a 1D array on the device of size 32*32, interpret it as 2D by row-major order.
- Each thread writes its computed global ID into the array at the position corresponding to its
  global (x, y) coordinate.
- After kernel execution, copy the array back to the host and print it in a 32x32 grid format.
- Include basic error checking for CUDA API calls.
- The kernel will be very simple: compute global linear index, compute 2D global indices,
  write to output array.
*/

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                           \
    do {                                                            \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            std::cerr << "CUDA error in " << __FILE__ << ":"       \
                      << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                    \
        }                                                           \
    } while (0)

__global__ void write_global_ids(int *output, int width, int height) {
    // Compute global 2D coordinates
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx < width && gy < height) {
        // Compute linear index in row-major order
        int idx = gy * width + gx;
        // Global flattened ID can be simply the linear index
        output[idx] = idx;
    }
}

int main() {
    const int blockDimX = 8;
    const int blockDimY = 8;
    const int gridDimX = 4;
    const int gridDimY = 4;

    const int width = gridDimX * blockDimX;   // 32
    const int height = gridDimY * blockDimY;  // 32
    const size_t totalElements = static_cast<size_t>(width) * height;

    int *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_output, totalElements * sizeof(int)));

    dim3 block(blockDimX, blockDimY);
    dim3 grid(gridDimX, gridDimY);

    write_global_ids<<<grid, block>>>(d_output, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back to host
    int *h_output = new int[totalElements];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, totalElements * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the matrix
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::cout << h_output[y * width + x];
            if (x < width - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
```