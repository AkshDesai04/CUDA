/*
Compare passing a struct with 3 floats vs. passing 3 separate float arguments to a kernel. Is there a performance difference? (Likely negligible, but good to think about).

In CUDA, arguments to a kernel are copied from host to device via the kernel launch configuration. The size and layout of the arguments determine how many bytes are transferred for each thread. A struct containing three floats occupies 12 bytes, the same as three separate float arguments (12 bytes total). However, the compiler may lay out the arguments differently: for separate floats, each argument is placed in a separate register or stack slot, while a struct might be passed as a single contiguous chunk. On the host side, the compiler will pack the arguments into the kernel launch payload; on the device side, the arguments are loaded into registers or local memory for each thread.

Because the total data size is identical and the GPU instruction pipeline handles register allocation efficiently, the performance difference between passing a struct and separate floats is generally negligible. The only potential differences are:
1. **Register allocation** – The compiler might allocate fewer registers for a struct if it sees it as a single entity, but in practice this is not a significant change for simple types.
2. **Instruction decoding** – A struct may require an extra load/store instruction to unpack into registers, but this cost is usually offset by the efficient use of registers.
3. **Memory alignment** – A struct may introduce padding, but with three floats alignment is already a multiple of 4, so no extra padding occurs.

Thus, the performance difference is typically within a few nanoseconds or less per thread, and for most applications it can be safely ignored. This program demonstrates two kernels, one that receives a struct and one that receives three separate floats, measures their execution times, and prints the results to confirm the negligible difference.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define N 1048576        // 1 million threads
#define BLOCK_SIZE 256

struct Vec3 {
    float x;
    float y;
    float z;
};

__global__ void kernelStruct(const Vec3 v, float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = v.x + v.y + v.z;
    }
}

__global__ void kernelSeparate(float x, float y, float z, float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = x + y + z;
    }
}

int main() {
    // Allocate output array on device
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    // Prepare struct and separate arguments
    Vec3 h_v = {1.0f, 2.0f, 3.0f};
    float h_x = 1.0f, h_y = 2.0f, h_z = 3.0f;

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Grid dimensions
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Timing for kernelStruct
    cudaEventRecord(start);
    kernelStruct<<<gridSize, BLOCK_SIZE>>>(h_v, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msStruct = 0.0f;
    cudaEventElapsedTime(&msStruct, start, stop);

    // Timing for kernelSeparate
    cudaEventRecord(start);
    kernelSeparate<<<gridSize, BLOCK_SIZE>>>(h_x, h_y, h_z, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msSeparate = 0.0f;
    cudaEventElapsedTime(&msSeparate, start, stop);

    // Print results
    printf("kernelStruct time:   %f ms\n", msStruct);
    printf("kernelSeparate time: %f ms\n", msSeparate);
    printf("Difference:          %f ms\n", msSeparate - msStruct);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);

    return 0;
}
