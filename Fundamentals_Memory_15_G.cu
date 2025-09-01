```cuda
/*
Is `__device__` memory typically fast or slow to access compared to global memory from `cudaMalloc`? (It is global memory, so the performance is the same).

To answer this question in practice we compare the execution time of two kernels that perform the same arithmetic operation, but one writes to a `__device__` array and the other writes to a global array allocated with `cudaMalloc`.  
The `__device__` array is defined at file scope and resides in global memory on the device, so it should have the same access latency and bandwidth characteristics as a regular global array.  

The program does the following:
1. Declares a `__device__` array `dev_arr` of size `SIZE`.  
2. Allocates a global array `d_global` via `cudaMalloc`.  
3. Launches two kernels:  
   - `write_to_device_array` writes each index value into `dev_arr`.  
   - `write_to_global_array` writes the same index values into `d_global`.  
4. Times each kernel using CUDA events and prints the elapsed time.  
5. Copies both arrays back to the host (via `cudaMemcpyFromSymbol` for the device array) and verifies that they contain the correct data.  

Because the `__device__` array is stored in the same global memory space as a `cudaMalloc` buffer, the measured times are expected to be very close, illustrating that `__device__` memory access performance is equivalent to that of regular global memory.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define SIZE (1 << 20) // 1M elements
#define BLOCK_SIZE 256

// Device global array declared with __device__
__device__ int dev_arr[SIZE];

// Kernel that writes index values into the __device__ array
__global__ void write_to_device_array(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dev_arr[idx] = idx;
    }
}

// Kernel that writes index values into a global array allocated by cudaMalloc
__global__ void write_to_global_array(int *d_global, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_global[idx] = idx;
    }
}

// Utility macro for error checking
#define CHECK_CUDA(call)                                          \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }

int main() {
    // Allocate host arrays for verification
    int *h_global = (int*)malloc(SIZE * sizeof(int));
    int *h_dev = (int*)malloc(SIZE * sizeof(int));
    if (!h_global || !h_dev) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Allocate device global array
    int *d_global;
    CHECK_CUDA(cudaMalloc((void**)&d_global, SIZE * sizeof(int)));

    // Configure kernel launch parameters
    int numBlocks = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // -----------------------
    // Time __device__ array kernel
    // -----------------------
    CHECK_CUDA(cudaEventRecord(start));
    write_to_device_array<<<numBlocks, BLOCK_SIZE>>>(SIZE);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float time_device;
    CHECK_CUDA(cudaEventElapsedTime(&time_device, start, stop));

    // Copy __device__ array back to host
    CHECK_CUDA(cudaMemcpyFromSymbol(h_dev, dev_arr, SIZE * sizeof(int), 0, cudaMemcpyDeviceToHost));

    // -----------------------
    // Time global array kernel
    // -----------------------
    CHECK_CUDA(cudaEventRecord(start));
    write_to_global_array<<<numBlocks, BLOCK_SIZE>>>(d_global, SIZE);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float time_global;
    CHECK_CUDA(cudaEventElapsedTime(&time_global, start, stop));

    // Copy global array back to host
    CHECK_CUDA(cudaMemcpy(h_global, d_global, SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Print timing results
    printf("Kernel writing to __device__ array:   %.3f ms\n", time_device);
    printf("Kernel writing to global (cudaMalloc) array: %.3f ms\n", time_global);

    // Verify correctness
    int mismatches = 0;
    for (int i = 0; i < SIZE; ++i) {
        if (h_dev[i] != i || h_global[i] != i) {
            ++mismatches;
            if (mismatches <= 10) {
                printf("Mismatch at index %d: dev=%d, global=%d\n", i, h_dev[i], h_global[i]);
            }
        }
    }
    if (mismatches == 0) {
        printf("Verification passed: both arrays contain correct values.\n");
    } else {
        printf("Verification failed: %d mismatches found.\n", mismatches);
    }

    // Clean up
    free(h_global);
    free(h_dev);
    CHECK_CUDA(cudaFree(d_global));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```