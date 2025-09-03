```cuda
/* 
Aim: How are synchronous errors reported? (Directly via the return code).

Thinking:
- CUDA synchronous errors (e.g., memory allocation failures, bad configuration for kernel launch, invalid API arguments) are reported immediately by the return value of the CUDA runtime API call.
- Kernel launches are asynchronous with respect to the host, but a failure to launch (such as an invalid grid/block configuration) is reported synchronously by `cudaGetLastError()` right after the launch.
- To demonstrate this, we will:
  1. Attempt a device memory allocation with an absurdly large size to trigger an out‑of‑memory error.
  2. Launch a simple kernel with an intentionally bad grid size (0) to trigger an invalid configuration error.
  3. After each operation, we will check the return code or call `cudaGetLastError()` to capture the synchronous error.
- We will use an error‑checking macro to simplify handling and to print descriptive messages.
- The program compiles with `nvcc` and can be run to observe the printed error codes/messages.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Macro for checking CUDA API calls
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Simple kernel that writes to an array
__global__ void write_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

int main(void) {
    const size_t n = 256;
    int *d_ptr = nullptr;

    // 1. Attempt to allocate an absurdly large amount of memory
    size_t large_size = (size_t)1 << 60;  // 1 exabyte, definitely too big
    cudaError_t mem_err = cudaMalloc((void**)&d_ptr, large_size);
    if (mem_err != cudaSuccess) {
        fprintf(stderr, "Synchronous allocation error caught: %s\n",
                cudaGetErrorString(mem_err));
    } else {
        // If allocation somehow succeeded (unlikely), free it
        cudaFree(d_ptr);
    }

    // 2. Allocate a reasonable amount of memory for the kernel
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, n * sizeof(int)));

    // 3. Launch kernel with an invalid grid size (0) to trigger a launch error
    dim3 blockSize(32);
    dim3 gridSize(0);  // Invalid grid dimension
    write_kernel<<<gridSize, blockSize>>>(d_ptr, n);

    // CUDA launch errors are reported via cudaGetLastError()
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "Synchronous kernel launch error caught: %s\n",
                cudaGetErrorString(launch_err));
    } else {
        // If launch succeeded (unlikely with gridSize=0), synchronize to catch runtime errors
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 4. Clean up
    CUDA_CHECK(cudaFree(d_ptr));

    printf("Program completed.\n");
    return 0;
}
```