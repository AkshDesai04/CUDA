```cpp
/*
Aim of the program:
If you launch more threads than elements N, the grid-stride loop will have some threads do no work. Verify this is handled correctly.

Thinking:
The goal is to demonstrate that a CUDA kernel employing a grid‑stride loop can be launched with more threads than there are elements in the data array and that this situation is handled safely—no out‑of‑bounds writes occur and all elements are processed exactly once. 

To test this we:
1. Allocate an array of N integers on the device and initialize it to 0.
2. Launch a kernel with a thread count that exceeds N.  
   The kernel uses a classic grid‑stride pattern:
   ```
   for (int i = tid; i < N; i += gridSize) { ... }
   ```
   Each thread processes elements starting at its unique thread index and then jumps ahead by the total number of threads in the grid (`gridSize`).  
3. Inside the loop each element is set to a known value (e.g., 1). Threads that find no valid `i` simply exit the loop without touching memory.
4. After the kernel finishes, copy the array back to host memory and verify that every element has the expected value. If any element remains 0 or is corrupted, the test fails.
5. Additionally, we print diagnostic information: the number of threads launched, the size of the array, and a success/failure message.

This simple verification confirms that the grid‑stride technique correctly handles the case where the launch configuration creates more threads than data elements.

*/
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel: grid‑stride loop that sets each element to 1
__global__ void set_to_one(int *d_arr, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += gridSize) {
        d_arr[i] = 1;  // work: set element to 1
    }
}

int main(int argc, char *argv[]) {
    // Default array size
    const int N_default = 1 << 20; // 1M elements
    int N = N_default;
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid array size: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    // Choose launch configuration that creates more threads than N
    const int threadsPerBlock = 256;
    // Intentionally launch many blocks
    const int blocks = 512; // 512*256 = 131072 threads

    // Allocate device memory
    int *d_arr = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    // Initialize to zero
    CHECK_CUDA(cudaMemset(d_arr, 0, N * sizeof(int)));

    // Launch kernel
    set_to_one<<<blocks, threadsPerBlock>>>(d_arr, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back to host
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify all elements set to 1
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != 1) {
            fprintf(stderr, "Error: element %d has value %d (expected 1)\n", i, h_arr[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("SUCCESS: All %d elements correctly set to 1 with %d threads launched.\n",
               N, blocks * threadsPerBlock);
    } else {
        printf("FAILURE: Some elements were not set correctly.\n");
    }

    // Clean up
    free(h_arr);
    CHECK_CUDA(cudaFree(d_arr));
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```