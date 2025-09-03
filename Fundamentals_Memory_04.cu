/*
Write a function that takes an integer `N`, allocates an array of `N` floats on the device, and initializes all its elements to zero using `cudaMemset`. The function should return the device pointer.

My thinking:
- We need a simple CUDA C++ (.cu) file that can compile with nvcc.
- The function signature will be:
      float* allocateAndZero(int N);
  It will allocate device memory using cudaMalloc and then initialize it with cudaMemset.
- cudaMemset expects the number of bytes, so we pass N * sizeof(float).
- It's good practice to check for errors after cudaMalloc and cudaMemset, but the prompt doesn't explicitly require error handling. Still, adding basic error checks makes the code robust and easier to debug.
- We'll include <cuda_runtime.h> for CUDA runtime API.
- For demonstration, a simple `main` will call the function with a small N, print a success message, and free the memory.
- The code should be self‑contained and compile as a single .cu file.
- No additional supporting files or text are output, only the .cu code itself.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/* Helper macro to check CUDA error codes */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/*
 * Allocates an array of N floats on the device and initializes it to zero.
 *
 * @param N Number of float elements to allocate.
 * @return Pointer to the allocated device memory.
 *         Caller is responsible for freeing with cudaFree().
 */
float* allocateAndZero(int N) {
    float* d_ptr = nullptr;
    // Allocate memory on the device
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_ptr), N * sizeof(float)));
    // Initialize memory to zero
    CHECK_CUDA(cudaMemset(d_ptr, 0, N * sizeof(float)));
    return d_ptr;
}

/* Simple test harness */
int main(void) {
    const int N = 10;
    float* d_arr = allocateAndZero(N);

    // (Optional) Verify that the memory was set to zero by copying back
    float h_arr[N];
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool allZero = true;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != 0.0f) {
            allZero = false;
            break;
        }
    }

    if (allZero) {
        printf("All %d elements are zero as expected.\n", N);
    } else {
        printf("Some elements were not zero.\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_arr));
    return 0;
}
