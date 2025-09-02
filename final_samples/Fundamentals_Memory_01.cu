```cpp
/* 
Aim of the program: Write a program that allocates an array of 256 integers on the host and initializes it with values from 0 to 255. Then, allocate memory for 256 integers on the GPU device using `cudaMalloc`. Finally, copy the host array to the device array using `cudaMemcpy` with the `cudaMemcpyHostToDevice` kind.

Thinking:
1. Include the necessary headers: <cstdio> for printf, <cuda_runtime.h> for CUDA API.
2. Define a macro `CUDA_CHECK` to wrap CUDA calls and report errors.
3. In main:
   a. Allocate an array of 256 integers on the host (`int *h_array = new int[256];`).
   b. Initialize it with values 0..255 using a loop.
   c. Allocate memory on the device (`int *d_array;` with `cudaMalloc((void**)&d_array, 256 * sizeof(int));`).
   d. Copy the data from host to device using `cudaMemcpy(d_array, h_array, 256 * sizeof(int), cudaMemcpyHostToDevice);`.
   e. (Optional) Verify by copying back to a second host array and printing a few values to confirm.
   f. Clean up: delete host array, cudaFree device memory.
4. Return 0.
5. Add error handling after each CUDA call using `CUDA_CHECK`.
6. Ensure the code compiles as a .cu file and uses standard CUDA runtime API.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main() {
    const int N = 256;
    int *h_array = new int[N];
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with values 0..255
    for (int i = 0; i < N; ++i) {
        h_array[i] = i;
    }

    // Allocate device memory
    int *d_array;
    CUDA_CHECK(cudaMalloc((void**)&d_array, N * sizeof(int)));

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice));

    // (Optional) Verify by copying back to host
    int *h_copy = new int[N];
    CUDA_CHECK(cudaMemcpy(h_copy, d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first 10 values to confirm
    printf("First 10 values after copy back from device:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_copy[i]);
    }
    printf("\n");

    // Clean up
    delete[] h_array;
    delete[] h_copy;
    CUDA_CHECK(cudaFree(d_array));

    return EXIT_SUCCESS;
}
```