/*
Add error checking to the `cudaMemset` call.

Thinking process:
1. The goal is to demonstrate how to incorporate error checking for a CUDA runtime API call, specifically `cudaMemset`. 
2. I decided to create a minimal yet complete CUDA program that allocates memory on the device, sets it with `cudaMemset`, copies it back to the host, prints the values, and finally frees the memory. 
3. To perform error checking consistently, I implemented a macro `CHECK_CUDA` that invokes the CUDA call, captures its return status, and prints a descriptive error message if the call fails. This pattern is idiomatic for CUDA error handling and keeps the code concise. 
4. The `cudaMemset` call is wrapped with `CHECK_CUDA`, ensuring that any failure (e.g., out‑of‑memory, invalid pointer, etc.) is immediately reported. 
5. The rest of the program also uses the macro for `cudaMalloc`, `cudaMemcpy`, and `cudaFree` to demonstrate consistent error handling across all runtime calls. 
6. The example sets each byte of an `int` array to the value `5`. Since `cudaMemset` operates byte‑wise, the resulting `int` values will be `0x05050505`. Printing them shows that the operation succeeded.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main() {
    const size_t N = 10;
    int *d_array = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_array, N * sizeof(int)));

    // Set all bytes of the array to 5, with error checking
    CHECK_CUDA(cudaMemset(d_array, 5, N * sizeof(int)));

    // Copy back to host for verification
    int h_array[N];
    CHECK_CUDA(cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the values
    printf("Device array contents after cudaMemset:\n");
    for (size_t i = 0; i < N; ++i) {
        printf("%08X ", static_cast<unsigned int>(h_array[i]));  // hex to see byte pattern
    }
    printf("\n");

    // Free device memory
    CHECK_CUDA(cudaFree(d_array));

    return 0;
}
