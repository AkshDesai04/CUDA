/*
Aim: Free the pitched memory using `cudaFree`.

Thinking:
- We need a complete .cu file that demonstrates allocation of pitched memory using cudaMallocPitch, optional initialization, and freeing it with cudaFree.
- Include error checking to ensure correct allocation and freeing.
- The code will allocate a 2D array of integers, use cudaMemset to initialize it to zero, then free it.
- No extra output or supporting files; just the .cu source code.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                           \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                    \
        }                                                           \
    } while (0)

int main(void) {
    size_t pitch;
    int width  = 256;   // number of columns
    int height = 256;   // number of rows
    int *d_arr = NULL;

    // Allocate pitched memory
    CHECK_CUDA(cudaMallocPitch((void **)&d_arr, &pitch,
                               width * sizeof(int), height));

    printf("Allocated pitched memory: pitch = %zu bytes\n", pitch);

    // Optional: initialize memory to zero
    CHECK_CUDA(cudaMemset(d_arr, 0, pitch * height));

    // (Optional) Here you could launch a kernel that uses d_arr

    // Free the pitched memory
    CHECK_CUDA(cudaFree(d_arr));
    printf("Pitched memory freed successfully.\n");

    return 0;
}
