/*
Aim: After a DtoD copy, modify `d_a`. Does `d_b` change? (No, it's a deep copy).

Thinking: The program demonstrates that cudaMemcpyDeviceToDevice performs a deep copy. We allocate two device arrays d_a and d_b, copy data from host to d_a, then use cudaMemcpyDeviceToDevice to copy d_a to d_b. After that, we modify d_a by writing new data to it. Finally, we copy both d_a and d_b back to host memory and print their contents. If d_b remains unchanged, it confirms that the copy was deep. We use standard CUDA runtime API calls and basic error checking.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                      \
        }                                                             \
    } while (0)

int main(void) {
    const int N = 10;
    int h_original[N] = {0,1,2,3,4,5,6,7,8,9};
    int h_modified[N] = {100,100,100,100,100,100,100,100,100,100};

    int *d_a = NULL;
    int *d_b = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(int)));

    // Copy original data from host to d_a
    CUDA_CHECK(cudaMemcpy(d_a, h_original, N * sizeof(int), cudaMemcpyHostToDevice));

    // Deep copy from d_a to d_b (Device-to-Device)
    CUDA_CHECK(cudaMemcpy(d_b, d_a, N * sizeof(int), cudaMemcpyDeviceToDevice));

    // Modify d_a by copying new data from host
    CUDA_CHECK(cudaMemcpy(d_a, h_modified, N * sizeof(int), cudaMemcpyHostToDevice));

    // Copy back to host for verification
    int h_a_back[N] = {0};
    int h_b_back[N] = {0};

    CUDA_CHECK(cudaMemcpy(h_a_back, d_a, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b_back, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Contents of d_a after modification (host copy):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_a_back[i]);
    }
    printf("\n\n");

    printf("Contents of d_b (should remain original) (host copy):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_b_back[i]);
    }
    printf("\n\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
