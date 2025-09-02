```cpp
/*
Aim: Use the cuRAND library to generate a large array of random numbers directly on the GPU. Create a generator, set its seed, and call `curandGenerateUniform` to fill a device array with floating-point numbers between 0.0 and 1.0.

Thinking:
1. Include the necessary headers: <stdio.h>, <stdlib.h>, and <curand.h>.
2. Define a macro for error checking of CUDA and cuRAND calls. This macro will print an error message and exit if an error occurs.
3. In `main`, set up the parameters:
   - Number of random numbers to generate (e.g., 1,000,000).
   - Device array allocation via `cudaMalloc`.
4. Create a cuRAND generator of type `CURAND_RNG_PSEUDO_DEFAULT` with `curandCreateGenerator`.
5. Set the seed using `curandSetPseudoRandomGeneratorSeed`. A fixed seed (e.g., 1234ULL) ensures reproducibility.
6. Generate uniform random numbers in the range [0.0, 1.0] using `curandGenerateUniform`, which fills the device array.
7. Copy the random numbers back to host memory with `cudaMemcpy`.
8. Print a small subset (e.g., first 10 values) to verify that numbers lie in the expected range.
9. Clean up: destroy the generator, free device and host memory.
10. Return 0 on success.

This straightforward program demonstrates how to perform GPU-based random number generation with cuRAND and transfer the results back to the CPU for further use or verification.
*/

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cuda_runtime.h>

// Macro for checking CUDA runtime API errors
#define CUDA_CHECK(err) do {                                 \
    cudaError_t e = (err);                                   \
    if (e != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                cudaGetErrorString(e), e, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                 \
    }                                                        \
} while (0)

// Macro for checking cuRAND API errors
#define CURAND_CHECK(err) do {                               \
    curandStatus_t e = (err);                                \
    if (e != CURAND_STATUS_SUCCESS) {                       \
        fprintf(stderr, "cuRAND Error: %d at %s:%d\n",      \
                e, __FILE__, __LINE__);                     \
        exit(EXIT_FAILURE);                                 \
    }                                                        \
} while (0)

int main(void)
{
    const size_t N = 1000000;      // Number of random numbers
    const unsigned long long seed = 1234ULL; // Seed for reproducibility

    float *d_rand = NULL;         // Device array
    float *h_rand = NULL;         // Host array

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_rand, N * sizeof(float)));

    // Allocate host memory
    h_rand = (float*)malloc(N * sizeof(float));
    if (h_rand == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Create cuRAND generator
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // Set the seed
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    // Generate uniform random numbers on the device
    CURAND_CHECK(curandGenerateUniform(gen, d_rand, N));

    // Copy the results back to host
    CUDA_CHECK(cudaMemcpy(h_rand, d_rand, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print first 10 random numbers
    printf("First 10 random numbers (uniform [0,1]):\n");
    for (int i = 0; i < 10; ++i) {
        printf("%0.6f\n", h_rand[i]);
    }

    // Clean up
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUDA_CHECK(cudaFree(d_rand));
    free(h_rand);

    return 0;
}
```