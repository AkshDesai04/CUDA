```cpp
/*
Use `curandGenerateNormal` to generate normally distributed (Gaussian) random numbers.

Thinking:
- The program demonstrates how to use the CURAND library to generate normally
  distributed random numbers on the GPU.  The main steps are:
  1. Create a CURAND generator (curandCreateGenerator).
  2. Set a seed for reproducibility (curandSetPseudoRandomGeneratorSeed).
  3. Allocate device memory for the random numbers.
  4. Call curandGenerateNormal to fill the device array with Gaussian numbers.
     The function takes the generator, a pointer to device memory, the number of
     random numbers, the mean, and the standard deviation.
  5. Copy the numbers back to host memory with cudaMemcpy.
  6. Print a few values to verify that the output looks Gaussian.
  7. Clean up: free device memory, destroy the generator, and check for
     errors at each step.
- We also provide a simple macro to check CUDA and CURAND API return codes
  and abort if any call fails.
- This example uses a small array of 10 numbers for clarity, but the code
  can be scaled to larger sizes without modification.
*/

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cuda_runtime.h>

// Macro to check CUDA runtime API calls
#define CUDA_CALL(call)                                                         \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Macro to check CURAND API calls
#define CURAND_CALL(call)                                                       \
    do {                                                                        \
        curandStatus_t err = (call);                                            \
        if (err != CURAND_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "CURAND error at %s:%d: %d\n",                      \
                    __FILE__, __LINE__, err);                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(void) {
    const int N = 10;          // number of random numbers to generate
    const double mean = 0.0;   // mean of Gaussian distribution
    const double stddev = 1.0; // standard deviation of Gaussian distribution

    // Allocate host memory to receive the random numbers
    double *h_random = (double *)malloc(N * sizeof(double));
    if (h_random == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory for the random numbers
    double *d_random = NULL;
    CUDA_CALL(cudaMalloc((void **)&d_random, N * sizeof(double)));

    // Create a CURAND generator
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // Set a seed for reproducibility
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    // Generate N normally distributed random numbers on the device
    CURAND_CALL(curandGenerateNormalDouble(gen, d_random, N, mean, stddev));

    // Copy the random numbers back to host
    CUDA_CALL(cudaMemcpy(h_random, d_random, N * sizeof(double), cudaMemcpyDeviceToHost));

    // Print the generated random numbers
    printf("Generated %d Gaussian random numbers (mean=%.2f, stddev=%.2f):\n", N, mean, stddev);
    for (int i = 0; i < N; ++i) {
        printf("x[%d] = %f\n", i, h_random[i]);
    }

    // Clean up
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(d_random));
    free(h_random);

    return 0;
}
```