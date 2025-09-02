/*
Use `curandGenerateNormal` to generate normally distributed (Gaussian) random numbers.
To achieve this I will create a simple CUDA program that demonstrates how to use the cuRAND
library's `curandGenerateNormal` function. The program will:
1. Initialize a cuRAND generator on the device.
2. Set a seed for reproducibility.
3. Allocate memory on both host and device for an array of floats.
4. Generate `N` normally distributed random numbers (mean = 0.0, stddev = 1.0)
   on the device using `curandGenerateNormal`.
5. Copy the generated numbers back to the host.
6. Print the first few numbers to verify the output.
7. Clean up allocated memory and destroy the generator.
I will include basic error checking for CUDA and cuRAND calls to make the program robust.
The code is self‑contained and can be compiled with `nvcc`:
    nvcc -o gaussian_random gaussian_random.cu
Then executed with `./gaussian_random`.
*/

// Include CUDA runtime and cuRAND headers
#include <cstdio>
#include <curand.h>
#include <cuda_runtime.h>

// Error checking macro for CUDA runtime API
#define CUDA_CALL(call)                                              \
    do {                                                             \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// Error checking macro for cuRAND API
#define CURAND_CALL(call)                                            \
    do {                                                             \
        curandStatus_t err = (call);                                 \
        if (err != CURAND_STATUS_SUCCESS) {                          \
            fprintf(stderr, "cuRAND error at %s:%d: %d\n",           \
                    __FILE__, __LINE__, err);                       \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

int main(void) {
    const int N = 10;            // Number of random numbers to generate
    const float mean = 0.0f;     // Mean of the Gaussian distribution
    const float stddev = 1.0f;   // Standard deviation

    // Host memory
    float h_random[N];

    // Device memory
    float *d_random = nullptr;
    CUDA_CALL(cudaMalloc((void**)&d_random, N * sizeof(float)));

    // Create cuRAND generator
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // Seed the generator (using time or a fixed seed for reproducibility)
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    // Generate normally distributed random numbers on the device
    CURAND_CALL(curandGenerateNormal(gen, d_random, N, mean, stddev));

    // Copy results back to host
    CUDA_CALL(cudaMemcpy(h_random, d_random, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Destroy cuRAND generator
    CURAND_CALL(curandDestroyGenerator(gen));

    // Free device memory
    CUDA_CALL(cudaFree(d_random));

    // Print the generated numbers
    printf("Generated %d Gaussian random numbers (mean=%.2f, stddev=%.2f):\n",
           N, mean, stddev);
    for (int i = 0; i < N; ++i) {
        printf("%f\n", h_random[i]);
    }

    return 0;
}
