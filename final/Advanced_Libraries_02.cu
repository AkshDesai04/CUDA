```cuda
/*
Aim of the program: Use the cuRAND library to generate a large array of random numbers directly on the GPU. Create a generator, set its seed, and call curandGenerateUniform to fill a device array with floating-point numbers between 0.0 and 1.0.

Thinking:
1. We need to include CUDA runtime, cuRAND headers and iostream for output.
2. Define a macro CURAND_CHECK to wrap cuRAND API calls for error checking.
3. In main, choose a large array size, e.g., 10 million floats.
4. Allocate device memory with cudaMalloc.
5. Create a cuRAND generator of type CURAND_RNG_PSEUDO_DEFAULT using curandCreateGenerator.
6. Set the seed with curandSetPseudoRandomGeneratorSeed.
7. Call curandGenerateUniform to fill the device array.
8. Copy a small portion back to host to verify randomness.
9. Clean up: free device memory, destroy generator, check for CUDA errors.
10. Print a few numbers to confirm they're in [0,1].
*/

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <vector>

// Macro for checking cuRAND API calls
#define CURAND_CHECK(call)                                                    \
    do {                                                                       \
        curandStatus_t err = call;                                            \
        if (err != CURAND_STATUS_SUCCESS) {                                   \
            std::cerr << "cuRAND error at " << __FILE__ << ":" << __LINE__    \
                      << " code=" << err << std::endl;                        \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

// Macro for checking CUDA Runtime API calls
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " : " << cudaGetErrorString(err)  \
                      << std::endl;                                          \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

int main() {
    const size_t N = 10'000'000; // 10 million random numbers
    float *dev_random = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&dev_random, N * sizeof(float)));

    // Create cuRAND generator
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // Set seed
    unsigned long long seed = 1234ULL;
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    // Generate uniform random numbers in [0,1)
    CURAND_CHECK(curandGenerateUniform(gen, dev_random, N));

    // Copy a few values back to host for verification
    std::vector<float> host_random(10);
    CUDA_CHECK(cudaMemcpy(host_random.data(), dev_random, 10 * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "First 10 random numbers:" << std::endl;
    for (size_t i = 0; i < host_random.size(); ++i) {
        std::cout << host_random[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(dev_random));
    CURAND_CHECK(curandDestroyGenerator(gen));

    return 0;
}
```