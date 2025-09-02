/*
Aim: Generate random integers instead of floats.

Thinking:
- The original problem requested generating random integers rather than floats.
- We'll use the CUDA CURAND library's host API to generate random unsigned integers directly on the GPU.
- The function `curandGenerate` outputs 32‑bit unsigned integers, which we can treat as random integers.
- We'll allocate device memory, generate N random integers, copy them back to the host, and print them.
- We'll include error checking for CURAND and CUDA calls.
- The program is self‑contained, compiles with `nvcc`, and prints the random integers to stdout.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 10  // Number of random integers to generate

int main(void) {
    curandGenerator_t gen;
    curandStatus_t status;

    // Create a pseudo‑random generator
    status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "Error: curandCreateGenerator failed (status %d)\n", status);
        return EXIT_FAILURE;
    }

    // Seed the generator
    status = curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    if (status != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "Error: curandSetPseudoRandomGeneratorSeed failed (status %d)\n", status);
        curandDestroyGenerator(gen);
        return EXIT_FAILURE;
    }

    // Allocate device memory for N unsigned ints
    unsigned int *d_random;
    cudaError_t cudaStatus = cudaMalloc((void **)&d_random, N * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error: cudaMalloc failed (status %d)\n", cudaStatus);
        curandDestroyGenerator(gen);
        return EXIT_FAILURE;
    }

    // Generate N random unsigned integers on the device
    status = curandGenerate(gen, d_random, N);
    if (status != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "Error: curandGenerate failed (status %d)\n", status);
        cudaFree(d_random);
        curandDestroyGenerator(gen);
        return EXIT_FAILURE;
    }

    // Copy the random integers back to host
    unsigned int *h_random = (unsigned int *)malloc(N * sizeof(unsigned int));
    if (!h_random) {
        fprintf(stderr, "Error: host allocation failed\n");
        cudaFree(d_random);
        curandDestroyGenerator(gen);
        return EXIT_FAILURE;
    }

    cudaStatus = cudaMemcpy(h_random, d_random, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error: cudaMemcpy failed (status %d)\n", cudaStatus);
        free(h_random);
        cudaFree(d_random);
        curandDestroyGenerator(gen);
        return EXIT_FAILURE;
    }

    // Print the random integers
    printf("Random integers:\n");
    for (int i = 0; i < N; ++i) {
        printf("%u ", h_random[i]);
    }
    printf("\n");

    // Clean up
    free(h_random);
    cudaFree(d_random);
    curandDestroyGenerator(gen);

    return EXIT_SUCCESS;
}
