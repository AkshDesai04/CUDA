```cpp
/*
Generate random integers instead of floats.
The aim of this program is to replace the generation of random floating-point numbers
with the generation of random integer values.  The original version of the
CUDA program used `curandGenerateUniform` to fill a device array with floats.
In this rewritten version, the program uses `curandGenerate`, which produces
random 32‑bit unsigned integers.  The program demonstrates how to set up a
CUDA random number generator, allocate memory on the device, generate an
array of random integers, copy the result back to the host, and display a
subset of the values.

Key points considered during the rewrite:

1. **Random number generator choice** – `curandGenerate` returns
   unsigned integers directly, so no conversion from floats is needed.
2. **Memory allocation** – both device and host arrays are allocated
   with the same size in terms of `unsigned int`.
3. **Error handling** – a helper function `checkCuda` and
   `checkCurand` are provided to wrap CUDA and CURAND API calls.
4. **Seed initialization** – the seed is set to the current time to
   ensure a different sequence on each run.
5. **Output** – the program prints the first 10 random integers
   to illustrate the result.

The resulting code is a complete, self‑contained CUDA program that can
be compiled with `nvcc`.  It demonstrates how to replace float
generation with integer generation while preserving the overall
structure of the original program.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>

/* Helper macro to check CUDA runtime API calls */
#define checkCuda(call)                                                    \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",  \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),       \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

/* Helper macro to check CURAND API calls */
#define checkCurand(call)                                                  \
    do {                                                                   \
        curandStatus_t err = call;                                         \
        if (err != CURAND_STATUS_SUCCESS) {                               \
            fprintf(stderr, "CURAND error at %s:%d code=%d \"%s\" \n",     \
                    __FILE__, __LINE__, err, getCurandErrorString(err));  \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

/* Translate CURAND error code to human readable string */
const char* getCurandErrorString(curandStatus_t err) {
    switch (err) {
        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
        default:
            return "Unknown CURAND error";
    }
}

int main(void) {
    /* Number of random integers to generate */
    const size_t N = 1000000;          /* 1 million integers */

    /* Device memory for random integers */
    unsigned int *d_randInts = NULL;
    checkCuda(cudaMalloc((void**)&d_randInts, N * sizeof(unsigned int)));

    /* Create CURAND generator */
    curandGenerator_t gen;
    checkCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    /* Set seed */
    unsigned long long seed = (unsigned long long)time(NULL);
    checkCurand(curandSetPseudoRandomGeneratorSeed(gen, seed));

    /* Generate N random unsigned integers on the device */
    checkCurand(curandGenerate(gen, d_randInts, N));

    /* Allocate host memory to copy back results */
    unsigned int *h_randInts = (unsigned int*)malloc(N * sizeof(unsigned int));
    if (h_randInts == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    /* Copy results from device to host */
    checkCuda(cudaMemcpy(h_randInts, d_randInts, N * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));

    /* Print the first 10 random integers */
    printf("First 10 random integers:\n");
    for (int i = 0; i < 10 && i < (int)N; ++i) {
        printf("%u\n", h_randInts[i]);
    }

    /* Clean up */
    free(h_randInts);
    checkCuda(cudaFree(d_randInts));
    checkCurand(curandDestroyGenerator(gen));

    printf("Random integer generation completed successfully.\n");
    return 0;
}
```