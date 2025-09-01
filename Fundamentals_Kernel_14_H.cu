/*
 * Query the device for its maxThreadsPerBlock and then deliberately launch a kernel with maxThreadsPerBlock + 1 threads to dynamically trigger the error.
 *
 * Thought process:
 * 1. Use cudaGetDeviceProperties to obtain the maxThreadsPerBlock value for the current device.
 * 2. Compute a thread count that is one greater than this maximum, guaranteeing an invalid launch configuration.
 * 3. Define a minimal kernel (empty body) because the intent is to trigger an error during launch, not during execution.
 * 4. Launch the kernel with a single block and the oversized number of threads.
 * 5. Immediately check for errors using cudaGetLastError() and, if necessary, cudaDeviceSynchronize() to surface any launch or execution errors.
 * 6. Print out the error string so that the failure is observable when the program runs.
 * 7. Wrap the error checking in a simple macro for readability.
 *
 * This program will compile with nvcc and, when run, will output a CUDA error indicating an invalid launch configuration.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple error-checking macro */
#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

/* Empty kernel to be launched with an invalid configuration */
__global__ void dummyKernel()
{
    // No operation; kernel body intentionally left empty.
}

int main(void)
{
    /* Get the properties of the current device */
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("Device %s: maxThreadsPerBlock = %d\n", prop.name, prop.maxThreadsPerBlock);

    /* Intentionally exceed the maximum number of threads per block */
    int invalidThreads = prop.maxThreadsPerBlock + 1;
    printf("Attempting to launch kernel with %d threads (max allowed is %d).\n",
           invalidThreads, prop.maxThreadsPerBlock);

    /* Launch the kernel with an invalid configuration */
    dummyKernel<<<1, invalidThreads>>>();

    /* Check for launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        /* Still attempt to synchronize to catch any potential execution errors */
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        }
        return EXIT_FAILURE;
    }

    /* If launch succeeded (unlikely), synchronize and report success */
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Kernel launched successfully (unexpected).\n");

    return EXIT_SUCCESS;
}
