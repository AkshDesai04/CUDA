```cuda
/*
Aim: On Linux, the OS may have a watchdog timer that kills the kernel after a few seconds, causing cudaDeviceSynchronize to return an error (cudaErrorLaunchTimeout). Try to catch and report this error.

Thinking:
- We create a kernel that performs a very long loop to exceed the watchdog timeout (typically 2–3 s).
- The kernel is launched, and then cudaDeviceSynchronize() is called.
- If the watchdog kills the kernel, cudaDeviceSynchronize() returns cudaErrorLaunchTimeout.
- We check the return code and print a descriptive message.
- A simple error checking macro is provided for other CUDA API calls.
- The program uses a volatile counter in the loop to prevent the compiler from optimizing the loop away.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple macro to check CUDA API return codes */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Kernel that runs for a long time to trigger the watchdog */
__global__ void longRunningKernel(void)
{
    volatile unsigned long long dummy = 0;
    /* Loop for a huge number of iterations; adjust the value to ensure
       the kernel runs longer than the watchdog timeout. */
    for (unsigned long long i = 0; i < (1ULL << 50); ++i) {
        dummy += i;
    }
    /* Use dummy to avoid compiler warning about unused variable */
    if (dummy == 0) printf("Never happens\n");
}

int main(void)
{
    /* Set device to use default GPU */
    CHECK_CUDA(cudaSetDevice(0));

    /* Launch kernel with 1 block of 1 thread */
    longRunningKernel<<<1, 1>>>();
    /* Check for launch errors */
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        return EXIT_FAILURE;
    }

    /* Synchronize and catch watchdog timeout */
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        if (syncErr == cudaErrorLaunchTimeout) {
            fprintf(stderr, "Error: Kernel launch timed out due to watchdog timer.\n");
        } else {
            fprintf(stderr, "Error during cudaDeviceSynchronize: %s\n", cudaGetErrorString(syncErr));
        }
        return EXIT_FAILURE;
    }

    printf("Kernel finished successfully.\n");

    /* Reset device to clean up */
    CHECK_CUDA(cudaDeviceReset());
    return EXIT_SUCCESS;
}
```