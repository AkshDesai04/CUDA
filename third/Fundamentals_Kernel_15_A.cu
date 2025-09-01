/*
 * aim: Launch the hanging kernel and try to use nvidia-smi in another terminal. You should see the GPU utilization at 100%.
 *
 * thinking: To achieve 100% GPU utilization we need to run a kernel that never finishes. The simplest way is to launch a kernel
 * that contains an infinite loop. The host will then wait for the kernel to finish using cudaDeviceSynchronize, which will block
 * forever. While the program is running we can open another terminal and run nvidia-smi to see the utilization spike. The code below
 * defines a kernel `hang_kernel` that loops infinitely. In `main` we launch this kernel with a minimal grid and block size, print a
 * message, and then call cudaDeviceSynchronize. The host program will hang at this point, keeping the GPU busy.
 *
 * The code includes a small error-checking macro `CHECK_CUDA` to catch any launch failures. Since the kernel never returns, the
 * program never reaches `cudaDeviceReset`, but that is acceptable for this demonstration.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel that runs forever
__global__ void hang_kernel() {
    // Infinite loop to keep the GPU busy
    while (true) {
        // Do nothing
    }
}

int main(void) {
    // Launch the kernel with 1 block and 1 thread
    hang_kernel<<<1, 1>>>();
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    printf("Hanging kernel launched. GPU should now be at 100%% utilization.\n");
    printf("Run 'nvidia-smi' in another terminal to verify.\n");

    // Wait forever for the kernel to finish (which it never will)
    CHECK_CUDA(cudaDeviceSynchronize());

    // This point will never be reached
    return 0;
}
