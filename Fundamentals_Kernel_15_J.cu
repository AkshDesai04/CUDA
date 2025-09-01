```cuda
/*
 * Launch the infinite loop kernel on a non-default stream. Does it still hang the application when you call `cudaDeviceSynchronize()`?
 *
 * ---- Thinking ----
 * The aim of this program is to answer the question: if we launch a kernel that never terminates (an infinite loop)
 * on a non-default CUDA stream, does a subsequent call to cudaDeviceSynchronize() still block the host program
 * until the kernel finishes? In CUDA, cudaDeviceSynchronize waits for all kernels in *all* streams to complete.
 * Therefore, even if the infinite kernel is launched on a user-created stream, the synchronization call will
 * still block indefinitely because the infinite kernel never returns.
 *
 * To demonstrate this, we:
 * 1. Create a CUDA stream using cudaStreamCreate().
 * 2. Launch a simple kernel that loops forever on that stream.
 * 3. Print a message after launching the kernel.
 * 4. Call cudaDeviceSynchronize() which should block.
 * 5. Print a message after synchronization (this line should never be reached).
 *
 * The program includes basic error checking for CUDA API calls. Because the infinite loop kernel never terminates,
 * the program will hang at the cudaDeviceSynchronize() call, which is the expected behaviour.
 *
 * This example is intentionally minimal and is meant for demonstration only; in real applications you would never
 * launch an infinite loop kernel.  Pressing Ctrl-C will terminate the program if you run it on a system that
 * allows interrupting the process.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Simple error checking macro
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel that never terminates
__global__ void infiniteLoopKernel(void) {
    while (true) { /* spin forever */ }
}

int main(void) {
    cudaStream_t stream;

    // Create a non-default stream
    CHECK_CUDA(cudaStreamCreate(&stream));

    printf("Launching infinite loop kernel on non-default stream...\n");
    infiniteLoopKernel<<<1, 1, 0, stream>>>();
    // Check for launch errors
    CHECK_CUDA(cudaGetLastError());

    printf("Kernel launched. Calling cudaDeviceSynchronize()...\n");
    // This call should block indefinitely because the kernel never completes
    CHECK_CUDA(cudaDeviceSynchronize());

    // This line will never be executed
    printf("cudaDeviceSynchronize() returned. This should not happen!\n");

    // Cleanup (unreachable, but good practice)
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
```