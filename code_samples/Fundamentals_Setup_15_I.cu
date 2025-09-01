/*
 * Aim of the program:
 * Explain the difference between `cudaDeviceReset()` and `cudaDeviceSynchronize()` in comments.
 *
 * My thinking:
 * The user wants a CUDA program that contains an explanation of the difference between
 * `cudaDeviceReset()` and `cudaDeviceSynchronize()`.  According to the developer
 * instructions, we must put the entire explanation in a multiline comment at the
 * top of the .cu file.  The comment should include the exact prompt text and then
 * the full reasoning.  After that, we need to provide a complete, compilable
 * CUDA C source file.  The code itself can be simple – it should just illustrate
 * the two API calls so that a reader can see where they are used, but the focus
 * is on the comments that explain the difference.
 *
 * Key points to explain:
 * 1. `cudaDeviceSynchronize()` blocks the host until all preceding CUDA work
 *    (kernel launches, memory copies, etc.) has finished.  It does *not* reset
 *    the device, clear resources, or change the device state.
 * 2. `cudaDeviceReset()` destroys all CUDA resources (streams, events, memory)
 *    associated with the current device, resets the device to a clean state,
 *    and returns the device to the same state as if the program had just
 *    started.  It is typically called at program termination or when a user
 *    needs to start over on the same device.
 * 3. Both functions can return errors, so the code checks the return value.
 * 4. Calling `cudaDeviceReset()` after all work is finished is required when
 *    the program uses persistent contexts or wants to free memory for other
 *    applications.  It should not be called repeatedly during normal
 *    execution because it would tear down the current context.
 * 5. `cudaDeviceSynchronize()` is often used in profiling, debugging, or when
 *    the host needs to read back results from device memory.
 *
 * The following example program demonstrates both calls.  It launches a tiny
 * kernel, then uses `cudaDeviceSynchronize()` to wait for completion.  After
 * that it resets the device with `cudaDeviceReset()`.  The comments in the
 * code describe what each call does in this context.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that does nothing but can be used to create a workload.
__global__ void dummyKernel() {
    // No-op
}

int main(void) {
    // Launch the dummy kernel asynchronously.
    dummyKernel<<<1, 1>>>();
    // Check for errors in kernel launch.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /*
     * At this point the kernel has been queued on the GPU, but the host
     * thread continues executing.  To ensure that all previously queued
     * operations have finished before we proceed (e.g., before reading
     * results from device memory), we call cudaDeviceSynchronize().
     *
     * This function blocks the host until all preceding device work has
     * completed.  It does *not* reset the device, clear memory, or
     * destroy any CUDA resources.  It is purely a synchronization point.
     */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("All device work has finished.\n");

    /*
     * Now we want to clean up the device state before the program exits.
     * Calling cudaDeviceReset() will destroy all CUDA resources
     * (streams, events, memory allocations) associated with the current
     * device, and reset the device to a clean state.  This is useful in
     * long-running applications that may need to reinitialize the device
     * or in cases where the device context must be released to allow
     * other applications to use the GPU.  It should not be called during
     * normal operation; only when you intend to free all resources.
     */
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Device has been reset.\n");

    return 0;
}
