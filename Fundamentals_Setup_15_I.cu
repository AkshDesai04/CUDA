/*
Explain the difference between `cudaDeviceReset()` and `cudaDeviceSynchronize()` in comments.

Thought process:
1. The prompt asks to explain the difference between the two CUDA runtime API functions, but it also wants the code to contain this explanation as comments.
2. According to the developer instruction, we must start the file with a multiline comment that includes the exact prompt text and then add our full reasoning there.
3. After the comment, the actual .cu code can be minimal but must compile and demonstrate (even if trivially) usage of both functions.
4. The code will:
   - Declare a trivial kernel to give some GPU work to synchronize on.
   - Call `cudaDeviceSynchronize()` to wait for all previously launched work to finish.
   - Call `cudaDeviceReset()` to clean up the device, freeing resources and returning the device to its initial state.
   - Print a message indicating both operations have completed.
5. All explanatory text will be in comments so the source file fulfills the requirement of explaining the difference inside the code.

The final code below follows those guidelines.
*/

#include <cuda_runtime.h>
#include <stdio.h>

/* Simple kernel that does nothing.
   It's only here so we have a GPU operation to synchronize on. */
__global__ void dummyKernel() {
    // No-op
}

int main(void) {
    // Launch dummy kernel on the default stream
    dummyKernel<<<1, 1>>>();

    /* 
     * cudaDeviceSynchronize() waits for all preceding GPU work
     * (including the dummy kernel launch) to complete. It does NOT
     * reset or clean up the device; it merely blocks the host until
     * the device has finished executing all queued operations.
     */
    cudaError_t errSync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(errSync));
        return 1;
    }

    /*
     * cudaDeviceReset() destroys all allocations and resets the device
     * to its initial state. It frees all memory, destroys streams,
     * events, and other resources associated with the device. This
     * function is typically called at the end of a program to ensure
     * that the device is left clean for subsequent runs or for
     * debugging purposes. Unlike cudaDeviceSynchronize, it does not
     * wait for pending operations; it should be called only after
     * ensuring that all work has finished (hence the call to
     * cudaDeviceSynchronize() above).
     */
    cudaError_t errReset = cudaDeviceReset();
    if (errReset != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(errReset));
        return 1;
    }

    printf("Device synchronized and reset successfully.\n");
    return 0;
}
