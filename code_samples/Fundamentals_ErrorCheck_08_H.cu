/*
Aim of the program: Place the CHECK(cudaDeviceSynchronize()) at the very end of main. Is this sufficient to know an error occurred? Is it good for debugging *where* it occurred?

Thinking:
To answer the question we write a simple CUDA program that launches a kernel and intentionally triggers a launch error by providing an invalid grid configuration (zero threads). We then use a CHECK macro that wraps CUDA runtime API calls. The macro reports any error and exits.  
If we only call CHECK(cudaDeviceSynchronize()) at the very end of main, the program will catch the error, but the error message will not tell us which kernel launch caused it. In contrast, inserting a CHECK after each launch (or using cudaGetLastError() immediately after a launch) will report the error immediately and the back‑trace will point to the exact line that triggered the fault, making debugging easier.  
Therefore, placing CHECK(cudaDeviceSynchronize()) only at the end is sufficient to know an error occurred, but it is not good for debugging where it occurred. To debug effectively, insert checks right after the problematic API calls or kernel launches.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro to check CUDA API calls
#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Simple kernel that does nothing
__global__ void do_nothing() {
    // Intentionally empty
}

int main(void) {
    // Allocate a trivial array on device
    int *d_arr = nullptr;
    size_t arr_size = 10 * sizeof(int);
    CHECK(cudaMalloc((void**)&d_arr, arr_size));

    // Intentionally trigger a launch error: zero threads per block
    // This will cause a launch failure that cudaDeviceSynchronize will catch
    dim3 blocks(1);
    dim3 threads(0); // Zero threads, invalid launch configuration
    do_nothing<<<blocks, threads>>>(d_arr);

    // We could check after the launch:
    // CHECK(cudaGetLastError()); // This would report the error immediately

    // Synchronize and check for errors
    CHECK(cudaDeviceSynchronize()); // This catches the error but doesn't pinpoint the launch

    // Cleanup (never reached if error occurs)
    CHECK(cudaFree(d_arr));

    printf("Program completed successfully.\n");
    return 0;
}
