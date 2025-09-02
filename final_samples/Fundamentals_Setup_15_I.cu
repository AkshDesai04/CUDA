```cuda
/*
Aim: Explain the difference between `cudaDeviceReset()` and `cudaDeviceSynchronize()` in comments.

Thinking:
The task is to produce a CUDA C source file (.cu) that contains a clear explanation, in comments, of how `cudaDeviceReset()` and `cudaDeviceSynchronize()` differ.  
The file must be a valid CUDA source file that can be compiled with `nvcc`.  
I will include a minimal kernel and a `main` function that demonstrates both calls.  
The explanatory comments will be inserted inline, near the calls, so a reader can understand the distinct semantics:

- `cudaDeviceSynchronize()` blocks the host until all previously issued device work completes, ensuring that the device is idle. It does **not** reset the device state; resources such as memory allocations and stream objects remain unchanged.

- `cudaDeviceReset()` cleans up the device state, freeing all allocations and destroying all streams that were created by the host for the current context. It also forces a synchronization because it cannot reset the device while it is busy. After a reset, the device returns to a state as if it had just been initialized, so subsequent launches must re‑establish any required resources.

I will also show that `cudaDeviceReset()` implicitly synchronizes, but its effect is global and irreversible until a new context is created. The comments will cover error checking and the typical usage pattern. No external files or additional data are required – the code is self‑contained. The goal is educational rather than functional, so a trivial kernel that does nothing is sufficient.

Now the code follows:
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Dummy kernel that does nothing */
__global__ void dummyKernel(void) {}

/* Helper to check CUDA API call results */
#define CHECK_CUDA(call)                                              \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

int main(void)
{
    /* Launch a kernel asynchronously on the default stream */
    dummyKernel<<<1, 1>>>();
    /* Check for launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* ------------------------------------------------------------------
       cudaDeviceSynchronize()
       ------------------------------------------
       Purpose:
           - Blocks the host thread until all previously issued
             device work (kernel launches, memory copies, etc.) is
             finished.
           - It ensures that the device is idle before the program
             proceeds.
           - It does **not** clean up any resources (memory, streams,
             events, etc.).  All CUDA objects remain valid and can be
             reused after the call.
       Typical usage:
           - When the host needs to guarantee that a kernel has
             completed before accessing results on the host.
           - When profiling or timing device work.
    ------------------------------------------------------------------- */
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("cudaDeviceSynchronize: all previous device work completed.\n");

    /* ------------------------------------------------------------------
       cudaDeviceReset()
       ------------------------------------------
       Purpose:
           - Cleans up the device state for the current process.
           - Frees all device memory allocations (global, constant,
             shared) and destroys all streams and events created by
             the host in the current context.
           - Forces a synchronization: it cannot reset the device
             while it is busy, so it implicitly waits for all
             outstanding work to finish.
           - After a reset, the device is in a state as if it had
             just been initialized; any previously allocated
             resources are gone.
       Typical usage:
           - At the end of a program that has finished all device
             work, when you want to explicitly release GPU resources
             before the process exits.
           - In a long‑running application that wants to reclaim all
             GPU memory after a particular phase.
       Note:
           - Unlike cudaDeviceSynchronize(), cudaDeviceReset()
             removes all CUDA context information for the process.
           - If you need to run more kernels after a reset, you must
             re‑allocate memory, recreate streams, etc.
    ------------------------------------------------------------------- */
    CHECK_CUDA(cudaDeviceReset());
    printf("cudaDeviceReset: device resources cleaned and context destroyed.\n");

    return 0;
}
```