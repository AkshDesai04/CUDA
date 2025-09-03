/*
Deliberately introduce an error (e.g., pass a null pointer to `cudaMemcpy`) and see your macro in action.
The goal of this program is to demonstrate how a simple CUDA error‑checking macro can capture
runtime errors and report them immediately.  I start by defining the macro `CUDA_CHECK` that
wraps any CUDA runtime call, checks the returned `cudaError_t`, and if the call fails, prints
the file name, line number, and the human‑readable error string before aborting the program.
This macro is commonly used to simplify error handling in CUDA applications.

Next, I allocate a small array on the host, initialize it, and declare a device pointer.
Intentionally, I set the device pointer to `nullptr` (or leave it uninitialized) and then
attempt to copy data from the host to this null device address using `cudaMemcpy`.  Because
the destination pointer is invalid, the CUDA runtime will return an error (usually
`cudaErrorInvalidDevicePointer`).  The `CUDA_CHECK` macro will catch this error, print a
diagnostic message, and terminate the program.  This deliberate mistake provides a clear
example of how the macro behaves under failure conditions.

The program is minimal: it includes the necessary headers, defines the macro, performs the
faulty copy, and exits.  No kernel is launched, as the focus is solely on the error
reporting mechanism.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(err);                                            \
        }                                                         \
    } while (0)

int main(void)
{
    const int N = 10;
    int *h_ptr = (int *)malloc(N * sizeof(int));
    if (!h_ptr) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array */
    for (int i = 0; i < N; ++i) {
        h_ptr[i] = i;
    }

    /* Intentionally introduce an error: pass a null device pointer to cudaMemcpy */
    int *d_ptr = nullptr; /* This is the deliberate mistake */

    /* Attempt to copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, N * sizeof(int), cudaMemcpyHostToDevice));

    /* If the error macro had not triggered, the following code would execute */
    printf("Copy succeeded (this line should not be printed).\n");

    free(h_ptr);
    return 0;
}
