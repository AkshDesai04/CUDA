/*
Deliberately introduce an error (e.g., pass a null pointer to `cudaMemcpy`) and see your macro in action.

Thinking:
- Create a simple CUDA error checking macro (CUDA_CHECK) that calls a CUDA API function,
  checks its return value, prints the file/line and error string if not cudaSuccess, 
  and exits the program.
- In main(), allocate a small block of host memory, initialize it.
- Intentionally set the device pointer to NULL to trigger an error in cudaMemcpy.
- Use the macro to wrap the cudaMemcpy call, which will capture the error and print
  a diagnostic message.
- Finally, clean up host memory and reset the device.
- This program demonstrates how the macro reacts to an intentional misuse
  (copying to/from a null device pointer) and provides clear error information.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err);                                            \
        }                                                          \
    } while (0)

int main(void) {
    // Allocate host memory
    int *h_ptr = (int *)malloc(sizeof(int));
    if (!h_ptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    *h_ptr = 42;

    // Intentionally set device pointer to NULL to cause error
    int *d_ptr = NULL;

    // This should trigger an error and be caught by CUDA_CHECK
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, sizeof(int), cudaMemcpyHostToDevice));

    // Clean up (this point will not be reached due to error exit)
    free(h_ptr);
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
