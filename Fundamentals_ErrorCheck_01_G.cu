/*
Change the macro to also print the original CUDA call as a string using the `#call` preprocessor directive.

My thinking:
- The goal is to modify an existing CUDA error‑checking macro so that, in addition to reporting the error code and message, it also prints the exact CUDA API call that caused the error as a string.
- This is conveniently done using the preprocessor stringification operator `#`. By wrapping the call in a macro and using `#call`, the macro will generate a string literal containing the source text of the call.
- The macro should capture the result of the CUDA call, check against `cudaSuccess`, and if an error occurs, print a message that includes the file, line, error string, and the original call string. Then it should exit or otherwise handle the failure.
- We'll implement `CUDA_SAFE_CALL` that does this. It will be used to wrap all CUDA runtime calls in a simple example program.
- The example will allocate memory on the host, copy data to the device, copy it back, print it, and free resources, with all CUDA calls wrapped by the macro.

The final .cu file includes the macro, the example main function, and necessary includes. No additional commentary is output, only the code as requested. */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* CUDA error‑checking macro that also prints the original CUDA call as a string. */
#define CUDA_SAFE_CALL(call)                                         \
    do {                                                             \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr,                                            \
                    "CUDA error at %s:%d: %s\n"                        \
                    "Original call: %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE);                                     \
        }                                                            \
    } while (0)

int main() {
    const int N = 10;
    size_t size = N * sizeof(int);

    /* Host memory allocation */
    int *h_ptr = (int*)malloc(size);
    if (!h_ptr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host data */
    for (int i = 0; i < N; ++i) h_ptr[i] = i;

    /* Device memory allocation */
    int *d_ptr = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_ptr, size));

    /* Copy data from host to device */
    CUDA_SAFE_CALL(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

    /* Copy data back from device to host */
    CUDA_SAFE_CALL(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));

    /* Print the results */
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_ptr[i]);
    }
    printf("\n");

    /* Clean up */
    CUDA_SAFE_CALL(cudaFree(d_ptr));
    free(h_ptr);

    return EXIT_SUCCESS;
}
