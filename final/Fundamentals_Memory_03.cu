```cuda
/*
Write a program that allocates two integer arrays, `d_a` and `d_b`, on the device. 
Initialize `d_a` by copying data from the host. Then, use `cudaMemcpy` with the `cudaMemcpyDeviceToDevice` kind to copy the contents of `d_a` directly to `d_b` without involving the host.

Thinking:
1. Include the necessary headers: <cuda_runtime.h> for CUDA runtime API and <stdio.h> for input/output.
2. Define a simple error-checking macro `CUDA_CHECK` that wraps a CUDA call and prints an error message along with the file and line number if the call fails. This keeps the code clean.
3. In `main()`, set a constant `N` (e.g., 10) for the size of the arrays. Allocate a host array `h_a` of size `N`, and initialize it with some values (for example, `h_a[i] = i`).
4. Allocate two device integer pointers `d_a` and `d_b`. Use `cudaMalloc` with the size `N * sizeof(int)` for each. Check the allocation with `CUDA_CHECK`.
5. Copy data from the host array `h_a` to the device array `d_a` using `cudaMemcpy` with `cudaMemcpyHostToDevice`. This initializes `d_a` on the device.
6. Perform the device-to-device copy: call `cudaMemcpy(d_b, d_a, N * sizeof(int), cudaMemcpyDeviceToDevice)`. This copies the contents of `d_a` directly into `d_b` without going through the host.
7. To verify the copy, allocate another host array `h_b` and copy the contents of `d_b` back to the host using `cudaMemcpy` with `cudaMemcpyDeviceToHost`. Print the elements of `h_b` to confirm they match the original `h_a`.
8. Clean up: free device memory with `cudaFree` for `d_a` and `d_b`; free host memory for `h_a` and `h_b`. Finally, call `cudaDeviceReset()` to clean up the CUDA context.
9. The program should be compilable with `nvcc` and run on any GPU-enabled system.

The final program is a straightforward demonstration of allocating device memory, copying from host to device, and performing a device-to-device copy using the CUDA runtime API.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API call results */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    const int N = 10;                     // Size of the arrays
    int *h_a = NULL;                     // Host array
    int *h_b = NULL;                     // Host array to receive copy
    int *d_a = NULL;                     // Device array a
    int *d_b = NULL;                     // Device array b

    /* Allocate and initialize host array */
    h_a = (int*)malloc(N * sizeof(int));
    h_b = (int*)malloc(N * sizeof(int));
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 10;  // arbitrary values
    }

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(int)));

    /* Copy from host to device: initialize d_a */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));

    /* Device-to-device copy: d_a -> d_b */
    CUDA_CHECK(cudaMemcpy(d_b, d_a, N * sizeof(int), cudaMemcpyDeviceToDevice));

    /* Copy result back to host to verify */
    CUDA_CHECK(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Original host array h_a: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_a[i]);
    }
    printf("\nCopied device array back to host h_b: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_b[i]);
    }
    printf("\n");

    /* Clean up */
    free(h_a);
    free(h_b);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```