/*
Pass a valid device pointer but a `nullptr` host pointer to `cudaMemcpy` HtoD. What error does this cause?
The aim of this program is to demonstrate the error that occurs when attempting to copy data from a host
pointer that is null to a device memory location. In CUDA, `cudaMemcpy` expects both source and destination
pointers to be valid for the direction of transfer. If the source pointer is `nullptr` for a
`cudaMemcpyHostToDevice` operation, the API should return `cudaErrorInvalidValue`. 
The program allocates device memory, sets the host pointer to `nullptr`, performs the copy, and
prints the resulting error code and string. This confirms that the error thrown is indeed
`cudaErrorInvalidValue`.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main(void) {
    const size_t N = 10;
    const size_t size = N * sizeof(int);

    // Allocate device memory
    int* d_ptr = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Host pointer is null
    int* h_ptr = nullptr;

    // Attempt to copy from null host pointer to device
    err = cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);

    // Print the error
    printf("cudaMemcpy returned error: %s (code %d)\n", cudaGetErrorString(err), err);

    // Clean up
    cudaFree(d_ptr);

    return (err == cudaSuccess) ? EXIT_SUCCESS : EXIT_FAILURE;
}
