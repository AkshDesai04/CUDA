/*
Pass a valid device pointer but a `nullptr` host pointer to `cudaMemcpy` HtoD. What error does this cause?
Thinking: In CUDA, `cudaMemcpy` requires non-NULL source and destination pointers. 
When the host source pointer is `nullptr` but the device destination pointer is valid, the runtime performs parameter validation and returns an error code. 
The expected error is `cudaErrorInvalidValue` (error code 0), indicating that one of the arguments is invalid. 
This program demonstrates this by attempting the copy and printing the resulting error string.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    const size_t N = 10;
    float *d_ptr = NULL;
    cudaError_t err = cudaMalloc((void**)&d_ptr, N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Intentionally use a null host pointer
    float *h_ptr = NULL;

    // Attempt to copy from null host to device
    err = cudaMemcpy(d_ptr, h_ptr, N * sizeof(float), cudaMemcpyHostToDevice);
    printf("cudaMemcpy returned: %s\n", cudaGetErrorString(err));

    // Clean up
    cudaFree(d_ptr);
    return 0;
}
