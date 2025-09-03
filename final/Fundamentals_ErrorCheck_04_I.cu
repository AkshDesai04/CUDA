```cuda
/*
Pass a valid device pointer but a `nullptr` host pointer to `cudaMemcpy` HtoD. What error does this cause?

Thought process:
- The program aims to demonstrate that passing a null pointer as the source (host side) while providing a valid destination (device side) to cudaMemcpy in a HostToDevice transfer results in an error.
- According to CUDA documentation, cudaMemcpy requires both source and destination pointers to be valid addresses for the transfer direction. Supplying a nullptr for the source violates this contract.
- The expected error code returned by cudaMemcpy (and retrievable via cudaGetLastError) is cudaErrorInvalidValue, which indicates that an argument is invalid.
- The code below allocates a device buffer, intentionally sets the host pointer to nullptr, performs cudaMemcpy, checks the returned error, and prints the error string. This confirms the expected behavior.
*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    // Allocate device memory (valid device pointer)
    int *d_ptr = nullptr;
    size_t size = 10 * sizeof(int);
    cudaError_t err = cudaMalloc((void**)&d_ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Intentionally set host pointer to nullptr
    int *h_ptr = nullptr;

    // Attempt to copy from host to device with nullptr host pointer
    err = cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);

    // Check the error
    if (err != cudaSuccess) {
        std::cout << "cudaMemcpy returned error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "cudaMemcpy succeeded unexpectedly." << std::endl;
    }

    // Clean up device memory
    cudaFree(d_ptr);

    return 0;
}
```