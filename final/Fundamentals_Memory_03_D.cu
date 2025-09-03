```cpp
/* 
Aim of the program:
Use `cudaMemset` to initialize `d_a` to a value (e.g., 5) instead of copying from the host. 
Then perform the DtoD copy.

Thinking:
- `cudaMemset` sets memory byte‑wise. When we pass 5 as the value, every byte in the
  allocated memory will be set to 0x05. For an `int` array this means each element
  will actually have the integer value 0x05050505 (84215045 decimal). 
- This is fine for demonstration; if we wanted a real integer value of 5 we'd need
  to use a kernel or `cudaMemcpy` from a host array.  
- We allocate two device arrays `d_a` and `d_b`.  
- We set `d_a` using `cudaMemset(d_a, 5, N*sizeof(int))`.  
- We then copy `d_a` to `d_b` using `cudaMemcpy` with the flag `cudaMemcpyDeviceToDevice`.  
- Finally, we copy `d_b` back to the host and print the values to verify that the
  data was correctly set and transferred.  
- Error checking macros are included to catch CUDA API errors.  
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",   \
                    #call, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

int main() {
    const int N = 10;
    const size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_a = nullptr; // Not used in this example
    int *h_b = (int*)malloc(size);
    if (!h_b) {
        fprintf(stderr, "Failed to allocate host memory for h_b\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_a = nullptr;
    int *d_b = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));

    // Use cudaMemset to set each byte of d_a to 5
    // This will set each int element to 0x05050505
    CHECK_CUDA(cudaMemset(d_a, 5, size));

    // Perform device-to-device copy from d_a to d_b
    CHECK_CUDA(cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Print the result
    printf("Contents of h_b after DtoD copy:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_b[i]);
    }
    printf("\n");

    // Clean up
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}
```