```cpp
/* 
Aim of the program: 
Allocate memory, but forget to copy the pointer's value back from the device. Print the host variable. What does it show and why?

Thinking:
1. On the host we declare a pointer (h_ptr) initialized to NULL (or 0).
2. We allocate device memory using cudaMalloc and store the device pointer in d_ptr.
3. Normally, if we want to use that allocated memory on the host, we would need to copy the device pointer back to the host using cudaMemcpy or by passing it directly.
4. In this exercise we intentionally skip that step – we do not copy d_ptr back to h_ptr.
5. When we print h_ptr, it will still contain its original value (NULL or 0), because the host variable was never updated with the device pointer value.
6. The printed value shows that the host pointer is still uninitialized/NULL, highlighting the importance of copying back the pointer value if we want to use it on the host side.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int *h_ptr = NULL;         // Host pointer, initially NULL
    int *d_ptr = NULL;         // Device pointer

    // Allocate memory on the device (e.g., 10 integers)
    size_t size = 10 * sizeof(int);
    cudaError_t err = cudaMalloc((void**)&d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // (Optional) Initialize device memory with a kernel or cudaMemset here
    // but we won't copy anything back to h_ptr.

    // Intentionally forget to copy the device pointer back to the host
    // cudaMemcpy(&h_ptr, &d_ptr, sizeof(int*), cudaMemcpyDeviceToHost);

    // Print the host pointer value
    printf("Host pointer h_ptr = %p\n", (void*)h_ptr);

    // Clean up device memory
    cudaFree(d_ptr);

    return 0;
}
```