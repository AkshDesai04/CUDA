```cuda
/*
Aim of the program:
If you print the value of a device pointer (`printf("%p", d_ptr);`), it prints an address. Explain that this address is only valid from the GPU's perspective.

Thinking process:
- The program demonstrates that a device pointer (obtained via cudaMalloc) represents a location in the GPU's address space, not the host's.
- We allocate an array on the GPU, then launch a kernel that uses device-side printf to print the pointer value.
- After the kernel, the host also prints the same pointer value using printf. The numeric value is the same, but the host interprets it as a virtual address in its own process space, which is meaningless for accessing the GPU memory.
- The code also attempts to dereference the pointer on the host (via cudaMemcpy) to show that the host cannot directly use the GPU pointer.
- Compile with: `nvcc -arch=sm_52 -o device_ptr_example device_ptr_example.cu`
- Run with: `./device_ptr_example`
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",        \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),           \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel that prints the device pointer value
__global__ void printDevicePtr(const int *d_arr, size_t size) {
    // Only thread 0 will print to avoid clutter
    if (threadIdx.x == 0) {
        // Print the pointer value; this is a GPU address
        printf("Kernel: device pointer value = %p\n", (void*)d_arr);
        // For demonstration, we also print the first element value
        if (size > 0) {
            printf("Kernel: first element = %d\n", d_arr[0]);
        }
    }
}

int main(void) {
    const size_t N = 10;
    const size_t bytes = N * sizeof(int);

    int *d_arr = NULL;
    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void**)&d_arr, bytes));

    // Initialize host array
    int h_arr[N];
    for (size_t i = 0; i < N; ++i) {
        h_arr[i] = static_cast<int>(i);
    }

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    // Launch kernel to print device pointer
    printDevicePtr<<<1, 1>>>(d_arr, N);
    // Ensure kernel finished
    CUDA_CHECK(cudaDeviceSynchronize());

    // Host-side printf of the same pointer
    printf("Host: device pointer value = %p\n", (void*)d_arr);

    // Attempting to dereference the device pointer on host would be incorrect.
    // We perform a cudaMemcpy to demonstrate correct host-device interaction.
    int h_copy[N];
    CUDA_CHECK(cudaMemcpy(h_copy, d_arr, bytes, cudaMemcpyDeviceToHost));
    printf("Host: first element after copy = %d\n", h_copy[0]);

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
```