```cuda
/*
Write a kernel that accepts two integer pointers, `a` and `b`. The kernel (with a single thread)
should read the value from `a`, add 10 to it, and write the result to `b`. The host must allocate,
initialize `a`, and verify the result in `b`.

Thinking:
- The kernel signature will be `__global__ void addTen(int* a, int* b)`.
- It should be launched with a single thread: <<<1,1>>>.
- Inside the kernel, we read `int val = *a;`, compute `val += 10;`, and store `*b = val;`.
- The host program will:
  1. Allocate host memory for the input integer and set it to a known value (e.g., 5).
  2. Allocate device memory for `a` and `b` using `cudaMalloc`.
  3. Copy the host input to device memory for `a` using `cudaMemcpy`.
  4. Launch the kernel.
  5. Copy the result back from device `b` to host memory.
  6. Verify that the host result equals the original value plus 10.
  7. Free device memory and return status.
- Basic error checking will be added for CUDA API calls.
- The program is self-contained and can be compiled with `nvcc` as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addTen(int *a, int *b)
{
    // Since we launch with a single thread, no need to check thread indices.
    int val = *a;      // Read from a
    val += 10;         // Add 10
    *b = val;          // Write to b
}

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void)
{
    const int hostInput = 5;     // Example input value
    int hostResult = 0;

    // Device pointers
    int *dev_a = NULL;
    int *dev_b = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&dev_a, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev_b, sizeof(int)));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(dev_a, &hostInput, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel with a single thread
    addTen<<<1, 1>>>(dev_a, dev_b);

    // Wait for kernel to finish and check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&hostResult, dev_b, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    if (hostResult == hostInput + 10) {
        printf("Success: %d + 10 = %d\n", hostInput, hostResult);
    } else {
        printf("Failure: Expected %d, got %d\n", hostInput + 10, hostResult);
        return EXIT_FAILURE;
    }

    // Clean up
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));

    return EXIT_SUCCESS;
}
```