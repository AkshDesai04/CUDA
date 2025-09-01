```cpp
/*
Write a kernel that accepts two integer pointers, `a` and `b`. The kernel (with a single thread) should read the value from `a`, add 10 to it, and write the result to `b`. The host must allocate, initialize `a`, and verify the result in `b`.

My thinking for this implementation:
1. We need a CUDA kernel that takes two int pointers. Since only a single thread is required, the kernel will be launched with <<<1,1>>>.
2. Inside the kernel we read the value pointed to by `a`, add 10, and write the result to `b`.
3. On the host side we allocate a single integer on the device for each pointer.
4. We set a known value for `a` on the host, copy it to the device, launch the kernel, copy back the result from `b`, and verify it matches the expected value (original + 10).
5. We'll use simple error checking via a macro that checks the return value of CUDA API calls for brevity.
6. The code is self‑contained in a .cu file and can be compiled with `nvcc`. 
*/

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Kernel that reads from 'a', adds 10, writes to 'b'
__global__ void add_ten_kernel(const int* a, int* b) {
    // Single thread performs the operation
    int val = *a;
    *b = val + 10;
}

int main(void) {
    const int host_value = 42;          // initial value for 'a'
    int host_result = 0;                // will hold the value from 'b'

    int *d_a = nullptr;                 // device pointer for 'a'
    int *d_b = nullptr;                 // device pointer for 'b'

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, sizeof(int)));

    // Copy input value to device
    CUDA_CHECK(cudaMemcpy(d_a, &host_value, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel with 1 block and 1 thread
    add_ten_kernel<<<1, 1>>>(d_a, d_b);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&host_result, d_b, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    if (host_result == host_value + 10) {
        printf("Success: %d + 10 = %d\n", host_value, host_result);
    } else {
        fprintf(stderr, "Error: expected %d but got %d\n", host_value + 10, host_result);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
```