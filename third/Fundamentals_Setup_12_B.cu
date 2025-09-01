```cpp
/*
Aim: Create an error by calling a regular C++ function (that is not marked __device__) from within your kernel.

Thinking:
1. Define a simple regular C++ function, e.g., int add(int a, int b) that returns a + b. 
   This function is only marked as a host function, not __device__.
2. Create a CUDA kernel that attempts to call this function. According to CUDA C++ rules, a kernel cannot call a host function; it must call only __device__ or __global__ functions. This should trigger a compile-time error.
3. Add a main function that launches the kernel, so the program compiles to an executable (though it won't run successfully due to the error).
4. No additional helper functions or headers beyond <iostream> and <cuda_runtime.h> are needed.
5. The code is intentionally wrong to demonstrate the error. This will compile until the kernel call and then fail with an error similar to "calling non-device function 'int add(int,int)' from __global__ function 'kernel' is not allowed".
*/

#include <iostream>
#include <cuda_runtime.h>

// Regular host function (not marked __device__)
int add(int a, int b) {
    return a + b;
}

// CUDA kernel that incorrectly calls the host function
__global__ void myKernel(int *d_out) {
    int idx = threadIdx.x;
    // Intentional error: calling a host function from a kernel
    d_out[idx] = add(idx, idx + 1);
}

int main() {
    const int N = 10;
    int h_out[N];
    int *d_out;
    cudaMalloc((void**)&d_out, N * sizeof(int));

    // Launch kernel
    myKernel<<<1, N>>>(d_out);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results (won't reach here if compilation fails)
    for (int i = 0; i < N; ++i) {
        std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;
    }

    cudaFree(d_out);
    return 0;
}
```