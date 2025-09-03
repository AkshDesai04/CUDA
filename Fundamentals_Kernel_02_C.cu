```cpp
/*
Aim of the program: Modify the printf to include the thread's ID: printf("Hello from thread %d!\n", threadIdx.x); Launch with 8 threads.

Thinking:
1. The program should be a simple CUDA C program that launches a kernel with 8 threads.
2. Each thread will print its own thread ID using device printf.
3. Device printf is supported in CUDA and requires <stdio.h>.
4. After launching the kernel, we must call cudaDeviceSynchronize() to wait for all threads to finish before the host program exits, otherwise the program may terminate before the device prints are flushed.
5. Use a single block (1, 8) to satisfy the requirement of launching 8 threads.
6. Ensure the code compiles with nvcc and runs correctly on a CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf("Hello from thread %d!\n", threadIdx.x);
}

int main() {
    // Launch the kernel with 1 block and 8 threads per block
    helloKernel<<<1, 8>>>();
    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    return 0;
}
```