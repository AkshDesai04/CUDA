```cuda
/*
Can you call a `__global__` function from a `__device__` function? (No, unless using Dynamic Parallelism). Can you call a `__device__` function from a `__global__` function? (Yes). Explain this in a comment.

I decided to illustrate the two situations in this small CUDA program.  
The program contains:
1. A `__device__` function `add()` that simply adds two integers.  
2. A `__global__` kernel `kernel_using_device()` that calls the device function `add()` and writes the result to global memory.  
   This demonstrates that a device function can be called from a kernel.

3. A `__global__` kernel `childKernel()` that is intended to be launched from the host or from another kernel.
4. A `__global__` kernel `parentKernel()` that shows dynamic parallelism: it launches `childKernel()` from within itself.  
   Dynamic parallelism is only available on devices of compute capability 3.5 and higher.  
   When compiled for such a device, this will work; otherwise the launch will fail at runtime with an error.

5. A commented-out snippet of a `__device__` function that attempts to launch a global kernel directly.  
   This would be a compile error unless dynamic parallelism is enabled and the device supports it.  
   The snippet is kept as a comment to illustrate the restriction.

The `main()` function allocates a single integer on the device, launches the kernels, copies the result back, and prints it.  
It also launches `parentKernel()` to demonstrate dynamic parallelism when available.  

By running this code on a recent GPU, you will see that the device function call succeeds, while a global call from a device function is disallowed unless you uncomment the dynamic parallelism part.  
The output will show the result of the `add()` call (which is 3) and will also confirm that the child kernel executed by dynamic parallelism wrote its own value (e.g., 42) to the output array.
*/
#include <stdio.h>
#include <cuda_runtime.h>

// Device function: can be called from any kernel or other device function
__device__ int add(int a, int b) {
    return a + b;
}

// Kernel that calls the device function
__global__ void kernel_using_device(int *out) {
    // Each thread writes the result of add(1,2) to its position
    int val = add(1, 2);
    out[threadIdx.x] = val;
}

// Child kernel to be launched via dynamic parallelism
__global__ void childKernel(int *out) {
    out[threadIdx.x] = 42;  // arbitrary value
}

// Parent kernel that launches childKernel from device side
__global__ void parentKernel(int *out) {
    // Launch childKernel only from the first thread to avoid redundant launches
    if (threadIdx.x == 0) {
        childKernel<<<1, 1>>>(out);
        // Check for errors from the child launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Dynamic parallelism launch failed: %s\\n", cudaGetErrorString(err));
        }
    }
}

// Uncommenting the following will result in a compile error unless dynamic parallelism
// and compute capability >= 3.5 is supported. It is left as a comment to illustrate the restriction.

/*
__device__ void tryCallKernel() {
    // This is illegal in standard CUDA: launching a kernel from device code
    kernel_using_device<<<1,1>>>();
}
*/

int main(void) {
    const int N = 1;
    int *d_out;
    int h_out[N];

    cudaMalloc((void **)&d_out, N * sizeof(int));

    // Launch kernel_using_device
    kernel_using_device<<<1, N>>>(d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result from kernel_using_device: %d\\n", h_out[0]);

    // Launch parentKernel to demonstrate dynamic parallelism (if supported)
    parentKernel<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result after dynamic parallelism childKernel: %d\\n", h_out[0]);

    cudaFree(d_out);
    return 0;
}
```