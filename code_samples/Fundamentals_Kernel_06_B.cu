/*
Modify the kernel to write the value of its `threadIdx.x` to `data[0]`. Launch with one thread and verify.

Thought process:
- The requirement is to create a CUDA program that modifies a kernel so that it writes the value of the thread index `threadIdx.x` into the first element of an array called `data`.
- The kernel will be launched with a single thread. This ensures that the only thread index is 0, so we can verify that the value written to `data[0]` is indeed 0.
- Implementation steps:
  1. Include necessary headers (`stdio.h`, `cuda_runtime.h`).
  2. Define the kernel that accepts an integer pointer `int *data` and writes `threadIdx.x` to `data[0]`.
  3. In `main`, allocate device memory for a single integer.
  4. Launch the kernel with a single block of a single thread: `kernel<<<1,1>>>(dev_data);`.
  5. Synchronize and copy the result back to host memory.
  6. Print the value on the host to confirm it is 0.
  7. Clean up memory and handle errors appropriately.
- Since we want to demonstrate correctness, the program prints the value read from the GPU array and indicates success if it matches the expected value.

The resulting code below implements all these steps.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that writes the threadIdx.x value to data[0]
__global__ void writeThreadIdx(int *data)
{
    // Only one thread will run; still use threadIdx.x for correctness
    data[0] = threadIdx.x;
}

int main()
{
    int *dev_data = NULL;
    int host_data = 0;
    size_t bytes = sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&dev_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel with 1 block and 1 thread
    writeThreadIdx<<<1, 1>>>(dev_data);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_data);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(&host_data, dev_data, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_data);
        return 1;
    }

    // Verify the result
    printf("Value of data[0] on GPU: %d\n", host_data);
    if (host_data == 0) {
        printf("Verification successful: threadIdx.x = 0 as expected.\n");
    } else {
        printf("Verification failed: unexpected value.\n");
    }

    // Clean up
    cudaFree(dev_data);
    return 0;
}
