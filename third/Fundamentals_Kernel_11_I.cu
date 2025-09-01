```cuda
/*
Use `atomicAdd` to safely have multiple threads increment a counter member within the single shared struct instance.

Thought process:
- The goal is to demonstrate safe concurrent modification of a shared data structure by multiple threads.
- We'll create a simple struct `Counter` that contains an integer counter.
- An instance of this struct will be allocated in shared memory (`__shared__`) so all threads in a block can access it.
- Each thread will perform `atomicAdd` on the `count` member to avoid race conditions.
- After all threads finish, the first thread will write the final counter value to global memory for the host to read.
- The kernel will be launched with a reasonable number of threads per block, and we will copy the result back to the host and print it.
- `atomicAdd` works on shared memory for compute capability 2.0 and above; the code assumes a modern GPU.
- Basic error checking will be omitted for brevity but could be added in production code.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct Counter {
    int count;
};

__global__ void increment_shared_counter(int *globalResult) {
    // Declare a shared instance of the struct
    __shared__ Counter s_counter;

    // Initialize the counter only once per block
    if (threadIdx.x == 0) {
        s_counter.count = 0;
    }
    __syncthreads();

    // Each thread atomically increments the counter
    atomicAdd(&(s_counter.count), 1);

    __syncthreads();

    // Thread 0 writes the final value to global memory
    if (threadIdx.x == 0) {
        globalResult[blockIdx.x] = s_counter.count;
    }
}

int main() {
    const int threadsPerBlock = 256;
    const int numberOfBlocks = 1;

    // Allocate space for the result on device
    int *d_result;
    cudaMalloc(&d_result, numberOfBlocks * sizeof(int));

    // Launch the kernel
    increment_shared_counter<<<numberOfBlocks, threadsPerBlock>>>(d_result);

    // Copy the result back to host
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the counter value
    printf("Final counter value: %d\n", h_result);

    // Clean up
    cudaFree(d_result);
    return 0;
}
```