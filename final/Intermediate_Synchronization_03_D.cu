```cuda
/*
Create a race condition on an array: have all threads increment `output[threadIdx.x % 4]`. Some locations will have races, others won't.

Thought process:
1. We need a simple CUDA kernel that increments a shared array entry based on the thread index modulo 4.
2. To introduce a race condition, we will avoid using any atomic operation when incrementing.
3. The pattern `threadIdx.x % 4` ensures that multiple threads may target the same array element.
4. To make only some array locations subject to races while others are safe, we choose a thread block size
   that does not evenly divide by 4. For example, with 5 threads in a block:
   - thread indices: 0, 1, 2, 3, 4
   - modulo 4 results: 0, 1, 2, 3, 0
   Here, array[0] will be accessed twice (threads 0 and 4), while array[1], array[2], and array[3] are
   accessed once each. Therefore, array[0] is the only element that can suffer from a race condition,
   while the others are deterministic.
5. The host code allocates memory, initializes the array to zero, launches the kernel,
   copies the result back, and prints it. Running the program multiple times may show
   that array[0] occasionally has a value less than 2 due to lost updates.
6. The program is written in plain CUDA C, compiled with `nvcc`, and requires no external libraries.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that increments output[threadIdx.x % 4] without atomic operations.
__global__ void raceKernel(int *output)
{
    int idx = threadIdx.x % 4;    // Compute the target index.
    output[idx]++;                // Non-atomic increment → race condition for duplicate indices.
}

int main(void)
{
    const int N = 4;              // Size of the array.
    int h_output[N] = {0, 0, 0, 0};   // Host array initialized to zero.
    int *d_output;

    // Allocate device memory.
    cudaMalloc((void **)&d_output, N * sizeof(int));
    cudaMemcpy(d_output, h_output, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 5 threads in a single block.
    // Adjusting block size to 5 ensures that only index 0 is accessed twice.
    raceKernel<<<1, 5>>>(d_output);

    // Synchronize to ensure kernel completion.
    cudaDeviceSynchronize();

    // Copy result back to host.
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results.
    printf("Resulting array values after raceKernel launch:\n");
    for (int i = 0; i < N; ++i) {
        printf("output[%d] = %d\n", i, h_output[i]);
    }

    // Clean up.
    cudaFree(d_output);
    return 0;
}
```