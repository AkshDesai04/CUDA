/*
Aim: Write a kernel to shift (rotate) an array by k positions using shared memory.

Thinking:
The requirement is to rotate an array by k positions (right rotation) using shared memory for the
intermediate storage. The simplest way to achieve this is to copy the entire array into shared
memory, perform the index arithmetic in shared memory, and write the results back to global
memory. Because shared memory is limited, the example is constrained to a small array that can
fit into the shared memory of a single block. For demonstration purposes, we use a single block
with a thread count equal to the array size. Each thread loads one element into shared memory,
computes its new rotated position, and writes the element to the output array at the new
position.

The kernel logic:
1. Allocate dynamic shared memory sized to hold the entire input array.
2. Each thread copies its element from global memory to shared memory.
3. Synchronize to ensure all loads complete.
4. Each thread calculates the rotated index `(idx + k) % N`.
5. Write the element from shared memory to the output array at the rotated index.

The host code:
- Initializes an example array.
- Allocates device memory for input and output.
- Launches the kernel with one block and dynamic shared memory.
- Copies back the result and prints both the original and rotated arrays.

Note: For larger arrays that do not fit into a single block's shared memory, a more complex
multi‑block strategy would be required. Here we keep the example simple and self‑contained.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void shiftKernel(const int *d_in, int *d_out, int N, int k)
{
    // Allocate dynamic shared memory sized to hold the entire input array
    extern __shared__ int s[];
    int idx = threadIdx.x;

    // Load data into shared memory
    if (idx < N) {
        s[idx] = d_in[idx];
    }

    // Ensure all threads have loaded data
    __syncthreads();

    // Compute rotated position and write to output
    if (idx < N) {
        int newIdx = (idx + k) % N;          // right rotation by k
        d_out[newIdx] = s[idx];
    }
}

int main(void)
{
    const int N = 10;          // Array size (must fit into shared memory)
    const int k = 3;           // Number of positions to rotate right
    int h_in[N], h_out[N];

    // Initialize input array
    printf("Input array: ");
    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
        printf("%d ", h_in[i]);
    }
    printf("\n");

    // Allocate device memory
    int *d_in, *d_out;
    cudaMalloc((void**)&d_in,  N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: one block, N threads, dynamic shared memory of N * sizeof(int)
    dim3 blockDim(N);
    dim3 gridDim(1);
    size_t sharedMemSize = N * sizeof(int);
    shiftKernel<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, N, k);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print output array
    printf("Rotated array (right by %d): ", k);
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
