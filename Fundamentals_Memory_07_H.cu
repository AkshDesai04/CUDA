/*
Explain how `cudaMemcpy` bridges the gap between these two separate memory spaces.

Thinking process:
- The user wants an explanation of `cudaMemcpy` and how it moves data between host (CPU) memory and device (GPU) memory.
- According to the instruction hierarchy, the developer's request to write a CUDA program overrides the user's request for a textual explanation.
- Therefore, I must produce a `.cu` file containing a complete program.
- In the initial multiline comment I will include the exact wording of the user's request ("Explain how `cudaMemcpy` bridges the gap between these two separate memory spaces.") and my reasoning as above.
- After the comment I will provide a minimal CUDA C program that demonstrates copying data from host to device and back using `cudaMemcpy`, printing the results to illustrate the bridge between the two memory spaces.
- No other output or explanation is provided outside the comment, per the developer's instruction.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that simply multiplies each element by 2
__global__ void multiply_by_two(const int *in, int *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = 2 * in[idx];
    }
}

int main(void) {
    const int N = 10;
    const size_t size = N * sizeof(int);

    // Allocate host memory
    int h_in[N];
    int h_out[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = i;          // Initialize input array
        h_out[i] = 0;         // Initialize output array
    }

    // Allocate device memory
    int *d_in = NULL;
    int *d_out = NULL;
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // Copy data from host to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiply_by_two<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back from device to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Input array on host:      ");
    for (int i = 0; i < N; i++) printf("%d ", h_in[i]);
    printf("\nOutput array after GPU:   ");
    for (int i = 0; i < N; i++) printf("%d ", h_out[i]);
    printf("\n");

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
