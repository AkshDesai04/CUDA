/*
Aim of the program: 
Is `cudaGetLastError()` a good way to check for kernel launch errors? (Yes, it's often used for this).

Thinking:
1. `cudaGetLastError()` is a CUDA runtime API function that returns the last error that has occurred on the device.
2. For kernel launches, which are asynchronous, errors may not be reported until a synchronization point or an explicit error query is made.
3. The typical pattern to detect kernel launch errors is:
   - Launch the kernel.
   - Call `cudaGetLastError()` immediately after to catch any launch errors.
   - Optionally call `cudaDeviceSynchronize()` to catch execution errors.
4. The program below demonstrates this pattern:
   - Defines a simple kernel that increments each element of an array.
   - Intentionally triggers a launch error by providing an invalid grid dimension (negative block size) to show that `cudaGetLastError()` catches it.
   - Also shows a successful launch where errors are checked via `cudaGetLastError()` and `cudaDeviceSynchronize()`.
5. The code prints out the results and any error messages, illustrating that `cudaGetLastError()` is indeed a good way to check for kernel launch errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that adds 1 to each element of an array
__global__ void increment_kernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

// Utility function to check CUDA errors
void check_cuda_error(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    const int N = 10;
    int h_arr[N];
    for (int i = 0; i < N; ++i) h_arr[i] = i;

    int *d_arr;
    size_t size = N * sizeof(int);
    check_cuda_error(cudaMalloc((void**)&d_arr, size), "cudaMalloc");

    // Copy data to device
    check_cuda_error(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // -----------------------------
    // 1. Successful kernel launch
    // -----------------------------
    printf("\n--- Successful kernel launch ---\n");
    int threadsPerBlock = 4;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launched successfully.\n");
    }

    // Wait for kernel to finish and check execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel executed successfully.\n");
    }

    // Copy result back
    check_cuda_error(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    printf("Result after successful kernel launch:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // -----------------------------
    // 2. Intentional launch error
    // -----------------------------
    printf("\n--- Intentional launch error ---\n");
    // Intentionally set block dimension to zero to cause a launch error
    int badThreadsPerBlock = 0;
    int badBlocksPerGrid = 1;

    increment_kernel<<<badBlocksPerGrid, badThreadsPerBlock>>>(d_arr, N);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error (expected): %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launched successfully (unexpected).\n");
    }

    // Clean up
    cudaFree(d_arr);
    return 0;
}
