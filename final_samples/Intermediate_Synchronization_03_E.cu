/*
Aim of the program: Run your racy kernel with `cuda-memcheck --tool racecheck` and analyze the output.
Thinking: The program is designed to purposely create a data race on a single integer variable. Each thread in a block increments the same global array element without using atomic operations. The race occurs because multiple threads try to read-modify-write the same memory location concurrently. We launch a large number of threads to increase the likelihood of contention. After the kernel finishes, the host prints the final value, which may be less than the expected number of increments due to lost updates. When run under cuda-memcheck with the racecheck tool, it will report a race condition on the memory location, list the conflicting threads, and provide the location in the code where the race occurs. The analysis of such output would confirm that the data race is indeed present and that atomic operations or proper synchronization would be required to avoid it.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that performs a non-atomic increment on a single array element
__global__ void racy_increment(int *d_arr, int num_increments)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < num_increments; ++i) {
        // No atomic, so multiple threads can race on d_arr[0]
        d_arr[0] += 1;
    }
}

// Utility to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int arraySize = 1;
    const int numBlocks = 64;
    const int threadsPerBlock = 128;
    const int incrementsPerThread = 1000;

    int *h_arr = (int *)malloc(arraySize * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    h_arr[0] = 0;

    int *d_arr = NULL;
    checkCudaError(cudaMalloc((void **)&d_arr, arraySize * sizeof(int)), "cudaMalloc");

    // Copy initial data to device
    checkCudaError(cudaMemcpy(d_arr, h_arr, arraySize * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Launch the racy kernel
    racy_increment<<<numBlocks, threadsPerBlock>>>(d_arr, incrementsPerThread);

    // Wait for GPU to finish
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_arr, d_arr, arraySize * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    // Compute expected value
    int expected = numBlocks * threadsPerBlock * incrementsPerThread;
    printf("Final value of d_arr[0] = %d (expected %d)\n", h_arr[0], expected);

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
