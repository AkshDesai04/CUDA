/*
Run `cuda-memcheck` with the `--tool synccheck` option to look for synchronization errors.

I decided to write a small CUDA program that intentionally creates a data race by accessing shared memory
without proper synchronization.  In the kernel, each thread writes its own thread index to a location in
__shared__ memory.  Immediately after the write, the thread reads a value from a neighboring index and writes
the result to a global array.  Because __syncthreads() is omitted, the read may occur before the write
by the other thread has completed, which is a classic synchronization error.  When the user runs
`cuda-memcheck --tool synccheck ./a.out` on this program, the tool should detect the unsynchronized
access to shared memory and report a synchronization error.  This program also includes basic CUDA
error checking and prints the resulting global array to the console.
*/

#include <cstdio>
#include <cstdlib>

#define N 32

// Kernel that creates a data race on shared memory
__global__ void raceKernel(int *out) {
    // Shared memory array
    __shared__ int sdata[N];

    int tid = threadIdx.x;

    // Each thread writes its thread id to shared memory
    sdata[tid] = tid;

    // Intentionally no __syncthreads() here

    // Each thread reads the value from the next thread's shared memory slot
    // This read may occur before the next thread has written its value
    int val = sdata[(tid + 1) % N];

    // Write the result to global memory
    out[tid] = val;
}

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    int *d_out;
    int h_out[N];

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // Launch kernel with N threads
    raceKernel<<<1, N>>>(d_out);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Output array (each element is the value read from shared memory of next thread):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
