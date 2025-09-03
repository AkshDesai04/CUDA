/*
Aim: 
Run `cuda-memcheck` with the `--tool racecheck` option to look for race conditions.

Thinking:
The goal is to provide a minimal CUDA C program that contains a clear data race, so that when the user runs
`cuda-memcheck --tool racecheck` on it, the tool will detect the race condition.  A simple and common way to
introduce a race is to have multiple threads update the same memory location without any synchronization
mechanism (e.g., without using atomics or locks).  In this example, a global counter is incremented by many
threads, each performing a loop of increments.  Because the write to `counter[0]` is not atomic, different
threads will interfere with each other, leading to a non-deterministic final value that is lower than the
expected one.  This makes it a perfect candidate for `cuda-memcheck --tool racecheck` to flag the race.

The code below:
- Allocates an integer on the device to act as a shared counter.
- Launches many threads (256 per block, 64 blocks) and each thread performs 1000 increments in a loop.
- After the kernel finishes, it copies the counter back to host and prints both the actual final value and the
  expected value (blocks * threadsPerBlock * iterations).
- The expected value is not reached due to the race, illustrating the problem.

To test, compile with `nvcc racecheck_example.cu -o racecheck_example` and then run
`cuda-memcheck --tool racecheck ./racecheck_example`.  The tool should report a race on the location
`counter[0]`.
*/

#include <stdio.h>
#include <cuda.h>

__global__ void raceKernel(int *counter, int iterations) {
    // Each thread performs a series of increments on the same memory location
    for (int i = 0; i < iterations; ++i) {
        // Intentional data race: no atomic operation
        counter[0] = counter[0] + 1;
    }
}

int main() {
    int *dev_counter;
    int host_counter = 0;

    // Allocate device memory for the counter
    cudaMalloc((void**)&dev_counter, sizeof(int));
    cudaMemcpy(dev_counter, &host_counter, sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocks = 64;
    const int iterations = 1000;

    // Launch kernel
    raceKernel<<<blocks, threadsPerBlock>>>(dev_counter, iterations);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&host_counter, dev_counter, sizeof(int), cudaMemcpyDeviceToHost);

    // Compute expected value
    int expected = blocks * threadsPerBlock * iterations;

    printf("Final counter value: %d (expected %d)\n", host_counter, expected);

    // Clean up
    cudaFree(dev_counter);
    return 0;
}
