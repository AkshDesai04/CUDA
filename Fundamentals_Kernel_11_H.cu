```cuda
/*
In a kernel with many threads, have every thread try to modify the same member of the single struct instance (e.g., d_data->a = threadIdx.x;). What is the final value after copying back? (Race condition).

The aim of this program is to illustrate a classic race condition in CUDA: multiple threads concurrently write to the same memory location without any synchronization. In this example, a single struct instance is allocated on the device, and each thread in a large block writes its threadIdx.x value to the struct's member 'a'. Because the writes happen simultaneously and the hardware has no guarantee of order for unordered concurrent writes, the final value of 'a' after copying back to the host is nondeterministic. It can be any of the values written by the threads, depending on which write happens last in the memory subsystem.

The code below:
1. Allocates a struct on the device.
2. Launches a kernel where each thread writes its thread index to the struct.
3. Copies the struct back to the host.
4. Prints the value of 'a' to show that it is unpredictable.
The program can be run multiple times to observe different outcomes, confirming the race condition.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct Data {
    int a;
};

__global__ void raceKernel(Data *d_data, int numThreads)
{
    // Each thread writes its threadIdx.x to the same struct member.
    // No synchronization is used, so this is a classic data race.
    d_data->a = threadIdx.x;
}

int main(void)
{
    const int numThreads = 256;      // Number of threads in the block
    const int blockSize  = 256;      // One block is enough for this demo

    Data *d_data = NULL;
    cudaError_t err;

    // Allocate memory for the struct on the device
    err = cudaMalloc((void**)&d_data, sizeof(Data));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch the kernel
    raceKernel<<<1, blockSize>>>(d_data, numThreads);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy the struct back to the host
    Data h_data;
    err = cudaMemcpy(&h_data, d_data, sizeof(Data), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Print the final value of 'a'
    printf("Final value of d_data->a after race: %d\n", h_data.a);

    // Clean up
    cudaFree(d_data);
    cudaDeviceReset();

    return 0;
}
```