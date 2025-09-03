/*
Aim of the program: Now do the reverse: allocate host memory with `malloc`, pass the host pointer to a kernel, and have the kernel try to dereference it. What happens? (This can cause an error or unspecified behavior).

Thinking:
1. In CUDA, device code can only access memory that is allocated on the device or memory that has been mapped for device access. 
2. If we allocate memory on the host using `malloc`, the resulting pointer refers to a location in the host's virtual address space. This address is not valid in the device's address space. 
3. Passing that host pointer to a kernel will not automatically map it for device use; the kernel will interpret the integer value of the pointer as a device address and attempt to dereference it. 
4. Since the address is invalid on the device, the kernel will either crash with an illegal address fault or produce unspecified behaviour. The CUDA runtime typically reports this as a `cudaErrorIllegalAddress` or `cudaErrorInvalidDevicePointer`. 
5. The program below demonstrates this by allocating a small host array, launching a kernel that dereferences the host pointer, and checking for errors using `cudaGetLastError`. The expected output is an error message indicating an invalid device pointer or illegal address. 
6. The program includes minimal error checking for CUDA API calls to clearly report any failure. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1

// Kernel that tries to dereference the passed pointer (which points to host memory)
__global__ void derefKernel(int *hostPtr, int *output) {
    // Attempt to read from the host memory address (invalid on device)
    int val = hostPtr[0];   // This is expected to fault
    output[0] = val;        // Write result back to device memory
}

int main(void) {
    // Allocate host memory with malloc
    int *hostData = (int *)malloc(sizeof(int) * N);
    if (hostData == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    hostData[0] = 42; // Some test value

    // Allocate device memory for output
    int *devOutput;
    cudaError_t err = cudaMalloc((void **)&devOutput, sizeof(int) * N);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(hostData);
        return EXIT_FAILURE;
    }

    // Launch kernel with host pointer (invalid device address)
    derefKernel<<<1, 1>>>(hostData, devOutput);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    } else {
        // Attempt to copy result back to host (this may also trigger an error)
        int outputValue;
        err = cudaMemcpy(&outputValue, devOutput, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy error: %s\n", cudaGetErrorString(err));
        } else {
            printf("Kernel output: %d\n", outputValue);
        }
    }

    // Clean up
    cudaFree(devOutput);
    free(hostData);

    // Ensure all CUDA calls have completed
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
