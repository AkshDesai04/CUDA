/*
Aim of the program: Extend the previous program. After copying data to the device, create a second host array of 256 integers. Copy the data from the device array back to this new host array using `cudaMemcpy` with the `cudaMemcpyDeviceToHost` kind. Verify that every element in the new host array matches the original.

Thinking:
- Start with an array of 256 integers on the host.
- Allocate device memory and copy the original array to the device.
- Launch a simple kernel (e.g., add 1 to each element) to demonstrate device processing.
- Allocate a second host array to receive data copied back from the device.
- Use cudaMemcpy with cudaMemcpyDeviceToHost to transfer the processed data back.
- Verify that each element in the second host array matches the expected result (original + 1).
- Use proper error checking after CUDA calls.
- Print success or error messages accordingly.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256

__global__ void add_one(int *d_arr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_arr[idx] += 1;
    }
}

void checkCudaError(const char *msg, cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Host arrays
    int h_orig[N];
    int h_new[N];

    // Initialize original host array
    for (int i = 0; i < N; ++i)
    {
        h_orig[i] = i;
    }

    // Device memory allocation
    int *d_arr = NULL;
    cudaError_t err = cudaMalloc((void **)&d_arr, N * sizeof(int));
    checkCudaError("cudaMalloc", err);

    // Copy original data from host to device
    err = cudaMemcpy(d_arr, h_orig, N * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy HostToDevice", err);

    // Launch kernel: add 1 to each element
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_one<<<blocksPerGrid, threadsPerBlock>>>(d_arr);
    err = cudaGetLastError();
    checkCudaError("Kernel launch", err);

    // Copy result from device back to second host array
    err = cudaMemcpy(h_new, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy DeviceToHost", err);

    // Verify that each element in h_new matches the original + 1
    int mismatch = 0;
    for (int i = 0; i < N; ++i)
    {
        if (h_new[i] != h_orig[i] + 1)
        {
            fprintf(stderr, "Mismatch at index %d: expected %d, got %d\n",
                    i, h_orig[i] + 1, h_new[i]);
            mismatch = 1;
            break;
        }
    }

    if (!mismatch)
    {
        printf("Verification succeeded: all elements match expected values.\n");
    }
    else
    {
        printf("Verification failed.\n");
    }

    // Clean up
    cudaFree(d_arr);

    return 0;
}
