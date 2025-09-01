/*
Aim: Write a __device__ function that calculates the distance between two 2D points, sqrt((x2-x1)^2 + (y2-y1)^2).
Thinking: To compute a Euclidean distance on the device we provide a simple __device__ helper that accepts four float coordinates.
The function will compute the horizontal and vertical differences, square them, sum the squares and then take the square root.
For floating‑point math on the device we use sqrtf from <math.h> (the device implementation).  
To demonstrate the function we also supply a tiny kernel that calls it for a pair of points and writes the result
to a device array. The host code allocates the array, launches the kernel, copies the result back and prints it.
Error checking is minimal but demonstrates typical CUDA practice.
*/

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// Device helper to compute Euclidean distance between two 2D points
__device__ float distance2D(float x1, float y1, float x2, float y2)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

// Simple kernel that uses the distance function
__global__ void compute_distance(float x1, float y1, float x2, float y2, float *out)
{
    // Each thread computes one distance; for demo we only use thread 0
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        out[0] = distance2D(x1, y1, x2, y2);
    }
}

int main(void)
{
    const float x1 = 1.0f, y1 = 2.0f;
    const float x2 = 4.0f, y2 = 6.0f;

    float *d_result;
    float h_result;

    // Allocate device memory for the result
    cudaMalloc((void **)&d_result, sizeof(float));

    // Launch kernel with one block of one thread
    compute_distance<<<1, 1>>>(x1, y1, x2, y2, d_result);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Distance between (%.2f, %.2f) and (%.2f, %.2f) is %.6f\n",
           x1, y1, x2, y2, h_result);

    // Clean up
    cudaFree(d_result);
    return 0;
}
