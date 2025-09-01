```cuda
/*
Aim: Write a `__device__` function that calculates the distance between two 2D points, `sqrt((x2-x1)^2 + (y2-y1)^2)`.

Thinking:
1. The distance formula is straightforward: sqrt((x2-x1)^2 + (y2-y1)^2). 
2. Since we are working on the device, we will implement a __device__ function that takes four floats (x1, y1, x2, y2) and returns the distance as a float.
3. Use the single-precision sqrt function `sqrtf` to keep the function simple and efficient on the GPU.
4. For demonstration purposes, provide a simple kernel that uses this function to compute distances for an array of point pairs.
5. Allocate memory on the host, copy data to the device, launch the kernel, copy the results back, and print them to verify correctness.
6. Use standard CUDA runtime API calls and error checking macros for clarity.

The resulting .cu file contains:
- The device distance function.
- A kernel that applies it to an array of point pairs.
- Host code that sets up data, runs the kernel, and prints results.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_POINTS 8

/* Device function to compute distance between two 2D points */
__device__ float distance2D(float x1, float y1, float x2, float y2)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

/* Kernel that computes distances for an array of point pairs */
__global__ void computeDistances(const float *x1, const float *y1,
                                 const float *x2, const float *y2,
                                 float *dist, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dist[idx] = distance2D(x1[idx], y1[idx], x2[idx], y2[idx]);
    }
}

/* Helper macro for error checking */
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void)
{
    /* Host data: pairs of points */
    float h_x1[NUM_POINTS] = {0, 1, 2, 3, 4, 5, 6, 7};
    float h_y1[NUM_POINTS] = {0, 1, 2, 3, 4, 5, 6, 7};
    float h_x2[NUM_POINTS] = {7, 6, 5, 4, 3, 2, 1, 0};
    float h_y2[NUM_POINTS] = {7, 6, 5, 4, 3, 2, 1, 0};

    float h_dist[NUM_POINTS] = {0};

    /* Device pointers */
    float *d_x1, *d_y1, *d_x2, *d_y2, *d_dist;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&d_x1, NUM_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y1, NUM_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_x2, NUM_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y2, NUM_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dist, NUM_POINTS * sizeof(float)));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_x1, h_x1, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y1, h_y1, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, h_x2, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y2, h_y2, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch kernel: 1 block of 32 threads is enough for 8 points */
    int threadsPerBlock = 32;
    int blocksPerGrid = (NUM_POINTS + threadsPerBlock - 1) / threadsPerBlock;
    computeDistances<<<blocksPerGrid, threadsPerBlock>>>(d_x1, d_y1, d_x2, d_y2, d_dist, NUM_POINTS);
    CUDA_CHECK(cudaGetLastError());

    /* Copy results back to host */
    CUDA_CHECK(cudaMemcpy(h_dist, d_dist, NUM_POINTS * sizeof(float), cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Distance results:\n");
    for (int i = 0; i < NUM_POINTS; ++i)
    {
        printf("Point %d: (%.1f, %.1f) to (%.1f, %.1f) => distance = %.6f\n",
               i, h_x1[i], h_y1[i], h_x2[i], h_y2[i], h_dist[i]);
    }

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_x1));
    CUDA_CHECK(cudaFree(d_y1));
    CUDA_CHECK(cudaFree(d_x2));
    CUDA_CHECK(cudaFree(d_y2));
    CUDA_CHECK(cudaFree(d_dist));

    return 0;
}
```