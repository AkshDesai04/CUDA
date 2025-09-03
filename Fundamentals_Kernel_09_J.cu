/*
Aim of the program: Write a __device__ function that calculates the distance between two 2D points, sqrt((x2-x1)^2 + (y2-y1)^2).

Thinking:
- The core requirement is a device function that can be invoked from CUDA kernels. 
- We will use float precision for simplicity, which is common in CUDA demos. 
- The function will compute the difference in x and y, square them, sum them, and return the square root using sqrtf for float. 
- The function will be declared as __device__ inline for potential inlining and to signal it runs on the GPU. 
- To demonstrate usage, a simple kernel will compute distances for an array of point pairs. 
- A host main will allocate test data, copy it to device memory, launch the kernel, retrieve the results, and print them. 
- Minimal includes are used: <cuda_runtime.h> for CUDA API, <cstdio> for printing, and <math.h> for sqrtf. 
- Error checking for CUDA calls is omitted for brevity but can be added in a real application. 
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

// Device function to compute Euclidean distance between two 2D points
__device__ inline float dist2D(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

// Kernel that computes distances for each pair of points
__global__ void computeDistances(const float *x1, const float *y1,
                                 const float *x2, const float *y2,
                                 float *dist, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dist[idx] = dist2D(x1[idx], y1[idx], x2[idx], y2[idx]);
    }
}

int main() {
    const size_t N = 10;
    float h_x1[N], h_y1[N], h_x2[N], h_y2[N], h_dist[N];

    // Initialize host data with some example points
    for (size_t i = 0; i < N; ++i) {
        h_x1[i] = float(i);
        h_y1[i] = float(i * 2);
        h_x2[i] = float(i + 1);
        h_y2[i] = float(i * 2 + 1);
    }

    // Device pointers
    float *d_x1, *d_y1, *d_x2, *d_y2, *d_dist;

    // Allocate device memory
    cudaMalloc((void**)&d_x1, N * sizeof(float));
    cudaMalloc((void**)&d_y1, N * sizeof(float));
    cudaMalloc((void**)&d_x2, N * sizeof(float));
    cudaMalloc((void**)&d_y2, N * sizeof(float));
    cudaMalloc((void**)&d_dist, N * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_x1, h_x1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, h_y1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, h_y2, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel: 1 block of 256 threads (sufficient for N <= 256)
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeDistances<<<blocks, threadsPerBlock>>>(d_x1, d_y1, d_x2, d_y2, d_dist, N);

    // Copy results back to host
    cudaMemcpy(h_dist, d_dist, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the distances
    for (size_t i = 0; i < N; ++i) {
        printf("Distance %zu: %f\n", i, h_dist[i]);
    }

    // Free device memory
    cudaFree(d_x1);
    cudaFree(d_y1);
    cudaFree(d_x2);
    cudaFree(d_y2);
    cudaFree(d_dist);

    return 0;
}
