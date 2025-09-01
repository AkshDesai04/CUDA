/*
Aim: Write a __device__ function that calculates the distance between two 2D points, sqrt((x2-x1)^2 + (y2-y1)^2).

Thinking:
- The goal is to provide a device-side helper that returns the Euclidean distance between two points in 2D space.
- Use double precision for accuracy. The CUDA math library supplies a device function sqrt for doubles.
- The function signature will be __device__ double distance2D(double x1, double y1, double x2, double y2).
- Inside the function we compute dx = x2 - x1, dy = y2 - y1, then return sqrt(dx*dx + dy*dy).
- To demonstrate usage, a simple kernel will call this function for a set of hard‑coded points and print the result using device printf (requires compute capability 2.0+).
- The host code launches the kernel with a single thread/block.
- The file is a complete .cu program that can be compiled with nvcc.
*/

#include <stdio.h>
#include <math.h>

__device__ double distance2D(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx * dx + dy * dy);
}

__global__ void testDistanceKernel() {
    // Example points
    double x1 = 0.0, y1 = 0.0;
    double x2 = 3.0, y2 = 4.0;

    double dist = distance2D(x1, y1, x2, y2);
    printf("Distance between (%.2f, %.2f) and (%.2f, %.2f) is %.6f\n",
           x1, y1, x2, y2, dist);
}

int main() {
    // Launch kernel with one thread
    testDistanceKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
