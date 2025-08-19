#include <stdio.h>

// CUDA kernel to print "Hello World"
__global__ void helloWorld() {
    printf("Hello World from GPU!\n");
}

int main() {
    // Launch the kernel with one block and one thread
    helloWorld<<<1, 1>>>();

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    return 0;
}
