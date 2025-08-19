#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1024  // Matrix size
#define BLOCK_SIZE 16  // Block size for kernel launch

// CUDA kernel to perform matrix multiplication
__global__ void matrixMultiply(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// Function to generate random integer matrix
void generateRandomMatrix(int *matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = rand() % 100;  // Random integers between 0 and 99
    }
}

// Function to print a matrix (for debugging)
void printMatrix(int *matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    int *A, *B, *C;  // Host matrices
    int *d_A, *d_B, *d_C;  // Device matrices
    size_t size = N * N * sizeof(int);

    // Initialize random seed
    srand(time(0));

    // Allocate memory for matrices on host
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // Generate random matrices A and B
    generateRandomMatrix(A, N);
    generateRandomMatrix(B, N);

    // Print matrices for debugging (optional)
    // printMatrix(A, N);
    // printMatrix(B, N);

    // Force to use GPU 3
    int deviceId = 3;  // Force to use GPU 3 (counting from 0)

    // Check if the selected device exists
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceId >= deviceCount) {
        printf("GPU %d not found. Available GPUs: %d\n", deviceId, deviceCount);
        return -1;
    }

    // Set the GPU device
    cudaSetDevice(deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    printf("Using GPU: %s\n", prop.name);

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel for matrix multiplication
    matrixMultiply<<<grid, block>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy result from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result (optional)
    // printMatrix(C, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
