```cpp
/*
Aim: 
The exchange involves GPU 0 sending its rightmost element to GPU 1's 'left halo' region, 
and GPU 1 sending its leftmost element to GPU 0's 'right halo' region.

Thinking:
1. Define a simple 1‑D array on each GPU that includes halo cells.
   For each array of size N=10, index 0 is the left halo, indices 1‑8 are the core data,
   and index 9 is the right halo. GPU 0 will send its core element at index 8 (the
   rightmost core value) to GPU 1’s left halo (index 0). GPU 1 will send its core
   element at index 1 (the leftmost core value) to GPU 0’s right halo (index 9).

2. Use CUDA peer‑to‑peer memory copy (cudaMemcpyPeer) to exchange the scalar
   values directly between GPUs. Peer access must be enabled for the two devices.

3. After the exchange, copy the device arrays back to host memory and print
   them to verify that the halo cells contain the expected values:
   - GPU 0 right halo should hold 200 (originated from GPU 1).
   - GPU 1 left halo should hold 100 (originated from GPU 0).

4. The code checks for at least two GPUs and verifies peer support before
   proceeding. Error checking is performed after each CUDA API call.

5. The program uses a single kernel launch per GPU to illustrate that the
   exchange is complete before any further device work. The kernel merely
   prints the array contents on the device (device‑side printf), but the final
   verification is performed on the host.

6. All allocation, initialization, exchange, and de‑allocation are handled
   explicitly, making the example self‑contained and suitable for educational
   purposes or as a minimal test harness for inter‑GPU halo exchange.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 10  // Total array size including halos
#define CHECK_CUDA(call)                                       \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

// Kernel to print array contents from each GPU
__global__ void printArray(const int *arr, int gpu_id) {
    printf("GPU %d array: ", gpu_id);
    for (int i = 0; i < N; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        fprintf(stderr, "This demo requires at least 2 GPUs.\n");
        return EXIT_FAILURE;
    }

    // Devices 0 and 1
    const int dev0 = 0;
    const int dev1 = 1;

    // Enable peer access between the two GPUs
    CHECK_CUDA(cudaSetDevice(dev0));
    cudaError_t err = cudaDeviceEnablePeerAccess(dev1, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        err = cudaSuccess;
    }
    CHECK_CUDA(err);

    CHECK_CUDA(cudaSetDevice(dev1));
    err = cudaDeviceEnablePeerAccess(dev0, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        err = cudaSuccess;
    }
    CHECK_CUDA(err);

    // Allocate device arrays
    int *d_arr0 = nullptr;
    int *d_arr1 = nullptr;
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMalloc(&d_arr0, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_arr0, 0, N * sizeof(int)));

    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMalloc(&d_arr1, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_arr1, 0, N * sizeof(int)));

    // Initialize core elements
    // GPU 0 core rightmost element (index 8) = 100
    CHECK_CUDA(cudaSetDevice(dev0));
    int val0 = 100;
    CHECK_CUDA(cudaMemcpy(&d_arr0[8], &val0, sizeof(int), cudaMemcpyHostToDevice));

    // GPU 1 core leftmost element (index 1) = 200
    CHECK_CUDA(cudaSetDevice(dev1));
    int val1 = 200;
    CHECK_CUDA(cudaMemcpy(&d_arr1[1], &val1, sizeof(int), cudaMemcpyHostToDevice));

    // Exchange halo values using peer copy
    // GPU 0 -> GPU 1 left halo (index 0)
    CHECK_CUDA(cudaMemcpyPeer(&d_arr1[0], dev1, &d_arr0[8], dev0, sizeof(int)));

    // GPU 1 -> GPU 0 right halo (index 9)
    CHECK_CUDA(cudaMemcpyPeer(&d_arr0[9], dev0, &d_arr1[1], dev1, sizeof(int)));

    // Launch kernels to print arrays from each GPU
    CHECK_CUDA(cudaSetDevice(dev0));
    printArray<<<1,1>>>(d_arr0, dev0);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(dev1));
    printArray<<<1,1>>>(d_arr1, dev1);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back to host to verify values
    int h_arr0[N], h_arr1[N];
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMemcpy(h_arr0, d_arr0, N * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMemcpy(h_arr1, d_arr1, N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("\nVerification from host:\n");
    printf("GPU 0 array: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_arr0[i]);
    printf("\n");

    printf("GPU 1 array: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_arr1[i]);
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaFree(d_arr0));

    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaFree(d_arr1));

    return EXIT_SUCCESS;
}
```