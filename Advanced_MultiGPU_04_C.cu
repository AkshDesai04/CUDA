```cuda
/*
Aim of the program:
Extend the halo exchange pattern to a ring of all available GPUs, where each GPU communicates with two neighbors.

Thinking:
- First, determine how many GPUs are available using cudaGetDeviceCount.
- For each GPU, allocate a 1‑D array of floats (or ints). The size of the array will be the interior domain size plus two halo elements (one at each end).  
- Each GPU will compute on its interior domain and then exchange halo elements with its left and right neighbors in a ring topology:
    * left neighbor = (device_id - 1 + num_devices) % num_devices
    * right neighbor = (device_id + 1) % num_devices
- The halo exchange is implemented with cudaMemcpyPeerAsync, which allows peer‑to‑peer copies between GPUs without staging to host.  
- We enable peer access between all pairs of GPUs with cudaDeviceEnablePeerAccess.  
- To overlap computation and communication, we use CUDA streams: one stream for the left copy and one for the right copy.  
- After the copies complete, we perform a simple kernel that uses the newly received halo values.  
- For demonstration, after the halo exchange we copy the device data back to host and print a few values to verify correctness.  
- The program is self‑contained and written in CUDA C, suitable for compilation with nvcc. No external libraries (e.g., MPI) are required; it relies solely on CUDA APIs for multi‑GPU communication.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 8          // size of interior domain
#define BLOCK_SIZE 256

// Simple kernel that adds the halo values to the interior
__global__ void halo_add_kernel(float *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx + 1] += d_arr[0] + d_arr[N + 1];
    }
}

// Check CUDA errors
void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    int device_count;
    checkCuda(cudaGetDeviceCount(&device_count), "Getting device count");
    if (device_count < 2) {
        fprintf(stderr, "Need at least 2 GPUs for ring exchange.\n");
        return EXIT_FAILURE;
    }
    printf("Number of GPUs: %d\n", device_count);

    // Host arrays to hold data from each device for verification
    float **h_data = (float**)malloc(device_count * sizeof(float*));
    for (int d = 0; d < device_count; ++d) {
        h_data[d] = (float*)malloc((N + 2) * sizeof(float));
    }

    // Device arrays
    float **d_arr = (float**)malloc(device_count * sizeof(float*));

    // Enable peer access between all pairs
    for (int d = 0; d < device_count; ++d) {
        checkCuda(cudaSetDevice(d), "Set device for peer access");
        for (int e = 0; e < device_count; ++e) {
            if (d != e) {
                cudaError_t p_err = cudaDeviceEnablePeerAccess(e, 0);
                if (p_err == cudaErrorPeerAccessAlreadyEnabled) {
                    // ignore, already enabled
                } else {
                    checkCuda(p_err, "Enabling peer access");
                }
            }
        }
    }

    // Allocate memory on each GPU and initialize interior values
    for (int d = 0; d < device_count; ++d) {
        checkCuda(cudaSetDevice(d), "Set device for allocation");
        checkCuda(cudaMalloc((void**)&d_arr[d], (N + 2) * sizeof(float)), "Allocating device memory");
        // Initialize: halo left = -1, halo right = -2, interior = d*10 + idx
        float *host_init = (float*)malloc((N + 2) * sizeof(float));
        host_init[0] = -1.0f; // left halo placeholder
        host_init[N + 1] = -2.0f; // right halo placeholder
        for (int i = 0; i < N; ++i) {
            host_init[i + 1] = (float)(d * 10 + i);
        }
        checkCuda(cudaMemcpy(d_arr[d], host_init, (N + 2) * sizeof(float), cudaMemcpyHostToDevice), "Initializing device data");
        free(host_init);
    }

    // Streams for left and right halo copies
    cudaStream_t *streams_left = (cudaStream_t*)malloc(device_count * sizeof(cudaStream_t));
    cudaStream_t *streams_right = (cudaStream_t*)malloc(device_count * sizeof(cudaStream_t));
    for (int d = 0; d < device_count; ++d) {
        checkCuda(cudaSetDevice(d), "Set device for stream creation");
        checkCuda(cudaStreamCreate(&streams_left[d]), "Creating left stream");
        checkCuda(cudaStreamCreate(&streams_right[d]), "Creating right stream");
    }

    // Perform halo exchange
    for (int d = 0; d < device_count; ++d) {
        int left = (d - 1 + device_count) % device_count;
        int right = (d + 1) % device_count;

        // Left copy: copy from left neighbor's right halo to this device's left halo
        checkCuda(cudaSetDevice(d), "Set device for left copy");
        checkCuda(cudaMemcpyPeerAsync(d_arr[d], d, d_arr[left] + N + 1, left,
                                      sizeof(float), streams_left[d]),
                   "Peer copy left");

        // Right copy: copy from right neighbor's left halo to this device's right halo
        checkCuda(cudaSetDevice(d), "Set device for right copy");
        checkCuda(cudaMemcpyPeerAsync(d_arr[d] + N + 1, d, d_arr[right] + 1, right,
                                      sizeof(float), streams_right[d]),
                   "Peer copy right");
    }

    // Synchronize all streams to ensure halo data is ready
    for (int d = 0; d < device_count; ++d) {
        checkCuda(cudaSetDevice(d), "Set device for sync");
        checkCuda(cudaStreamSynchronize(streams_left[d]), "Sync left stream");
        checkCuda(cudaStreamSynchronize(streams_right[d]), "Sync right stream");
    }

    // Launch kernel on each device to use halo values
    for (int d = 0; d < device_count; ++d) {
        checkCuda(cudaSetDevice(d), "Set device for kernel launch");
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        halo_add_kernel<<<grid, BLOCK_SIZE>>>(d_arr[d], N);
        checkCuda(cudaGetLastError(), "Kernel launch");
    }

    // Copy back data to host for verification
    for (int d = 0; d < device_count; ++d) {
        checkCuda(cudaSetDevice(d), "Set device for copy back");
        checkCuda(cudaMemcpy(h_data[d], d_arr[d], (N + 2) * sizeof(float), cudaMemcpyDeviceToHost),
                  "Copying data back to host");
    }

    // Print results
    for (int d = 0; d < device_count; ++d) {
        printf("Device %d data after halo exchange and kernel:\n", d);
        for (int i = 0; i < N + 2; ++i) {
            printf("%0.2f ", h_data[d][i]);
        }
        printf("\n");
    }

    // Cleanup
    for (int d = 0; d < device_count; ++d) {
        checkCuda(cudaSetDevice(d), "Set device for cleanup");
        checkCuda(cudaFree(d_arr[d]), "Free device memory");
        checkCuda(cudaStreamDestroy(streams_left[d]), "Destroy left stream");
        checkCuda(cudaStreamDestroy(streams_right[d]), "Destroy right stream");
    }
    free(d_arr);
    free(h_data);
    free(streams_left);
    free(streams_right);

    return 0;
}
```