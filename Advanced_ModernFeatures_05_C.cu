```cuda
/* 
 * Aim of the program (exact wording from prompt):
 * Use IPC to implement a simple client-server model where a server process manages a large dataset on the GPU, and multiple client processes can attach to its memory to perform read-only computations.
 *
 * My thinking while designing this program:
 * 1. The goal is to demonstrate GPU IPC (inter‑process communication) in CUDA, where one process (the server) allocates a big array on the GPU, writes a handle for it, and other processes (the clients) attach to that memory and perform read‑only operations.
 * 2. CUDA provides `cudaIpcGetMemHandle` to obtain a `cudaIpcMemHandle_t` that uniquely identifies a GPU allocation. Clients can then call `cudaIpcOpenMemHandle` to map the same GPU memory into their address space.
 * 3. For simplicity we store the handle in a plain text file (`ipc_handle.txt`). In a real system one might use shared memory, sockets, or a database. Writing to a file keeps the code portable and easy to understand.
 * 4. The dataset is a large 1‑dimensional float array. The server initializes it with a simple pattern (value equals its index) using a CUDA kernel. This eliminates the need for a host array copy and keeps the data entirely on the GPU.
 * 5. Clients read the handle, map the memory, and run a simple reduction kernel to compute the sum of all elements. The reduction uses a block‑wise partial sum and then a final CPU accumulation. The result is printed to stdout. This demonstrates that multiple clients can attach to the same memory and perform independent computations.
 * 6. No write access is allowed on the client side, and we pass `cudaIpcMemLazyEnablePeerAccess` to `cudaIpcOpenMemHandle` to allow lazy enabling of peer access if necessary.
 * 7. Error checking is provided via a macro `CHECK_CUDA` to make the code concise.
 * 8. The program can be compiled as a server (`nvcc ipc.cu -D SERVER`) or a client (`nvcc ipc.cu -D CLIENT`). The two modes are isolated with `#ifdef SERVER` / `#ifdef CLIENT` sections.
 * 9. The handle file size is fixed at 32 bytes (size of `cudaIpcMemHandle_t`), and we simply write/read the raw binary handle.
 * 10. Because the handle is binary, the file is opened in binary mode to avoid newline translation issues.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>

/* Helper macro for error checking */
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* Size of the dataset: 1M floats (~4 MB) */
#define N (1 << 20)

/* Path to the file that will contain the IPC handle */
#define HANDLE_FILE "ipc_handle.bin"

/* =======================
 * Server code
 * =======================
 */
#ifdef SERVER

/* Kernel to fill array with its index value */
__global__ void initArray(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = (float)idx;
    }
}

/* Simple reduction kernel to sum a portion of the array */
__global__ void sumKernel(const float *arr, float *partialSums, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) sum += arr[idx];
    if (idx + blockDim.x < n) sum += arr[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partialSums[blockIdx.x] = sdata[0];
}

int main(void) {
    /* Allocate GPU memory */
    float *d_array = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_array, N * sizeof(float)));

    /* Initialize array */
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    initArray<<<blocks, threads>>>(d_array, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Get IPC handle */
    cudaIpcMemHandle_t handle;
    CHECK_CUDA(cudaIpcGetMemHandle(&handle, d_array));

    /* Write handle to file */
    FILE *fp = fopen(HANDLE_FILE, "wb");
    if (!fp) {
        perror("Failed to open handle file for writing");
        exit(EXIT_FAILURE);
    }
    size_t written = fwrite(&handle, 1, sizeof(handle), fp);
    if (written != sizeof(handle)) {
        perror("Failed to write complete handle");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    fclose(fp);
    printf("Server: IPC handle written to %s\n", HANDLE_FILE);

    /* Keep server alive to allow clients to attach */
    printf("Server: Press Enter to exit...\n");
    getchar();

    /* Clean up */
    CHECK_CUDA(cudaFree(d_array));
    return 0;
}

#endif /* SERVER */

/* =======================
 * Client code
 * =======================
 */
#ifdef CLIENT

/* Kernel to compute sum of array elements */
__global__ void sumKernel(const float *arr, float *partialSums, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) sum += arr[idx];
    if (idx + blockDim.x < n) sum += arr[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partialSums[blockIdx.x] = sdata[0];
}

int main(void) {
    /* Read handle from file */
    FILE *fp = fopen(HANDLE_FILE, "rb");
    if (!fp) {
        perror("Failed to open handle file for reading");
        exit(EXIT_FAILURE);
    }
    cudaIpcMemHandle_t handle;
    size_t read = fread(&handle, 1, sizeof(handle), fp);
    fclose(fp);
    if (read != sizeof(handle)) {
        fprintf(stderr, "Failed to read complete IPC handle\n");
        exit(EXIT_FAILURE);
    }
    printf("Client: Read IPC handle from %s\n", HANDLE_FILE);

    /* Open the shared device memory */
    float *d_array = NULL;
    CHECK_CUDA(cudaIpcOpenMemHandle((void**)&d_array, handle,
                                    cudaIpcMemLazyEnablePeerAccess));

    /* Allocate memory for partial sums on device */
    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    float *d_partial = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_partial, blocks * sizeof(float)));

    /* Launch reduction kernel */
    size_t sharedMemSize = threads * sizeof(float);
    sumKernel<<<blocks, threads, sharedMemSize>>>(d_array, d_partial, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy partial sums back to host and accumulate */
    float *h_partial = (float*)malloc(blocks * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += h_partial[i];
    printf("Client: Sum of array elements = %f\n", (float)sum);

    /* Clean up */
    free(h_partial);
    CHECK_CUDA(cudaFree(d_partial));
    CHECK_CUDA(cudaIpcCloseMemHandle(d_array));

    return 0;
}

#endif /* CLIENT */
```