```cuda
/*
 * Aim of the program:
 * Demonstrate that after Process A exits, the memory it shared via IPC is no longer valid for Process B to access.
 *
 * Thinking process:
 * 1. CUDA IPC allows two separate processes to share device memory.  Process A allocates a device array,
 *    writes data into it, obtains an IPC handle with cudaIpcGetMemHandle, and writes the 64‑byte handle
 *    to a file that Process B can read.
 *
 * 2. Process B opens the file, reads the handle, maps the shared memory with cudaIpcOpenMemHandle,
 *    and copies the data back to the host for verification.  This shows that IPC works while both
 *    processes are running.
 *
 * 3. To demonstrate that the memory becomes invalid after Process A exits, Process B will wait for
 *    a short period (so the user can terminate Process A manually or allow the program to exit)
 *    and then attempt to copy the memory again.  Because the driver cleans up the allocation when
 *    Process A terminates, the second cudaMemcpy should fail with an error such as
 *    cudaErrorInvalidDevicePointer or cudaErrorInvalidResourceHandle.  The program prints the
 *    error code and message to show that the memory is no longer valid.
 *
 * 4. The code is written as a single .cu file.  Compile it with:
 *        nvcc -o ipc_demo ipc_demo.cu
 *    Run it twice:
 *        ./ipc_demo A    // Process A
 *        ./ipc_demo B    // Process B
 *
 * 5. Notes:
 *    - The IPC handle is stored in a simple binary file "ipc_handle.bin".
 *    - Process B sleeps for 5 seconds before attempting the second copy to allow Process A
 *      to exit.  If Process A exits earlier, the second copy will immediately fail.
 *    - Error checking is performed after each CUDA API call.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>      // for sleep()
#include <errno.h>
#include <sys/stat.h>    // for open()
#include <fcntl.h>       // for open()
#include <sys/types.h>

#include <cuda_runtime.h>

// Macro for error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel to initialize array with values 0..N-1
__global__ void init_array(int *arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) arr[idx] = idx;
}

// Function for Process A
void processA()
{
    const int N = 256;
    int *d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    // Launch kernel to fill array
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    init_array<<<blocks, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get IPC handle
    cudaIpcMemHandle_t ipc_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handle, d_arr));

    // Write handle to file
    int fd = open("ipc_handle.bin", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        exit(EXIT_FAILURE);
    }
    ssize_t written = write(fd, &ipc_handle, sizeof(ipc_handle));
    if (written != sizeof(ipc_handle)) {
        perror("write");
        close(fd);
        exit(EXIT_FAILURE);
    }
    close(fd);
    printf("[Process A] IPC handle written to ipc_handle.bin.  Exiting in 10 seconds...\n");
    // Wait to allow Process B to read
    sleep(10);
    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    printf("[Process A] Exiting.\n");
}

// Function for Process B
void processB()
{
    // Read IPC handle from file
    int fd = open("ipc_handle.bin", O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(EXIT_FAILURE);
    }
    cudaIpcMemHandle_t ipc_handle;
    ssize_t read_bytes = read(fd, &ipc_handle, sizeof(ipc_handle));
    close(fd);
    if (read_bytes != sizeof(ipc_handle)) {
        fprintf(stderr, "Failed to read complete IPC handle from file.\n");
        exit(EXIT_FAILURE);
    }

    // Open the IPC handle
    int *d_shared = NULL;
    cudaError_t err = cudaIpcOpenMemHandle((void **)&d_shared, ipc_handle,
                                           cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaIpcOpenMemHandle failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("[Process B] IPC handle opened.  Accessing data...\n");

    // Copy data to host
    const int N = 256;
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(h_arr, d_shared, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "First cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("[Process B] First read successful.  Sample values:\n");
    for (int i = 0; i < 10; i++) {
        printf("  h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Wait for Process A to exit
    printf("[Process B] Sleeping 5 seconds to allow Process A to exit...\n");
    sleep(5);

    // Attempt to read again after Process A has exited
    printf("[Process B] Attempting second read after Process A exit...\n");
    err = cudaMemcpy(h_arr, d_shared, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[Process B] Second cudaMemcpy failed as expected: %s (Error code %d)\n",
               cudaGetErrorString(err), err);
    } else {
        printf("[Process B] Second cudaMemcpy succeeded unexpectedly.  Sample values:\n");
        for (int i = 0; i < 10; i++) {
            printf("  h_arr[%d] = %d\n", i, h_arr[i]);
        }
    }

    // Clean up
    free(h_arr);
    CUDA_CHECK(cudaIpcCloseMemHandle(d_shared));
    printf("[Process B] Exiting.\n");
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [A|B]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "A") == 0) {
        processA();
    } else if (strcmp(argv[1], "B") == 0) {
        processB();
    } else {
        fprintf(stderr, "Invalid argument. Use 'A' for Process A or 'B' for Process B.\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```