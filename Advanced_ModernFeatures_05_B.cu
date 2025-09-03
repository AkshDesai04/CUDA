```cuda
/********************************************************************
 * Aim of the program: Demonstrate that after Process A exits,
 * the memory it shared via IPC is no longer valid for Process B to access.
 *
 * Thinking process:
 * 1. CUDA IPC uses cudaIpcGetMemHandle to obtain a handle to a device
 *    allocation that can be shared with other processes. The handle
 *    can be transferred via a file or a socket.
 * 2. The documentation states that if the process that created the
 *    IPC handle terminates, the memory becomes invalid for other
 *    processes.  We will verify this by:
 *    - Process A (parent) allocates device memory, writes a value,
 *      obtains an IPC handle, writes the handle to a temporary file,
 *      then sleeps for a while before exiting.
 *    - Process B (child) reads the handle from the file, opens the
 *      memory with cudaIpcOpenMemHandle, copies the value to host
 *      and prints it (success). After Process A exits, Process B
 *      attempts to copy again; we expect this to fail with an error.
 * 3. Implementation details:
 *    - We use a single executable that runs in two modes: parent
 *      (default) and child (argv[1] == "child").
 *    - The parent spawns the child via fork+exec.  They communicate
 *      the IPC handle through a temporary file "/tmp/ipc_handle.bin".
 *    - Synchronization is achieved with simple sleep calls for
 *      demonstration purposes.
 *    - Error checking is performed after each CUDA call; on failure,
 *      we print the CUDA error string and exit.
 * 4. Expected output (approximate):
 *    Process A: allocated device memory and wrote IPC handle.
 *    Process A: sleeping 5s before exit.
 *    Process B: read IPC handle and opened memory.
 *    Process B: read value before parent exit: 42
 *    Process B: sleeping 3s (parent exited).
 *    Process B: attempted to read after parent exit: failed with error.
 *
 * This program is self-contained and can be compiled with nvcc:
 *   nvcc -o ipc_demo ipc_demo.cu
 * Run:
 *   ./ipc_demo
 ********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/* Temporary file path for sharing the IPC handle */
#define HANDLE_FILE "/tmp/ipc_handle.bin"

/* Function to check CUDA errors */
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Write the IPC handle to a file */
void write_handle_to_file(const cudaIpcMemHandle_t *handle) {
    int fd = open(HANDLE_FILE, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("Failed to open handle file for writing");
        exit(EXIT_FAILURE);
    }
    ssize_t written = write(fd, handle, sizeof(cudaIpcMemHandle_t));
    if (written != sizeof(cudaIpcMemHandle_t)) {
        perror("Failed to write complete handle to file");
        close(fd);
        exit(EXIT_FAILURE);
    }
    close(fd);
}

/* Read the IPC handle from a file */
void read_handle_from_file(cudaIpcMemHandle_t *handle) {
    int fd = open(HANDLE_FILE, O_RDONLY);
    if (fd < 0) {
        perror("Failed to open handle file for reading");
        exit(EXIT_FAILURE);
    }
    ssize_t read_bytes = read(fd, handle, sizeof(cudaIpcMemHandle_t));
    if (read_bytes != sizeof(cudaIpcMemHandle_t)) {
        perror("Failed to read complete handle from file");
        close(fd);
        exit(EXIT_FAILURE);
    }
    close(fd);
}

/* Parent process (Process A) */
void run_parent() {
    int *dev_ptr = NULL;
    int host_val = 42;

    /* Allocate device memory */
    checkCudaError(cudaMalloc((void**)&dev_ptr, sizeof(int)), "cudaMalloc");

    /* Copy a value to device */
    checkCudaError(cudaMemcpy(dev_ptr, &host_val, sizeof(int), cudaMemcpyHostToDevice),
                   "cudaMemcpy host->device");

    /* Obtain IPC handle */
    cudaIpcMemHandle_t handle;
    checkCudaError(cudaIpcGetMemHandle(&handle, dev_ptr), "cudaIpcGetMemHandle");

    /* Write handle to file for child to read */
    write_handle_to_file(&handle);
    printf("Process A: allocated device memory and wrote IPC handle to %s.\n", HANDLE_FILE);

    /* Sleep to allow child to read before exit */
    printf("Process A: sleeping 5 seconds before exiting.\n");
    sleep(5);

    /* Clean up (optional) */
    cudaFree(dev_ptr);

    /* Remove the handle file */
    unlink(HANDLE_FILE);

    printf("Process A: exiting.\n");
    exit(EXIT_SUCCESS);
}

/* Child process (Process B) */
void run_child() {
    /* Allow parent to write the handle */
    sleep(1);

    /* Read handle from file */
    cudaIpcMemHandle_t handle;
    read_handle_from_file(&handle);
    printf("Process B: read IPC handle from %s.\n", HANDLE_FILE);

    /* Open the IPC memory handle */
    int *dev_ptr = NULL;
    checkCudaError(cudaIpcOpenMemHandle((void**)&dev_ptr, handle, cudaIpcMemLazyEnablePeerAccess),
                   "cudaIpcOpenMemHandle");

    /* Copy value from device to host */
    int host_val = 0;
    checkCudaError(cudaMemcpy(&host_val, dev_ptr, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy device->host before parent exit");
    printf("Process B: read value before parent exit: %d\n", host_val);

    /* Sleep to let parent exit */
    printf("Process B: sleeping 3 seconds (parent should exit during this time).\n");
    sleep(3);

    /* Attempt to copy again after parent exit */
    int host_val2 = 0;
    cudaError_t err = cudaMemcpy(&host_val2, dev_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Process B: failed to copy after parent exit: %s\n",
                cudaGetErrorString(err));
    } else {
        printf("Process B: read value after parent exit: %d (unexpectedly succeeded)\n", host_val2);
    }

    /* Close the IPC handle */
    checkCudaError(cudaIpcCloseMemHandle(dev_ptr), "cudaIpcCloseMemHandle");

    printf("Process B: exiting.\n");
    exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "child") == 0) {
        /* Child mode */
        run_child();
    } else {
        /* Parent mode */
        /* Spawn child process */
        pid_t pid = fork();
        if (pid < 0) {
            perror("Failed to fork");
            exit(EXIT_FAILURE);
        } else if (pid == 0) {
            /* In child process: exec same binary with "child" argument */
            execlp(argv[0], argv[0], "child", (char *)NULL);
            perror("Failed to exec child");
            exit(EXIT_FAILURE);
        } else {
            /* In parent process */
            run_parent();
        }
    }
    return 0;
}
```