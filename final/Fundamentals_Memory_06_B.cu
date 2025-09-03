```cuda
/*
 * Does the leaked memory get freed when the application exits? Use `nvidia-smi` to confirm.
 *
 * Thinking:
 *   The goal of this program is to demonstrate a deliberate CUDA memory leak
 *   and allow the user to verify whether the memory is reclaimed when the
 *   process terminates.  We will:
 *     1. Allocate a large chunk of device memory (e.g., 1 GB) using cudaMalloc.
 *     2. Optionally perform a trivial kernel launch to ensure the allocation
 *        is truly used (although not strictly necessary for the leak).
 *     3. Intentionally omit cudaFree on the allocation.
 *     4. Print a message indicating that the program is about to exit.
 *     5. Exit the program normally.
 *
 *   When the program terminates, the CUDA driver should automatically
 *   free any resources associated with the process.  To confirm this,
 *   the user can run `nvidia-smi` before starting the program and again
 *   after it has exited to see whether the device memory usage has
 *   returned to its previous level.
 *
 *   In the code below, we also use cudaMemGetInfo to print the available
 *   memory before and after allocation to give the user some immediate
 *   feedback.  However, this feedback is only from the program itself;
 *   the definitive confirmation comes from `nvidia-smi`.
 *
 *   Note: If you run this on a system with multiple GPUs, the allocation
 *   will occur on the default device (device 0).  Feel free to set a
 *   different device with cudaSetDevice if desired.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    size_t free_before, total_before;
    cudaError_t err;

    // Query and print available memory before allocation
    err = cudaMemGetInfo(&free_before, &total_before);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed before allocation: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Before allocation: free=%zu bytes, total=%zu bytes\n", free_before, total_before);

    // Allocate 1 GB of device memory
    size_t alloc_size = 1ULL * 1024 * 1024 * 1024; // 1 GB
    void *d_ptr = NULL;
    err = cudaMalloc(&d_ptr, alloc_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Allocated 1 GB of device memory at %p\n", d_ptr);

    // Optionally launch a trivial kernel to touch the memory
    // Here we skip the kernel launch because it is not required for the leak demonstration.

    // Query and print available memory after allocation
    size_t free_after, total_after;
    err = cudaMemGetInfo(&free_after, &total_after);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed after allocation: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("After allocation: free=%zu bytes, total=%zu bytes\n", free_after, total_after);

    // Intentionally omit cudaFree(d_ptr) to create a memory leak

    // Inform the user that the program is about to exit
    printf("Program exiting without freeing device memory. Use nvidia-smi to confirm memory is reclaimed.\n");

    // Exiting without cudaDeviceReset or cudaFree
    return 0;
}
```