```cpp
/*
In a loop, allocate memory and overwrite the pointer variable in the next iteration. Explain why this is a leak.

Thought process:
- The aim is to demonstrate a classic memory leak scenario in CUDA C where memory is allocated
  repeatedly inside a loop but the pointer reference to that allocation is overwritten on the
  next iteration without freeing the previously allocated block.
- Each iteration calls cudaMalloc (or a host allocation such as malloc) and stores the returned
  pointer in the same variable. On the next iteration, that variable is assigned a new pointer,
  making the old pointer inaccessible. Since no cudaFree (or free) call is made for the old
  allocation, the memory remains allocated until program termination, leading to a leak.
- The program below allocates a small chunk of device memory in a loop of 10 iterations.
  After the loop, it frees only the last allocation. The earlier 9 allocations are lost,
  which is the memory leak.
- The code also prints the size of the allocated memory to keep the compiler happy and to
  illustrate that the allocations are happening. In a real debugging scenario, tools like
  cuda-memcheck or Nsight would detect the leak.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    const int NUM_ITERS = 10;
    const size_t ALLOC_SIZE = 1024 * sizeof(float); // 1 KB of floats

    // Pointer that will be overwritten in each loop iteration
    float *d_ptr = NULL;

    for (int i = 0; i < NUM_ITERS; ++i) {
        // Allocate memory on the device
        cudaError_t err = cudaMalloc((void **)&d_ptr, ALLOC_SIZE);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed at iteration %d: %s\n",
                    i, cudaGetErrorString(err));
            // Attempt to free any previously allocated memory before exit
            if (d_ptr != NULL) {
                cudaFree(d_ptr);
            }
            return -1;
        }

        // Just to use the pointer so that the compiler does not optimize away the call
        printf("Iteration %d: Allocated %zu bytes on device. Pointer = %p\n",
               i, ALLOC_SIZE, (void *)d_ptr);
    }

    // At this point, d_ptr points to the last allocated block.
    // The previous NUM_ITERS-1 allocations are lost because we never freed them.
    // This is a memory leak.

    // Free the last allocation (to avoid leaking that one)
    cudaFree(d_ptr);
    d_ptr = NULL;

    // Normally we would also check for errors on cudaFree, omitted here for brevity.

    printf("Done. Only the last allocation was freed.\n");
    return 0;
}
```