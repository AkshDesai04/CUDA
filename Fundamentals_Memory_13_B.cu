/*
Write a loop that keeps allocating 1GB chunks of memory until `cudaMalloc` fails. Print how many GB were successfully allocated before failure.

Thinking:
- Use cudaMalloc to allocate 1GB each iteration.
- Keep a counter.
- Stop when cudaMalloc returns an error other than cudaSuccess.
- Print the counter value (GB allocated) after the loop.
- We don't free each allocation; the memory stays allocated until program exit, which is fine for this test.
- Use size_t chunkSize = 1ULL << 30 for 1GB.
- Include <stdio.h> and <cuda_runtime.h>.
- Implement main() returning int.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    const size_t chunkSize = 1ULL << 30; // 1GB
    void *ptr = NULL;
    int count = 0;
    cudaError_t err;

    while (true) {
        err = cudaMalloc(&ptr, chunkSize);
        if (err != cudaSuccess) {
            // Allocation failed; break out of the loop
            break;
        }
        count++;
        // We intentionally do not free the allocation to keep the memory allocated.
        // The pointer is overwritten each iteration, but the allocated memory remains allocated.
    }

    printf("Successfully allocated %d GB before failure.\n", count);

    return 0;
}
