```cpp
/*
Aim of the program: Launch with a 1D block of 64 threads. Have each thread print its lane ID within its warp (`threadIdx.x % 32`).

Thinking:
1. The goal is to demonstrate the concept of a lane ID, which is the thread's index within its warp. Since a warp consists of 32 threads, the lane ID is simply `threadIdx.x % 32`.
2. We need a 1D block of 64 threads. That means the grid has one block and each block has 64 threads: `<<<1, 64>>>`.
3. Each thread will compute its lane ID and print it using `printf` inside the kernel. CUDA allows device-side `printf` on compute capability 2.0+.
4. The kernel will be very simple: compute `int laneId = threadIdx.x % 32;` and then `printf("Thread %d lane %d\n", threadIdx.x, laneId);`.
5. In `main()`, we launch the kernel, then call `cudaDeviceSynchronize()` to ensure the device has finished before the program exits.
6. Include `<stdio.h>` for `printf`. The code is minimal and self‑contained, with no error checking to keep it concise.
7. No additional headers or utilities are needed. The program compiles with `nvcc` and runs on any CUDA capable GPU that supports device `printf`.

End of thinking.
*/

#include <stdio.h>

__global__ void print_lane_id() {
    int lane_id = threadIdx.x % 32;
    printf("Thread %d lane %d\n", threadIdx.x, lane_id);
}

int main() {
    // Launch a single block of 64 threads
    print_lane_id<<<1, 64>>>();

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    return 0;
}
```