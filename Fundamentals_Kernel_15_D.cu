```cpp
/*
 * Aim: 
 * Write a kernel that has a very long but finite loop. Call it without cudaDeviceSynchronize. 
 * The host program finishes almost instantly. Now add cudaDeviceSynchronize. The host program now waits for the kernel to complete.
 *
 * Thinking process:
 * 1. The core demonstration is to show that launching a kernel is asynchronous by default; the host thread returns immediately after the launch, even if the kernel is still running on the device. 
 * 2. To make the effect noticeable, the kernel must perform a long computation that takes a measurable amount of time. A simple way is to run a large loop inside the kernel. Since the loop does nothing useful, we store the loop counter in a volatile variable or in shared memory to prevent the compiler from optimizing it away.
 * 3. We'll use a single thread (or a few threads) to keep the demonstration simple. One thread performing a loop of, say, 1e9 iterations should take a few seconds on a typical GPU.
 * 4. The host program will first launch this kernel *without* calling cudaDeviceSynchronize(). It will then print a message and exit. We will time the host program to show that it ends quickly.
 * 5. Next, we will launch the same kernel again *with* cudaDeviceSynchronize() after the launch. The host program will block until the kernel finishes. We will measure the time to confirm the delay.
 * 6. For clarity, both parts are included in one executable: the first launch is asynchronous, the second is synchronous. We use simple prints to indicate which phase is running. Optionally, you could comment/uncomment the synchronize call to observe the behavior yourself.
 *
 * Implementation details:
 * - The kernel `longLoopKernel` runs a loop from 0 to a large constant. The loop counter is stored in a `volatile` local variable to prevent optimization.
 * - The kernel is launched with 1 block and 1 thread (configurable if desired). 
 * - We use `std::chrono` on the host to measure elapsed time for each phase.
 * - No error checking macros are used for brevity; in production code you should check the return status of each CUDA call.
 */

#include <cstdio>
#include <chrono>
#include <iostream>

__global__ void longLoopKernel(unsigned long long loopCount, volatile unsigned long long* dummy)
{
    // Each thread runs a long loop; store result in dummy to avoid optimization.
    unsigned long long sum = 0;
    for (unsigned long long i = 0; i < loopCount; ++i)
    {
        sum += i;
    }
    *dummy = sum;
}

int main()
{
    const unsigned long long LOOP_COUNT = 1000000000ULL; // 1e9 iterations (~seconds depending on GPU)
    volatile unsigned long long* d_dummy;
    cudaMalloc(&d_dummy, sizeof(volatile unsigned long long));

    // ---------- Asynchronous launch ----------
    std::cout << "Launching kernel asynchronously (no cudaDeviceSynchronize)...\n";
    auto asyncStart = std::chrono::high_resolution_clock::now();

    longLoopKernel<<<1, 1>>>(LOOP_COUNT, d_dummy);

    auto asyncAfterLaunch = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> asyncElapsed = asyncAfterLaunch - asyncStart;
    std::cout << "Kernel launched. Host time elapsed after launch: " << asyncElapsed.count() << " s\n";
    std::cout << "Host program ends here (async part).\n\n";

    // Small delay to separate outputs
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // ---------- Synchronous launch ----------
    std::cout << "Launching kernel synchronously (with cudaDeviceSynchronize)...\n";
    auto syncStart = std::chrono::high_resolution_clock::now();

    longLoopKernel<<<1, 1>>>(LOOP_COUNT, d_dummy);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    auto syncAfterSync = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> syncElapsed = syncAfterSync - syncStart;
    std::cout << "Kernel finished. Host time elapsed including synchronization: " << syncElapsed.count() << " s\n";

    // Clean up
    cudaFree((void*)d_dummy);
    return 0;
}
```