/*
Because kernel launches are asynchronous, the CPU can continue working on other tasks while the GPU is busy.
When a CUDA kernel is launched, the call returns immediately; the CPU thread does not wait for the GPU to finish executing the kernel.
Instead, the kernel is queued in the GPU's execution stream, and the CPU can proceed to perform other computations, I/O, or launch additional kernels.
Only when the program explicitly synchronizes (e.g., with cudaDeviceSynchronize(), cudaStreamSynchronize(), or by copying data back from the device) does the CPU block until the GPU has completed its work.
This asynchronous behavior enables overlapping of CPU and GPU workloads, improving overall application performance when designed correctly. 
*/
