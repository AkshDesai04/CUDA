Here are three common asynchronous CUDA API calls:

1. **`cudaMemcpyAsync`** – Asynchronously copies data between host and device (or device to device) while the host can continue execution.
2. **Kernel launch** – Launching a CUDA kernel (e.g., `myKernel<<<grid, block, 0, stream>>>`) is asynchronous with respect to the host; it returns immediately and runs on the GPU in the specified stream.
3. **`cudaStreamSynchronize`** – While itself a synchronization call, it can be used asynchronously to wait for all operations in a particular stream to finish, allowing other streams or host code to proceed concurrently.

These calls are often combined to overlap data transfer and computation, improving overall performance.