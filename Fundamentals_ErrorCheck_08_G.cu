No.  
`cudaPeekAtLastError()` only reports errors that occur **at launch time** (e.g. an invalid grid/block size, bad device pointer passed to the kernel, etc.). It does not wait for the kernel to run, so it will not catch errors that happen **inside the kernel** (such as an out‑of‑bounds global‑memory read/write).  

To detect such in‑kernel errors you must either:

```cpp
kernel<<<grid,block>>>(...);   // launch
cudaDeviceSynchronize();       // wait for completion
cudaError_t err = cudaGetLastError();  // or cudaPeekAtLastError()
```

or use `cudaGetLastError()` after a synchronous kernel call. In short, `cudaPeekAtLastError()` right after a launch cannot catch in‑kernel memory errors.