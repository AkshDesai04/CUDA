When you launch a kernel in CUDA you pass *pointers* to device memory, not the actual data.  
If you declare a structure on the host, allocate a device pointer for it, and then call

```cpp
myKernel<<<blocks, threads>>>(deviceStructPtr);
```

but **do not** first copy the host structure into `deviceStructPtr` (e.g. with `cudaMemcpy`), the kernel will read the contents of whatever bytes happen to be at that device address.

That memory region on the device is uninitialized unless you explicitly zero it or set it in some other way. Consequently the kernel will see **indeterminate values** (often garbage) for the struct members. It may read whatever happened to be there from a previous allocation or kernel launch, or if the device memory happens to have been zeroed by the runtime it could read zeros. In any case, the values are not the ones you intended to pass from the host. This leads to undefined behavior and can cause crashes or incorrect results.