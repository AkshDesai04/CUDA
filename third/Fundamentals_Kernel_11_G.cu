When you launch a kernel you normally do this:

1. Allocate a structure on the host.  
2. Allocate memory on the device (or use a device‑pointer that already points to device memory).  
3. Copy the host structure to the device (`cudaMemcpy`).  
4. Pass the device pointer to the kernel.  
5. After the kernel finishes, copy the structure back from the device to the host (`cudaMemcpy`).  
6. Verify the results on the host.

If you **omit step 5** (the copy back), the host copy of the structure never sees the changes made by the kernel. The host memory remains exactly as it was after step 3, so any comparison between the expected modified values and the actual host values will fail.

Only in special cases does the host see the updated data without an explicit `cudaMemcpy`:

* **Unified (managed) memory** – the same memory object is accessible on both host and device, so modifications by the kernel are immediately visible to the host (subject to synchronization).
* **Device pointer used directly on the host** – if the host somehow has a pointer to device memory (e.g., a device pointer that is also valid on the host because of zero‑copy memory or a mapped array), reading from it on the host will give the updated values. But this is not the usual “allocate on host, copy to device” pattern.

Under the typical scenario you described—allocating a struct on the host, copying it to device, modifying it in the kernel, and then forgetting to copy it back—the verification on the host will **not** pass. The host will still contain the original values.