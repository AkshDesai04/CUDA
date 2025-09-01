/*
Aim of the program: Copy only a portion of `d_a` to `d_b` using pointer arithmetic.

My plan for this program is to demonstrate how to copy a contiguous sub‑array from one device buffer to another
by using pointer arithmetic in a CUDA kernel.  The host side will set up a source array `h_a` with known
contents, allocate two device buffers `d_a` and `d_b`, and copy `h_a` to `d_a`.  The kernel `copyPortion` receives
the device pointers, an offset into `d_a`, and the number of elements to copy.  Inside the kernel, I compute
the source address with `d_a + offset + idx` and the destination address simply with `d_b + idx`.  The
kernel launches enough threads to cover the requested length and copies each element with a simple assignment.

After the kernel finishes, the host copies the contents of `d_b` back to a host array `h_b` and prints the
original source array, the portion that was copied, and the destination array to confirm the operation
completed correctly.  The program includes a simple error‑checking macro `CUDA_CHECK` for clarity.  The code
is fully self‑contained and ready to be compiled with `nvcc`.