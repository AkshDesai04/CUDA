When a CUDA program exits, the CUDA driver automatically destroys all contexts that were created for that process.  
When a context is destroyed the driver frees **all** memory that was allocated in that context, including any “leaked” device pointers that the program never freed.  
In other words, the leaked memory is reclaimed automatically when the process terminates.

You can confirm this with `nvidia-smi`:

1. **Check memory before launching the program**  
   ```bash
   $ nvidia-smi
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 525.125.02    Driver Version: 525.125.02    CUDA Version: 12.2 |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                               |                      |               MIG M. |
   |===============================+======================+======================|
   |   0  GeForce RTX 3080    Off | 0000:01:00.0 Off |                  N/A |
   | 30%   32C    P8   13W / 350W |      0MiB / 10018MiB |      0%      Default |
   +-----------------------------------------------------------------------------+
   ```
   Here the memory usage is 0 MiB.

2. **Run a program that intentionally leaks device memory**  
   ```bash
   $ ./leak_program
   ```
   While it runs, `nvidia-smi` will show a higher Memory-Usage, e.g. 2000 MiB.

3. **After the program exits**  
   ```bash
   $ nvidia-smi
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 525.125.02    Driver Version: 525.125.02    CUDA Version: 12.2 |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                               |                      |               MIG M. |
   |===============================+======================+======================|
   |   0  GeForce RTX 3080    Off | 0000:01:00.0 Off |                  N/A |
   | 30%   32C    P8   13W / 350W |      0MiB / 10018MiB |      0%      Default |
   +-----------------------------------------------------------------------------+
   ```
   The memory usage has returned to 0 MiB, indicating that the leaked memory was automatically freed when the process ended.

**Bottom line:**  
CUDA automatically releases all allocated device memory when a process terminates, so you do not need to manually free leaked memory at program exit. However, it is still good practice to free memory explicitly to avoid leaks during long‑running or multi‑context applications.