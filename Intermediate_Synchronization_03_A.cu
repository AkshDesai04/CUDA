The “incorrect increment” you’re describing is a classic example of a **read‑modify‑write race** that occurs when multiple threads update a shared variable without proper synchronization. The sequence unfolds as follows:

1. **All threads read the old value**  
   Each thread independently loads the current value of the shared variable from memory into its own register or local cache. At this point every thread is working with the same snapshot of the data, even though the value may be about to change.

2. **All threads compute the new value**  
   Using the value it just read, each thread performs its arithmetic operation (e.g., `new_val = old_val + 1`). The result is stored in a temporary local variable or register. Importantly, none of the threads have yet written this new value back to the shared memory.

3. **All threads write their new value**  
   Finally, each thread writes its locally computed result back to the shared memory location. Since the writes are unordered relative to each other, the order in which the threads commit their results is nondeterministic.

Because each thread is writing *its own* computed value, the write from the last thread to reach that memory location wins, overwriting any previously written values. The net effect is that only a single increment (or possibly none, if the last write happens to be a no‑op) takes place, and the intended collective update (e.g., a sum of all increments) is lost.

This pattern is why atomic operations, locks, or other synchronization mechanisms are required whenever multiple threads must update the same memory location in a coordinated way.