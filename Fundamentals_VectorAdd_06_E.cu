A fused multiply‑add (FMA) is a single instruction that performs a multiplication and an addition in one go, producing the exact mathematical result of  
**(a × b) + c** with only one rounding step. In IEEE‑754 floating‑point arithmetic this means the product a × b is computed to infinite precision, then c is added, and only the final sum is rounded to the machine format.  
This has two key advantages:

1. **Higher accuracy** – By rounding only once the error introduced by separate multiply and add operations is avoided.  
2. **Performance** – Most modern CPUs and GPUs expose an FMA instruction that executes faster and with lower latency than a separate multiply followed by an add.

The SAXPY operation (Single‑precision A X Plus Y) is defined as

```
Y[i] = a * X[i] + Y[i]
```

for every element *i*.  Each iteration performs exactly the pattern that an FMA implements: one scalar *a* multiplied by a vector element *X[i]*, then added to the current value of *Y[i]*.  Because the same scalar *a* is reused for all elements, the cost of loading it is negligible compared with the per‑element computation.  Using an FMA allows SAXPY to compute the product and sum in a single fused instruction, gaining both speed and improved numerical precision compared with a separate `mul` followed by an `add`.