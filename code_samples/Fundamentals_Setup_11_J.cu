```makefile
# Simple Makefile for CUDA project
# Compile the main.cu file using nvcc and produce an executable named "main".
# Target: all (default)
# Clean up generated files with: make clean

# Compiler and flags
CC      = nvcc
CFLAGS  = -arch=sm_61  # Adjust GPU architecture as needed
TARGET  = main
SRCS    = main.cu

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Clean up
clean:
	rm -f $(TARGET)
```