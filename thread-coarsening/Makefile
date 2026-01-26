# Compiler
NVCC = nvcc

# Target executable
TARGET = cnn_example

# Source files
SOURCES = main.cu

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++11

# CUDA libraries
CUDA_LIBS = -lcurand

# Include directories (if needed)
INCLUDES = 

# Build target
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(CUDA_LIBS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Phony targets
.PHONY: all run clean
