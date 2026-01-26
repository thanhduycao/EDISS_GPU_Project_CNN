# Compiler
NVCC = nvcc

# Target executable
TARGET = shared_cnn

# Source files
SOURCES = shared_cnn.cu

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++11

# CUDA libraries (removed curand - not needed)
CUDA_LIBS = 

# Include directories
INCLUDES = -I.

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