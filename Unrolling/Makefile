# Compiler
NVCC = nvcc

# Target executable name
TARGET = xray_unrolled

# Source files
# Change this to match your actual filename
SOURCES = xray_unrolled.cu

# Compiler flags
# -O3: Max optimization
# -arch=sm_80: Target A100 GPU (Ampere)
# -I..: Look in parent directory for header files (stb_image.h)
# -std=c++11: Standard C++ support
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++11 -I..

# CUDA libraries
CUDA_LIBS = -lcurand

# Build target
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET) $(CUDA_LIBS)

# Run the program 
# WARNING: Only run 'make run' inside an interactive GPU session (sinteractive).
# Do not run this on the login node.
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
# Added cleanup for the .txt map files so you can start fresh
clean:
	rm -f $(TARGET) *.o pooled_feature_maps_*.txt

# Phony targets
.PHONY: all run clean