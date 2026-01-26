/**
 * Multi-Config CNN with Thread Coarsening and Real X-Ray Images
 * 
 * Combines:
 * - Thread coarsening optimization for performance
 * - Multi-configuration benchmarking
 * - Real chest X-ray image loading
 */

 #define STB_IMAGE_IMPLEMENTATION
 #include "../stb_image.h"
 
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <float.h>
 #include <string.h>
 #include <dirent.h>
 #include <cuda_runtime.h>
 
 // Configuration structure
 typedef struct {
     int inputSize;
     int inputChannels;
     int kernelSize;
     int kernelCount;
     int padding;
     int stride;
     int batchSize;
     int coarseFactor;  // Thread coarsening factor
     const char *name;
 } CNNConfig;
 
 // CUDA error checking
 #define CHECK_CUDA_ERROR(call) { \
     cudaError_t err = call; \
     if (err != cudaSuccess) { \
         fprintf(stderr, "CUDA error: %s, in file '%s', line %d\n", \
                 cudaGetErrorString(err), __FILE__, __LINE__); \
         exit(EXIT_FAILURE); \
     } \
 }
 
 // Initialize convolution kernels
 void initializeKernels(float *kernels, const CNNConfig *cfg) {
     float scale = sqrtf(6.0f / (cfg->kernelSize * cfg->kernelSize * 
                                 (cfg->inputChannels + cfg->kernelCount)));
 
     for (int k = 0; k < cfg->kernelCount; k++) {
         for (int c = 0; c < cfg->inputChannels; c++) {
             for (int i = 0; i < cfg->kernelSize; i++) {
                 for (int j = 0; j < cfg->kernelSize; j++) {
                     int idx = k * cfg->inputChannels * cfg->kernelSize * cfg->kernelSize +
                               c * cfg->kernelSize * cfg->kernelSize +
                               i * cfg->kernelSize + j;
                     kernels[idx] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * scale;
                 }
             }
         }
     }
 }
 
 // Load and resize a single X-ray image
 int loadXrayImage(float *input, const char *imagePath, int batchIndex, 
                   int inputSize, int inputChannels) {
     int width, height, channels;
     unsigned char *img = stbi_load(imagePath, &width, &height, &channels, 1);
 
     if (!img) {
         fprintf(stderr, "Failed to load: %s\n", imagePath);
         return 0;
     }
 
     int offset = batchIndex * inputChannels * inputSize * inputSize;
 
     // Simple resize using nearest neighbor
     for (int y = 0; y < inputSize; y++) {
         for (int x = 0; x < inputSize; x++) {
             int src_x = (x * width) / inputSize;
             int src_y = (y * height) / inputSize;
             int src_idx = src_y * width + src_x;
             int dst_idx = offset + y * inputSize + x;
 
             // Normalize to [0, 1]
             input[dst_idx] = img[src_idx] / 255.0f;
         }
     }
 
     stbi_image_free(img);
     return 1;
 }
 
 // Comparator for qsort
 static int cmpstringp(const void *p1, const void *p2) {
     const char *a = *(const char * const *)p1;
     const char *b = *(const char * const *)p2;
     return strcmp(a, b);
 }
 
 // Load images from directory (DETERMINISTIC)
 int loadImagesFromDirectory(float *input, const char *dirPath, int maxImages, 
                             int startIdx, int inputSize, int inputChannels) {
     DIR *dir;
     struct dirent *entry;
     char *files[4096];
     int nfiles = 0;
     int count = 0;
 
     dir = opendir(dirPath);
     if (!dir) {
         fprintf(stderr, "Cannot open directory: %s\n", dirPath);
         return 0;
     }
 
     // Collect filenames
     while ((entry = readdir(dir)) != NULL) {
         if (entry->d_name[0] == '.') continue;
 
         if (strstr(entry->d_name, ".jpeg") ||
             strstr(entry->d_name, ".jpg") ||
             strstr(entry->d_name, ".png")) {
 
             if (nfiles >= (int)(sizeof(files) / sizeof(files[0]))) {
                 fprintf(stderr, "Too many files in directory (max 4096): %s\n", dirPath);
                 break;
             }
             files[nfiles++] = strdup(entry->d_name);
         }
     }
     closedir(dir);
 
     // Sort filenames alphabetically
     qsort(files, nfiles, sizeof(char *), cmpstringp);
 
     printf("  Loading images from: %s\n", dirPath);
 
     // Load first maxImages
     for (int i = 0; i < nfiles && count < maxImages; i++) {
         char fullPath[512];
         snprintf(fullPath, sizeof(fullPath), "%s/%s", dirPath, files[i]);
 
         if (loadXrayImage(input, fullPath, startIdx + count, inputSize, inputChannels)) {
             count++;
         }
     }
 
     // Free all allocated filenames
     for (int i = 0; i < nfiles; i++) {
         free(files[i]);
     }
 
     printf("    Loaded %d images\n", count);
     return count;
 }
 
 // ------------------- CUDA Kernels -------------------
 
 // Direct convolution kernel (baseline)
 __global__ void convolutionDirectKernel(
     float *input, float *kernels, float *output,
     int batchSize, int inputChannels, int inputSize,
     int kernelSize, int kernelCount, int outputSize,
     int padding, int stride) 
 {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     int k = blockIdx.z % kernelCount;
     int b = blockIdx.z / kernelCount;
 
     if (x >= outputSize || y >= outputSize || k >= kernelCount || b >= batchSize)
         return;
 
     float sum = 0.0f;
 
     for (int c = 0; c < inputChannels; c++) {
         for (int ky = 0; ky < kernelSize; ky++) {
             for (int kx = 0; kx < kernelSize; kx++) {
                 int in_x = x * stride - padding + kx;
                 int in_y = y * stride - padding + ky;
 
                 if (in_x >= 0 && in_x < inputSize && in_y >= 0 && in_y < inputSize) {
                     float in_val = input[
                         b * inputChannels * inputSize * inputSize +
                         c * inputSize * inputSize +
                         in_y * inputSize + in_x
                     ];
 
                     float kernel_val = kernels[
                         k * inputChannels * kernelSize * kernelSize +
                         c * kernelSize * kernelSize +
                         ky * kernelSize + kx
                     ];
 
                     sum += in_val * kernel_val;
                 }
             }
         }
     }
 
     output[
         b * kernelCount * outputSize * outputSize +
         k * outputSize * outputSize +
         y * outputSize + x
     ] = sum;
 }
 
 // Shared memory optimized convolution kernel WITH thread coarsening
 __global__ void convolutionSharedKernel(
     float *input, float *kernels, float *output,
     int batchSize, int inputChannels, int inputSize,
     int kernelSize, int kernelCount, int outputSize,
     int padding, int stride, int coarseFactor) 
 {
     extern __shared__ float sharedData[];
 
     int tileSize = blockDim.x * coarseFactor;
     int tileSizeWithPadding = tileSize + kernelSize - 1;
 
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int bx = blockIdx.x;
     int by = blockIdx.y;
     int k = blockIdx.z % kernelCount;
     int b = blockIdx.z / kernelCount;
 
     int base_out_x = (bx * blockDim.x + tx) * coarseFactor;
     int base_out_y = (by * blockDim.y + ty) * coarseFactor;
 
     int in_x_base = bx * blockDim.x * coarseFactor * stride - padding;
     int in_y_base = by * blockDim.y * coarseFactor * stride - padding;
 
     float *sharedInput = sharedData;
 
     // Collaborative loading of shared memory
     int totalElements = tileSizeWithPadding * tileSizeWithPadding;
     int threadsPerBlock = blockDim.x * blockDim.y;
     int flatThreadIdx = ty * blockDim.x + tx;
 
     for (int c = 0; c < inputChannels; c++) {
         for (int i = flatThreadIdx; i < totalElements; i += threadsPerBlock) {
             int tile_y = i / tileSizeWithPadding;
             int tile_x = i % tileSizeWithPadding;
 
             int in_y = in_y_base + tile_y;
             int in_x = in_x_base + tile_x;
 
             float value = 0.0f;
             if (in_y >= 0 && in_y < inputSize && in_x >= 0 && in_x < inputSize) {
                 value = input[
                     b * inputChannels * inputSize * inputSize +
                     c * inputSize * inputSize +
                     in_y * inputSize + in_x
                 ];
             }
 
             sharedInput[
                 c * tileSizeWithPadding * tileSizeWithPadding +
                 tile_y * tileSizeWithPadding + tile_x
             ] = value;
         }
     }
 
     __syncthreads();
 
     // Each thread computes coarseFactor x coarseFactor outputs
     if (b < batchSize) {
         for (int dy = 0; dy < coarseFactor; dy++) {
             for (int dx = 0; dx < coarseFactor; dx++) {
                 int out_x = base_out_x + dx;
                 int out_y = base_out_y + dy;
 
                 if (out_x < outputSize && out_y < outputSize) {
                     float sum = 0.0f;
 
                     int tile_x = tx * coarseFactor + dx;
                     int tile_y = ty * coarseFactor + dy;
 
                     for (int c = 0; c < inputChannels; c++) {
                         for (int ky = 0; ky < kernelSize; ky++) {
                             for (int kx = 0; kx < kernelSize; kx++) {
                                 int shared_y = tile_y * stride + ky;
                                 int shared_x = tile_x * stride + kx;
 
                                 float in_val = sharedInput[
                                     c * tileSizeWithPadding * tileSizeWithPadding +
                                     shared_y * tileSizeWithPadding + shared_x
                                 ];
 
                                 float kernel_val = kernels[
                                     k * inputChannels * kernelSize * kernelSize +
                                     c * kernelSize * kernelSize +
                                     ky * kernelSize + kx
                                 ];
 
                                 sum += in_val * kernel_val;
                             }
                         }
                     }
 
                     output[
                         b * kernelCount * outputSize * outputSize +
                         k * outputSize * outputSize +
                         out_y * outputSize + out_x
                     ] = sum;
                 }
             }
         }
     }
 }
 
 // ReLU activation kernel
 __global__ void reluActivationKernel(float *data, int size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
 }
 
 // Max pooling kernel
 __global__ void maxPoolingKernel(
     float *input, float *output,
     int batchSize, int channels, int inputSize,
     int poolSize, int outputSize, int stride)
 {
     int out_x = blockIdx.x * blockDim.x + threadIdx.x;
     int out_y = blockIdx.y * blockDim.y + threadIdx.y;
     int c = blockIdx.z % channels;
     int b = blockIdx.z / channels;
 
     if (out_x >= outputSize || out_y >= outputSize || c >= channels || b >= batchSize)
         return;
 
     int in_x_base = out_x * stride;
     int in_y_base = out_y * stride;
 
     float maxVal = -FLT_MAX;
 
     for (int dy = 0; dy < poolSize; dy++) {
         for (int dx = 0; dx < poolSize; dx++) {
             int in_y = in_y_base + dy;
             int in_x = in_x_base + dx;
 
             if (in_y < inputSize && in_x < inputSize) {
                 float value = input[
                     b * channels * inputSize * inputSize +
                     c * inputSize * inputSize +
                     in_y * inputSize + in_x
                 ];
                 maxVal = fmaxf(maxVal, value);
             }
         }
     }
 
     output[
         b * channels * outputSize * outputSize +
         c * outputSize * outputSize +
         out_y * outputSize + out_x
     ] = maxVal;
 }
 
 // Forward pass
 void forwardCNN(
     float *d_input,
     float *d_kernels,
     float *d_conv_output,
     float *d_activation,
     float *d_pooling_output,
     const CNNConfig *cfg,
     bool useSharedMemory,
     float *timing)
 {
     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
 
     int outputSize = (cfg->inputSize + 2*cfg->padding - cfg->kernelSize) / cfg->stride + 1;
 
     // Convolution
     cudaEventRecord(start);
 
     if (useSharedMemory) {
         dim3 blockDim(8, 8, 1);
         dim3 gridDim(
             (outputSize + blockDim.x * cfg->coarseFactor - 1) / (blockDim.x * cfg->coarseFactor),
             (outputSize + blockDim.y * cfg->coarseFactor - 1) / (blockDim.y * cfg->coarseFactor),
             cfg->kernelCount * cfg->batchSize
         );
 
         int tileSize = blockDim.x * cfg->coarseFactor;
         int tileSizeWithPadding = tileSize + cfg->kernelSize - 1;
         int sharedMemSize = cfg->inputChannels * tileSizeWithPadding * tileSizeWithPadding * sizeof(float);
 
         convolutionSharedKernel<<<gridDim, blockDim, sharedMemSize>>>(
             d_input, d_kernels, d_conv_output,
             cfg->batchSize, cfg->inputChannels, cfg->inputSize,
             cfg->kernelSize, cfg->kernelCount, outputSize,
             cfg->padding, cfg->stride, cfg->coarseFactor
         );
     } else {
         dim3 blockDim(8, 8, 1);
         dim3 gridDim(
             (outputSize + blockDim.x - 1) / blockDim.x,
             (outputSize + blockDim.y - 1) / blockDim.y,
             cfg->kernelCount * cfg->batchSize
         );
 
         convolutionDirectKernel<<<gridDim, blockDim>>>(
             d_input, d_kernels, d_conv_output,
             cfg->batchSize, cfg->inputChannels, cfg->inputSize,
             cfg->kernelSize, cfg->kernelCount, outputSize,
             cfg->padding, cfg->stride
         );
     }
 
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&timing[0], start, stop);
 
     // ReLU
     int totalElements = cfg->batchSize * cfg->kernelCount * outputSize * outputSize;
     int blockSize = 256;
     int gridSize = (totalElements + blockSize - 1) / blockSize;
 
     CHECK_CUDA_ERROR(cudaMemcpy(d_activation, d_conv_output, totalElements * sizeof(float), cudaMemcpyDeviceToDevice));
 
     cudaEventRecord(start);
     reluActivationKernel<<<gridSize, blockSize>>>(d_activation, totalElements);
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&timing[1], start, stop);
 
     // Max Pooling
     int poolSize = 2;
     int poolStride = 2;
     int poolOutputSize = outputSize / poolStride;
 
     dim3 poolBlockDim(8, 8);
     dim3 poolGridDim(
         (poolOutputSize + poolBlockDim.x - 1) / poolBlockDim.x,
         (poolOutputSize + poolBlockDim.y - 1) / poolBlockDim.y,
         cfg->batchSize * cfg->kernelCount
     );
 
     cudaEventRecord(start);
     maxPoolingKernel<<<poolGridDim, poolBlockDim>>>(
         d_activation, d_pooling_output,
         cfg->batchSize, cfg->kernelCount, outputSize,
         poolSize, poolOutputSize, poolStride
     );
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&timing[2], start, stop);
 
     cudaEventDestroy(start);
     cudaEventDestroy(stop);
 }
 
 // Run single configuration
 void runConfiguration(const CNNConfig *cfg) {
     printf("\n========================================\n");
     printf("CONFIGURATION: %s\n", cfg->name);
     printf("========================================\n");
     printf("Input Size: %dx%d\n", cfg->inputSize, cfg->inputSize);
     printf("Kernel: %dx%d, Count: %d\n", cfg->kernelSize, cfg->kernelSize, cfg->kernelCount);
     printf("Batch Size: %d\n", cfg->batchSize);
     printf("Padding: %d, Stride: %d\n", cfg->padding, cfg->stride);
     printf("Thread Coarsening Factor: %d\n\n", cfg->coarseFactor);
 
     int outputSize = (cfg->inputSize + 2*cfg->padding - cfg->kernelSize) / cfg->stride + 1;
     int poolOutputSize = outputSize / 2;
 
     // Allocate host memory
     size_t inputBytes = cfg->batchSize * cfg->inputChannels * cfg->inputSize * cfg->inputSize * sizeof(float);
     size_t kernelBytes = cfg->kernelCount * cfg->inputChannels * cfg->kernelSize * cfg->kernelSize * sizeof(float);
     size_t outputBytes = cfg->batchSize * cfg->kernelCount * outputSize * outputSize * sizeof(float);
     size_t poolBytes = cfg->batchSize * cfg->kernelCount * poolOutputSize * poolOutputSize * sizeof(float);
 
     float *h_input = (float*)malloc(inputBytes);
     float *h_kernels = (float*)malloc(kernelBytes);
     float *h_output_direct = (float*)malloc(outputBytes);
     float *h_output_shared = (float*)malloc(outputBytes);
     float *h_pooling_output = (float*)malloc(poolBytes);
 
     if (!h_input || !h_kernels || !h_output_direct || !h_output_shared || !h_pooling_output) {
         fprintf(stderr, "Host malloc failed for config: %s\n", cfg->name);
         free(h_input);
         free(h_kernels);
         free(h_output_direct);
         free(h_output_shared);
         free(h_pooling_output);
         return;
     }
 
     // Load real X-ray images
     printf("Loading chest X-ray images...\n");
     int normalCount = loadImagesFromDirectory(
         h_input,
         "../data/chest_xray/train/NORMAL",
         cfg->batchSize / 2,
         0,
         cfg->inputSize,
         cfg->inputChannels
     );
 
     int pneumoniaCount = loadImagesFromDirectory(
         h_input,
         "../data/chest_xray/train/PNEUMONIA",
         cfg->batchSize / 2,
         cfg->batchSize / 2,
         cfg->inputSize,
         cfg->inputChannels
     );
 
     printf("Loaded: %d NORMAL + %d PNEUMONIA = %d total\n\n",
            normalCount, pneumoniaCount, normalCount + pneumoniaCount);
 
     // Initialize kernels
     initializeKernels(h_kernels, cfg);
 
     // Allocate device memory
     float *d_input, *d_kernels;
     float *d_conv_output_direct, *d_activation_direct, *d_pooling_output_direct;
     float *d_conv_output_shared, *d_activation_shared, *d_pooling_output_shared;
 
     CHECK_CUDA_ERROR(cudaMalloc(&d_input, inputBytes));
     CHECK_CUDA_ERROR(cudaMalloc(&d_kernels, kernelBytes));
     CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output_direct, outputBytes));
     CHECK_CUDA_ERROR(cudaMalloc(&d_activation_direct, outputBytes));
     CHECK_CUDA_ERROR(cudaMalloc(&d_pooling_output_direct, poolBytes));
     CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output_shared, outputBytes));
     CHECK_CUDA_ERROR(cudaMalloc(&d_activation_shared, outputBytes));
     CHECK_CUDA_ERROR(cudaMalloc(&d_pooling_output_shared, poolBytes));
 
     // Copy to device
     CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice));
     CHECK_CUDA_ERROR(cudaMemcpy(d_kernels, h_kernels, kernelBytes, cudaMemcpyHostToDevice));
 
     // Run direct convolution
     printf("Running direct convolution...\n");
     float direct_timing[3] = {0};
     forwardCNN(d_input, d_kernels, d_conv_output_direct, d_activation_direct, 
                d_pooling_output_direct, cfg, false, direct_timing);
 
     // Run shared memory + thread coarsening
     printf("Running shared memory with thread coarsening...\n");
     float shared_timing[3] = {0};
     forwardCNN(d_input, d_kernels, d_conv_output_shared, d_activation_shared,
                d_pooling_output_shared, cfg, true, shared_timing);
 
     // Copy results back
     CHECK_CUDA_ERROR(cudaMemcpy(h_output_direct, d_conv_output_direct, outputBytes, cudaMemcpyDeviceToHost));
     CHECK_CUDA_ERROR(cudaMemcpy(h_output_shared, d_conv_output_shared, outputBytes, cudaMemcpyDeviceToHost));
     CHECK_CUDA_ERROR(cudaMemcpy(h_pooling_output, d_pooling_output_shared, poolBytes, cudaMemcpyDeviceToHost));
 
     // Verify results
     float maxDiff = 0.0f;
     for (size_t i = 0; i < cfg->batchSize * cfg->kernelCount * outputSize * outputSize; i++) {
         float diff = fabs(h_output_direct[i] - h_output_shared[i]);
         maxDiff = fmaxf(maxDiff, diff);
     }
 
     // Save feature maps
     char filename[256];
     snprintf(filename, sizeof(filename), "pooled_feature_maps_%s.txt", cfg->name);
     FILE *fp = fopen(filename, "w");
     if (fp) {
         int batch0 = 0;
         int numChannelsToSave = (cfg->kernelCount < 4) ? cfg->kernelCount : 4;
 
         for (int k = 0; k < numChannelsToSave; k++) {
             fprintf(fp, "# batch=%d channel=%d\n", batch0, k);
 
             for (int y = 0; y < poolOutputSize; y++) {
                 for (int x = 0; x < poolOutputSize; x++) {
                     int idx = batch0 * cfg->kernelCount * poolOutputSize * poolOutputSize +
                               k * poolOutputSize * poolOutputSize +
                               y * poolOutputSize + x;
                     fprintf(fp, "%.6f ", h_pooling_output[idx]);
                 }
                 fprintf(fp, "\n");
             }
             fprintf(fp, "\n");
         }
         fclose(fp);
         printf("Feature maps saved to %s\n", filename);
     }
 
     // Print results
     printf("\n=== PERFORMANCE RESULTS ===\n\n");
     printf("Performance comparison:\n");
     printf("  Direct Convolution: %.3f ms\n", direct_timing[0]);
     printf("  Shared Memory + Thread Coarsening: %.3f ms\n", shared_timing[0]);
     printf("  Speedup: %.2fx\n\n", direct_timing[0] / shared_timing[0]);
 
     printf("Layer timings (Shared Memory + Thread Coarsening):\n");
     printf("  Convolution: %.3f ms\n", shared_timing[0]);
     printf("  ReLU Activation: %.3f ms\n", shared_timing[1]);
     printf("  Max Pooling: %.3f ms\n", shared_timing[2]);
     printf("  Total: %.3f ms\n\n", shared_timing[0] + shared_timing[1] + shared_timing[2]);
 
     printf("Verification:\n");
     printf("  Max difference: %e\n", maxDiff);
     printf("  Results match: %s\n\n", (maxDiff < 1e-4) ? "YES" : "NO");
 
     // Calculate throughput
     float totalTimeSeconds = (shared_timing[0] + shared_timing[1] + shared_timing[2]) / 1000.0f;
     float imagesPerSecond = cfg->batchSize / totalTimeSeconds;
     printf("Throughput: %.2f images/second\n", imagesPerSecond);
 
     printf("\nSample feature map (first image, first channel, top-left 5x5):\n");
     int displaySize = (outputSize < 5) ? outputSize : 5;
     for (int y = 0; y < displaySize; y++) {
         for (int x = 0; x < displaySize; x++) {
             printf("%.3f ", h_output_shared[y * outputSize + x]);
         }
         printf("\n");
     }
 
     // Free memory
     free(h_input);
     free(h_kernels);
     free(h_output_direct);
     free(h_output_shared);
     free(h_pooling_output);
 
     CHECK_CUDA_ERROR(cudaFree(d_input));
     CHECK_CUDA_ERROR(cudaFree(d_kernels));
     CHECK_CUDA_ERROR(cudaFree(d_conv_output_direct));
     CHECK_CUDA_ERROR(cudaFree(d_activation_direct));
     CHECK_CUDA_ERROR(cudaFree(d_pooling_output_direct));
     CHECK_CUDA_ERROR(cudaFree(d_conv_output_shared));
     CHECK_CUDA_ERROR(cudaFree(d_activation_shared));
     CHECK_CUDA_ERROR(cudaFree(d_pooling_output_shared));
 
     printf("\n✓ Configuration complete!\n");
 }
 
 int main() {
     srand(42);
 
     printf("=== Multi-Configuration CNN with Thread Coarsening ===\n");
     printf("Testing three configurations with real chest X-ray images\n\n");
 
     // Define three configurations
     CNNConfig configs[3] = {
         // Config 1: Small (MNIST-like)
         {
             .inputSize = 32,
             .inputChannels = 1,
             .kernelSize = 3,
             .kernelCount = 16,
             .padding = 1,
             .stride = 1,
             .batchSize = 64,
             .coarseFactor = 2,
             .name = "Small"
         },
         // Config 2: Medium (ResNet Standard)
         {
             .inputSize = 256,
             .inputChannels = 1,
             .kernelSize = 3,
             .kernelCount = 16,
             .padding = 1,
             .stride = 1,
             .batchSize = 64,
             .coarseFactor = 2,
             .name = "Medium"
         },
         // Config 3: Large (High-Res X-Ray)
         {
             .inputSize = 1024,
             .inputChannels = 1,
             .kernelSize = 7,
             .kernelCount = 16,
             .padding = 3,
             .stride = 1,
             .batchSize = 64,
             .coarseFactor = 2,
             .name = "Large"
         }
     };
 
     // Run all three configurations
     for (int i = 0; i < 3; i++) {
         runConfiguration(&configs[i]);
     }
 
     printf("\n========================================\n");
     printf("ALL CONFIGURATIONS COMPLETE\n");
     printf("========================================\n");
 
     return 0;
 }
 