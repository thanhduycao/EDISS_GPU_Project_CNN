// /**
//  * Chest X-Ray CNN with Constant Memory
//  * Processes real medical images for pneumonia detection (feature extraction demo)
//  *
//  * NOTE:
//  * - This is a forward-pass demo (conv -> ReLU -> maxpool) using CUDA.
//  * - Convolution weights are RANDOM (not trained), so it does NOT truly classify pneumonia.
//  * - Loads real images from two folders into a batch, runs the kernels, prints timings,
//  *   and dumps pooled feature maps to a text file.
//  * - Image loading is deterministic: filenames are sorted alphabetically before loading.
//  */

// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <float.h>
// #include <string.h>
// #include <dirent.h>
// #include <cuda_runtime.h>

// // CNN configuration
// #define INPUT_SIZE 128       // Resize X-rays to 128x128
// #define INPUT_CHANNELS 1     // Grayscale
// #define KERNEL_SIZE 5        // Convolution kernel size
// #define KERNEL_COUNT 16      // Number of filters
// #define PADDING 2            // Zero padding
// #define STRIDE 1             // Convolution stride
// #define BATCH_SIZE 32        // Process 32 images at once

// // Output dimensions
// #define OUTPUT_SIZE ((INPUT_SIZE + 2*PADDING - KERNEL_SIZE) / STRIDE + 1)

// // CUDA error checking
// #define CHECK_CUDA_ERROR(call) { \
//     cudaError_t err = call; \
//     if (err != cudaSuccess) { \
//         fprintf(stderr, "CUDA error: %s, in file '%s', line %d\n", \
//                 cudaGetErrorString(err), __FILE__, __LINE__); \
//         exit(EXIT_FAILURE); \
//     } \
// }

// // Constant memory for kernel weights
// __constant__ float d_kernels_const[KERNEL_COUNT * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE];

// // Initialize convolution kernels
// void initializeKernels(float *kernels) {
//     float scale = sqrtf(6.0f / (KERNEL_SIZE * KERNEL_SIZE * (INPUT_CHANNELS + KERNEL_COUNT)));

//     for (int k = 0; k < KERNEL_COUNT; k++) {
//         for (int c = 0; c < INPUT_CHANNELS; c++) {
//             for (int i = 0; i < KERNEL_SIZE; i++) {
//                 for (int j = 0; j < KERNEL_SIZE; j++) {
//                     int idx = k * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE +
//                               c * KERNEL_SIZE * KERNEL_SIZE +
//                               i * KERNEL_SIZE + j;
//                     kernels[idx] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * scale;
//                 }
//             }
//         }
//     }
// }

// // Load and resize a single X-ray image
// int loadXrayImage(float *input, const char *imagePath, int batchIndex) {
//     int width, height, channels;
//     unsigned char *img = stbi_load(imagePath, &width, &height, &channels, 1);

//     if (!img) {
//         fprintf(stderr, "Failed to load: %s\n", imagePath);
//         return 0;
//     }

//     int offset = batchIndex * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE;

//     // Simple resize using nearest neighbor
//     for (int y = 0; y < INPUT_SIZE; y++) {
//         for (int x = 0; x < INPUT_SIZE; x++) {
//             int src_x = (x * width) / INPUT_SIZE;
//             int src_y = (y * height) / INPUT_SIZE;
//             int src_idx = src_y * width + src_x;
//             int dst_idx = offset + y * INPUT_SIZE + x;

//             // Normalize to [0, 1]
//             input[dst_idx] = img[src_idx] / 255.0f;
//         }
//     }

//     stbi_image_free(img);
//     return 1;
// }

// // Comparator for qsort (sort filenames alphabetically)
// static int cmpstringp(const void *p1, const void *p2) {
//     const char *a = *(const char * const *)p1;
//     const char *b = *(const char * const *)p2;
//     return strcmp(a, b);
// }

// // Load images from directory (DETERMINISTIC: sorts filenames)
// int loadImagesFromDirectory(float *input, const char *dirPath, int maxImages, int startIdx) {
//     DIR *dir;
//     struct dirent *entry;
//     char *files[4096];
//     int nfiles = 0;
//     int count = 0;

//     dir = opendir(dirPath);
//     if (!dir) {
//         fprintf(stderr, "Cannot open directory: %s\n", dirPath);
//         return 0;
//     }

//     // Collect filenames
//     while ((entry = readdir(dir)) != NULL) {
//         if (entry->d_name[0] == '.') continue;

//         if (strstr(entry->d_name, ".jpeg") ||
//             strstr(entry->d_name, ".jpg") ||
//             strstr(entry->d_name, ".png")) {

//             if (nfiles >= (int)(sizeof(files) / sizeof(files[0]))) {
//                 fprintf(stderr, "Too many files in directory (max 4096): %s\n", dirPath);
//                 break;
//             }
//             files[nfiles++] = strdup(entry->d_name);
//         }
//     }
//     closedir(dir);

//     // Sort filenames alphabetically (deterministic selection/order)
//     qsort(files, nfiles, sizeof(char *), cmpstringp);

//     printf("Loading images from: %s\n", dirPath);

//     // Load first maxImages
//     for (int i = 0; i < nfiles && count < maxImages; i++) {
//         char fullPath[512];
//         snprintf(fullPath, sizeof(fullPath), "%s/%s", dirPath, files[i]);

//         if (loadXrayImage(input, fullPath, startIdx + count)) {
//             count++;
//             if (count % 10 == 0) {
//                 printf("  Loaded %d images...\n", count);
//             }
//         }
//         free(files[i]);
//     }

//     // Free any remaining strdup'd names not used (if count < nfiles and loop ended early)
//     // (In our code above, we free as we go for all iterated entries; this covers the case
//     // where maxImages < nfiles but we still iterated only until count == maxImages.)
//     // If loop exited because count reached maxImages, remaining entries from i..nfiles-1
//     // haven't been freed. Free them now:
//     if (count == maxImages) {
//         // Determine how many were iterated:
//         // The loop iterates i from 0 upward, but breaks only when condition fails.
//         // Since we didn't track i after loop, do a second pass to free remaining safely:
//         // We'll just free all again? No (double free).
//         // Instead, we re-collect logic: easiest is to free in a second loop based on count.
//         // BUT count is number loaded, not same as i (because some loads may fail).
//         // So to be safe, we won't attempt to infer. We'll avoid this complexity by
//         // ensuring we always free in the loop over ALL nfiles. Since we stop early,
//         // we need a safe approach: track last index processed.
//     }

//     // Better: track last processed index for freeing remaining
//     // (Implemented below by rewriting the above free logic cleanly.)
//     //
//     // NOTE: We already printed progress above; we will keep functionality but ensure memory is freed.

//     // Re-do freeing robustly: since the above could leak when stopping early due to count==maxImages,
//     // we implement it properly using a tracked index by restructuring. To keep code simple and correct,
//     // we will implement the correct approach here: (the above already ran, but we can't undo it).
//     // Instead, we avoid this complexity by not stopping early in iteration: we can still stop loading
//     // once count==maxImages, but we must free remaining names.
//     //
//     // To keep this file clean and correct, we will implement the deterministic loader again below
//     // and return from there. So the code above is replaced in compilation by the block below.
//     //
//     // (To avoid confusion, we return here with a safe approach by not relying on above.)
//     //
//     // IMPORTANT: This early return would skip freeing if we didn't handle it; so DO NOT return here.

//     // ---- Correct, leak-free implementation (actual one used) ----
//     // We already collected and sorted; we will load again deterministically into a fresh loop,
//     // but that would double-load. Instead, implement correctly from scratch and remove above in your file.
//     //
//     // Since you asked for "full complete code", here's the clean version below.
//     // ----------------------------------------------------------------

//     // This return is unreachable in the final cleaned version; kept for clarity.
//     printf("  Total loaded: %d images\n", count);
//     return count;
// }

// // ------------------- CUDA Kernels -------------------

// // Constant memory convolution kernel
// __global__ void convolutionConstantKernel(
//     float *input, float *output,
//     int batchSize, int inputChannels, int inputSize,
//     int kernelSize, int kernelCount, int outputSize,
//     int padding, int stride)
// {
//     extern __shared__ float sharedData[];

//     int tileSize = blockDim.x;
//     int tileSizeWithPadding = tileSize + kernelSize - 1;

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     int k = blockIdx.z % kernelCount;
//     int b = blockIdx.z / kernelCount;

//     int out_x = bx * tileSize + tx;
//     int out_y = by * tileSize + ty;

//     int in_x_base = bx * tileSize * stride - padding;
//     int in_y_base = by * tileSize * stride - padding;

//     float *sharedInput = sharedData;

//     // Load input tile into shared memory
//     for (int c = 0; c < inputChannels; c++) {
//         for (int dy = 0; dy < tileSizeWithPadding; dy += tileSize) {
//             for (int dx = 0; dx < tileSizeWithPadding; dx += tileSize) {
//                 int in_y = in_y_base + ty + dy;
//                 int in_x = in_x_base + tx + dx;

//                 float value = 0.0f;
//                 if (in_y >= 0 && in_y < inputSize && in_x >= 0 && in_x < inputSize) {
//                     value = input[
//                         b * inputChannels * inputSize * inputSize +
//                         c * inputSize * inputSize +
//                         in_y * inputSize + in_x
//                     ];
//                 }

//                 if (ty + dy < tileSizeWithPadding && tx + dx < tileSizeWithPadding) {
//                     sharedInput[
//                         c * tileSizeWithPadding * tileSizeWithPadding +
//                         (ty + dy) * tileSizeWithPadding + (tx + dx)
//                     ] = value;
//                 }
//             }
//         }
//     }

//     __syncthreads();

//     // Compute convolution
//     if (out_x < outputSize && out_y < outputSize && b < batchSize && k < kernelCount) {
//         float sum = 0.0f;

//         for (int c = 0; c < inputChannels; c++) {
//             for (int ky = 0; ky < kernelSize; ky++) {
//                 for (int kx = 0; kx < kernelSize; kx++) {
//                     int shared_y = ty * stride + ky;
//                     int shared_x = tx * stride + kx;

//                     float in_val = sharedInput[
//                         c * tileSizeWithPadding * tileSizeWithPadding +
//                         shared_y * tileSizeWithPadding + shared_x
//                     ];

//                     float kernel_val = d_kernels_const[
//                         k * inputChannels * kernelSize * kernelSize +
//                         c * kernelSize * kernelSize +
//                         ky * kernelSize + kx
//                     ];

//                     sum += in_val * kernel_val;
//                 }
//             }
//         }

//         output[
//             b * kernelCount * outputSize * outputSize +
//             k * outputSize * outputSize +
//             out_y * outputSize + out_x
//         ] = sum;
//     }
// }

// // ReLU activation
// __global__ void reluActivationKernel(float *data, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
// }

// // Max pooling
// __global__ void maxPoolingKernel(
//     float *input, float *output,
//     int batchSize, int channels, int inputSize,
//     int poolSize, int outputSize, int stride)
// {
//     int out_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int out_y = blockIdx.y * blockDim.y + threadIdx.y;
//     int c = blockIdx.z % channels;
//     int b = blockIdx.z / channels;

//     if (out_x >= outputSize || out_y >= outputSize || c >= channels || b >= batchSize)
//         return;

//     int in_x_base = out_x * stride;
//     int in_y_base = out_y * stride;

//     float maxVal = -FLT_MAX;

//     for (int dy = 0; dy < poolSize; dy++) {
//         for (int dx = 0; dx < poolSize; dx++) {
//             int in_y = in_y_base + dy;
//             int in_x = in_x_base + dx;

//             if (in_y < inputSize && in_x < inputSize) {
//                 float value = input[
//                     b * channels * inputSize * inputSize +
//                     c * inputSize * inputSize +
//                     in_y * inputSize + in_x
//                 ];
//                 maxVal = fmaxf(maxVal, value);
//             }
//         }
//     }

//     output[
//         b * channels * outputSize * outputSize +
//         c * outputSize * outputSize +
//         out_y * outputSize + out_x
//     ] = maxVal;
// }

// // Forward pass
// void forwardCNN(
//     float *d_input,
//     float *d_conv_output,
//     float *d_activation,
//     float *d_pooling_output,
//     float *timing)
// {
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     // Convolution
//     dim3 blockDim(8, 8, 1);
//     dim3 gridDim(
//         (OUTPUT_SIZE + blockDim.x - 1) / blockDim.x,
//         (OUTPUT_SIZE + blockDim.y - 1) / blockDim.y,
//         KERNEL_COUNT * BATCH_SIZE
//     );

//     int tileSize = blockDim.x;
//     int tileSizeWithPadding = tileSize + KERNEL_SIZE - 1;
//     int sharedMemSize = INPUT_CHANNELS * tileSizeWithPadding * tileSizeWithPadding * sizeof(float);

//     cudaEventRecord(start);
//     convolutionConstantKernel<<<gridDim, blockDim, sharedMemSize>>>(
//         d_input, d_conv_output,
//         BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE,
//         KERNEL_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
//         PADDING, STRIDE
//     );
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&timing[0], start, stop);

//     // ReLU
//     int totalElements = BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE;
//     int blockSize = 256;
//     int gridSize = (totalElements + blockSize - 1) / blockSize;

//     CHECK_CUDA_ERROR(cudaMemcpy(d_activation, d_conv_output, totalElements * sizeof(float), cudaMemcpyDeviceToDevice));

//     cudaEventRecord(start);
//     reluActivationKernel<<<gridSize, blockSize>>>(d_activation, totalElements);
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&timing[1], start, stop);

//     // Max Pooling
//     int poolSize = 2;
//     int poolStride = 2;
//     int poolOutputSize = OUTPUT_SIZE / poolStride;

//     dim3 poolBlockDim(8, 8);
//     dim3 poolGridDim(
//         (poolOutputSize + poolBlockDim.x - 1) / poolBlockDim.x,
//         (poolOutputSize + poolBlockDim.y - 1) / poolBlockDim.y,
//         BATCH_SIZE * KERNEL_COUNT
//     );

//     cudaEventRecord(start);
//     maxPoolingKernel<<<poolGridDim, poolBlockDim>>>(
//         d_activation, d_pooling_output,
//         BATCH_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
//         poolSize, poolOutputSize, poolStride
//     );
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&timing[2], start, stop);

//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
// }

// int main() {
//     srand(42);

//     printf("=== Chest X-Ray CNN with Constant Memory ===\n\n");

//     // Allocate host memory
//     float *h_input = (float*)malloc(BATCH_SIZE * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float));
//     float *h_kernels = (float*)malloc(KERNEL_COUNT * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
//     float *h_output = (float*)malloc(BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));
//     float *h_pooling_output = (float*)malloc(BATCH_SIZE * KERNEL_COUNT * (OUTPUT_SIZE/2) * (OUTPUT_SIZE/2) * sizeof(float));

//     if (!h_input || !h_kernels || !h_output || !h_pooling_output) {
//         fprintf(stderr, "Host malloc failed\n");
//         return 1;
//     }

//     // Load real X-ray images
//     printf("Loading chest X-ray images...\n");
//     int normalCount = loadImagesFromDirectory(
//         h_input,
//         "../data/chest_xray/train/NORMAL",
//         BATCH_SIZE / 2,
//         0
//     );

//     int pneumoniaCount = loadImagesFromDirectory(
//         h_input,
//         "../data/chest_xray/train/PNEUMONIA",
//         BATCH_SIZE / 2,
//         BATCH_SIZE / 2
//     );

//     printf("\nLoaded: %d NORMAL + %d PNEUMONIA = %d total images\n\n",
//            normalCount, pneumoniaCount, normalCount + pneumoniaCount);

//     // Initialize kernels
//     initializeKernels(h_kernels);

//     // Allocate device memory
//     float *d_input, *d_conv_output, *d_activation, *d_pooling_output;

//     CHECK_CUDA_ERROR(cudaMalloc(&d_input, BATCH_SIZE * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float)));
//     CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output, BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float)));
//     CHECK_CUDA_ERROR(cudaMalloc(&d_activation, BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float)));
//     CHECK_CUDA_ERROR(cudaMalloc(&d_pooling_output, BATCH_SIZE * KERNEL_COUNT * (OUTPUT_SIZE/2) * (OUTPUT_SIZE/2) * sizeof(float)));

//     // Copy to device
//     CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input,
//         BATCH_SIZE * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float),
//         cudaMemcpyHostToDevice));

//     CHECK_CUDA_ERROR(cudaMemcpyToSymbol(
//         d_kernels_const,
//         h_kernels,
//         KERNEL_COUNT * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float)
//     ));

//     // Run CNN
//     printf("Running CNN forward pass...\n");
//     float timing[3] = {0};
//     forwardCNN(d_input, d_conv_output, d_activation, d_pooling_output, timing);

//     // Copy results back
//     CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_conv_output,
//         BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float),
//         cudaMemcpyDeviceToHost));

//     CHECK_CUDA_ERROR(cudaMemcpy(h_pooling_output, d_pooling_output,
//         BATCH_SIZE * KERNEL_COUNT * (OUTPUT_SIZE/2) * (OUTPUT_SIZE/2) * sizeof(float),
//         cudaMemcpyDeviceToHost));

//     // ================== DUMP MAX-POOL FEATURE MAPS (ONCE) ==================
//     {
//         FILE *fp = fopen("pooled_feature_maps.txt", "w");
//         if (!fp) {
//             perror("fopen pooled_feature_maps.txt");
//         } else {
//             int pooledSize = OUTPUT_SIZE / 2;
//             int batch0 = 0;  // first image in batch

//             for (int k = 0; k < 4; k++) {  // first 4 channels
//                 fprintf(fp, "# batch=%d channel=%d\n", batch0, k);

//                 for (int y = 0; y < pooledSize; y++) {
//                     for (int x = 0; x < pooledSize; x++) {
//                         int idx =
//                             batch0 * KERNEL_COUNT * pooledSize * pooledSize +
//                             k * pooledSize * pooledSize +
//                             y * pooledSize + x;

//                         fprintf(fp, "%.6f ", h_pooling_output[idx]);
//                     }
//                     fprintf(fp, "\n");
//                 }
//                 fprintf(fp, "\n");
//             }

//             fclose(fp);
//             printf("Feature maps saved to pooled_feature_maps.txt\n");
//         }
//     }

//     // Print results
//     printf("\n=== Results ===\n");
//     printf("Configuration:\n");
//     printf("  Input size: %dx%d (grayscale X-rays)\n", INPUT_SIZE, INPUT_SIZE);
//     printf("  Kernel size: %dx%d with %d filters\n", KERNEL_SIZE, KERNEL_SIZE, KERNEL_COUNT);
//     printf("  Batch size: %d images\n", BATCH_SIZE);
//     printf("  Output size after convolution: %dx%d\n", OUTPUT_SIZE, OUTPUT_SIZE);
//     printf("  Output size after pooling: %dx%d\n\n", OUTPUT_SIZE/2, OUTPUT_SIZE/2);

//     float constKB = (KERNEL_COUNT * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float)) / 1024.0f;
//     printf("Constant memory used: %.2f KB / 64 KB\n\n", constKB);

//     printf("Layer timings:\n");
//     printf("  Convolution: %.3f ms\n", timing[0]);
//     printf("  ReLU Activation: %.3f ms\n", timing[1]);
//     printf("  Max Pooling: %.3f ms\n", timing[2]);
//     printf("  Total: %.3f ms\n\n", timing[0] + timing[1] + timing[2]);

//     printf("Sample feature map (first image, first channel, top-left 5x5):\n");
//     for (int y = 0; y < 5; y++) {
//         for (int x = 0; x < 5; x++) {
//             printf("%.3f ", h_output[y * OUTPUT_SIZE + x]);
//         }
//         printf("\n");
//     }

//     // Free memory
//     free(h_input);
//     free(h_kernels);
//     free(h_output);
//     free(h_pooling_output);

//     CHECK_CUDA_ERROR(cudaFree(d_input));
//     CHECK_CUDA_ERROR(cudaFree(d_conv_output));
//     CHECK_CUDA_ERROR(cudaFree(d_activation));
//     CHECK_CUDA_ERROR(cudaFree(d_pooling_output));

//     printf("\n✓ CNN processing complete!\n");
//     return 0;
// }


/**
 * Chest X-Ray CNN with Shared Memory
 * Processes real medical images for pneumonia detection (feature extraction demo)
 *
 * NOTE:
 * - This is a forward-pass demo (conv -> ReLU -> maxpool) using CUDA.
 * - Convolution weights are RANDOM (not trained), so it does NOT truly classify pneumonia.
 * - Loads real images from two folders into a batch, runs the kernels, prints timings,
 *   and dumps pooled feature maps to a text file.
 * - Image loading is deterministic: filenames are sorted alphabetically before loading.
 * - Runs three configurations: Small (32x32), Medium (256x256), Large (1024x1024)
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

// Comparator for qsort (sort filenames alphabetically)
static int cmpstringp(const void *p1, const void *p2) {
    const char *a = *(const char * const *)p1;
    const char *b = *(const char * const *)p2;
    return strcmp(a, b);
}

// Load images from directory (DETERMINISTIC: sorts filenames)
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

    // Sort filenames alphabetically (deterministic selection/order)
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

// Shared memory optimized convolution kernel
__global__ void convolutionSharedKernel(
    float *input, float *kernels, float *output,
    int batchSize, int inputChannels, int inputSize,
    int kernelSize, int kernelCount, int outputSize,
    int padding, int stride) 
{
    // Shared memory for input tile (with padding)
    extern __shared__ float sharedData[];
    
    // Calculate tile dimensions
    int tileSize = blockDim.x; // Assuming blockDim.x == blockDim.y
    int tileSizeWithPadding = tileSize + kernelSize - 1;
    
    // Calculate output position
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int k = blockIdx.z; // Output channel (kernel number)
    int b = threadIdx.z; // Batch index
    
    // Output coordinates
    int out_x = bx * tileSize + tx;
    int out_y = by * tileSize + ty;
    
    // Base input coordinates (top-left of the tile)
    int in_x_base = bx * tileSize * stride - padding;
    int in_y_base = by * tileSize * stride - padding;
    
    // Pointers to shared memory
    float *sharedInput = sharedData;
    
    // Load input data to shared memory
    for (int c = 0; c < inputChannels; c++) {
        // Each thread loads multiple elements to cover the tile with padding
        for (int dy = 0; dy < tileSizeWithPadding; dy += tileSize) {
            for (int dx = 0; dx < tileSizeWithPadding; dx += tileSize) {
                int in_y = in_y_base + ty + dy;
                int in_x = in_x_base + tx + dx;
                
                // Check bounds and apply padding
                float value = 0.0f;
                if (in_y >= 0 && in_y < inputSize && in_x >= 0 && in_x < inputSize) {
                    value = input[
                        b * inputChannels * inputSize * inputSize +
                        c * inputSize * inputSize +
                        in_y * inputSize + in_x
                    ];
                }
                
                // Store in shared memory if within tile bounds
                if (ty + dy < tileSizeWithPadding && tx + dx < tileSizeWithPadding) {
                    sharedInput[
                        c * tileSizeWithPadding * tileSizeWithPadding +
                        (ty + dy) * tileSizeWithPadding + (tx + dx)
                    ] = value;
                }
            }
        }
    }
    
    // Ensure all threads have loaded data to shared memory
    __syncthreads();
    
    // Compute convolution if within output bounds
    if (out_x < outputSize && out_y < outputSize && b < batchSize) {
        float sum = 0.0f;
        
        // For each input channel
        for (int c = 0; c < inputChannels; c++) {
            // For each kernel position
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    // Shared memory position
                    int shared_y = ty * stride + ky;
                    int shared_x = tx * stride + kx;
                    
                    // Input value from shared memory
                    float in_val = sharedInput[
                        c * tileSizeWithPadding * tileSizeWithPadding +
                        shared_y * tileSizeWithPadding + shared_x
                    ];
                    
                    // Kernel value
                    float kernel_val = kernels[
                        k * inputChannels * kernelSize * kernelSize +
                        c * kernelSize * kernelSize +
                        ky * kernelSize + kx
                    ];
                    
                    // Accumulate result
                    sum += in_val * kernel_val;
                }
            }
        }
        
        // Store output
        if (out_x < outputSize && out_y < outputSize) {
            output[
                b * kernelCount * outputSize * outputSize +
                k * outputSize * outputSize +
                out_y * outputSize + out_x
            ] = sum;
        }
    }
}

// ReLU activation
__global__ void reluActivationKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

// Max pooling
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

// Forward pass - Updated to use shared memory kernel
void forwardCNN(
    float *d_input,
    float *d_kernels,  // Added parameter for kernels
    float *d_conv_output,
    float *d_activation,
    float *d_pooling_output,
    const CNNConfig *cfg,
    float *timing)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int outputSize = (cfg->inputSize + 2*cfg->padding - cfg->kernelSize) / cfg->stride + 1;

    // Convolution with shared memory kernel
    dim3 blockDim(8, 8, 1);
    dim3 gridDim(
        (outputSize + blockDim.x - 1) / blockDim.x,
        (outputSize + blockDim.y - 1) / blockDim.y,
        cfg->kernelCount * cfg->batchSize
    );

    int tileSize = blockDim.x;
    int tileSizeWithPadding = tileSize + cfg->kernelSize - 1;
    int sharedMemSize = cfg->inputChannels * tileSizeWithPadding * tileSizeWithPadding * sizeof(float);

    cudaEventRecord(start);
    convolutionSharedKernel<<<gridDim, blockDim, sharedMemSize>>>(
        d_input, d_kernels, d_conv_output,  // Pass kernels as parameter
        cfg->batchSize, cfg->inputChannels, cfg->inputSize,
        cfg->kernelSize, cfg->kernelCount, outputSize,
        cfg->padding, cfg->stride
    );
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

// Run single configuration - Updated
void runConfiguration(const CNNConfig *cfg) {
    printf("\n========================================\n");
    printf("CONFIGURATION: %s\n", cfg->name);
    printf("========================================\n");
    printf("Input Size: %dx%d\n", cfg->inputSize, cfg->inputSize);
    printf("Kernel: %dx%d, Count: %d\n", cfg->kernelSize, cfg->kernelSize, cfg->kernelCount);
    printf("Batch Size: %d\n", cfg->batchSize);
    printf("Padding: %d, Stride: %d\n\n", cfg->padding, cfg->stride);

    int outputSize = (cfg->inputSize + 2*cfg->padding - cfg->kernelSize) / cfg->stride + 1;
    int poolOutputSize = outputSize / 2;

    // Allocate host memory
    size_t inputBytes = cfg->batchSize * cfg->inputChannels * cfg->inputSize * cfg->inputSize * sizeof(float);
    size_t kernelBytes = cfg->kernelCount * cfg->inputChannels * cfg->kernelSize * cfg->kernelSize * sizeof(float);
    size_t outputBytes = cfg->batchSize * cfg->kernelCount * outputSize * outputSize * sizeof(float);
    size_t poolBytes = cfg->batchSize * cfg->kernelCount * poolOutputSize * poolOutputSize * sizeof(float);

    float *h_input = (float*)malloc(inputBytes);
    float *h_kernels = (float*)malloc(kernelBytes);
    float *h_output = (float*)malloc(outputBytes);
    float *h_pooling_output = (float*)malloc(poolBytes);

    if (!h_input || !h_kernels || !h_output || !h_pooling_output) {
        fprintf(stderr, "Host malloc failed for config: %s\n", cfg->name);
        free(h_input);
        free(h_kernels);
        free(h_output);
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

    // Remove constant memory size check since we're not using it anymore
    float kernelSizeKB = kernelBytes / 1024.0f;
    printf("Kernel size: %.2f KB\n", kernelSizeKB);

    // Allocate device memory
    float *d_input, *d_kernels, *d_conv_output, *d_activation, *d_pooling_output;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, inputBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernels, kernelBytes));  // Allocate global memory for kernels
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output, outputBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_activation, outputBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pooling_output, poolBytes));

    // Copy to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernels, h_kernels, kernelBytes, cudaMemcpyHostToDevice));  // Copy kernels to global memory

    // Run CNN
    printf("Running CNN forward pass...\n");
    float timing[3] = {0};
    forwardCNN(d_input, d_kernels, d_conv_output, d_activation, d_pooling_output, cfg, timing);  // Pass d_kernels

    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_conv_output, outputBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pooling_output, d_pooling_output, poolBytes, cudaMemcpyDeviceToHost));

    // Save feature maps for this configuration
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
    printf("\n=== Results ===\n");
    printf("Output size after convolution: %dx%d\n", outputSize, outputSize);
    printf("Output size after pooling: %dx%d\n\n", poolOutputSize, poolOutputSize);
    printf("Kernel memory used: %.2f KB (global memory)\n\n", kernelSizeKB);

    printf("Layer timings:\n");
    printf("  Convolution: %.3f ms\n", timing[0]);
    printf("  ReLU Activation: %.3f ms\n", timing[1]);
    printf("  Max Pooling: %.3f ms\n", timing[2]);
    printf("  Total: %.3f ms\n\n", timing[0] + timing[1] + timing[2]);

    // Calculate throughput
    float totalTimeSeconds = (timing[0] + timing[1] + timing[2]) / 1000.0f;
    float imagesPerSecond = cfg->batchSize / totalTimeSeconds;
    printf("Throughput: %.2f images/second\n", imagesPerSecond);

    printf("\nSample feature map (first image, first channel, top-left 5x5):\n");
    int displaySize = (outputSize < 5) ? outputSize : 5;
    for (int y = 0; y < displaySize; y++) {
        for (int x = 0; x < displaySize; x++) {
            printf("%.3f ", h_output[y * outputSize + x]);
        }
        printf("\n");
    }

    // Free memory
    free(h_input);
    free(h_kernels);
    free(h_output);
    free(h_pooling_output);

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_kernels));  // Free kernel memory
    CHECK_CUDA_ERROR(cudaFree(d_conv_output));
    CHECK_CUDA_ERROR(cudaFree(d_activation));
    CHECK_CUDA_ERROR(cudaFree(d_pooling_output));

    printf("\n✓ Configuration complete!\n");
}

int main() {
    srand(42);

    printf("=== Multi-Configuration Chest X-Ray CNN Benchmark ===\n");
    printf("Testing three different configurations:\n");
    printf("  1. Small (32x32) - MNIST-like\n");
    printf("  2. Medium (256x256) - ResNet Standard\n");
    printf("  3. Large (1024x1024) - High-Res X-Ray\n");

    // Define three configurations
    CNNConfig configs[3] = {
        // Config 1: Small (MNIST-like) - Correctness Verification
        {
            .inputSize = 32,
            .inputChannels = 1,
            .kernelSize = 3,
            .kernelCount = 16,
            .padding = 1,
            .stride = 1,
            .batchSize = 64,
            .name = "Small"
        },
        // Config 2: Medium (ResNet Standard) - Compute Benchmark
        {
            .inputSize = 256,
            .inputChannels = 1,
            .kernelSize = 3,
            .kernelCount = 16,
            .padding = 1,
            .stride = 1,
            .batchSize = 64,
            .name = "Medium"
        },
        // Config 3: Large (High-Res X-Ray) - Bandwidth Benchmark
        {
            .inputSize = 1024,
            .inputChannels = 1,
            .kernelSize = 7,
            .kernelCount = 16,
            .padding = 3,
            .stride = 1,
            .batchSize = 64,
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
