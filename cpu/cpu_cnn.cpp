/**
 * Multi-Config CNN CPU implementation
 * 
 * Combines:
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

#include <chrono>

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

// ------------------- CPU Baseline Layers -------------------

static inline float getKernel(const float *kernels, const CNNConfig *cfg,
                              int k, int c, int ky, int kx) {
    return kernels[
        k * cfg->inputChannels * cfg->kernelSize * cfg->kernelSize +
        c * cfg->kernelSize * cfg->kernelSize +
        ky * cfg->kernelSize + kx
    ];
}

static inline float getInput(const float *input, const CNNConfig *cfg,
                             int b, int c, int y, int x) {
    return input[
        b * cfg->inputChannels * cfg->inputSize * cfg->inputSize +
        c * cfg->inputSize * cfg->inputSize +
        y * cfg->inputSize + x
    ];
}

static inline void setOutput(float *output,
                             int kernelCount, int outSize,
                             int b, int k, int y, int x,
                             float v) {
    output[
        b * kernelCount * outSize * outSize +
        k * outSize * outSize +
        y * outSize + x
    ] = v;
}

static inline float getOutput(const float *output,
                              int kernelCount, int outSize,
                              int b, int k, int y, int x) {
    return output[
        b * kernelCount * outSize * outSize +
        k * outSize * outSize +
        y * outSize + x
    ];
}

// CPU convolution
void convolutionCPU(const float *input, float *output,
                    const float *kernels, const CNNConfig *cfg,
                    int outputSize) {
    for (int b = 0; b < cfg->batchSize; b++) {
        for (int k = 0; k < cfg->kernelCount; k++) {
            for (int out_y = 0; out_y < outputSize; out_y++) {
                for (int out_x = 0; out_x < outputSize; out_x++) {

                    float sum = 0.0f;

                    int in_y_base = out_y * cfg->stride - cfg->padding;
                    int in_x_base = out_x * cfg->stride - cfg->padding;

                    for (int c = 0; c < cfg->inputChannels; c++) {
                        for (int ky = 0; ky < cfg->kernelSize; ky++) {
                            for (int kx = 0; kx < cfg->kernelSize; kx++) {

                                int in_y = in_y_base + ky;
                                int in_x = in_x_base + kx;

                                float in_val = 0.0f;
                                if (in_y >= 0 && in_y < cfg->inputSize &&
                                    in_x >= 0 && in_x < cfg->inputSize) {
                                    in_val = getInput(input, cfg, b, c, in_y, in_x);
                                }

                                float w = getKernel(kernels, cfg, k, c, ky, kx);
                                sum += in_val * w;
                            }
                        }
                    }

                    setOutput(output, cfg->kernelCount, outputSize, b, k, out_y, out_x, sum);
                }
            }
        }
    }
}

// ReLU
void reluCPU(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] > 0.0f) ? data[i] : 0.0f;
    }
}

// Max Pooling
void maxPoolingCPU(const float *input, float *output,
                   int batchSize, int channels,
                   int inputSize, int poolSize,
                   int outputSize, int stride) {
    for (int b = 0; b < batchSize; b++) {
        for (int c = 0; c < channels; c++) {
            for (int out_y = 0; out_y < outputSize; out_y++) {
                for (int out_x = 0; out_x < outputSize; out_x++) {

                    int in_y_base = out_y * stride;
                    int in_x_base = out_x * stride;

                    float maxVal = -FLT_MAX;

                    for (int dy = 0; dy < poolSize; dy++) {
                        for (int dx = 0; dx < poolSize; dx++) {
                            int in_y = in_y_base + dy;
                            int in_x = in_x_base + dx;

                            if (in_y < inputSize && in_x < inputSize) {
                                float v = input[
                                    b * channels * inputSize * inputSize +
                                    c * inputSize * inputSize +
                                    in_y * inputSize + in_x
                                ];
                                maxVal = (v > maxVal) ? v : maxVal;
                            }
                        }
                    }

                    output[
                        b * channels * outputSize * outputSize +
                        c * outputSize * outputSize +
                        out_y * outputSize + out_x
                    ] = maxVal;
                }
            }
        }
    }
}

// Forward pass
void forwardCNN_CPU(float *h_input,
                    float *h_conv_output,
                    float *h_activation,
                    float *h_pooling_output,
                    const CNNConfig *cfg,
                    float *timing_ms) {
    int outputSize = (cfg->inputSize + 2 * cfg->padding - cfg->kernelSize) / cfg->stride + 1;
    int totalElements = cfg->batchSize * cfg->kernelCount * outputSize * outputSize;

    // Convolution
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    timing_ms[0] = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // ReLU
    auto t2 = std::chrono::high_resolution_clock::now();
    reluCPU(h_activation, totalElements);
    auto t3 = std::chrono::high_resolution_clock::now();
    timing_ms[1] = std::chrono::duration<float, std::milli>(t3 - t2).count();

    // Pool
    int poolSize = 2;
    int poolStride = 2;
    int poolOutputSize = outputSize / poolStride;

    auto t4 = std::chrono::high_resolution_clock::now();
    maxPoolingCPU(h_activation, h_pooling_output,
                  cfg->batchSize, cfg->kernelCount,
                  outputSize, poolSize,
                  poolOutputSize, poolStride);
    auto t5 = std::chrono::high_resolution_clock::now();
    timing_ms[2] = std::chrono::duration<float, std::milli>(t5 - t4).count();
}

// Run single configuration
void runConfiguration(const CNNConfig *cfg) {
    printf("\n========================================\n");
    printf("CONFIGURATION: %s\n", cfg->name);
    printf("========================================\n");
    printf("Input Size: %dx%d\n", cfg->inputSize, cfg->inputSize);
    printf("Kernel: %dx%d, Count: %d\n", cfg->kernelSize, cfg->kernelSize, cfg->kernelCount);
    printf("Batch Size: %d\n", cfg->batchSize);
    printf("Padding: %d, Stride: %d\n\n", cfg->padding, cfg->stride);

    int outputSize = (cfg->inputSize + 2 * cfg->padding - cfg->kernelSize) / cfg->stride + 1;
    int poolOutputSize = outputSize / 2;

    // Allocate host memory
    size_t inputBytes  = (size_t)cfg->batchSize * cfg->inputChannels * cfg->inputSize * cfg->inputSize * sizeof(float);
    size_t kernelBytes = (size_t)cfg->kernelCount * cfg->inputChannels * cfg->kernelSize * cfg->kernelSize * sizeof(float);
    size_t outputBytes = (size_t)cfg->batchSize * cfg->kernelCount * outputSize * outputSize * sizeof(float);
    size_t poolBytes   = (size_t)cfg->batchSize * cfg->kernelCount * poolOutputSize * poolOutputSize * sizeof(float);

    float *h_input          = (float*)malloc(inputBytes);
    float *h_kernels        = (float*)malloc(kernelBytes);
    float *h_output         = (float*)malloc(outputBytes);
    float *h_activation     = (float*)malloc(outputBytes);
    float *h_pooling_output = (float*)malloc(poolBytes);

    if (!h_input || !h_kernels || !h_output || !h_activation || !h_pooling_output) {
        fprintf(stderr, "Host malloc failed for config: %s\n", cfg->name);
        free(h_input);
        free(h_kernels);
        free(h_output);
        free(h_activation);
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

    // Run forward pass
    printf("Running CNN forward pass (CPU baseline)...\n");

    // Convolution timing
    float timing[3] = {0};

    auto c0 = std::chrono::high_resolution_clock::now();
    convolutionCPU(h_input, h_output, h_kernels, cfg, outputSize);
    auto c1 = std::chrono::high_resolution_clock::now();
    timing[0] = std::chrono::duration<float, std::milli>(c1 - c0).count();

    // Activation buffer
    memcpy(h_activation, h_output, outputBytes);

    auto r0 = std::chrono::high_resolution_clock::now();
    reluCPU(h_activation, (int)(cfg->batchSize * cfg->kernelCount * outputSize * outputSize));
    auto r1 = std::chrono::high_resolution_clock::now();
    timing[1] = std::chrono::duration<float, std::milli>(r1 - r0).count();

    auto p0 = std::chrono::high_resolution_clock::now();
    maxPoolingCPU(h_activation, h_pooling_output,
                  cfg->batchSize, cfg->kernelCount,
                  outputSize, 2,
                  poolOutputSize, 2);
    auto p1 = std::chrono::high_resolution_clock::now();
    timing[2] = std::chrono::duration<float, std::milli>(p1 - p0).count();

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
    printf("\n=== Results ===\n");
    printf("Output size after convolution: %dx%d\n", outputSize, outputSize);
    printf("Output size after pooling: %dx%d\n\n", poolOutputSize, poolOutputSize);

    printf("Layer timings:\n");
    printf("  Convolution: %.3f ms\n", timing[0]);
    printf("  ReLU Activation: %.3f ms\n", timing[1]);
    printf("  Max Pooling: %.3f ms\n", timing[2]);
    printf("  Total: %.3f ms\n\n", timing[0] + timing[1] + timing[2]);

    // Calculate throughput
    float totalTimeSeconds = (timing[0] + timing[1] + timing[2]) / 1000.0f;
    float imagesPerSecond = cfg->batchSize / totalTimeSeconds;
    printf("Throughput: %.2f images/second\n", imagesPerSecond);

    // Sample feature map (first image, first channel, top-left 5x5)
    printf("\nSample feature map (first image, first channel, top-left 5x5):\n");
    int displaySize = (outputSize < 5) ? outputSize : 5;
    for (int y = 0; y < displaySize; y++) {
        for (int x = 0; x < displaySize; x++) {
            float v = getOutput(h_output, cfg->kernelCount, outputSize, 0, 0, y, x);
            printf("%.3f ", v);
        }
        printf("\n");
    }

    // Free
    free(h_input);
    free(h_kernels);
    free(h_output);
    free(h_activation);
    free(h_pooling_output);

    printf("\n✓ Configuration complete!\n");
}

int main() {
    srand(42);

    printf("=== Multi-Configuration Chest X-Ray CNN Benchmark (CPU Baseline) ===\n");
    printf("Testing three different configurations:\n");
    printf("  1. Small (32x32) - MNIST-like\n");
    printf("  2. Medium (256x256) - ResNet Standard\n");
    printf("  3. Large (1024x1024) - High-Res X-Ray\n");

    // Define three configurations
    CNNConfig configs[3] = {
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

    for (int i = 0; i < 3; i++) {
        runConfiguration(&configs[i]);
    }

    printf("\n========================================\n");
    printf("ALL CONFIGURATIONS COMPLETE\n");
    printf("========================================\n");

    return 0;
}
