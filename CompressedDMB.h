// CompressedDMB.h - Add this new file to your project
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <zlib.h>

namespace CompressedDMB {

// ============================================================================
// OCTAHEDRAL NORMAL ENCODING
// ============================================================================

inline float signNotZero(float v) {
    return (v >= 0.0f) ? 1.0f : -1.0f;
}

inline void encodeOctahedral(float nx, float ny, float nz, float& ox, float& oy) {
    float l1norm = std::abs(nx) + std::abs(ny) + std::abs(nz);
    if (l1norm < 1e-10f) {
        ox = oy = 0.0f;
        return;
    }
    ox = nx / l1norm;
    oy = ny / l1norm;
    
    if (nz < 0.0f) {
        float tmpX = (1.0f - std::abs(oy)) * signNotZero(ox);
        float tmpY = (1.0f - std::abs(ox)) * signNotZero(oy);
        ox = tmpX;
        oy = tmpY;
    }
}

inline void decodeOctahedral(float ox, float oy, float& nx, float& ny, float& nz) {
    nz = 1.0f - std::abs(ox) - std::abs(oy);
    
    if (nz < 0.0f) {
        float tmpX = (1.0f - std::abs(oy)) * signNotZero(ox);
        float tmpY = (1.0f - std::abs(ox)) * signNotZero(oy);
        nx = tmpX;
        ny = tmpY;
    } else {
        nx = ox;
        ny = oy;
    }
    
    float len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len > 1e-10f) {
        nx /= len;
        ny /= len;
        nz /= len;
    }
}

// ============================================================================
// QUANTIZATION HELPERS
// ============================================================================

inline uint16_t quantizeFloat16(float value, float minVal, float maxVal) {
    if (maxVal <= minVal) return 0;
    float normalized = (value - minVal) / (maxVal - minVal);
    normalized = std::max(0.0f, std::min(1.0f, normalized));
    return static_cast<uint16_t>(normalized * 65535.0f + 0.5f);
}

inline float dequantizeFloat16(uint16_t value, float minVal, float maxVal) {
    return minVal + (static_cast<float>(value) / 65535.0f) * (maxVal - minVal);
}

inline uint8_t quantizeFloat8(float value, float minVal, float maxVal) {
    if (maxVal <= minVal) return 0;
    float normalized = (value - minVal) / (maxVal - minVal);
    normalized = std::max(0.0f, std::min(1.0f, normalized));
    return static_cast<uint8_t>(normalized * 255.0f + 0.5f);
}

inline float dequantizeFloat8(uint8_t value, float minVal, float maxVal) {
    return minVal + (static_cast<float>(value) / 255.0f) * (maxVal - minVal);
}

// ============================================================================
// ZLIB COMPRESSION
// ============================================================================

inline std::vector<uint8_t> compressZlib(const void* data, size_t size, int level = 3) {
    uLongf compressedSize = compressBound(size);
    std::vector<uint8_t> compressed(compressedSize + sizeof(uint32_t));
    
    *reinterpret_cast<uint32_t*>(compressed.data()) = static_cast<uint32_t>(size);
    
    int result = compress2(
        compressed.data() + sizeof(uint32_t), 
        &compressedSize,
        reinterpret_cast<const Bytef*>(data), 
        size, 
        level
    );
    
    if (result != Z_OK) {
        return {};
    }
    
    compressed.resize(compressedSize + sizeof(uint32_t));
    return compressed;
}

inline std::vector<uint8_t> decompressZlib(const void* data, size_t compressedSize) {
    uint32_t originalSize = *reinterpret_cast<const uint32_t*>(data);
    std::vector<uint8_t> decompressed(originalSize);
    
    uLongf destLen = originalSize;
    int result = uncompress(
        decompressed.data(),
        &destLen,
        reinterpret_cast<const Bytef*>(data) + sizeof(uint32_t),
        compressedSize - sizeof(uint32_t)
    );
    
    if (result != Z_OK) {
        return {};
    }
    
    return decompressed;
}

// ============================================================================
// COMPRESSED DEPTH I/O
// ============================================================================

inline int writeDepthCompressed(const std::string& path, const cv::Mat_<float>& depth) {
    if (depth.empty()) return -1;
    
    int32_t width = depth.cols;
    int32_t height = depth.rows;
    
    float depthMin = std::numeric_limits<float>::max();
    float depthMax = std::numeric_limits<float>::lowest();
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float d = depth(y, x);
            if (d > 0.0f && d < 1e6f) {
                depthMin = std::min(depthMin, d);
                depthMax = std::max(depthMax, d);
            }
        }
    }
    
    if (depthMin >= depthMax) {
        depthMin = 0.0f;
        depthMax = 1.0f;
    }
    
    float range = depthMax - depthMin;
    depthMin -= range * 0.01f;
    depthMax += range * 0.01f;
    
    std::vector<uint16_t> quantized(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float d = depth(y, x);
            if (d <= 0.0f || d >= 1e6f) {
                quantized[y * width + x] = 0;
            } else {
                uint16_t q = quantizeFloat16(d, depthMin, depthMax);
                quantized[y * width + x] = (q == 0) ? 1 : q;
            }
        }
    }
    
    auto compressed = compressZlib(quantized.data(), quantized.size() * sizeof(uint16_t));
    if (compressed.empty()) return -1;
    
    FILE* file = fopen(path.c_str(), "wb");
    if (!file) return -1;
    
    fwrite("CDMB", 1, 4, file);
    uint8_t version = 1;
    uint8_t type = 1;
    fwrite(&version, 1, 1, file);
    fwrite(&type, 1, 1, file);
    fwrite(&width, sizeof(int32_t), 1, file);
    fwrite(&height, sizeof(int32_t), 1, file);
    fwrite(&depthMin, sizeof(float), 1, file);
    fwrite(&depthMax, sizeof(float), 1, file);
    
    uint32_t compSize = static_cast<uint32_t>(compressed.size());
    fwrite(&compSize, sizeof(uint32_t), 1, file);
    fwrite(compressed.data(), 1, compSize, file);
    
    fclose(file);
    return 0;
}

inline int readDepthCompressed(const std::string& path, cv::Mat_<float>& depth) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) return -1;
    
    char magic[4];
    if (fread(magic, 1, 4, file) != 4 || memcmp(magic, "CDMB", 4) != 0) {
        fclose(file);
        return -1;
    }
    
    uint8_t version, type;
    int32_t width, height;
    float depthMin, depthMax;
    
    fread(&version, 1, 1, file);
    fread(&type, 1, 1, file);
    fread(&width, sizeof(int32_t), 1, file);
    fread(&height, sizeof(int32_t), 1, file);
    fread(&depthMin, sizeof(float), 1, file);
    fread(&depthMax, sizeof(float), 1, file);
    
    if (type != 1) {
        fclose(file);
        return -1;
    }
    
    uint32_t compSize;
    fread(&compSize, sizeof(uint32_t), 1, file);
    
    std::vector<uint8_t> compressed(compSize);
    fread(compressed.data(), 1, compSize, file);
    fclose(file);
    
    auto decompressed = decompressZlib(compressed.data(), compSize);
    if (decompressed.empty()) return -1;
    
    const uint16_t* quantized = reinterpret_cast<const uint16_t*>(decompressed.data());
    
    depth = cv::Mat_<float>(height, width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint16_t q = quantized[y * width + x];
            if (q == 0) {
                depth(y, x) = 0.0f;
            } else {
                depth(y, x) = dequantizeFloat16(q, depthMin, depthMax);
            }
        }
    }
    
    return 0;
}

// ============================================================================
// COMPRESSED NORMAL I/O
// ============================================================================

inline int writeNormalCompressed(const std::string& path, const cv::Mat_<cv::Vec3f>& normal) {
    if (normal.empty()) return -1;
    
    int32_t width = normal.cols;
    int32_t height = normal.rows;
    
    std::vector<uint16_t> encoded(width * height * 2);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const cv::Vec3f& n = normal(y, x);
            float ox, oy;
            encodeOctahedral(n[0], n[1], n[2], ox, oy);
            
            int idx = (y * width + x) * 2;
            encoded[idx + 0] = quantizeFloat16(ox, -1.0f, 1.0f);
            encoded[idx + 1] = quantizeFloat16(oy, -1.0f, 1.0f);
        }
    }
    
    auto compressed = compressZlib(encoded.data(), encoded.size() * sizeof(uint16_t));
    if (compressed.empty()) return -1;
    
    FILE* file = fopen(path.c_str(), "wb");
    if (!file) return -1;
    
    fwrite("CDMB", 1, 4, file);
    uint8_t version = 1;
    uint8_t type = 2;
    fwrite(&version, 1, 1, file);
    fwrite(&type, 1, 1, file);
    fwrite(&width, sizeof(int32_t), 1, file);
    fwrite(&height, sizeof(int32_t), 1, file);
    
    float unused = 0.0f;
    fwrite(&unused, sizeof(float), 1, file);
    fwrite(&unused, sizeof(float), 1, file);
    
    uint32_t compSize = static_cast<uint32_t>(compressed.size());
    fwrite(&compSize, sizeof(uint32_t), 1, file);
    fwrite(compressed.data(), 1, compSize, file);
    
    fclose(file);
    return 0;
}

inline int readNormalCompressed(const std::string& path, cv::Mat_<cv::Vec3f>& normal) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) return -1;
    
    char magic[4];
    if (fread(magic, 1, 4, file) != 4 || memcmp(magic, "CDMB", 4) != 0) {
        fclose(file);
        return -1;
    }
    
    uint8_t version, type;
    int32_t width, height;
    float unused1, unused2;
    
    fread(&version, 1, 1, file);
    fread(&type, 1, 1, file);
    fread(&width, sizeof(int32_t), 1, file);
    fread(&height, sizeof(int32_t), 1, file);
    fread(&unused1, sizeof(float), 1, file);
    fread(&unused2, sizeof(float), 1, file);
    
    if (type != 2) {
        fclose(file);
        return -1;
    }
    
    uint32_t compSize;
    fread(&compSize, sizeof(uint32_t), 1, file);
    
    std::vector<uint8_t> compressed(compSize);
    fread(compressed.data(), 1, compSize, file);
    fclose(file);
    
    auto decompressed = decompressZlib(compressed.data(), compSize);
    if (decompressed.empty()) return -1;
    
    const uint16_t* encoded = reinterpret_cast<const uint16_t*>(decompressed.data());
    
    normal = cv::Mat_<cv::Vec3f>(height, width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 2;
            float ox = dequantizeFloat16(encoded[idx + 0], -1.0f, 1.0f);
            float oy = dequantizeFloat16(encoded[idx + 1], -1.0f, 1.0f);
            
            float nx, ny, nz;
            decodeOctahedral(ox, oy, nx, ny, nz);
            normal(y, x) = cv::Vec3f(nx, ny, nz);
        }
    }
    
    return 0;
}

// ============================================================================
// COMPRESSED COST I/O
// ============================================================================

inline int writeCostCompressed(const std::string& path, const cv::Mat_<float>& cost) {
    if (cost.empty()) return -1;
    
    int32_t width = cost.cols;
    int32_t height = cost.rows;
    
    const float costMin = 0.0f;
    const float costMax = 2.5f;
    
    std::vector<uint8_t> quantized(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            quantized[y * width + x] = quantizeFloat8(cost(y, x), costMin, costMax);
        }
    }
    
    auto compressed = compressZlib(quantized.data(), quantized.size());
    if (compressed.empty()) return -1;
    
    FILE* file = fopen(path.c_str(), "wb");
    if (!file) return -1;
    
    fwrite("CDMB", 1, 4, file);
    uint8_t version = 1;
    uint8_t type = 3;
    fwrite(&version, 1, 1, file);
    fwrite(&type, 1, 1, file);
    fwrite(&width, sizeof(int32_t), 1, file);
    fwrite(&height, sizeof(int32_t), 1, file);
    
    float params[2] = {costMin, costMax};
    fwrite(params, sizeof(float), 2, file);
    
    uint32_t compSize = static_cast<uint32_t>(compressed.size());
    fwrite(&compSize, sizeof(uint32_t), 1, file);
    fwrite(compressed.data(), 1, compSize, file);
    
    fclose(file);
    return 0;
}

inline int readCostCompressed(const std::string& path, cv::Mat_<float>& cost) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) return -1;
    
    char magic[4];
    if (fread(magic, 1, 4, file) != 4 || memcmp(magic, "CDMB", 4) != 0) {
        fclose(file);
        return -1;
    }
    
    uint8_t version, type;
    int32_t width, height;
    float costMin, costMax;
    
    fread(&version, 1, 1, file);
    fread(&type, 1, 1, file);
    fread(&width, sizeof(int32_t), 1, file);
    fread(&height, sizeof(int32_t), 1, file);
    fread(&costMin, sizeof(float), 1, file);
    fread(&costMax, sizeof(float), 1, file);
    
    if (type != 3) {
        fclose(file);
        return -1;
    }
    
    uint32_t compSize;
    fread(&compSize, sizeof(uint32_t), 1, file);
    
    std::vector<uint8_t> compressed(compSize);
    fread(compressed.data(), 1, compSize, file);
    fclose(file);
    
    auto decompressed = decompressZlib(compressed.data(), compSize);
    if (decompressed.empty()) return -1;
    
    cost = cv::Mat_<float>(height, width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cost(y, x) = dequantizeFloat8(decompressed[y * width + x], costMin, costMax);
        }
    }
    
    return 0;
}

// ============================================================================
// FORMAT DETECTION
// ============================================================================

inline bool isCompressedFormat(const std::string& path) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) return false;
    
    char magic[4];
    size_t read = fread(magic, 1, 4, file);
    fclose(file);
    
    return (read == 4 && memcmp(magic, "CDMB", 4) == 0);
}

} // namespace CompressedDMB