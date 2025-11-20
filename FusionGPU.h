#ifndef FUSION_GPU_H
#define FUSION_GPU_H

#include <vector>
#include <string>

// Forward declarations
struct Problem;

// Main fusion function declaration
void RunFusionCuda(const std::string &dense_folder,
                   const std::vector<Problem> &problems,
                   bool geom_consistency = false,
                   size_t max_images_per_chunk = 80);

#endif // FUSION_GPU_H