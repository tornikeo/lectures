#include <torch/torch.h>
#include <torch/types.h>

__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {

}

torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);
    auto result = torch::empty_like(matrix);
    dim3 threads_per_block(16,16);
    dim3 number_of_blocks(
        (width + threads_per_block.x - 1) / threads_per_block.x, 
        (height + threads_per_block.y - 1) / threads_per_block.y
    );

}
