import torch
from torch.utils.cpp_extension import load_inline

# Define the C++ source code
cpp_source = """
#include <torch/extension.h>

at::Tensor add_one_to_tensor(at::Tensor input_tensor) {
    // Ensure the tensor is a contiguous CPU tensor
    auto tensor = input_tensor.contiguous();
    auto data_ptr = tensor.data_ptr<float>();
    
    // Iterate over all elements and add one
    for (int i = 0; i < tensor.numel(); ++i) {
        data_ptr[i] += 1;
    }

    return tensor;
}
"""

# Compile the C++ code as a PyTorch extension
my_module = load_inline(
    name='my_module',
    cpp_sources=[cpp_source],
    functions=['add_one_to_tensor'],
    verbose=True,
    build_directory='./load_add_one'
)

# Create a 1000x1000 tensor of zeros
input_tensor = torch.zeros((1000, 1000), dtype=torch.float32)

# Call the C++ function to modify the tensor
output_tensor = my_module.add_one_to_tensor(input_tensor)

# Print the result
print(output_tensor)
