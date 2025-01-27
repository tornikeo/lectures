#include <torch/extension.h>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("add_one_to_tensor", torch::wrap_pybind_function(add_one_to_tensor), "add_one_to_tensor");
}