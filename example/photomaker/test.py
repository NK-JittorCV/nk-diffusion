import torch

def masked_scatter_using_fill(input_tensor, mask, src):
    # 检查 mask 的形状
    assert mask.dim() == 1, "Mask must be a 1D tensor"
    assert input_tensor.shape[0] == mask.size(0), "The first dimension of input_tensor must match the size of mask"
    
    # 检查源张量的形状
    assert src.dim() == 2, "Source tensor must be a 2D tensor"
    assert src.size(0) == mask.sum(), "The number of rows in src must match the number of True values in mask"

    # 创建一个拷贝，避免修改原始张量
    result = input_tensor.clone()
    
    # 获取掩码为 True 的行索引
    mask_indices = mask.nonzero().squeeze()

    # 将源张量对应行复制到目标张量
    for i, idx in enumerate(mask_indices):
        result[idx] = src[i]

    return result

# 示例用法
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
mask = torch.tensor([True, False, True, False])
src = torch.tensor([[13, 14, 15], [16, 17, 18]])

output_tensor = masked_scatter_using_fill(input_tensor, mask, src)
print(output_tensor)

