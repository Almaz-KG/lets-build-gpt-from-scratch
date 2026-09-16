import torch
from torch import Tensor
from torch.nn import functional as F

torch.manual_seed(1337) # type: ignore

B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

print(f"shape: {x.shape}")
# print(f"values: {x}")


def naive_python_implementation(x: Tensor) -> Tensor:
    xbow = torch.zeros((B, T, C))

    for b in range(B):
        for t in range(T):
            xprev = x[b, :t + 1]
            xbow[b, t] = xprev.mean(dim=0)
    return xbow


def mat_mul_implementation(x: Tensor) -> Tensor:
    wei = torch.tril(torch.ones((T, T)))
    wei = wei / wei.sum(dim=1, keepdim=True)
    return wei @ x


def softmax_implementation(x: Tensor) -> Tensor:
    tril = torch.tril(torch.ones(T, T))

    wei = torch.zeros((T, T))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    return wei @ x




print("==" * 20)
naive_impl = naive_python_implementation(x)
print(naive_impl.shape)
# print(naive_impl)


print("==" * 20)
mat_mul_impl = mat_mul_implementation(x)
print(mat_mul_impl.shape)
# print(mat_mul_impl)
print(f"EQUAL: {torch.allclose(naive_impl, mat_mul_impl)}")


print("==" * 20)
softmax_impl = softmax_implementation(x)
print(softmax_impl.shape)
# print(mat_mul_impl)
print(f"EQUAL: {torch.allclose(naive_impl, softmax_impl)}")


