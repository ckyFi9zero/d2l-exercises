import torch

# 1.运行本节中的代码。将本节中的条件语句X == Y更改为X < Y或X > Y，然后看看你可以得到什么样的张量。
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X)
print(Y)
print(X==Y)
print(X<Y)
print(X>Y)

# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])
# tensor([[2., 1., 4., 3.],
#         [1., 2., 3., 4.],
#         [4., 3., 2., 1.]])
# tensor([[False,  True, False,  True],
#         [False, False, False, False],
#         [False, False, False, False]])
# tensor([[ True, False,  True, False],
#         [False, False, False, False],
#         [False, False, False, False]])
# tensor([[False, False, False, False],
#         [ True,  True,  True,  True],
#         [ True,  True,  True,  True]])


# 2.用其他形状（例如三维张量）替换广播机制中按元素操作的两个张量。结果是否与预期相同？
# 成功
a = torch.arange(12).reshape((3, 2, 2))
b = torch.arange(4).reshape((1, 2, 2)) 
print(a)
print(b)
print(a+b)
# # 失败
# a = torch.arange(12).reshape((3, 2, 2))
# b = torch.arange(4).reshape((2, 2, 1))
# print(a)
# print(b)
# print(a+b)

# 用三维等其他形状替换后，广播机制的规则不变：从最后一维对齐，维度必须相等或其中一个为 1。
# 满足规则时会自动扩展维度为 1 的轴并进行逐元素运算，结果形状为各维取最大值，数值与“复制扩展后相加”的预期一致。
# 不满足规则时会报错，无法得到结果。
