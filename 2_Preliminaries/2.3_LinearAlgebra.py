import torch

# 1.证明一个矩阵A的转置的转置是A
A = torch.randn(3, 4)
X = A.t().t()
print(torch.allclose(X, A))  # True

# 2.给出两个矩阵A和B，证明“它们转置的和”等于“它们和的转置”
B = torch.randn(3, 4)
left = A.t() + B.t()
right = (A + B).t()
print(torch.allclose(left, right))  # True

# 3.给定任意方阵A，A+A转置总是对称的吗?为什么?
A = torch.randn(4, 4)
S = A + A.t()
print(torch.allclose(S, S.t()))  # True

# 4.本节中定义了形状(2,3,4)的张量X。len(X)的输出结果是什么？
X = torch.zeros(2, 3, 4)
print(len(X))       # 2
print(X.shape[0])   # 2

# 5.对于任意形状的张量X,len(X)是否总是对应于X特定轴的长度?这个轴是什么?
X = torch.zeros(5, 6, 7)
print(len(X), X.shape[0])  # 5 5 对应轴0（第一维）

# 6.运行A/A.sum(axis=1)，看看会发生什么。请分析一下原因？
A = torch.arange(12, dtype=torch.float32).reshape(3, 4)
try:
    print(A / A.sum(dim=1))
except Exception as e:
    print("Error:", e)
# 报错原因：
# A 是形状 (3,4)
# A.sum(dim=1) 是形状 (3,)
# 广播会从末尾对齐：(3,4) 和 (3,) 等价尝试对齐成 (3,4) vs (1,3)，最后一维 4 和 3 对不上 → 报错。
# 正确写法：
row_sum = A.sum(dim=1, keepdim=True)  # (3,1)
print(row_sum.shape)                 # torch.Size([3, 1])
print(A / row_sum)                   # 每行除以该行和

# 7.考虑一个具有形状(2,3,4)的张量，在轴0、1、2上的求和输出是什么形状?
X = torch.zeros(2, 3, 4)

print(X.sum(dim=0).shape)  # (3,4)
print(X.sum(dim=1).shape)  # (2,4)
print(X.sum(dim=2).shape)  # (2,3)

# 8.为linalg.norm函数提供3个或更多轴的张量，并观察其输出。对于任意形状的张量这个函数计算得到什么?
X = torch.arange(2*3*4, dtype=torch.float32).reshape(2, 3, 4)

# 对全部三个轴求范数 -> 得到一个标量
# 对 3 个轴一起求 norm：相当于把这些轴上的元素当成一个大向量，求默认 2-范数（平方和开根号）。
# 3轴及以上使用dim会报错
n_all = torch.linalg.norm(X.reshape(-1), ord=2)  # 或 X.flatten()
print(n_all)

# 对 (1,2) 两个轴求范数 -> 每个 batch(轴0) 得到一个值，形状 (2,)
n_12 = torch.linalg.norm(X, ord=2, dim=(1, 2))
print(n_12, n_12.shape)    # torch.Size([2])
