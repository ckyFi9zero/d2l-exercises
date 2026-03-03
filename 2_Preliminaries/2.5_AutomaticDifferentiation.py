# 1. 为什么计算二阶导数比一阶导数的开销要更大？

# 在深度学习框架（如 PyTorch/TensorFlow）中，计算一阶导数（梯度）只需要进行一次反向传播。
# 计算图：二阶导数实际上是“导数的导数”。为了计算它，系统必须先构建一阶导数的计算图，然后再对这个图进行求导。
# 内存与时间：二阶导数的计算量通常与参数数量的平方相关（Hessian 矩阵），而一阶导数只与参数数量成线性关系。
# 因此，二阶导数在计算开销和内存占用上都要显著高于一阶。

# 2. 在运行反向传播函数之后，立即再次运行它，看看会发生什么。

# 报错：如果在运行 `backward()` 后不清除计算图，再次运行会报错，因为中间变量在第一次反向传播后为了节省内存已经被释放了。
# 梯度累加：如果设置了 `retain_graph=True`，连续运行两次反向传播，梯度会**累加**到同一个变量上（$grad = grad_1 + grad_2$）。

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 强制指定使用 Agg 后端，确保在 Docker 等无显示器环境下运行不报错
import matplotlib
matplotlib.use('Agg')

# 3.在控制流的例子中，我们计算d关于a的导数，如果将变量a更改为随机向量或矩阵，会发生什么？
# 如果 a 是向量或矩阵，d 也会变成向量或矩阵。
# 在自动微分中，通常需要计算的是标量对张量的导数。
# 如果 d 是张量，我们需要提供一个“梯度向量”来进行雅可比向量积运算。
# 结论：a 变为矩阵时，计算变为 Jacobians（雅可比矩阵）计算，通常需要指定梯度方向。

# 4.重新设计一个求控制流梯度的例子，运行并分析结果。
def control_flow_func(a, val):
    # 包含条件分支的控制流
    # 修改点：使用传入的数值 val 进行判断，而不是直接判断符号对象 a
    if val > 0: 
        return a**2
    else:
        return sp.exp(a)

a_sym = sp.symbols('a')

# 情况 A: a > 0 (传入数值 1 来引导控制流走向分支一)
res_pos = control_flow_func(a_sym, 1)
grad_pos = sp.diff(res_pos, a_sym)
print(f"当 a > 0 时, 函数为 a^2, 梯度为: {grad_pos}")

# 情况 B: a <= 0 (传入数值 -1 来引导控制流走向分支二)
res_neg = control_flow_func(a_sym, -1)
grad_neg = sp.diff(res_neg, a_sym)
print(f"当 a <= 0 时, 函数为 exp(a), 梯度为: {grad_neg}")

# 5.使 f(x) = sin(x)，绘制 f(x) 和 df(x)/dx 的图像，其中后者不使用 f'(x) = cos(x)。
# 准备数据
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)
f_vals = np.sin(x_vals)

# 使用数值微分 (Finite Difference) 而不使用 cos(x)
# 公式: f'(x) = [f(x+h) - f(x)] / h
h = 1e-5
df_vals = (np.sin(x_vals + h) - np.sin(x_vals)) / h

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f_vals, label=r'$f(x) = \sin(x)$', color='blue')
plt.plot(x_vals, df_vals, '--', label=r'Numerical $df(x)/dx$', color='orange')
plt.axhline(0, color='black', linewidth=0.5)
plt.title(r'Function $\sin(x)$ and its Numerical Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

output_file = '2.5.5.png'
plt.savefig(output_file)
plt.close()
print(f"图像已保存至: {output_file}")
