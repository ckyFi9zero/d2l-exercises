import sympy as sp
import numpy as np

import matplotlib
# 强制指定使用 Agg 后端，确保在 Docker/无 GUI 环境下运行不报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 绘制函数 y = f(x) = x^3 - 1/x 和其在 x = 1 处切线的图像。
x_sym = sp.symbols('x')
f_expr = x_sym**3 - 1/x_sym

# 计算切点 (1, f(1))
y1 = f_expr.subs(x_sym, 1)
# 计算导数 f'(x) 并求出 x=1 处的斜率 k
df_expr = sp.diff(f_expr, x_sym)
slope = df_expr.subs(x_sym, 1)

# 切线方程: y = k(x - 1) + y1
print(f"函数表达式: f(x) = {f_expr}")
print(f"在 x=1 处的斜率: k = {slope}")
print(f"切线方程: y = {slope}*x + ({y1 - slope * 1})")

# 可视化准备
x_vals = np.linspace(0.5, 2, 400)
y_vals = x_vals**3 - 1/x_vals
tangent_vals = float(slope) * x_vals + float(y1 - slope * 1)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label=r'$f(x) = x^3 - \frac{1}{x}$')
plt.plot(x_vals, tangent_vals, '--', label='Tangent at x=1')
plt.scatter([1], [float(y1)], color='red', label='Point (1,0)')
plt.title('Function and its Tangent Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 保存图像
output_file1 = '2.4.1.png'
plt.savefig(output_file1)
plt.close()
print(f"图像已保存至: {output_file1}")

# 2. 求函数 f(x) = 3x1^2 + 5e^(x2) 的梯度。
x1, x2 = sp.symbols('x1 x2')
f2 = 3*x1**2 + 5*sp.exp(x2)
# 计算偏导数组成梯度向量
grad_f2 = [sp.diff(f2, x1), sp.diff(f2, x2)]
print(f"函数: f(x1, x2) = {f2}")
print(f"梯度 ∇f = {grad_f2}")



# 3. 函数 f(x) = ||x||_2 的梯度是什么？
# 以三维向量 x = [x1, x2, x3] 为例进行符号推导
v = sp.symbols('x1 x2 x3')
f3 = sp.sqrt(sum(xi**2 for xi in v))
grad_f3 = [sp.diff(f3, xi) for xi in v]

print("函数 (L2范数): f(x) = sqrt(x1^2 + x2^2 + x3^2)")
print("各分量的梯度为:")
for i, g in enumerate(grad_f3):
    print(f"  ∂f/∂x{i+1} = {g}")
print("通用结论: ∇f(x) = x / ||x||_2 (当 x ≠ 0 时)")


# 4. 尝试写出函数 u = f(x, y, z)，其中 x = x(a, b), y = y(a, b), z = z(a, b) 的链式法则。
a, b = sp.symbols('a b')
# 定义抽象函数关系
x_func = sp.Function('x')(a, b)
y_func = sp.Function('y')(a, b)
z_func = sp.Function('z')(a, b)
f_func = sp.Function('f')(x_func, y_func, z_func)

# 自动推导链式法则结构
du_da = sp.diff(f_func, a)
du_db = sp.diff(f_func, b)

print("多元函数链式法则公式:")
print(f"∂u/∂a = {du_da}")
print(f"∂u/∂b = {du_db}")
