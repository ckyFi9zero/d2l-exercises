import os
import pandas as pd
import torch

os.makedirs(os.path.join('data'), exist_ok=True)
data_file = os.path.join('data', 'house_more.csv')

# 读数据
data = pd.read_csv(data_file)
print("Original shape:", data.shape)
print(data.head())

# 1.删除缺失值最多的列。
# 说明：如果缺失值最多的列不止一个，则全部删除
na_count = data.isna().sum()
max_na = na_count.max()
cols_to_drop = na_count[na_count == max_na].index.tolist()

print("\nMissing values per column:\n", na_count)
print("\nMax missing:", max_na)
print("Drop columns:", cols_to_drop)

data = data.drop(columns=cols_to_drop)
print("After drop shape:", data.shape)

# 2.将预处理后的数据集转换为张量格式。
# 假设 Price 是标签列（回归）
label_col = "Price"
if label_col not in data.columns:
    raise ValueError(f"Label column '{label_col}' not found. Columns: {list(data.columns)}")

# 分离特征和标签
y = data[label_col]
X = data.drop(columns=[label_col])

# 数值列/类别列分别处理缺失值
numeric_cols = X.select_dtypes(include=["number"]).columns
categorical_cols = X.select_dtypes(exclude=["number"]).columns

# 数值列：用均值填充
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

# 类别列：用众数填充（若整列都是 NA，则用字符串 "Unknown"）
for c in categorical_cols:
    if X[c].isna().all():
        X[c] = X[c].fillna("Unknown")
    else:
        X[c] = X[c].fillna(X[c].mode(dropna=True)[0])

# one-hot 编码类别特征
X = pd.get_dummies(X, dummy_na=False)

print("\nProcessed feature shape:", X.shape)
print("Processed columns:", list(X.columns)[:10], "...")

# 转为 torch 张量
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

print("\nX_tensor shape:", X_tensor.shape)
print("y_tensor shape:", y_tensor.shape)
