import os

# 0.创建包含更多行和列的原始数据集。

# 当前 .py 文件所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# ../data 目录
data_dir = os.path.join(base_dir, ".", "data")
os.makedirs(data_dir, exist_ok=True)

# 更大一点的数据集
data_file = os.path.join(data_dir, "house_more.csv")

with open(data_file, "w", encoding="utf-8") as f:
    # 行列更多
    f.write("Id,NumRooms,Alley,LotArea,YearBuilt,RoofStyle,Heating,HasGarage,SaleCondition,Price\n")
    f.write("1,NA,Pave,8450,2003,Gable,GasA,Yes,Normal,208500\n")
    f.write("2,3,NA,9600,1976,Gable,GasA,Yes,Normal,181500\n")
    f.write("3,4,NA,11250,2001,Hip,GasA,Yes,Normal,223500\n")
    f.write("4,3,Pave,9550,1915,Gable,GasA,No,Abnorml,140000\n")
    f.write("5,4,NA,14260,2000,Gable,GasA,Yes,Normal,250000\n")
    f.write("6,2,Grvl,14115,1993,Gable,GasA,Yes,Normal,143000\n")
    f.write("7,3,NA,10084,2004,Gable,GasA,Yes,Normal,307000\n")
    f.write("8,3,Pave,10382,1973,Gable,GasA,Yes,Normal,200000\n")
    f.write("9,2,NA,6120,1931,Gable,GasA,No,Normal,129900\n")
    f.write("10,2,NA,7420,1939,Gable,GasA,Yes,Normal,118000\n")
    f.write("11,3,NA,11200,1965,Hip,GasA,Yes,Normal,140000\n")
    f.write("12,NA,NA,11924,2005,Hip,GasA,Yes,Partial,345000\n")

print("Created:", os.path.abspath(data_file))
