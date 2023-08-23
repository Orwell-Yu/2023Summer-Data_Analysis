import os  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib
matplotlib.rc("font",family='YouYuan')
import numpy as np  
from sklearn.linear_model import LinearRegression  
  
# 读取CSV文件  
data = pd.read_csv('one_hot.csv')  
  
# 获取输入变量和因变量  
X_columns = data.columns[6:]  # 选择第7列及之后的列作为输入变量  
y_columns = data.columns[2:6]  # 选择第3~6列作为因变量  
  
# 创建存储图表的文件夹  
output_folder = 'LinearRegression_Analysis'  
if not os.path.exists(output_folder):  
    os.makedirs(output_folder)  
  
# 用于存储每个因变量的斜率和对应的自变量名称  
max_slope_vars = {}  
  
# 逐列进行线性回归分析  
for y_column in y_columns:  
    y = data[y_column]  # 选择当前列作为因变量  
  
    # 创建存储散点图数据的列表  
    scatter_data = []  
  
    # 遍历输入变量列  
    for X_column in X_columns:  
        X = data[X_column].values.reshape(-1, 1)  # 将自变量转换为二维数组形式  
  
        # 创建并拟合线性回归模型  
        model = LinearRegression()  
        model.fit(X, y)  
  
        # 生成预测数据  
        X_pred = np.array([[0], [1]])  # 生成自变量的预测数据  
        y_pred = model.predict(X_pred)  # 根据模型进行预测  
  
        # 将散点图数据添加到列表中  
        scatter_data.append((X_pred, y_pred))  
  
    # 绘制散点图和回归线  
    fig, ax = plt.subplots()  
    for X_pred, y_pred in scatter_data:  
        ax.scatter(X_pred, y_pred, color='blue', label='Data Points')  
        ax.plot(X_pred, y_pred, color='red', linewidth=2, label='Linear Regression')  
  
    # 设置图表标题和标签  
    ax.set_title(f'Linear Regression for {y_column}')  
    ax.set_xlabel('Input Variable')  
    ax.set_ylabel('Target Variable')  
  
    # 调整图例的位置和字体大小  
    ax.legend(prop={'size': 4}, bbox_to_anchor=(1.02, 1), loc='upper left').remove()  
  
    # 保存图表图片  
    output_file = os.path.join(output_folder, f'{y_column}_analysis.png')  
    plt.savefig(output_file, bbox_inches='tight')  
  
    # 计算每个自变量的斜率  
    slopes = []  
    for i, (X_pred, y_pred) in enumerate(scatter_data):  
        slope = (y_pred[1] - y_pred[0]) / (X_pred[1] - X_pred[0])  
        slopes.append((X_columns[i], abs(slope)))  # 使用斜率的绝对值  
    
    # 根据斜率的绝对值对自变量进行排序，并获取斜率绝对值最大的两个自变量名称  
    slopes.sort(key=lambda x: x[1], reverse=True)  
    max_slope_vars[y_column] = [var for var, _ in slopes[:2]]    
  
# 打印每个因变量的名称和斜率最大的两个自变量名称  
for y_column, vars in max_slope_vars.items():  
    print(f"Dependent Variable: {y_column}")  
    print(f"Variables with Maximum Slope: {vars[0]}, {vars[1]}")  
    print()  
