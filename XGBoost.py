from XGBoost_testdata import Generate
import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
matplotlib.rc("font", family='YouYuan')

# 创建保存结果的目录
result_dir = "XGBoost_Analysis"
if os.path.exists(result_dir):
    # 如果目录已经存在，删除目录及其中的文件
    for file_name in os.listdir(result_dir):
        file_path = os.path.join(result_dir, file_name)
        os.remove(file_path)
    os.rmdir(result_dir)
os.mkdir(result_dir)

# 读取CSV文件
data = pd.read_csv("one_hot1.csv")

# -------------------------------------------------------------------------------------------------------------------------
# 提取因变量列和自变量列,关于“消耗”对时间进行降权处理
target_column = data.columns[1]
feature_columns = data.columns[7:]

# 设定终止日期
today = datetime(2023, 8, 15)

# 创建目标变量和特征变量的数据集
X = data[feature_columns]
y = data[target_column]

# 划分数据集和测试集
train_size = len(data)  # 使用所有数据作为训练集
X_train = X[:train_size]  # 将所有数据用于训练和测试
y_train = y[:train_size]  # 将所有数据用于训练和测试
X_test = Generate()  # 函数接口，尚未实现

# 根据日期计算样本权重
train_dates = pd.to_datetime(data['创建时间'][:train_size])

# 线性降权
min_date = train_dates.min()
train_weights = (today - train_dates).dt.days.apply(lambda x: max(0,
                                                                  1 - x / (today - min_date).days))

# 创建DMatrix对象，并传入样本权重
dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
dtest = xgb.DMatrix(X_test)

# 设置参数
params = {
    'objective': 'reg:squarederror',
    'eta': 0.1,
    'max_depth': 3
}

# 训练模型
model = xgb.train(params, dtrain)

# 进行预测，后续如果实现接口可以改成dtest
predictions = model.predict(dtrain)

# 找到使因变量最大和最小的一组自变量为True和False，后续如果实现的话可以改成X_test
max_idx = predictions.argmax()
min_idx = predictions.argmin()
max_row = X_train.iloc[max_idx]  # 后续如果实现的话可以改成X_test
min_row = X_train.iloc[min_idx]  # 后续如果实现的话可以改成X_test

# 创建结果DataFrame，并转置
result_df = pd.DataFrame({'Max': max_row, 'Min': min_row}).T

# 保存结果为Excel文件
result_file = os.path.join(result_dir, f"{target_column}_analysis.xlsx")
result_df.to_excel(result_file, index=False)

# 获取特征重要性分数
importance = model.get_score(importance_type='gain')
feature_scores = pd.DataFrame(
    list(importance.items()), columns=['Feature', 'Score'])
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# 创建柱状图
plt.figure(figsize=(15, 10))
plt.barh(feature_scores['Feature'], feature_scores['Score'])
plt.xlabel('Feature Score')
plt.ylabel('Features')
plt.title(f'Feature Importance - {target_column}')
plt.tight_layout()

# 保存柱状图为图片
result_file = os.path.join(
    result_dir, f"{target_column}_feature_importance.png")
plt.savefig(result_file)
plt.close()

# 保存特征重要性分数到Excel文件
feature_scores_file = os.path.join(
    result_dir, f"{target_column}_feature_scores.xlsx")
feature_scores.to_excel(feature_scores_file, index=False)

# -------------------------------------------------------------------------------------------------------------------------
# 提取因变量列和自变量列
target_column = data.columns[2]
feature_columns = data.columns[7:]

# 创建目标变量和特征变量的数据集
X = data[feature_columns]
y = data[target_column]

# 划分数据集和测试集
train_size = len(data)  # 使用所有数据作为训练集
X_train = X[:train_size]  # 将所有数据用于训练和测试
y_train = y[:train_size]  # 将所有数据用于训练和测试
X_test = Generate()  # 函数接口，尚未实现

# 创建DMatrix对象
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)  

# 设置参数
params = {
    'objective': 'reg:squarederror',
    'eta': 0.1,
    'max_depth': 3
}

# 训练模型
model = xgb.train(params, dtrain)

# 进行预测，后续如果实现接口可以改成dtest
predictions = model.predict(dtrain)

# 找到使因变量最大和最小的一组自变量为True和False，后续如果实现的话可以改成X_test
max_idx = predictions.argmax()
min_idx = predictions.argmin()
max_row = X_train.iloc[max_idx]  # 后续如果实现的话可以改成X_test
min_row = X_train.iloc[min_idx]  # 后续如果实现的话可以改成X_test

# 创建结果DataFrame，并转置
result_df = pd.DataFrame({'Max': max_row, 'Min': min_row}).T

# 保存结果为Excel文件
result_file = os.path.join(result_dir, f"{target_column}_analysis.xlsx")
result_df.to_excel(result_file, index=False)

# 获取特征重要性分数
importance = model.get_score(importance_type='gain')
feature_scores = pd.DataFrame(
    list(importance.items()), columns=['Feature', 'Score'])
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# 创建柱状图
plt.figure(figsize=(15, 10))
plt.barh(feature_scores['Feature'], feature_scores['Score'])
plt.xlabel('Feature Score')
plt.ylabel('Features')
plt.title(f'Feature Importance - {target_column}')
plt.tight_layout()

# 保存柱状图为图片
result_file = os.path.join(
    result_dir, f"{target_column}_feature_importance.png")
plt.savefig(result_file)
plt.close()

# 保存特征重要性分数到Excel文件
feature_scores_file = os.path.join(
    result_dir, f"{target_column}_feature_scores.xlsx")
feature_scores.to_excel(feature_scores_file, index=False)

# -------------------------------------------------------------------------------------------------------------------------
# 提取因变量列和自变量列，“点击率”和“播放率”都要降权
target_columns = data.columns[3:6]
feature_columns = data.columns[7:]

# 对每个因变量逐列进行分析
for target_col in target_columns:
    # 创建目标变量和特征变量的数据集
    X = data[feature_columns]
    y = data[target_col]

    # 划分数据集和测试集
    train_size = len(data)  # 使用所有数据作为训练集
    X_train = X[:train_size]  # 将所有数据用于训练和测试
    y_train = y[:train_size]  # 将所有数据用于训练和测试
    X_test = Generate()  # 函数接口，尚未实现

    # 根据日期计算样本权重
    train_dates = pd.to_datetime(data['创建时间'][:train_size])

    # 指数降权
    train_weights = (today - train_dates).dt.days.apply(lambda x: 0.9 ** x)

    # 创建DMatrix对象
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)  

    # 设置参数
    params = {
        'objective': 'reg:squarederror',
        'eta': 0.1,
        'max_depth': 3
    }

    # 训练模型
    model = xgb.train(params, dtrain)

    # 进行预测，后续如果实现接口可以改成dtest
    predictions = model.predict(dtrain)

    # 找到使因变量最大和最小的一组自变量为True和False，后续如果实现的话可以改成X_test
    max_idx = predictions.argmax()
    min_idx = predictions.argmin()
    max_row = X_train.iloc[max_idx]  # 后续如果实现的话可以改成X_test
    min_row = X_train.iloc[min_idx]  # 后续如果实现的话可以改成X_test

    # 创建结果DataFrame，并转置
    result_df = pd.DataFrame({'Max': max_row, 'Min': min_row}).T

    # 保存结果为Excel文件
    result_file = os.path.join(result_dir, f"{target_col}_analysis.xlsx")
    result_df.to_excel(result_file, index=False)

    # 获取特征重要性分数
    importance = model.get_score(importance_type='gain')
    feature_scores = pd.DataFrame(
        list(importance.items()), columns=['Feature', 'Score'])
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)

    # 创建柱状图
    plt.figure(figsize=(15, 10))
    plt.barh(feature_scores['Feature'], feature_scores['Score'])
    plt.xlabel('Feature Score')
    plt.ylabel('Features')
    plt.title(f'Feature Importance - {target_col}')
    plt.tight_layout()

    # 保存柱状图为图片
    result_file = os.path.join(
        result_dir, f"{target_col}_feature_importance.png")
    plt.savefig(result_file)
    plt.close()

    # 保存特征重要性分数到Excel文件
    feature_scores_file = os.path.join(
        result_dir, f"{target_col}_feature_scores.xlsx")
    feature_scores.to_excel(feature_scores_file, index=False)

print("分析完成并保存结果文件。")
