import os  
import pandas as pd  
import xgboost as xgb  
import plotly.graph_objects as go  
from datetime import datetime  
  
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
  
# 提取因变量列和自变量列,关于“消耗”对时间进行降权处理  
target_column = data.columns[1]  
feature_columns = data.columns[7:]  
  
# 设定终止日期  
today = datetime(2023, 8, 15)  
  
# 创建目标变量和特征变量的数据集  
X = data[feature_columns]  
y = data[target_column]  
  
# 划分数据集和测试集  
train_size = int(0.8 * len(data))  
X_train, X_test = X[:train_size], X[train_size:]  
y_train, y_test = y[:train_size], y[train_size:]  
  
# 根据日期计算样本权重  
train_dates = pd.to_datetime(data['创建时间'][:train_size])  
test_dates = pd.to_datetime(data['创建时间'][train_size:])  
train_weights = (today - train_dates).dt.days.apply(lambda x: 0.9 ** x)  
test_weights = (today - test_dates).dt.days.apply(lambda x: 0.9 ** x)  
  
# 创建DMatrix对象，并传入样本权重  
dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)  
dtest = xgb.DMatrix(X_test, weight=test_weights)  
  
# 设置参数  
params = {  
    'objective': 'reg:squarederror',  
    'eta': 0.1,  
    'max_depth': 3  
}  
  
# 训练模型  
model = xgb.train(params, dtrain)  
  
# 进行预测  
predictions = model.predict(dtest)  
  
# 找到使因变量最大和最小的一组自变量为True和False  
max_idx = predictions.argmax()  
min_idx = predictions.argmin()  
max_row = X_test.iloc[max_idx]  
min_row = X_test.iloc[min_idx]  
  
# 创建结果DataFrame，并转置  
result_df = pd.DataFrame({'Max': max_row, 'Min': min_row}).T  
  
# 保存结果为Excel文件  
result_file = os.path.join(result_dir, f"{target_column}_analysis.xlsx")  
result_df.to_excel(result_file, index=False)  
  
# 获取特征重要性分数  
importance = model.get_score(importance_type='gain')  
feature_scores = pd.DataFrame(list(importance.items()), columns=['Feature', 'Score'])  
feature_scores = feature_scores.sort_values(by='Score', ascending=False)  
  
# 创建柱状图  
fig = go.Figure(data=[go.Bar(x=feature_scores['Score'], y=feature_scores['Feature'], orientation='h')])  
fig.update_layout(  
    title=f'Feature Importance - {target_column}',  
    xaxis_title='Feature Score',  
    yaxis_title='Features',  
    height=1000,  
    width=1500  
)  
fig.write_image(os.path.join(result_dir, f"{target_column}_feature_importance.png")) 

# print(len(feature_columns))   
  
# 提取因变量列和自变量列  
target_columns = data.columns[2:6]  
feature_columns = data.columns[7:]  
  
# 对每个因变量逐列进行分析  
for target_col in target_columns:  
    # 创建目标变量和特征变量的数据集  
    X = data[feature_columns]  
    y = data[target_col]  
  
    # 划分数据集和测试集  
    train_size = int(0.8 * len(data))  
    X_train, X_test = X[:train_size], X[train_size:]  
    y_train, y_test = y[:train_size], y[train_size:]  
  
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
  
    # 进行预测  
    predictions = model.predict(dtest)  
  
    # 找到使因变量最大和最小的一组自变量为True和False  
    max_idx = predictions.argmax()  
    min_idx = predictions.argmin()  
    max_row = X_test.iloc[max_idx]  
    min_row = X_test.iloc[min_idx]  
  
    # 创建结果DataFrame，并转置  
    result_df = pd.DataFrame({'Max': max_row, 'Min': min_row}).T  
  
    # 保存结果为Excel文件  
    result_file = os.path.join(result_dir, f"{target_col}_analysis.xlsx")  
    result_df.to_excel(result_file, index=False)  
  
    # 获取特征重要性分数  
    importance = model.get_score(importance_type='gain')  
    feature_scores = pd.DataFrame(list(importance.items()), columns=['Feature', 'Score'])  
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)  
  
    # 创建柱状图  
    fig = go.Figure(data=[go.Bar(x=feature_scores['Score'], y=feature_scores['Feature'], orientation='h')])  
    fig.update_layout(  
        title=f'Feature Importance - {target_col}',  
        xaxis_title='Feature Score',  
        yaxis_title='Features',  
        height=1000,  
        width=1500  
    )  
    fig.write_image(os.path.join(result_dir, f"{target_col}_feature_importance.png")) 
    
    # 保存特征重要性分数到Excel文件    
    feature_scores_file = os.path.join(result_dir, f"{target_col}_feature_scores.xlsx")    
    feature_scores.to_excel(feature_scores_file, index=False) 
    
    # print(len(feature_columns))  
  
print("分析完成并保存结果文件。")  