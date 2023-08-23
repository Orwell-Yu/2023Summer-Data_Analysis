import pandas as pd  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import train_test_split  
import os  
import joblib
  
# 读取CSV数据集，指定第一行为标题行  
data = pd.read_csv('one_hot.csv',header=0)  
  
# 提取因变量列和自变量列  
dependent_vars = data.iloc[:, 1:6]  
independent_vars = data.iloc[:, 6:]  
  
# 创建保存结果的目录  
output_folder = 'DecisionTree_Analysis'  
if not os.path.exists(output_folder):  
    os.makedirs(output_folder)  
   
  
# 对每个因变量进行分析  
for col in dependent_vars.columns:  
    dependent_var = dependent_vars[col]  
      
    # 划分数据集和测试集  
    X_train, X_test, y_train, y_test = train_test_split(independent_vars, dependent_var, test_size=0.2)  
      
    # 构建回归树模型  
    model = DecisionTreeRegressor()  
    model.fit(X_train, y_train)  
      
    # 保存模型  
    model_filename = f'{output_folder}/model_{col}.joblib'  
    joblib.dump(model, model_filename)  
