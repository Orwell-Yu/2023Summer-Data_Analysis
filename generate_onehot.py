import pandas as pd  
  
# 读取Excel文件  
data = pd.read_excel('matrix_ex02.xlsx')  
  
# 对每一列进行独热编码  
encoded_data = pd.get_dummies(data, columns=data.columns[7:])  
  
# 将清洗后的数据输出为CSV文件  
encoded_data.to_csv('one_hot1.csv', index=False)  