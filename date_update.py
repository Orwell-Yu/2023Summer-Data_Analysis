import pandas as pd  

filepth='2023素材标签分析.xlsx'
# 读取 Excel 文件  
df = pd.read_excel(filepth)  
  
# 将"创建时间"列转换为日期时间格式  
df['创建时间'] = pd.to_datetime(df['创建时间'])  
  
# 修改日期格式为"xxx年xxx月"  
df['创建时间'] = df['创建时间'].dt.strftime('%Y年%m月')  
  
# 保存修改后的数据回原表  
df.to_excel(filepth, index=False)  