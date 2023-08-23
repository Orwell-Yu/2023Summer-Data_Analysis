import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib  
matplotlib.rc("font", family='YouYuan')  
import shutil  
import os  
  
def clean_outliers(data, output_folder):  
    outliers = []  
      
    for column in data.columns:  
        if data[column].nunique() <= 12:  # 仅对下拉菜单列进行离群值检测  
            value_counts = data[column].value_counts(normalize=True)  
            if len(value_counts) <= 12:  # 确保仅对下拉菜单列进行离群值检测  
                min_frequency = value_counts.min()  
                if min_frequency < 0.05:  # 小于5%的频率视为离群值  
                    outliers.extend(data[data[column] != value_counts.idxmax()].index)  
                    if outliers:  
                        # 保存离群值文件  
                        outliers_file_path = os.path.join(output_folder, str(column) + '_outliers.csv')  
                        data.loc[outliers].to_csv(outliers_file_path, index=False)  
                    else:  
                        print("没有离群值")  
  
def generate_bar_charts(data, output_folder):  
    # 生成柱状图  
    for column_name in data.columns[6:]:  
        data[column_name].value_counts().plot(kind='bar')  
        plt.xlabel('Category')  
        plt.ylabel('Count')  
        plt.title(column_name)  
        chart_file_path = os.path.join(output_folder, str(column_name) + '.png')  
        plt.savefig(chart_file_path)  
        plt.close()  
  
# 创建保存图表的文件夹  
output_folder = 'Outlier_Analysis'  
if os.path.exists(output_folder):  
    shutil.rmtree(output_folder)  
os.makedirs(output_folder, exist_ok=True)  
  
# 读取CSV文件  
data = pd.read_csv('one_hot.csv')  
  
# 生成柱状图  
generate_bar_charts(data, output_folder)  
  
# 数据清洗 - CSV文件  
clean_outliers(data, output_folder)  
  
# 读取Excel文件  
data = pd.read_excel('matrix_ex01.xlsx')  
  
# 数据清洗 - Excel文件  
clean_outliers(data, output_folder)  