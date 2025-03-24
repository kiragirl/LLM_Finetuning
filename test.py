import chardet
import pandas as pd

with open('chat_message_info(1).csv', 'rb') as f:
    result = chardet.detect(f.read())
    print(result)

# 加载原始CSV文件，指定其编码为'Windows-1252'
df = pd.read_csv('chat_message_info(1).csv', encoding='Windows-1252')

# 将DataFrame保存为新的CSV文件，指定目标编码格式为'UTF-8'
df.to_csv('converted_to_utf8.csv', encoding='utf-8', index=False)

# 如果需要转换为GBK编码，则可以如下操作：
df.to_csv('converted_to_gbk.csv', encoding='gbk', index=False)