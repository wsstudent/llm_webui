import pandas as pd

# 假设你的数据集是一个 JSON 格式的文件，读取数据集
data = pd.read_json(r'D:\1code\minicpm_2B\CJEval\data\CJEval_data\train\train_初中生物.json', encoding='utf-8',
                    lines=True)
print(data.head())


# 定义转换函数，将每条记录转换为目标格式
def convert_record(record):
    # 提取信息
    subject = record['subject']
    ques_type = record['ques_type']
    ques_difficulty = record['ques_difficulty']
    ques_content = record['ques_content']
    ques_analyze = record['ques_analyze']
    ques_knowledges = record['ques_knowledges']

    # 构建目标格式的内容字段
    content = f"类型#{subject}*题型#{ques_type}*难度#{ques_difficulty}*内容#{ques_content}*考察的知识点#{ques_knowledges}"



    # 构建目标格式的summary字段
    summary = f"{ques_analyze}"

    return {
        "content": content,
        "summary": summary
    }


# 对每一条记录应用转换函数
converted_data = data.apply(convert_record, axis=1)

# 将转换后的数据转换为 DataFrame 以方便后续处理
converted_df = pd.DataFrame(converted_data.tolist())

# 逐行写入 JSON 文件，每个对象独立成一行
with open('train_single_line.json', 'w', encoding='utf-8') as f:
    for _, row in converted_df.iterrows():
        # 将每行数据转为 JSON 字符串，并写入文件，每行一个独立的 JSON 对象
        json_str = row.to_json(force_ascii=False)
        f.write(json_str + '\n')  # 写入 JSON 对象并换行
