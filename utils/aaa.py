import os

folder_path = 'D:/BaiduNetdiskDownload/data_obb/val/labelTxt'  # 替换成你的文件夹路径

# 获取文件夹中所有txt文件的列表
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 遍历每个txt文件
for txt_file in txt_files:
    file_path = os.path.join(folder_path, txt_file)

    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 删除前两行内容
    lines = lines[2:]

    # 写回文件
    with open(file_path, 'w') as file:
        file.writelines(lines)

print('处理完成。')
