import os


def process_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        processed_lines = []
        for line in lines:
            data = line.strip().split()
            if len(data) >= 6:
                processed_data = data[:1] + [f'{float(x):.6f}' for x in data[1:5]] + data[5:]
                processed_lines.append(' '.join(processed_data) + '\n')
            else:
                processed_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(processed_lines)


def process_all_txt_files(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            process_txt_file(file_path)


# 指定文件夹路径
folder_path = 'C:/Users/zhr/Desktop/labels'
process_all_txt_files(folder_path)
