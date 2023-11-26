
import os

lujing = r"D:\Dataset\个人学习行为\睡觉图片"
file_list = os.listdir(lujing)
y = input('要改的类别数字：')
for file in file_list:
    try:
        list = []
        if file[-3:] == 'txt':
            cur_file = os.path.join(lujing, file)
            with open(cur_file, 'r+'):
                _r = open(cur_file)
                content = _r.readline()
                for i in range(len(content)):
                    list.append(content[i])
                print(list)
                list[0] = y
            with open(cur_file,'a+',encoding='utf-8') as test:
                test.truncate(0)
            new_content = ''.join(list)
            print(new_content)
            with open(cur_file,'a+',encoding='utf-8') as test:
                test.write(new_content)
    except:
        print("file")
