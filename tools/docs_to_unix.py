#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py
"""
import pickle
import json
import os


def dos2unix(original, destination):
    content = ''
    outsize = 0
    with open(original, 'rb') as infile:
        content = infile.read()
    with open(destination, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))

    print("Done. Saved %s bytes." % (len(content) - outsize))


class StrToBytes:
    def __init__(self, file_obj):
        self.file_obj = file_obj

    def read(self, size):
        return self.file_obj.read(size).encode()

    def readline(self, size=-1):
        return self.file_obj.readline(size).encode()


def write_list(file_name, landmarks):
    fp = open(file_name, 'w+')
    fp.write("[" + '\n')
    for i in range(len(landmarks)):
        fp.write(str(landmarks[i]) + ',\n')

    fp.write("]")
    fp.close()
    return True


def write_list_to_json(my_list, json_file_name, json_file_save_path):
    """
    将list写入到json文件
    :param my_list:
    :param json_file_name: 写入的json文件名字
    :param json_file_save_path: json文件存储路径
    :return:
    """
    os.chdir(json_file_save_path)
    with open(json_file_name, 'w') as f:
        json.dump(my_list, f)


def str2bytes():
    try:
        with open('email_authors.pkl', 'r') as read_file:
            data_dict = pickle.load(StrToBytes(read_file))
            print(data_dict)

            with open('email_authors.pkl', 'wb') as write_file:
                pickle.dump(data_dict, write_file)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    print(1)
    # str2bytes()
    #
    # print(write_list('email_authors.txt', data_dict))
    #
    # write_list_to_json(data_dict, 'email_authors.json', "./")

    original = "../final_project/final_project_dataset.pkl"
    destination = "../final_project/final_project_dataset1.pkl"

    original = "../outliers/practice_outliers_net_worths.pkl"
    destination = "../outliers/practice_outliers_net_worths1.pkl"

    original = "../final_project/final_project_dataset.pkl"
    destination = "../final_project/final_project_dataset1.pkl"

    original = "../tools/python2_lesson14_keys.pkl"
    destination = "../tools/python2_lesson14_keys1.pkl"



    dos2unix(original, destination)

