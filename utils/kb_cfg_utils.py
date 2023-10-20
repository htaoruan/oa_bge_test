'''
Author: astor-oss
Date: 2023-07-22 22:09:09
LastEditors: huzx huzx@zetyun.com
LastEditTime: 2023-07-24 14:25:20
Description: 请填写简介
'''

import configparser
import random
import re

config = configparser.ConfigParser()
file='configs/kb_server_cfg.ini'
config.read(file)

def get_value_by_key(section:str, key:str) -> any:
    return config.get(section, key)

## Get KeyWords from key_wors_section by Key_words_list
def get_keywords() -> list:
    key_word_list = get_value_by_key('key_words_section', 'key_words_list').split()
    return key_word_list


def get_often_questions() -> list:
    question_dict = dict(config['question_section'])
    question_list = []
    for key,value in question_dict.items():
        question_list.append(value)
    return question_list


def get_top3_questions() -> list:
    question_list = get_often_questions()
    random_number = random.randint(0, 8)

    result_list = []
    for i in range(0, 3):
        index = (random_number + i) % 9
        result_list.append(question_list[index])
    return result_list


## 基于预制的问题查询对应的问答
def get_built_in_question(input:str) -> str:
    question_list = get_often_questions() 

    input_without_symbol =  re.sub(r'[^\w\s]|[，。！？]$', '', input)

    for item in question_list:
        if input_without_symbol in item:
            return item.split("##")[1]
        elif input_without_symbol.lower() in item.lower():
            return item.split("##")[1]
        else :
            continue

    return ""


##########配置文件的
def get_datanames() -> list:
    datanames_dict = dict(config['database_index'])

    return datanames_dict


def set_datanames(data_index) -> str:
    config.set('database_index', 'kb_vector_index_name', data_index['kb_vector_index_name'])
    config.set('database_index', 'kb_analy_index_name', data_index['kb_analy_index_name'])
    config.set('database_index', 'kb_vector_table_answer_name', data_index['kb_vector_table_answer_name'])
    with open(file, 'w') as configfile:
        config.write(configfile)
