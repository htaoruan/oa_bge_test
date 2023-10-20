'''
Author: astor-oss
Date: 2023-07-21 07:23:17
LastEditors: huzx huzx@zetyun.com
LastEditTime: 2023-07-24 14:04:52
Description: 请填写简介
'''
import torch
import os
import re
from typing import List
from configs.kb_configs import *
from utils import kb_cfg_utils
from utils.timing import log_function_time
from langchain.document_transformers import (
    LongContextReorder,
)
# define logger object to write log
from utils.logger import setup_logger
logger = setup_logger()

'''
资源回收
'''
@log_function_time()
def torch_gc():
    if torch.cuda.is_available():
        print("Will clean cache")
        # with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")

'''
获取文件夹下的所有路径和文件名称
'''
def tree(filepath, ignore_dir_names=None, ignore_file_names=None):

    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]

'''
prompt_id 的对应关系
A:PROMPT_TEMPLATE
B:
C:
'''
def generate_prompt(related_docs: List[str],
                    query: str =None,
                    prompt_id: str = "ELSE", ) -> str:

    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(related_docs)
    PROMPT_TEMPLATE=PROMPT[prompt_id]

    if  prompt_id=='ANALY_ZJ':
        text = "\n".join(f" {text}" for text in enumerate(related_docs))
        prompt=PROMPT_TEMPLATE.replace("{context}", text)
    elif  prompt_id=='ANAY_TY':
        text = "\n".join(f" {text}" for text in enumerate(related_docs))
        prompt=PROMPT_TEMPLATE.replace("{context}", text).replace("{question}", query)
    else:
        context = "\n".join([doc.page_content for doc in reordered_docs])
        # 如果是预制问题，则将对应的Context增加预制答案
        built_in_answer = kb_cfg_utils.get_built_in_question(query)
        if (len(built_in_answer) > 3):
            context += built_in_answer
        logger.info(f"Current question is builtIn: {built_in_answer}")
        prompt = PROMPT_TEMPLATE.replace("{question}", query).replace("{context}", context).\
                    replace("{contextLen}", str(len(context)))

    return prompt

###正则匹配对数据进行分类
def generate_prompt_regex(file_name):
    import re
    mark_pro=''
    for  str_re in PRODUCT_ID:
        res_=re.search(str_re.upper(), file_name.upper())
        if res_ is not None:
            mark_pro=str_re
        if mark_pro=='':
            mark_pro='ELSE'
    return mark_pro




def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def get_vs_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "vector_store")

'''
获取文档分类
'''
def  branch_index(str_tmp):
    list_br=[]
    for pro in PRODUCT_ID:
        if pro in str_tmp :
            list_br.append(pro)
            break
    if len(list_br)>0:
        return list_br[0]
    else:
        return 'ELSE'
'''
功能：基于关键词进行匹配，确实关键词是否命中对应的文本内容。
参数：
    1. input 输入文本
    2. key_list 输入关键词
'''
def get_match_keywords(input: str, key_list: list) -> str:
    for item in key_list:
        print(item)
        print(input)
        match = re.search(item, input, re.IGNORECASE)
        if match:
            return item
    return "";