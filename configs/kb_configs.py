import torch.backends
import os
import logging
import uuid


########################

FLAG_USER_NAME = uuid.uuid4().hex  ##使用主机ID, 序列号, 和当前时间来生成UUID

# 知识库默认存储路径(提供文件上传的功能，存储对应的文件信息)
KB_DEFAULT_ROOT_PATH = "/home/huzongxing/knowledge_base/OA知识库文档to宗星-训练-20230626"  ##docker
KB_ROOT_PATH = os.environ.get("KB_ROOT_PATH", KB_DEFAULT_ROOT_PATH)

KB_REMOTE_IP='http://172.20.1.32:8988/'   ##docker
FOLDER_NAME='Selfload'
# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
##PRODUCT_ID=['RT', 'APS' ,'DINGODB', 'BAP','DEP']

PROMPT={
"APS" :"""
情境(Situation):
       当前场景为面向企业的知识库系统，业务人员基于输入的知识进行问题问答。
       所问问题对应的上下文信息: {context}
任务(Task):
       基于用户的输入问题，进行回答。
要求:
       所有问题的回答，必须采用中文。
       if ({contextLen} > 0) 时，不允许直接回答 "根据已知信息，无法回答该问题"。
输入问题:
       {question}
回答:  """,

"RT" :"""
情境(Situation):
       当前场景为面向企业的知识库系统，业务人员基于输入的知识进行问题问答。
       所问问题对应的上下文信息: {context}
任务(Task):
       基于用户的输入问题，进行回答。
要求:
       所有问题的回答，必须采用中文。
       if ({contextLen} > 0) 时，不允许直接回答 "根据已知信息，无法回答该问题"。
输入问题:
       {question}
回答:  """,
"DINGODB" :"""
情境(Situation):
       当前场景为面向企业的知识库系统，业务人员基于输入的知识进行问题问答。
       所问问题对应的上下文信息: {context}
任务(Task):
       基于用户的输入问题，进行回答。
要求:
       所有问题的回答，必须采用中文。
       if ({contextLen} > 0) 时，不允许直接回答 "根据已知信息，无法回答该问题"。
输入问题:
       {question}
回答:  """,

"BAP" :"""
情境(Situation):
       当前场景为面向企业的知识库系统，业务人员基于输入的知识进行问题问答。
       所问问题对应的上下文信息: {context}
任务(Task):
       基于用户的输入问题，进行回答。
要求:
       所有问题的回答，必须采用中文。
       if ({contextLen} > 0) 时，不允许直接回答 "根据已知信息，无法回答该问题"。
输入问题:
       {question}
回答:  """,

"DEP" :"""
情境(Situation):
       当前场景为面向企业的知识库系统，业务人员基于输入的知识进行问题问答。
       所问问题对应的上下文信息: {context}
任务(Task):
       基于用户的输入问题，进行回答。
要求:
       所有问题的回答，必须采用中文。
       if ({contextLen} > 0) 时，不允许直接回答 "根据已知信息，无法回答该问题"。
输入问题:
       {question}
回答:  """,

    "ELSE" :"""
 情境(Situation):
      你作为一名严谨的企业知识专家，
      你只有权限看到以下可能有用的信息: {context}，你的回答准确性应当放在首位。
      现在需要你严谨地回答用户问题。
任务(Task):
      基于用户的输入问题，借助权限内信息进行回答。
要求:
     所有问题的回答，必须采用中文。回答要精简且准确。
     if ({contextLen} > 0) 时，不允许直接回答 "根据已知信息，无法回答该问题"。
     如果已知信息并不能解决当前问题，请直接回答"根据已知信息，无法回答该问题"。
     你的回答要严格准确且专业，不允许包含任何编造的成分。要注意的你的权限。
输入问题:
            {question}
        回答:
回答:  """,
"ANALY_ZJ" : """
现在开始你作为一个文章写作大师，现在需要你帮助我在从文章中主要内容段落，总结出来文章只要得内容，主要得段落内容是:/n
{context}。
按照上边段落中得内容，总结一下这篇文章主要内容是什么""",
"ANAY_TY":"""
从现在现在开始你作为技术文章大师，我现在需要你帮助我在从文章中段落回答的我的问题。
文章的段落内容是:/n
 {context}。
按照上边段落中得内容，回答一下我的问题：
我的问题是：
    {question} """
}

#################################### Parameters about Vector Database ####################################  ##docker

KB_DEFAULT_VECTOR_INDEX_NAME = "zetyun_new" #"jiaojian_poc"
KB_VECTOR_INDEX_NAME = os.environ.get("KB_VECTOR_INDEX_NAME", KB_DEFAULT_VECTOR_INDEX_NAME)

KB_ANALY_INDEX_NAME = "poc1_pdf"

KB_DEFAULT_VECTOR_USERNAME = 'root'
KB_DEFAULT_VECTOR_PASSWORD = '123456'

KB_DEFAULT_VECTOR_ADDRESS = "172.20.31.10:14000"
KB_VECTOR_ADDRESS = os.environ.get("KB_VECTOR_ADDRESS", KB_DEFAULT_VECTOR_ADDRESS)


###历史/主要问题的数据
KB_DEFAULT_TABLE_ANSWER_NAME = "jiaojian_answer"
KB_VECTOR_TABLE_ANSWER_NAME = os.environ.get("KB_VECTOR_TABLE_ANSWER_NAME", KB_DEFAULT_TABLE_ANSWER_NAME)
###历史/主要问题的数据
KB_DEFAULT_TABLE_HOST = "172.20.31.10"
KB_TABLE_HOST = os.environ.get("KB_TABLE_HOST", KB_DEFAULT_TABLE_HOST)
KB_DEFAULT_TABLE_PORT = 3308
KB_TABLE_PORT = os.environ.get("KB_TABLE_PORT", KB_DEFAULT_TABLE_PORT)
###历史/主要问题的数据
KB_DEFAULT_TABLE_USERNAME = 'root'
KB_DEFAULT_TABLE_PASSWORD = '123123'

# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 5

# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 1.1

##缓存知识库
LRUCACHE_NUM=1000

##########################################################################################################

########################################################################################################## #################################### Parameters about Document Parser ####################################
# 文本分句长度
SENTENCE_SIZE = 250
# 匹配后单段上下文长度
CHUNK_SIZE = 180
# 传入LLM的历史记录长度
LLM_HISTORY_LEN = 3
# LLM streaming reponse 是否是流式会带
STREAMING = True #False

####文档的标记列表
PRODUCT_ID=['RT', 'APS' ,'DINGODB', 'BAP','DEP']


NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

#################################################################################
######################### LOG CONFIGURATION #####################################
# notset, debug, info, warning, error, or critical
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")

LOG_DEFAULT_FILE = 'api_server.log'
LOG_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs_out")
if not os.path.exists(LOG_ROOT_PATH):
    os.makedirs(LOG_ROOT_PATH)
#################################################################################