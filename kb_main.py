from vectorstores.IndivDingo import *
from vectorstores.dingo import *
from models.model_load import KbModel


# 数据库相关参数
index_name = "zetyun_new" 
KB_DEFAULT_VECTOR_USERNAME = 'root'
KB_DEFAULT_VECTOR_PASSWORD = '123456'
KB_DEFAULT_VECTOR_ADDRESS = "172.20.31.10:14000"

embeddings = KbModel("172.20.1.32:9911","172.20.1.32:9911")
# search_params=None
# search_params = {}
# 连接数据库初始化
vector_store = IndivDingo(
        embedding_function=embeddings.embed_documents,
        text_key="text",
        index_name=index_name,
        host=[KB_DEFAULT_VECTOR_ADDRESS],
        user=KB_DEFAULT_VECTOR_USERNAME,
        password=KB_DEFAULT_VECTOR_PASSWORD,
    )

query =  'dingodb的特点？'

qa_docs,related_docs_with_score = vector_store.similarity_search_with_chunk_conent_sx(query,k=6,index_name=index_name)#,search_params=search_params)
print(type(related_docs_with_score))
print(len(related_docs_with_score))
print(type(related_docs_with_score[0]))
print(related_docs_with_score[0].metadata)
print(type(related_docs_with_score[0].metadata))