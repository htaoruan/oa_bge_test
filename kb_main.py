# from vectorstores.IndivDingo import *
# from vectorstores.dingo import *
# from models.model_load import KbModel


# # 数据库相关参数
# index_name = "zetyun_new" 
# KB_DEFAULT_VECTOR_USERNAME = 'root'
# KB_DEFAULT_VECTOR_PASSWORD = '123456'
# KB_DEFAULT_VECTOR_ADDRESS = "172.20.31.10:14000"

# # 编码模型相关服务地址
# embeddings = KbModel("172.20.1.32:9911","172.20.1.32:9911")

# # 连接数据库初始化
# vector_store = IndivDingo(
#         embedding_function=embeddings.embed_documents,
#         text_key="text",
#         index_name=index_name,
#         host=[KB_DEFAULT_VECTOR_ADDRESS],
#         user=KB_DEFAULT_VECTOR_USERNAME,
#         password=KB_DEFAULT_VECTOR_PASSWORD,
#     )


# if __name__ == "__main__":
#     # 简单测试
#     # query =  'dingodb的特点？'
#     # qa_docs,related_docs_with_score = vector_store.similarity_search_with_chunk_conent_sx(
#     #         query,
#     #         k=5,
#     #         index_name=index_name
#     #         )
#     # print(qa_docs)
#     # print(related_docs_with_score)
#     # for chunk in related_docs_with_score:
#     #     print(chunk.metadata['text'])
#     query_list =  []
#     answer_list =  []
#     hit = 0
#     with open('/data/rhtao/multi-modal/bge_fine_tune/FlagEmbedding/datasets/query_val.txt', 'r') as file , open("/data/rhtao/multi-modal/bge_fine_tune/FlagEmbedding/datasets/answer_val.txt", "r") as answer_file:
#         queries =  file.readlines()
#         answers =  answer_file.readlines()
#         for query, answer in zip(queries, answers):
#             query_list.append(query)
#             answer_list.append(answer)
#     for i in range(len(query_list)):
#         query = query_list[i]
#         answer = answer_list[i]
#         qa_docs,related_docs_with_score = vector_store.similarity_search_with_chunk_conent_sx(
#             query,
#             k=6,
#             index_name=index_name
#             )
#         for chunk in related_docs_with_score[::-1]:
#             if chunk.metadata['text'] in answer:
#                 hit += 1
#                 # break
#     print(hit/len(query_list))

        

import csv
from vectorstores.IndivDingo import *
from vectorstores.dingo import *
from models.model_load import KbModel


# 数据库相关参数
index_name = "zetyun_new" 
KB_DEFAULT_VECTOR_USERNAME = 'root'
KB_DEFAULT_VECTOR_PASSWORD = '123456'
KB_DEFAULT_VECTOR_ADDRESS = "172.20.31.10:14000"

# 编码模型相关服务地址
embeddings = KbModel("172.20.1.32:9911","172.20.1.32:9911")

# 连接数据库初始化
vector_store = IndivDingo(
    embedding_function=embeddings.embed_documents,
    text_key="text",
    index_name=index_name,
    host=[KB_DEFAULT_VECTOR_ADDRESS],
    user=KB_DEFAULT_VECTOR_USERNAME,
    password=KB_DEFAULT_VECTOR_PASSWORD,
)

if __name__ == "__main__":
    query_list = []
    answer_list = []
    # correct_results = []
    # error_results = []
    correct_num = 0

    with open('/data/rhtao/multi-modal/bge_fine_tune/FlagEmbedding/datasets/query_val.txt', 'r') as file, open("/data/rhtao/multi-modal/bge_fine_tune/FlagEmbedding/datasets/answer_val.txt", "r") as answer_file:
        queries = file.readlines()
        answers = answer_file.readlines()
        for query, answer in zip(queries, answers):
            query_list.append(query)
            answer_list.append(answer)

    for i in range(len(query_list)):#len(query_list)
        query = query_list[i]
        answer = answer_list[i]
        qa_docs, related_docs_with_score = vector_store.similarity_search_with_chunk_conent_sx(
            query,
            k=6,
            index_name=index_name
        )
        top5chunks = [chunk.metadata['text'] for chunk in related_docs_with_score[::-1]]
        # print(top5chunks)

        for chunk in top5chunks:
            if chunk.strip() in answer:
                correct_num += 1
                # correct_results.append([answer, top5chunks])
                break
            # else:
            #     # error_results.append([answer, top5chunks])
            #     break
    print(correct_num/len(query_list))
    # # 将结果写入到correct.csv文件
    # with open('correct.csv', 'w', newline='') as correct_file:
    #     writer = csv.writer(correct_file)
    #     writer.writerow(['Answer', 'Chunk'])
    #     writer.writerows(correct_results)

    # # 将结果写入到error.csv文件
    # with open('error.csv', 'w', newline='') as error_file:
    #     writer = csv.writer(error_file)
    #     writer.writerow(['Answer', 'Chunk'])
    #     writer.writerows(error_results)


# 全匹配模式下的Top5正确率为:0.38822525597269625
# 忽略空格模式下的Top5正确率为: