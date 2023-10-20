from typing import Optional, List, Tuple, Callable, Any, Iterable, Dict

import numpy as np
from langchain.vectorstores import VectorStore
from langchain.docstore.document import Document
from langchain.vectorstores.utils import maximal_marginal_relevance

from configs.kb_configs import VECTOR_SEARCH_TOP_K
from utils import log_function_time
from vectorstores import Dingo
from datetime import datetime

class IndivDingo(Dingo, VectorStore):

    def __init__(
            self,
            embedding_function: Callable,
            text_key: str,
            client: Any = None,
            index_name: Optional[str] = None,
            index_params: Optional[dict] = None,
            search_params: Optional[dict] = None,
            host: List[str] = ["172.20.31.10:13000"],
            user: str = "root",
            password: str = "123123",
            self_id=True

    ):
        super().__init__(embedding_function=embedding_function,
            text_key=text_key,
            index_name=index_name ,
            index_params= index_params,
            search_params=search_params,
            host =host,
            user = user,
            password =password,
            self_id = self_id
        )



    #删表 重新建表
    def clearn_index(self,index_name:str ='kb_test' ,self_id=True,**kwargs: Any):

        if index_name is not None and index_name  in  self._client.get_index():
            self._client.delete_index(index_name)

        if self_id is True:
            self._client.create_index(index_name, 1024, index_type="flat", auto_id=False)
        else:
            self._client.create_index(index_name, 1024, index_type="flat")



    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            text_key: str = "text",
            **kwargs: Any,
    ) -> List[str]:

        if kwargs != None and "index_name" in kwargs.keys():
            index_name = kwargs.get("index_name")
            if  index_name  not  in self._client.get_index():
                self._client.create_index(index_name, 1024, index_type="flat", auto_id=False)
        else :
            index_name = self._index_name
        ###自动生成
        max_index = []
        if kwargs != None and kwargs.get("self_id") is True:
            ##生成ids
            ##时间戳到毫秒
            now = datetime.now()
            timestamp = datetime.timestamp(now)
            ts13 = str(timestamp * 1000).split('.')[0]
            ids = [ts13 + str(i).zfill(6) for i in range(0, len(texts))]
            max_index = ids[-1]


        if kwargs != None and "product_id" in kwargs.keys() :
            product_id=kwargs.get("product_id")
        else:
            product_id='SCWD'

        embeds = []
        print(len(texts))
        for i in range(0, len(texts)):
            lines_batch = [texts[i]]
            embed = self._embedding_function(lines_batch)
            embeds.append(embed[0])
            metadatas[i][text_key] = lines_batch[0]
            metadatas[i]['product_id'] = product_id
            metadatas[i]['max_index'] = max_index
            ###生成主键

        # upsert to Dingo
        for i in range(0, len(texts), 1000):
            j = i + 1000
            self._client.vector_add(index_name, metadatas[i:j], embeds[i:j], ids[i:j])
        return embeds,metadatas

    '''
    解析的数据的方法
    '''
    def del_results(self,results):
        docs = []
        for res in results:
            id = res["id"]
            metadatas = res["scalarData"]
            score = res['distance']
            product_id = metadatas["product_id"]['fields'][0]['data']
            text = metadatas[self._text_key]['fields'][0]['data']
            if "answer_id" in metadatas.keys():
                answer_id = metadatas["answer_id"]['fields'][0]['data']
                metadata = {'source': product_id, 'score': score, "id": id, "text": text, 'product_id': product_id,
                            'answer_id': answer_id}
            elif "page" in metadatas.keys():
                source_id = metadatas["source_id"]['fields'][0]['data']
                page_id = metadatas["page"]['fields'][0]['data']
                max_index = metadatas["max_index"]['fields'][0]['data']
                metadata = {'source_id': source_id, 'score': score, "id": id, "text": text, 'product_id': product_id,
                            'page': page_id,"max_index": max_index}
            else:
                source = metadatas["source"]['fields'][0]['data']
                max_index = metadatas["max_index"]['fields'][0]['data']
                metadata = {'source': source, 'score': score, "max_index": max_index, "id": id, "text": text,
                            'product_id': product_id}

            docs.append(Document(page_content=text, metadata=metadata))
        return   docs


    ##相似度查询
    @log_function_time()
    def similarity_search_with_score(
            self,
            query: str,
            k: int = VECTOR_SEARCH_TOP_K,  ###相似性的
            search_params: Optional[dict] = None,
            **kwargs: Any
    ) -> List[Tuple[Document, float]]:

        if kwargs != None and "index_name" in kwargs.keys():
            index_name = kwargs.get("index_name")
        else:
            index_name = self._index_name

        query_obj = self._embedding_function([query])
        results = self._client.vector_search(index_name, xq=query_obj, top_k=k, search_params=search_params)

        if results == []:
            return []
        else:
            return  self.del_results(results[0]["vectorWithDistances"])






    ##mmr数据筛选
    @log_function_time()
    def max_marginal_relevance_search_by_vector(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 15,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        search_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:

        if kwargs != None and "index_name" in kwargs.keys():
            index_name = kwargs.get("index_name")
        else:
            index_name = self._index_name


        query_obj = self._embedding_function([query])
        # print('***********',query_obj)

        results = self._client.vector_search(index_name, xq=query_obj, top_k=fetch_k, search_params=search_params)
        if results == []:
            return []
        else:
            mmr_selected = maximal_marginal_relevance(
                np.array(query_obj, dtype=np.float32),
                [i['vector']['floatValues'] for i in   results[0]["vectorWithDistances"]],
                k=k,
                lambda_mult=lambda_mult,
            )

            selected_results = [r for i, r in enumerate(results[0]["vectorWithDistances"]) if i in mmr_selected]
            return self.del_results(selected_results)

    ###上下文查询
    @log_function_time()
    def similarity_search_with_chunk_conent_sx(self,
                                                 query: str,
                                                 k: int = VECTOR_SEARCH_TOP_K,
                                                 search_params: Optional[dict] = None,
                                                 **kwargs: Any):

        if kwargs != None and "index_name" in kwargs.keys():
            index_name = kwargs.get("index_name")
        else:
            index_name = self._index_name

        related_docs = self.similarity_search_with_score(query=query, k=k, search_params=search_params,index_name=index_name)
        ##mmr临近搜索程序
        #
        # related_docs = self.max_marginal_relevance_search_by_vector(query=query, k=k, search_params=search_params,
                                                        #  index_name=index_name)

        if related_docs == []:
            return [],[]

        indices = []  # doc在里面对应的index！！
        doc_end_index_all = []
        docs = []  # 文档中的返回数据
        docs_qa=[]  ##QA的返回数据
        #if_answer = False  # 是否需要重排找到的上下文
        ##
        for doc_tmp in related_docs:
            score = doc_tmp.metadata["score"]
            ##相似问题的查找
            if "answer_id" in doc_tmp.metadata.keys():
                if abs(score) < 0.25: ##后期写方法进行 ##匹配度的问题
                     return [doc_tmp], []
                if abs(score)  >= 0 and score <= self.score_threshold:
                    docs_qa.append(doc_tmp)
            else:
                if score >= 0 and score <= self.score_threshold:
                    doc_end_index = int(str(doc_tmp.metadata["max_index"])[-6:])
                    indice = str(doc_tmp.metadata["id"])
                    doc_end_index_all.append(doc_end_index)
                    indices.append(indice)
                    docs.append(doc_tmp)

        if not self.chunk_conent:
            return docs_qa,docs

        doc_start_index = 0
        xs = 10  ##定量
        index_all = []
        id_set = set()
        for i, indexs in enumerate(indices):
            doc_end_index = doc_end_index_all[i]
            index_tmp = []
            for tmps in range(max(int(indexs[-6:]) - xs, doc_start_index),
                              min(int(indexs[-6:]) + xs, doc_end_index) + 1):
                index_tmp.append(str(indexs)[:13] + str(tmps).zfill(6))
            index_all = index_all + index_tmp
        id_set.update(index_all)

        docxs_all = self._client.vector_get(index_name=index_name, ids=id_set, vector=False)

        dict_all = {}
        for doc in docxs_all:
            id = str(doc.get("id"))
            dict_all[id] = doc.get("scalarData").get("text").get("fields")[0]['data']

        index_alls = []
        for i, indice in enumerate(indices):
            doc_end_index = doc_end_index_all[i]
            doc = docs[i]
            ids_q = str(indice[:13])
            ids = int(indice[-6:])
            for j in range(1, xs):
                s = ids_q + str(max(doc_start_index, ids - j)).zfill(6)
                x = ids_q + str(min(doc_end_index, ids + j)).zfill(6)
                if s not in index_alls and s != str(indice):
                    page_content_s = dict_all.get(s, "")
                    index_alls.append(s)
                    doc.page_content = page_content_s + doc.page_content
                if x not in index_alls and x != str(indice):
                    pass_content_x = dict_all.get(x, "")
                    index_alls.append(x)
                    doc.page_content = doc.page_content + pass_content_x
                docs[i] = doc
                if len(docs) == 1:
                    if len(doc.page_content) > 2 * self.chunk_size:
                        break
                else:
                    if len(doc.page_content) > self.chunk_size:
                        break

        return docs_qa,docs

