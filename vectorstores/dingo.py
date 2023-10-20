"""Wrapper around Dingo vector database."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance
import dingodb
from configs.kb_configs import CHUNK_SIZE, VECTOR_SEARCH_SCORE_THRESHOLD, VECTOR_SEARCH_TOP_K
import copy
from dingodb import DingoDB

from utils import branch_index
from utils.timing import log_function_time

# define logger object to write log
from utils.logger import setup_logger
logger = setup_logger()


class Dingo(VectorStore):
    """Wrapper around Dingo vector database.

    To use, you should have the ``dingodb`` python package installed.

    Example:
        .. code-block:: python

            from langchain.vectorstores import Dingo
            from langchain.embeddings.openai import OpenAIEmbeddings
            import dingodb
    """

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
        """Initialize with Dingo client."""
        # try:
        #     import dingodb
        # except ImportError:
        #     raise ValueError(
        #         "Could not import dingo python package. "
        #         "Please install it with `pip install dingodb."
        #     )

        # collection
        if client is not None:
            dingo_client = client
        else:
            try:
                # connect to dingo db
                dingo_client = dingodb.DingoDB(user, password, host)
            except ValueError as e:
                raise ValueError(f"Dingo failed to connect: {e}")

        self._text_key = text_key
        self._client = dingo_client
        # if exsits index_name

        if index_name is not None and index_name not in dingo_client.get_index():
            if self_id is True:
                dingo_client.create_index(index_name, 1024, index_type="flat", auto_id=False)
            else:
                dingo_client.create_index(index_name, 1024, index_type="flat")

        self._index_name = index_name
        self._embedding_function = embedding_function
        self.chunk_conent = True  # 是否考虑上下文
        self.chunk_size = CHUNK_SIZE
        self.score_threshold = VECTOR_SEARCH_SCORE_THRESHOLD

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            text_key: str = "text",
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
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

        ##product_id=branch_index()
        product_id='SCWD'
        embeds = []
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
            self._client.vector_add(self._index_name, metadatas[i:j], embeds[i:j], ids[i:j])
            # index.upsert(vectors=list(to_upsert), namespace=namespace)
        return ids

    @log_function_time()
    def similarity_search_with_score(
            self,
            query: str,
            k: int = VECTOR_SEARCH_TOP_K,  ###相似性的
            search_params: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return Dingo documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_params: Dictionary of argument(s) to filter on metadata

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs = []
        query_obj = self._embedding_function([query])
        #  index_name, xq, search_params, topk
        results = self._client.vector_search(self._index_name, xq=query_obj, top_k=k, search_params=search_params)

        if results == []:
            return []

        for res in results[0]["vectorWithDistances"]:
            metadatas = res["scalarData"]
            id = res["id"]
            score = res['distance']
            text = metadatas[self._text_key]['fields'][0]['data']
            if "answer_id" not in metadatas.keys():
                source = metadatas["source"]['fields'][0]['data']
                max_index = metadatas["max_index"]['fields'][0]['data']
                metadata = {'source': source, 'score': score, "max_index": max_index, "id": id, "text": text}
            else:
                answer_id = metadatas["answer_id"]
                metadata = {'answer_id': answer_id, "id": id, "text": text}
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    @log_function_time()
    def similarity_search(
            self,
            query: str,
            k: int = 4,
            search_params: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return dingo documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_params : Dictionary of argument(s) to filter on metadata

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, search_params=search_params
        )
        return [doc for doc, _ in docs_and_scores]

    def _similarity_search_with_relevance_scores(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self.similarity_search_with_score(query, k)

    @log_function_time()
    def max_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            search_params: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        results = self._client.vector_search(self._index_name, [embedding], search_params, k)

        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            [item["floatValues"] for item in results[0]["vectorWithDistances"]],
            k=k,
            lambda_mult=lambda_mult,
        )
        selected = [results[0]["vectorWithDistances"][i]["metaData"] for i in mmr_selected]
        return [
            Document(page_content=metadata.pop((self._text_key)), metadata=metadata)
            for metadata in selected
        ]

    @log_function_time()
    def max_marginal_relevance_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            search_params: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self._embedding_function(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, search_params
        )

    @classmethod
    def from_texts(
            cls,
            embedding: Embeddings,
            texts: List[str],
            metadatas: Optional[List[dict]] = None,
            #doc_list: List[Document],
            ids: Optional[List[str]] = None,
            text_key: str = "text",
            index_name: Optional[str] = None,
            client: Any = None,
            host: List[str] = ["172.20.31.10:13000"],
            user: str = "root",
            password: str = "123123",
            **kwargs: Any,
    ):
        """Construct Dingsssso wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Dingo index

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import Dingo
                from langchain.embeddings import OpenAIEmbeddings
                import dingodb
sss
                embeddings = OpenAIEmbeddings()
                dingo = Dingo.from_texts(
                    texts,
                    embeddings,
                    index_name="langchain-demo"
                )
        """
        # try:
        #     import dingodb
        # except ImportError:
        #     raise ValueError(
        #         "Could not import dingo python package. "
        #         "Please install it with `pip install dingodb`."
        #     )

        # indexes = pinecone.list_indexes()  # checks if provided index exists
        #
        # if index_name in indexes:
        #     index = pinecone.Index(index_name)
        # elif len(indexes) == 0:
        #     raise ValueError(
        #         "No active indexes found in your Pinecone project, "
        #         "are you sure you're using the right API key and environment?"
        #     )
        # else:
        #     raise ValueError(
        #         f"Index '{index_name}' not found in your Pinecone project. "
        #         f"Did you mean one of the following indexes: {', '.join(indexes)}"
        #     )

        # collection

        if client is not None:
            dingo_client = client
        else:
            try:
                # connect to dingo db
                dingo_client = dingodb.DingoDB(user, password, host)
            except ValueError as e:
                raise ValueError(f"Dingo failed to connect: {e}")
        max_index = None
        if kwargs != None and kwargs.get("self_id") is True:
            ##生成ids
            ##时间戳到毫秒
            ###
            now = datetime.now()
            timestamp = datetime.timestamp(now)
            ts13 = str(timestamp * 1000).split('.')[0]
            ids = [ts13 + str(i).zfill(6) for i in range(0, len(texts))]
            max_index = ids[-1]
            if index_name not in dingo_client.get_index():
                dingo_client.create_index(index_name, 1024, index_type="flat", auto_id=False)
        else:
            if index_name not in dingo_client.get_index():
                dingo_client.create_index(index_name, 1024, index_type="flat")
            # dingo_client.create_index(index_name, 1024, index_type="hnsw")

        product_id='SCWD'
        embeds = []
        for i in range(0, len(texts)):
            lines_batch = [texts[i]]

            embed = embedding.embed_documents(lines_batch)
            embeds.append(embed[0])
            metadatas[i][text_key] = lines_batch[0]
            metadatas[i]['product_id'] = product_id
            metadatas[i]['max_index'] = max_index  ###获取最大值
            ###生成主键

        # upsert to Dingo
        for i in range(0, len(texts), 1000):
            j = i + 1000
            dingo_client.vector_add(index_name, metadatas[i:j], embeds[i:j], ids[i:j])
            # index.upsert(vectors=list(to_upsert), namespace=namespace)
        return cls(embedding.embed_query, text_key, dingo_client, index_name)

    def delete(
            self,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        """Delete by vector IDs or filter.
        Args:
            ids: List of ids to delete.
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        return self._client.vector_delete(ids=ids)

    def seperate_list(self, ls: List[int]) -> List[List[int]]:  # 分句
        lists = []
        ls1 = [ls[0]]
        for i in range(1, len(ls)):
            if str(ls[i - 1])[:-6] == str(ls[i])[:-6]:  # 时间戳一样，代表着是同一个文件
                ls1.append(ls[i])
            else:
                lists.append(ls1)
                ls1 = [ls[i]]
        lists.append(ls1)
        return lists  # 返回划分后的句子列表

    @log_function_time()
    def similarity_search_with_chunk_conent(self,
                                            query: str,
                                            k: int = VECTOR_SEARCH_TOP_K,
                                            search_params: Optional[dict] = None):
        related_docs = self.similarity_search_with_score(query=query, k=k, search_params=search_params)
        if related_docs == []:
            return []

        scores = [doc.metadata["score"] for doc in related_docs]
        indices = [doc.metadata["id"] for doc in related_docs]  # doc在里面对应的index！！
        docs = []  # 最后要输出的文档全体
        id_set = set()
        rearrange_id_list = False  # 是否需要重排找到的上下文

        if not self.chunk_conent:
            docs = [doc for doc in related_docs if 0 <= doc.metadata["score"] <= self.score_threshold]
            return docs

        # 下面讨论包含上下文的情况
        for i in range(k):
            doc_score = scores[i]
            doc = related_docs[i]  # Document[i]
            if 0 < self.score_threshold < doc_score:
                continue
            doc_index = doc.metadata["id"]
            docs_len = 0  # 记录当前句子的上下文长度
            doc_start_index = 0
            doc_end_index = int(doc.metadata["max_index"]["fields"][0]["data"][-6:])
            doc_loc = int(str(doc.metadata["id"])[-6:])
            for kk in range(0, max(doc_loc - doc_start_index, doc_end_index - doc_loc)):
                break_flag = False
                expand_range = []
                if "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "forward":
                    if kk <= (doc_end_index - doc_loc):
                        expand_range = [doc_loc + kk]  # 向下文寻找
                elif "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "backward":
                    if kk <= (doc_loc - doc_start_index):
                        expand_range = [doc_loc - kk]  # 向上文寻找
                else:
                    if kk <= (doc_end_index - doc_loc):
                        expand_range.append(doc_loc + kk)
                    if kk <= (doc_loc - doc_start_index):
                        expand_range.append(doc_loc - kk)

                # expand_range: List[int]
                for l in expand_range:
                    expand_id = int(str(doc_index)[:-6] + str(l).zfill(6))  # 上下文所在id
                    if expand_id not in id_set:
                        doc0 = self._client.vector_get(index_name=self._index_name, ids=[expand_id])
                        PageContent = doc0[0].get("scalarData").get("text")["fields"][0]["data"]
                        if docs_len + len(PageContent) > self.chunk_size:
                            break_flag = True  # 如果找到的句子长度超出限制,终止循环
                            break
                        docs_len += len(PageContent)
                        id_set.add(expand_id)  # id_set： 当前query找到的相关上下文的句子在faiss库中的index
                        rearrange_id_list = True
                if break_flag:
                    break

        if not rearrange_id_list:  # 没有增加任何一个上下文信息，就直接退出
            docs = [d for d in related_docs if 0 <= doc.metadata["score"] <= self.score_threshold]
            return docs

        id_list = sorted(list(id_set))
        id_lists = self.seperate_list(id_list)  # 把得到的语句按照连续与否划分成多个子list
        for id_seq in id_lists:
            score = 2 * VECTOR_SEARCH_SCORE_THRESHOLD
            for id in id_seq:  # id_seq是属于相同文章的那部分答案
                if id == id_seq[0]:
                    doc00 = self._client.vector_get(index_name=self._index_name, ids=[id])
                    PageContent = doc00[0].get("scalarData").get("text")["fields"][0]["data"]
                    Source = doc00[0].get("scalarData").get("source")["fields"][0]["data"]
                    doc00 = Document(page_content=PageContent, metadata={"source": Source})
                else:
                    doc0 = self._client.vector_get(index_name=self._index_name, ids=[id])
                    PageContent0 = doc0[0].get("scalarData").get("text")["fields"][0]["data"]
                    doc00.page_content += " " + PageContent0

            if not isinstance(doc00, Document):
                raise ValueError(f"Could not find document for id {id_seq}, got {doc}")

            for item in related_docs:
                if str(item.metadata["id"])[:-6] == str(list(id_seq)[0])[:-6]:
                    score = min(score, item.metadata["score"])
            doc00.metadata["score"] = int(score)
            docs.append(doc00)
        return docs

    ##上下文重启






