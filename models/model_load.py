import json
import requests


class KbModel:
    def __init__(self, llm_server, emb_server):
        self.llm_server = llm_server
        self.emb_server = emb_server
        self.llm_url = f"http://{self.llm_server}/llm"
        self.llm_chat_url = f"http://{self.llm_server}/llm_chat"
        self.emb_documents_url = f"http://{self.emb_server}/embeddings"
        self.emb_query_url = f"http://{self.emb_server}/embedding"
        
    def llm_chat(self, prompt):
        data = {"prompt": prompt}
        res = requests.post(self.llm_chat_url, data=json.dumps(data)
        )
        return res.json()
    
    def stream_chat(self, prompt):
        data = {"prompt": prompt}
        res = requests.post(self.llm_url, data=json.dumps(data), stream=True
        )
        return res
        
    def embed_query(self, text):
        data = {"text": text}
        res = requests.post(self.emb_query_url, data=json.dumps(data)
        )
        return res.json()
    
    def embed_documents(self, texts):
        data = {"texts": texts}
        res = requests.post(self.emb_documents_url, data=json.dumps(data)
        )
        return res.json()
        