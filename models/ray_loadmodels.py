import os

from ray.serve.drivers import DAGDriver

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from transformers import AutoTokenizer, AutoModel

import torch
import time
from typing import Generator, Dict

import requests
from starlette.responses import StreamingResponse
from starlette.requests import Request

from ray import serve



def chatglm_auto_configure_device_map( num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: PEFT加载lora模型出现的层命名不同
    layer_prefix = 'transformer'

    device_map = {
            f"{layer_prefix}.embedding.word_embeddings": 0,
            f"{layer_prefix}.rotary_pos_emb": 0,
            f"{layer_prefix}.output_layer": 0,
            f"{layer_prefix}.encoder.final_layernorm": 0,
            f"base_model.model.output_layer": 0
        }
    encode = ".encoder"

    ##这个不是很懂
    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'{layer_prefix}{encode}.layers.{i}'] = gpu_target
        used += 1

    return device_map


class chatGLM():
    def __init__(self, model_name) -> None:
        from accelerate import dispatch_model
        model = AutoModel.from_pretrained(model_name,
                                          torch_dtype=torch.bfloat16,
                                          trust_remote_code=True).half()
        self.model = dispatch_model(model, device_map=chatglm_auto_configure_device_map(2)).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def __call__(self, prompt) -> any:
        max_length = 10000
        temperature = 0.01
        response, _ = self.model.chat(self.tokenizer, prompt, max_length=max_length, temperature=temperature)
        return response




@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus":2})
class StreamingResponder:

    def __init__(self):
        # import  sys
        st = time.time()
        self.llm = chatGLM(model_name="/home/huzongxing/load_model/6B")
        et = time.time() - st
        print(f" {et} seconds.")
        print("===== end =====")

    def generate_numbers(self, prompt: str) -> Generator[str, None, None]:
        for inum, (stream_resp, _) in enumerate(self.llm.model.stream_chat(
                self.llm.tokenizer ,
                prompt,
                history= [],
                max_length=10000,
                temperature= 0.01)):
            yield stream_resp
    async def __call__(self, request: Request) -> StreamingResponse:
        prompt = request.query_params.get("prompt", "你是谁")
        gen = self.generate_numbers(prompt)
        return StreamingResponse(gen, status_code=200, media_type="text/plain")

deployment_graph = StreamingResponder.bind()
# serve.run(StreamingResponder.bind())
# if __name__ == "__main__":
#     #serve.run(StreamingResponder.bind())
#     ##存在的问题  模型的加载 还有ip和端口是否可以改变
#     r = requests.get("http://localhost:8000/?prompt=你好", stream=True)
#     start = time.time()
#     r.raise_for_status()
#     for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
#         print(f"Got result {round(time.time()-start, 1)}s after start: '{chunk}'")

# def ceshi_test(pors='九章云极是什么'):
#     r =requests.get(f"http://172.20.1.32:7789?prompt={pors}", stream=True)
#     start = time.time()
#     r.raise_for_status()
#     for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
#         print(f"Got result {round(time.time()-start, 1)}s after start: '{chunk}'")
# #
