import json
import logging
import argparse
import openai
from openai import OpenAI
import time
import socket
import os

class RemoteLLMs:

    def init_local_client(self):
        """
        初始化用户的客户端
        :param args:
        :return:
        """
        raise NotImplementedError()

    def __load_args(self, config_path):
        # 首先读取Config
        self.args = json.load(open(config_path))

    def __init__(self, config_path):
        # 首先读取Config
        self.__load_args(config_path)
        self.max_retries = self.args.get("max_retries", 5)
        self.client = None
        for idx in range(self.max_retries):
            model = self.init_local_client()
            if model is not None:
                self.client = model
                break
        if self.client is None:
            raise ModuleNotFoundError()

    def create_prompt(self, history, current_query):
        pass

    def request_llm(self, context, seed=1234, sleep_time=1, repeat_times=0):
        pass

    def fit_case(self, pattern: str, data: dict, meta_dict: dict = None):

        if meta_dict is not None:
            for k,v in meta_dict.items():
                pattern = pattern.replace(k, str(v))

        for k, v in data.items():
            pattern = pattern.replace(k, str(v))

        assert '{{' not in pattern, pattern
        assert '}}' not in pattern, pattern
        return pattern


    def interactive_dialogue(self):
        """
        进行交互式的对话，进行模型检查
        :return:
        """
        contexts = []
        while True:
            current_query = input("请输入当前你的对话(输入'CLEAN'清除上下文，'END'离开对话)：")
            if current_query == "CLEAN":
                contexts = []
                print("已经清除上下文")
                continue
            elif current_query == "END":
                return
            contexts = self.create_prompt(current_query, contexts)
            results = self.request_llm(contexts)
            print("%s\t%s" % (results[-1]['role'], results[-1]['content']))


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=r"llms\applications\remote\configs\wsx_gpt35.json")
    args = parser.parse_args()
    return args


class ChatGPTLLM(RemoteLLMs):
    def init_local_client(self):
        try:
            self.model = self.args['model']
            client = OpenAI(api_key=self.args['api_key'], base_url=self.args['base_url'])
            return client
        except:
            return None

    def create_prompt(self, current_query, context=None):
        if context is None:
            context = []
        context.append(
            {
                "role": "user",
                "content": current_query,
            }
        )
        return context

    def request_llm(self, context, seed=1234, sleep_time=1, repeat_times=0):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=context,
                    stream=False,
                    seed=seed+repeat_times
                )
                context.append(
                    {
                        'role': response.choices[0].message.role,
                        'content': response.choices[0].message.content
                    }
                )
                return context
            except openai.RateLimitError as e:
                logging.error(str(e))
                raise e
            except (openai.APIError, openai.InternalServerError, socket.timeout) as e:
                logging.error(str(e))
                raise e
            except Exception as e:
                # 捕捉未预料的异常，考虑是否终止循环或做其他处理
                logging.error(f"An unexpected error occurred: {str(e)}")
                raise e
            time.sleep(sleep_time)

if __name__ == '__main__':
    # https://platform.openai.com/docs/api-reference
    args = read_args()
    chat_gpt = ChatGPTLLM(args.config_path)
    chat_gpt.interactive_dialogue()