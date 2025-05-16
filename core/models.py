# 输入: args.model_name, args.model_url, args.model_key, args.temperature
# 输出: model config(url, key, prompt)

import requests
from openai import OpenAI
from retry import retry
from core.logger import logger
from typing import List, Dict, Tuple, Callable


def default_post_process_func(reply: str) -> str:
    # deepseek api already handles this issue: it is within the 'reasoning_content' field
    if "</answer>" in reply:
        reply = reply.split("</answer>")[-1]
    elif "</think>" in reply:
        reply = reply.rsplit("</think>", 1)[1].strip()
    elif "---" in reply:
        reply = reply.rsplit("---", 1)[0].strip()
    return reply


class GPTModel:
    def __init__(
        self, config: dict, post_process_func: Callable = default_post_process_func
    ):
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 512)
        self.seed = config.get("seed", 42)
        self.post_process_func = post_process_func
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["api_url"],
        )

    @retry(tries=3, delay=1, backoff=2, logger=logger)
    def chat(self, messages: List[Dict], slow_think_prompt: str = None) -> Tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        reply = response.choices[0].message.content
        in_tokens = response.usage.prompt_tokens
        out_tokens = response.usage.completion_tokens

        reply = self.post_process(reply)
        return reply, in_tokens, out_tokens

    def post_process(self, reply: str) -> str:
        if self.post_process_func:
            return self.post_process_func(reply)
        return reply



class AgentModel:
    def __init__(self, configs: List[dict], post_process_funcs: List[Callable] = None):
        if post_process_funcs is None:
            post_process_funcs = [None] * len(configs)
        self.agents = [
            GPTModel(config, post_process_func)
            for config, post_process_func in zip(configs, post_process_funcs)
        ]

    def chat(self, messages_list: List[List[Dict]]):
        last_response = None
        in_tokens_total = 0
        out_tokens_total = 0
        for messages, model in zip(messages_list, self.agents):
            reply, in_tokens, out_tokens = model.chat(messages)
            reply = self.post_process(reply)
            messages.append({"role": "assistant", "content": reply})
            last_response = reply

        return last_response, in_tokens_total, out_tokens_total

    def post_process(self, reply: str) -> str:
        if self.post_process_func:
            return self.post_process_func(reply)
        return reply


class HuatuoModel:
    def __init__(self, config: dict, post_process_func: Callable = None):
        self.model_name = config["model_name"]
        self.api_url = config["api_url"]

        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 8192)
        self.seed = config.get("seed", 42)

    def chat(self, messages: List[Dict], slow_think_prompt: str = None) -> Tuple[str, int, int]:
        if messages[0]["role"] == "system":
            messages = messages[1:]

        in_tokens = 0
        out_tokens = 0

        # 如果最后一条消息是医生填写预问诊单的指令，则调用summary接口
        if messages[-1]["content"].startswith(
            "基于对话历史，你现在必须填写预问诊单。"
        ):
            response = requests.post(
                self.api_url + "/summary",
                json={
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "seed": self.seed,
                },
            )
            response = response.json()
            reply = response["choices"][0]["message"]["content"]
            return reply, in_tokens, out_tokens

        response = requests.post(
            self.api_url + "/chat/completions",
            json={
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
            },
        )
        response = response.json()
        reply = response["choices"][0]["message"]["content"]
        if "[医生]: " not in reply:
            reply = "[医生]: " + reply
        if "End of Dialog" in reply:
            # 如果最后一条消息是医生填写预问诊单的指令，则调用summary接口
            response = requests.post(
                self.api_url + "/summary",
                json={
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "seed": self.seed,
                },
            )
            response = response.json()
            reply = response["choices"][0]["message"]["content"]
        return reply, in_tokens, out_tokens



class PatientSlowThinkModel(GPTModel):
    def __init__(self, config: dict):
        super().__init__(config)

    def chat(self, messages: List[Dict], slow_think_prompt: str) -> Tuple[str, int, int]:
        if not messages or not isinstance(messages, list):
            raise ValueError("messages must be a non-empty list")
        
        if not slow_think_prompt or not isinstance(slow_think_prompt, str):
            raise ValueError("slow_think_prompt must be a non-empty string")

        in_tokens = 0
        out_tokens = 0
        try:
            first_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            in_tokens += first_response.usage.prompt_tokens
            out_tokens += first_response.usage.completion_tokens
            first_response = self.post_process(first_response.choices[0].message.content)
            first_response = self.patient_response_process(first_response)
            if not first_response:
                raise ValueError("First response is empty")
            doc_response = messages[-1]["content"]
            second_response, in_token_2, out_token_2 = self.slow_think(first_response, slow_think_prompt, doc_response)
            second_response = self.post_process(second_response)
            second_response = self.patient_response_process(second_response)
            in_tokens += in_token_2
            out_tokens += out_token_2
            return second_response, in_tokens, out_tokens
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise

    def slow_think(self, first_response: str, slow_think_prompt: str, doc_response: str) -> Tuple[str, int, int]:
        if not first_response or not isinstance(first_response, str):
            raise ValueError("first_response must be a non-empty string")
            
        if not slow_think_prompt or not isinstance(slow_think_prompt, str):
            raise ValueError("slow_think_prompt must be a non-empty string")
        doc_response_performance_prompt = f'''
[本轮对话中医生的问题]:
{doc_response}

[本轮对话中病人的回答]:
{first_response}


        '''
        slow_think_prompt_full = slow_think_prompt + doc_response_performance_prompt.format(
            doc_response=doc_response,
            patient_response=first_response
        )
        print(slow_think_prompt_full)
        messages = [
            {
                'role': 'user', 
                'content': slow_think_prompt_full
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            reply = response.choices[0].message.content
            in_token = response.usage.prompt_tokens
            out_token = response.usage.completion_tokens
            return reply, in_token, out_token
        except Exception as e:
            logger.error(f"Error in slow_think: {str(e)}")
            raise

    def patient_response_process(self, response: str) -> str:
        if '[病人]' in response:
            response = '[病人]' + response.rsplit('[病人]', 1)[1].strip()
        elif '[病人]' not in response:
            response = '[病人]: ' + response.strip()
        return response.strip()


class DoctorSlowThinkModel(GPTModel):
    def __init__(self, config: dict, post_process_func: Callable = default_post_process_func):
        super().__init__(config, post_process_func)

    def chat(self, messages: List[Dict], slow_think_prompt: str = None) -> Tuple[str, int, int]:
        if not messages or not isinstance(messages, list):
            raise ValueError("messages must be a non-empty list")
            
        if slow_think_prompt is None:
            # 如果没有提供slow_think_prompt，直接返回第一次的响应
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens
            reply = self.post_process_func(response.choices[0].message.content)
            reply = self.doctor_response_process(reply)
            return reply, tokens_in, tokens_out
            
        tokens_in_all = 0
        tokens_out_all = 0
        first_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )        
        tokens_in_all += first_response.usage.prompt_tokens
        tokens_out_all += first_response.usage.completion_tokens
        first_response = self.post_process_func(first_response.choices[0].message.content)
        first_response = self.doctor_response_process(first_response)
        if not first_response:
            raise ValueError("First response is empty")
        second_response, in_token_2, out_token_2 = self.slow_think(first_response, slow_think_prompt)
        if not second_response:
            raise ValueError("Second response is empty")
        second_response = self.post_process_func(second_response)
        second_response = self.doctor_response_process(second_response)
        tokens_in_all += in_token_2
        tokens_out_all += out_token_2
        return second_response, tokens_in_all, tokens_out_all


    def slow_think(self, first_response: str, slow_think_prompt: str) -> Tuple[str, int, int]:
        if not first_response or not isinstance(first_response, str):
            raise ValueError("first_response must be a non-empty string")
            
        if not slow_think_prompt or not isinstance(slow_think_prompt, str):
            raise ValueError("slow_think_prompt must be a non-empty string")
        messages = [
            {
                'role': 'system',
                'content': slow_think_prompt
            },
            {
                'role': 'user',
                'content': first_response
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            reply = response.choices[0].message.content
            in_token = response.usage.prompt_tokens
            out_token = response.usage.completion_tokens
            return reply, in_token, out_token
        except Exception as e:
            logger.error(f"Error in slow_think: {str(e)}")
            return None, 0, 0
            
            

    def doctor_response_process(self, response: str) -> str:
        if '[医生]' in response:
            response = '[医生]' + response.rsplit('[医生]', 1)[1].strip()
        elif '[医生]' not in response:
            response = '[医生]: ' + response.strip()
        return response.strip()

