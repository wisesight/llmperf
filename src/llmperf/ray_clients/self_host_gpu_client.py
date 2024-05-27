import io
import json
import os
import time
from typing import Any, Dict
import requests
import boto3
import ray
from transformers import LlamaTokenizerFast

from llmperf import common_metrics
from llmperf.models import RequestConfig
from llmperf.ray_llm_client import LLMClient


@ray.remote
class SelfHostGPUClient(LLMClient):
    """Client for OpenAI Chat Completions API."""

    def __init__(self):
        # Sagemaker doesn't return the number of tokens that are generated so we approximate it by
        # using the llama tokenizer.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )
    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            pass
        if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            pass
        if not os.environ.get("AWS_REGION_NAME"):
            pass

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        model = request_config.model
        sampling_params = request_config.sampling_params
        message = {
            "prompt": prompt,
            "n": 1,
            "use_beam_search": False,
            }
        
        message["max_tokens"] = sampling_params["max_tokens"]
        
        tokens_received = 0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        metrics = {}

        start_time = time.monotonic()

        try:
            response = requests.post(model, json= message)
            total_request_time = time.monotonic() - start_time
            generated_text = response.json()["text"][0]
            tokens_received = len(self.tokenizer.encode(generated_text))
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            print(f"Warning Or Error: {e}")
            print(error_response_code)
            error_msg = str(e)
            error_response_code = 500

        metrics[common_metrics.ERROR_MSG] = error_msg
        metrics[common_metrics.ERROR_CODE] = error_response_code
        metrics[common_metrics.INTER_TOKEN_LAT] = 0
        metrics[common_metrics.TTFT] = 0
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config

