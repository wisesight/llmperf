import io
import json
import os
import time
from typing import Any, Dict

import boto3
import ray
import json

from llmperf import common_metrics
from llmperf.models import RequestConfig
from llmperf.ray_llm_client import LLMClient


@ray.remote
class BedrockClient(LLMClient):
    """Client for AWS Bedrock Foundation Model on Llama-2-13b-chat"""

    def __init__(self):
        # Sagemaker doesn't return the number of tokens that are generated so we approximate it by
        # using the llama tokenizer.
        # self.tokenizer = LlamaTokenizerFast.from_pretrained(
        #     "hf-internal-testing/llama-tokenizer"
        # )

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            raise ValueError("AWS_ACCESS_KEY_ID must be set.")
        if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            raise ValueError("AWS_SECRET_ACCESS_KEY must be set.")
        if not os.environ.get("AWS_REGION_NAME"):
            raise ValueError("AWS_REGION_NAME must be set.")

        prompt = request_config.prompt
        prompt, _ = prompt
        model = request_config.model

        bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
        
        sampling_params = request_config.sampling_params
        
        if "max_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = sampling_params["max_tokens"]
            del sampling_params["max_tokens"]
        
        body = {
            "prompt": prompt,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_gen_len": 512,
        }
        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        metrics = {}

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()
        try:
            response = bedrock_runtime.invoke_model(modelId="meta.llama2-13b-chat-v1", body = json.dumps(body))
            total_request_time = time.monotonic() - start_time
            
            response_body = json.loads(response["body"].read())
            tokens_received = response_body["generation_token_count"]
            prompt_token = response_body["prompt_token_count"]
            
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
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_token
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_token

        return metrics, generated_text, request_config