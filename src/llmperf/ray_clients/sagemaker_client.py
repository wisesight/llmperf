import io
import json
import os
import time
from typing import Any, Dict

import boto3
import ray
from transformers import LlamaTokenizerFast

from llmperf import common_metrics
from llmperf.models import RequestConfig
from llmperf.ray_llm_client import LLMClient


@ray.remote
class SageMakerNeuronClient(LLMClient):
    """Client for OpenAI Chat Completions API."""

    def __init__(self):
        # Sagemaker doesn't return the number of tokens that are generated so we approximate it by
        # using the llama tokenizer.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            raise ValueError("AWS_ACCESS_KEY_ID must be set.")
        if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            raise ValueError("AWS_SECRET_ACCESS_KEY must be set.")
        if not os.environ.get("AWS_REGION_NAME"):
            raise ValueError("AWS_REGION_NAME must be set.")

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        model = request_config.model
        sm_runtime = boto3.client(
            "sagemaker-runtime", region_name=os.environ.get("AWS_REGION_NAME")
        )

        sampling_params = request_config.sampling_params

        if "max_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = sampling_params["max_tokens"]
            del sampling_params["max_tokens"]

        msg_input = [{"role": "user", "content": prompt}]

        message = {
            "inputs": msg_input,
            "parameters": {
                **sampling_params,
            },
        }
        print(
            "this is sampling_params",
            sampling_params,
            flush=True,
        )

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
            response = sm_runtime.invoke_endpoint(
                EndpointName=model,
                ContentType="application/json",
                Body=json.dumps(message),
                CustomAttributes="accept_eula=true",
            )
            print("this is response", response, flush=True)

            total_request_time = time.monotonic() - start_time
            generated_text = response["Body"].read().decode("utf8")
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

