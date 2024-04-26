import requests
import json
from loguru import logger
from datetime import datetime
import string
import time

from src.subnet.miner.llm.utils import get_message_token_count


class CorcelClient:
    def __init__(self, api_key: str, base_url: str = "https://api.corcel.io/v1/text/cortext/chat"):
        self.api_key = api_key
        self.base_url = base_url

    def send_prompt(self, model: str, prompt: str, question: str = None, result: any = None) -> str:
        # Convert the result to a JSON string if it is not already
        result_str = ""
        if result:
            try:
                # Use a custom serialization method to handle datetime objects
                result_str = json.dumps(result, indent=2, default=self.json_serializer)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error during result formatting: {e}")
                raise Exception("Error formatting result as JSON") from e

        # Use string.Template to safely substitute the result into the prompt
        template = string.Template(prompt)
        try:
            full_prompt = template.safe_substitute(result=result_str)
        except KeyError as e:
            logger.error(f"KeyError during prompt formatting: {e}")
            logger.error(f"Prompt: {prompt}")
            logger.error(f"Result: {result_str}")
            raise Exception("Error formatting prompt with result") from e

        # If question is provided, append it to the prompt
        if question:
            full_prompt = f"{full_prompt}\n\nActual question is: {question}"
        prompt_token_count = get_message_token_count(full_prompt)

        payload = {
            "model": model,
            "stream": False,
            "top_p": 1,
            "temperature": 0,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        logger.info(f"Sending request to {self.base_url} with payload: {json.dumps(payload, indent=2)}")

        try:
            # Record the start time
            start_time = time.time()

            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Record the end time
            end_time = time.time()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            raise
        except Exception as err:
            logger.error(f"An error occurred: {err}")
            raise

        response_data = response.json()
        logger.info(f"Response from API: {json.dumps(response_data, indent=2)}")

        # Debugging the response structure
        try:
            if not isinstance(response_data, list) or len(response_data) == 0:
                raise Exception("Invalid response structure: response data is empty or not a list")

            choices_data = response_data[0].get('choices')
            if not isinstance(choices_data, list) or len(choices_data) == 0:
                raise Exception("Invalid response structure: 'choices' list is empty or not a list")

            # Calculate the duration
            duration = end_time - start_time

            # Extract the number of tokens from the response
            tokens = sum(len(choice.get('delta', {}).get('content', '').split()) for choice in choices_data)

            # Calculate tokens per second
            tokens_per_second = tokens / duration if duration > 0 else 0

            logger.info(f"Tokens per second: {tokens_per_second}")
            completion_token_count = get_message_token_count(choices_data[0]['delta']['content'])

            return choices_data[0]['delta']['content'], {'completion_tokens': completion_token_count, 'prompt_tokens': prompt_token_count, 'total_tokens': prompt_token_count + completion_token_count}
        except (KeyError, TypeError, IndexError) as e:
            logger.error(f"Unexpected response structure: {response_data}")
            raise Exception("Invalid response structure") from e

    @staticmethod
    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
