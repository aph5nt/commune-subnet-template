from typing import List
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from loguru import logger

from src.subnet.validator._config import ValidatorSettings
from src.subnet.validator.llm.base_llm import BaseLLM
from src.subnet.validator.llm.prompt_reader import read_local_file
from src.subnet.validator.llm.utils import split_messages_into_chunks
from src.subnet.protocol.llm_engine import LLM_ERROR_PROMPT_GENERATION_FAILED


class OpenAILLM(BaseLLM):
    def __init__(self, settings: ValidatorSettings) -> None:
        self.settings = settings
        self.chat_gpt4o = ChatOpenAI(api_key=settings.LLM_API_KEY, model="gpt-4o", temperature=0)
        self.MAX_TOKENS = 128000

    def build_prompt_from_txid_and_block(self, txid: str, block: str, network: str) -> str:
        local_file_path = f"openai/prompts/{network}/prompt_generation/prompt_generation_prompt.txt"
        prompt = read_local_file(local_file_path)
        if not prompt:
            raise Exception("Failed to read prompt content")

        if not txid:
            logger.warning("The transaction ID is empty. Cannot generate a prompt without a valid txid.")
            return "Prompt generation failed: Transaction ID is required but not provided."
        if not block:
            logger.warning("The block is empty. Cannot generate a prompt without a valid block.")
            return "Prompt generation failed: Block is required but not provided."

        try:
            # Substitute txid and block into the prompt
            full_prompt = prompt.format(txid=txid, block=block)
            logger.info(f"Full prompt: {full_prompt}")
        except KeyError as e:
            logger.error(f"KeyError during prompt formatting: {e}")
            logger.error(f"Prompt: {prompt}")
            logger.error(f"Transaction ID: {txid}, Block: {block}")
            raise Exception("Error formatting prompt with txid and block") from e

        # Prepare the messages
        messages = [SystemMessage(content=full_prompt)]

        try:
            message_chunks = split_messages_into_chunks(messages)
            ai_responses = []
            for chunk in message_chunks:
                ai_message = self.chat_gpt4o.invoke(chunk)
                ai_responses.append(ai_message.content)
            combined_response = "\n".join(ai_responses)
            return combined_response

        except Exception as e:
            logger.error(f"LlmQuery prompt generation error: {e}")
            raise Exception(LLM_ERROR_PROMPT_GENERATION_FAILED)
