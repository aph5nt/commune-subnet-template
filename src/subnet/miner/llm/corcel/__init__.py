from typing import List
from loguru import logger

from src.subnet.miner._config import MinerSettings
from src.subnet.miner.llm.base_llm import BaseLLM
from src.subnet.miner.llm.corcel.corcel_client import CorcelClient
from src.subnet.miner.llm.prompt_reader import read_local_file
from src.subnet.protocol.llm_engine import LlmMessage, LLM_ERROR_QUERY_BUILD_FAILED, LLM_ERROR_INTERPRETION_FAILED, \
    LLM_ERROR_NOT_APPLICAPLE_QUESTIONS, LLM_ERROR_GENERAL_RESPONSE_FAILED, MODEL_TYPE_FUNDS_FLOW, \
    MODEL_TYPE_BALANCE_TRACKING


class CorcelLLM(BaseLLM):
    def __init__(self, settings: MinerSettings) -> None:
        self.settings = settings
        self.corcel_client = CorcelClient(settings.LLM_API_KEY)

    def build_query_from_messages_balance_tracker(self, llm_messages: List[LlmMessage], network: str) -> str:
        return self._build_query_from_messages(llm_messages, network, "balance_tracking")

    def build_cypher_query_from_messages(self, llm_messages: List[LlmMessage],  network: str) -> str:
        return self._build_query_from_messages(llm_messages, network, "funds_flow")

    def _build_query_from_messages(self, llm_messages: List[LlmMessage], network: str, subfolder: str) -> str:
        local_file_path = f"corcel/prompts/{network}/{subfolder}/query_prompt.txt"
        prompt = read_local_file(local_file_path)
        if not prompt:
            raise Exception("Failed to read prompt content")

        question = "\n".join([message['content'] for message in llm_messages])

        try:
            ai_response, _ = self.corcel_client.send_prompt(model="gpt-4o", prompt=prompt, question=question)
            logger.info(f'ai_response using GPT-4: {ai_response}')

            # Log the entire response content
            logger.debug(f"AI response content: {ai_response}")

            # Handle both cases: with and without triple backticks
            if ai_response.startswith("```") and ai_response.endswith("```"):
                # Extract the SQL code from the response
                query = ai_response.strip("```sql\n").strip("```")
            else:
                # Directly use the content as the query
                query = ai_response.strip()

            return query
        except Exception as e:
            logger.error(f"LlmQuery build error: {e}")
            raise Exception(LLM_ERROR_QUERY_BUILD_FAILED)

    def interpret_result_balance_tracker(self, llm_messages: List[LlmMessage], result: list, network: str) -> str:
        return self._interpret_result(llm_messages, result, network, "balance_tracking")

    def interpret_result_funds_flow(self, llm_messages: List[LlmMessage], result: list, network: str) -> str:
        return self._interpret_result(llm_messages, result, network, "funds_flow")

    def _interpret_result(self, llm_messages: List[LlmMessage], result: list, network: str, subfolder: str) -> str:
        local_file_path = f"corcel/prompts/{network}/{subfolder}/interpretation_prompt.txt"
        prompt = read_local_file(local_file_path)
        if not prompt:
            raise Exception("Failed to read prompt content")

        prompt = prompt.format(result=result)
        question = "\n".join([message.content for message in llm_messages])

        try:
            ai_response, _ = self.corcel_client.send_prompt(model="gpt-4o", prompt=prompt, result=result)
            ai_response = ai_response.strip('"')
            return ai_response
        except Exception as e:
            logger.error(f"LlmQuery interpret result error: {e}")
            raise Exception(LLM_ERROR_INTERPRETION_FAILED)

    def determine_model_type(self, llm_messages: List[LlmMessage], network: str) -> str:
        local_file_path = f"corcel/prompts/{network}/classification/classification_prompt.txt"
        prompt = read_local_file(local_file_path)
        if not prompt:
            raise Exception("Failed to read prompt content")

        question = "\n".join([message['content'] for message in llm_messages])
        logger.info(f"Formed question: {question}")

        try:
            ai_response, _ = self.corcel_client.send_prompt(model="gpt-4o", prompt=prompt, question=question)
            if ai_response in [MODEL_TYPE_FUNDS_FLOW, MODEL_TYPE_BALANCE_TRACKING]:
                return ai_response
            else:
                raise Exception("LLM_ERROR_CLASSIFICATION_FAILED")

        except Exception as e:
            logger.error(f"LlmQuery classification error: {e}")
            raise Exception("LLM_ERROR_CLASSIFICATION_FAILED")

    def generate_general_response(self, llm_messages: List[LlmMessage]) -> str:
        general_prompt = "Your general prompt here"
        question = "\n".join([message.content for message in llm_messages])

        try:
            ai_response, _ = self.corcel_client.send_prompt(model="gpt-4o", prompt=general_prompt, question=question)
            if ai_response == "not applicable questions":
                raise Exception(LLM_ERROR_NOT_APPLICAPLE_QUESTIONS)
            else:
                return ai_response
        except Exception as e:
            logger.error(f"LlmQuery general response error: {e}")
            raise Exception(LLM_ERROR_GENERAL_RESPONSE_FAILED)

