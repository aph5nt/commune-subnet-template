from abc import ABC, abstractmethod
from src.subnet.miner._config import MinerSettings


class BaseLLM(ABC):
    @abstractmethod
    def __init__(self, settings: MinerSettings) -> None:
        """
        Initialize LLM
        """

    @abstractmethod
    def build_query_from_messages_balance_tracker(self, llm_messages, llm_type: str, network: str):
        """
        Build query synapse from natural language query for balance tracker
        """

    @abstractmethod
    def build_cypher_query_from_messages(self, llm_messages, llm_type: str, network: str):
        """
        Build query synapse from natural language query for funds flow
        """

    @abstractmethod
    def interpret_result_funds_flow(self, llm_messages, result: list, llm_type: str, network: str):
        """
        Interpret result into natural language based on user's query and structured result dict for funds flow
        """

    @abstractmethod
    def interpret_result_balance_tracker(self, llm_messages, result: list, llm_type: str, network: str):
        """
        Interpret result into natural language based on user's query and structured result dict for balance tracker
        """

    @abstractmethod
    def determine_model_type(self, llm_messages, llm_type: str, network: str):
        """
        Determine model type based on messages
        """

    @abstractmethod
    def generate_general_response(self, llm_messages):
        """
        Generate general response based on chat history
        """
