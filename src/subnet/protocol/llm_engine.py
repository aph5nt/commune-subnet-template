from typing import List, Optional, Dict, Literal
from typing_extensions import Any
from pydantic import BaseModel, Field

from src.subnet.protocol.blockchain import NETWORK_BITCOIN

# Query Types
QUERY_TYPE_SEARCH = "search"
QUERY_TYPE_FLOW = "flow"
QUERY_TYPE_AGGREGATION = "aggregation"

# Model types
MODEL_TYPE_FUNDS_FLOW = "funds_flow"
MODEL_TYPE_BALANCE_TRACKING = "balance_tracking"


def get_model_types():
    return [MODEL_TYPE_FUNDS_FLOW, MODEL_TYPE_BALANCE_TRACKING]


ERROR_TYPE = int
LLM_TYPE_OPENAI = "openai"
LLM_TYPE_CUSTOM = "custom"
LLM_TYPE_CORCEL = "corcel"

# LLM MESSAGE TYPE
LLM_MESSAGE_TYPE_USER = 1
LLM_MESSAGE_TYPE_AGENT = 2

# LLM Error Codes
LLM_ERROR_NO_ERROR = 0
LLM_ERROR_TYPE_NOT_SUPPORTED = 1
LLM_ERROR_SEARCH_TARGET_NOT_SUPPORTED = 2
LLM_ERROR_SEARCH_LIMIT_NOT_SPECIFIED = 3
LLM_ERROR_SEARCH_LIMIT_EXCEEDED = 4
LLM_ERROR_INTERPRETION_FAILED = 5
LLM_ERROR_EXECUTION_FAILED = 6
LLM_ERROR_QUERY_BUILD_FAILED = 7
LLM_ERROR_GENERAL_RESPONSE_FAILED = 8
LLM_ERROR_NOT_APPLICAPLE_QUESTIONS = 9
LLM_CLIENT_ERROR = 10
LLM_ERROR_INVALID_SEARCH_PROMPT = 11
LLM_ERROR_MODIFICATION_NOT_ALLOWED = 12
LLM_UNKNOWN_ERROR = 999

# LLM Error Messages
LLM_ERROR_MESSAGES = {
    LLM_ERROR_NO_ERROR: "No Error",
    LLM_ERROR_TYPE_NOT_SUPPORTED: "Not supported query type",
    LLM_ERROR_SEARCH_TARGET_NOT_SUPPORTED: "Please let us know what you want to search.",
    LLM_ERROR_SEARCH_LIMIT_NOT_SPECIFIED: "Because there are too many results, you need to let us know how many results you want to get.",
    LLM_ERROR_SEARCH_LIMIT_EXCEEDED: "We cannot provide that many results.",
    LLM_ERROR_INTERPRETION_FAILED: "Unexpected error occurs while interpreting results.",
    LLM_ERROR_EXECUTION_FAILED: "Unexpected error occurs during database interaction.",
    LLM_ERROR_QUERY_BUILD_FAILED: "Unexpected error occurs while inferencing AI models.",
    LLM_ERROR_GENERAL_RESPONSE_FAILED: "Unexpected error occurs while answering general questions.",
    LLM_ERROR_NOT_APPLICAPLE_QUESTIONS: "Your question is not applicable to our subnet. We only answer questions related blockchain or cryptocurrency.",
    LLM_CLIENT_ERROR: "LLM client error",
    LLM_UNKNOWN_ERROR: "LLM unknown error"
}


class Challenge(BaseModel):
    in_total_amount: Optional[int] = None
    out_total_amount: Optional[int] = None
    tx_id_last_6_chars: Optional[str] = None
    checksum: Optional[str] = None
    block_height: Optional[int] = None
    output: Optional[Any] = None


class LlmMessage(BaseModel):
    type: int = Field(0, title="The type of the message")
    content: str = Field("", title="The content of the message")

    def to_dict(self):
        return {
            "type": self.type,
            "content": self.content
        }


class QueryOutput(BaseModel):
    type: Literal["graph", "text", "table", "error"] = Field(..., title="The type of the output")
    result: Optional[List[object]] = None
    interpreted_result: Optional[str] = None
    error: Optional[ERROR_TYPE] = None

    class Config:
        arbitrary_types_allowed = True


class LlmQuery(BaseModel):
    network: str = Field(NETWORK_BITCOIN, title="The network to query")
    messages: List[LlmMessage] = None
    output: Optional[QueryOutput] = None


class Query(BaseModel):
    network: str = None
    type: str = None

    # search query
    target: str = None
    where: Optional[Dict] = None
    limit: Optional[int] = None
    skip: Optional[int] = 0

    # output
    output: Optional[QueryOutput] = None


class LlmQueryRequest(BaseModel):
    network: str
    prompt: List[LlmMessage]
    miner_key: Optional[str] = None


class SwitchResponse(BaseModel):
    model: str = Field(default="funds_flow", title="The model to be prompted")