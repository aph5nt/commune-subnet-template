import os
import signal
import traceback
from typing import List

from communex._common import get_node_url
from communex.client import CommuneClient
from communex.module import Module, endpoint
from fastapi import Depends, Body
from keylimiter import TokenBucketLimiter
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from src.subnet.miner._config import MinerSettings, load_environment
from src.subnet.miner.blockchain import GraphSearchFactory
from src.subnet.miner.blockchain import BalanceSearchFactory
from src.subnet.miner.blockchain import GraphSummaryTransformerFactory
from src.subnet.miner.blockchain import GraphTransformerFactory, ChartTransformerFactory, TabularTransformerFactory
from src.subnet.miner.llm.factory import LLMFactory
from src.subnet.protocol.llm_engine import LLM_UNKNOWN_ERROR, LLM_ERROR_MESSAGES, \
    QueryOutput, LLM_ERROR_MODIFICATION_NOT_ALLOWED, LLM_ERROR_INVALID_SEARCH_PROMPT, MODEL_TYPE_FUNDS_FLOW, \
    MODEL_TYPE_BALANCE_TRACKING, LlmQueryRequest, LlmMessage, Challenge
from src.subnet.validator.database import db_manager


class Miner(Module):

    def __init__(self, settings: MinerSettings):
        super().__init__()
        self.settings = settings
        self.llm = LLMFactory().create_llm(self.settings)
        self.graph_transformer_factory = GraphTransformerFactory()
        self.chart_transformer_factory = ChartTransformerFactory()
        self.tabular_transformer_factory = TabularTransformerFactory()
        self.graph_search_factory = GraphSearchFactory()
        self.balance_search_factory = BalanceSearchFactory()
        self.graph_summary_transformer_factory = GraphSummaryTransformerFactory()

    @endpoint
    def discovery(self) -> dict:
        return {
            "network": self.settings.NETWORK
        }

    @endpoint
    async def challenge(self, challenge: Challenge):

        if challenge['kind'] == MODEL_TYPE_FUNDS_FLOW:
            search = GraphSearchFactory().create_graph_search(self.settings)
            tx_id = search.solve_challenge(
                in_total_amount=challenge['in_total_amount'],
                out_total_amount=challenge['out_total_amount'],
                tx_id_last_6_chars=challenge['tx_id_last_6_chars']
            )
            return {
                "tx_id": tx_id
            }
        else:
            search = BalanceSearchFactory().create_balance_search(self.settings.NETWORK)
            output = await search.solve_challenge([challenge['block_height']])
            return {
                "sum": output
            }

    # integrity_challenge
    # prompt_challenge (this expects funds flow model type
    # prompt_challenge_cross_check (this expects same graph structure across all miners)
    # prompt

    @endpoint
    def cross_check_query(self, request: LlmQueryRequest):

        # we need to validate incoming query, we support only querying here
        # we need to LIMIT the number of returned data by query to some number (ist that possible?)

        return {
            "error": "Not implemented"
        }

    @endpoint
    async def llm_query(self, prompt: List[LlmMessage] = Body(...)):

        logger.info(f"llm query received: messages={prompt}")

        output = None
        token_usage = None
        start_time = time.time()

        try:
            model_type, token_usage_classification = self.llm.determine_model_type(prompt, self.settings.LLM_TYPE, self.settings.NETWORK)
            logger.info(f"Determined model type: {model_type}")

            if model_type == MODEL_TYPE_FUNDS_FLOW:
                output, token_usage_query_interpret = await self._handle_funds_flow_query(request)
            elif model_type == MODEL_TYPE_BALANCE_TRACKING:
                output, token_usage_query_interpret = await self._handle_balance_tracking_query(request)
            else:
                output = {'error': 'Unsupported model type'}
            token_usage = {
                'completion_tokens': token_usage_classification['completion_tokens'] + token_usage_query_interpret['completion_tokens'],
                'prompt_tokens': token_usage_classification['prompt_tokens'] + token_usage_query_interpret['prompt_tokens'],
                'total_tokens': token_usage_classification['total_tokens'] + token_usage_query_interpret['total_tokens']
            }

        except Exception as e:
            logger.error(traceback.format_exc())
            error_code = e.args[0] if len(e.args) > 0 and isinstance(e.args[0], int) else LLM_UNKNOWN_ERROR
            output = [
                QueryOutput(type="error", error=error_code, result=LLM_ERROR_MESSAGES.get(error_code, 'An error occurred'))]

        logger.info(f"Serving miner llm query output: {output} (Total time taken: {time.time() - start_time} seconds)")

        return output, token_usage

    async def _handle_funds_flow_query(self, prompt: List[LlmMessage]):
        try:
            graph_search = self.graph_search_factory.create_graph_search(self.settings)
            query_start_time = time.time()

            query, token_usage_query = self.llm.build_cypher_query_from_messages(prompt, self.settings.LLM_TYPE, self.settings.NETWORK)
            query = query.strip('`')
            logger.info(f"Generated Cypher query: {query} (Time taken: {time.time() - query_start_time} seconds)")

            if query == 'modification_error':
                error_code = LLM_ERROR_MODIFICATION_NOT_ALLOWED
                error_message = LLM_ERROR_MESSAGES[error_code]
                logger.error(f"Error {error_code}: {error_message}")
                return [{'type': 'error', 'result': error_message, 'error': error_code}], {
                    'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

            if query == 'invalid_prompt_error':
                error_code = LLM_ERROR_INVALID_SEARCH_PROMPT
                error_message = LLM_ERROR_MESSAGES[error_code]
                logger.error(f"Error {error_code}: {error_message}")
                return [{'type': 'error', 'result': error_message, 'error': error_code}], {
                    'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

            execute_query_start_time = time.time()
            result = graph_search.execute_query(query)
            logger.info(f"Query execution time: {time.time() - execute_query_start_time} seconds")
            logger.info(f"Result: {result}")
            graph_search.close()

            # Use transformer for graph result
            graph_transformer = self.graph_transformer_factory.create_graph_transformer(self.settings.NETWORK)
            graph_summary_transformer = self.graph_summary_transformer_factory.create_graph_summary_transformer(self.settings.NETWORK)
            graph_transformed_result = graph_transformer.transform_result(result)
            graph_summary_result = graph_summary_transformer.transform_result(result)
            logger.info(f"Summary Result: {graph_summary_result}")

            # Use transformer for chart result
            chart_transformer = self.chart_transformer_factory.create_chart_transformer(self.settings.NETWORK)
            chart_transformed_result = None
            if chart_transformer.is_chart_applicable(result):
                chart_transformed_result = chart_transformer.convert_funds_flow_to_chart(result)

            interpret_result_start_time = time.time()
            interpreted_result, token_usage_interpret = self.llm.interpret_result_funds_flow(
                llm_messages=prompt,
                result=graph_summary_result,
                llm_type=self.settings.LLM_TYPE,
                network=self.settings.NETWORK
            )
            logger.info(f"Result interpretation time: {time.time() - interpret_result_start_time} seconds")

            output = [
                QueryOutput(type="graph", result=graph_transformed_result),
                QueryOutput(type="text", result=interpreted_result)
            ]

            token_usage = {
                'completion_tokens': token_usage_query.get('completion_tokens', 0) + token_usage_interpret.get('completion_tokens', 0),
                'prompt_tokens': token_usage_query.get('prompt_tokens', 0) + token_usage_interpret.get('prompt_tokens', 0),
                'total_tokens': token_usage_query.get('total_tokens', 0) + token_usage_interpret.get('total_tokens', 0)
            }
        except Exception as e:
            logger.error(traceback.format_exc())
            error_code = e.args[0] if len(e.args) > 0 and isinstance(e.args[0], int) else LLM_UNKNOWN_ERROR
            error_message = LLM_ERROR_MESSAGES.get(error_code, 'An unknown error occurred')
            output = [
                QueryOutput(type="error", error=error_code, result=error_message)
            ]
            token_usage = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

        return output, token_usage

    async def _handle_balance_tracking_query(self, request):
        try:
            query_start_time = time.time()
            query, token_usage_query = self.llm.build_query_from_messages_balance_tracker(request.messages, settings.LLM_TYPE, settings.NETWORK)
            logger.info(f"extracted query: {query} (Time taken: {time.time() - query_start_time} seconds)")

            if query in ['modification_error', 'invalid_prompt_error']:
                error_code = LLM_ERROR_MODIFICATION_NOT_ALLOWED if query == 'modification_error' else LLM_ERROR_INVALID_SEARCH_PROMPT
                error_message = LLM_ERROR_MESSAGES.get(error_code)
                logger.error(f"Error {error_code}: {error_message}")
                return [{'type': 'error', 'result': error_message, 'error': error_code}], {
                    'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

            execute_query_start_time = time.time()
            balance_search = self.balance_search_factory.create_balance_search(self.settings.NETWORK)
            result = await balance_search.execute_query(query)

            logger.info(f"Query execution time: {time.time() - execute_query_start_time} seconds")

            # Use transformer for tabular result
            tabular_transformer = self.tabular_transformer_factory.create_tabular_transformer(request.network)
            tabular_transformed_result = tabular_transformer.transform_result_set(result)

            # Use transformer for chart result
            chart_transformer = self.chart_transformer_factory.create_chart_transformer(request.network)
            chart_transformed_result = None
            if chart_transformer.is_chart_applicable(result):
                chart_transformed_result = chart_transformer.convert_balance_tracking_to_chart(result)

            interpret_result_start_time = time.time()
            interpreted_result, token_usage_interpret = self.llm.interpret_result_balance_tracker(
                llm_messages=request.messages,
                result=tabular_transformed_result,
                llm_type=request.llm_type,
                network=request.network
            )
            logger.info(f"Result interpretation time: {time.time() - interpret_result_start_time} seconds")

            output = [
                QueryOutput(type="text", result=interpreted_result),
                QueryOutput(type="table", result=tabular_transformed_result)
            ]

            token_usage = {
                'completion_tokens': token_usage_query.get('completion_tokens', 0) + token_usage_interpret.get(
                    'completion_tokens', 0),
                'prompt_tokens': token_usage_query.get('prompt_tokens', 0) + token_usage_interpret.get('prompt_tokens',
                                                                                                       0),
                'total_tokens': token_usage_query.get('total_tokens', 0) + token_usage_interpret.get('total_tokens', 0)
            }
        except Exception as e:
            logger.error(traceback.format_exc())
            error_code = e.args[0] if len(e.args) > 0 and isinstance(e.args[0], int) else LLM_UNKNOWN_ERROR
            error_message = LLM_ERROR_MESSAGES.get(error_code, 'An error occurred')
            output = [
                QueryOutput(type="error", error=error_code, result=error_message)
            ]
            token_usage = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}

        return output, token_usage


if __name__ == "__main__":
    from communex.module.server import ModuleServer
    from communex.compat.key import classic_load_key
    import uvicorn
    import time
    import sys

    if len(sys.argv) != 2:
        logger.error("Usage: python -m subnet.cli <environment> ; where <environment> is 'testnet' or 'mainnet'")
        sys.exit(1)

    env = sys.argv[1]
    use_testnet = env == 'testnet'
    load_environment(env)

    logger.remove()
    logger.add(
        "../logs/miner.log",
        rotation="500 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG"
    )

    settings = MinerSettings()
    keypair = classic_load_key(settings.MINER_KEY)
    c_client = CommuneClient(get_node_url(use_testnet=use_testnet))
    miner = Miner(settings=settings)
    refill_rate: float = 1 / 1000
    bucket = TokenBucketLimiter(
        refill_rate=refill_rate,
        bucket_size=1000,
        time_func=time.time,
    )

    db_manager.init(settings.DATABASE_URL)

    server = ModuleServer(miner, keypair, subnets_whitelist=[settings.NET_UID], use_testnet=use_testnet)
    app = server.get_fastapi_app()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def shutdown_handler(signal, frame):
        uvicorn_server.should_exit = True
        uvicorn_server.force_exit = True

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    uvicorn_server = uvicorn.Server(config=uvicorn.Config(app, host="0.0.0.0", port=9962, workers=4))
    uvicorn_server.run()

