from neo4j import GraphDatabase
from loguru import logger

from src.subnet.miner._config import MinerSettings
from src.subnet.miner.blockchain import BaseGraphSearch
from src.subnet.miner.blockchain.bitcoin.query_builder import QueryBuilder
from src.subnet.protocol.llm_engine import Query


class BitcoinGraphSearch(BaseGraphSearch):
    def __init__(self, settings: MinerSettings):
        logger.info(f'Here is loaded configs {settings.GRAPH_DATABASE_URL}')
        self.driver = GraphDatabase.driver(
            settings.GRAPH_DATABASE_URL,
            auth=(settings.GRAPH_DATABASE_USER, settings.GRAPH_DATABASE_PASSWORD),
            connection_timeout=60,
            max_connection_lifetime=60,
            max_connection_pool_size=128,
            encrypted=False,
        )

    def close(self):
        self.driver.close()

    def execute_predefined_query(self, query: Query):
        cypher_query = QueryBuilder.build_query(query)
        logger.info(f"Executing cypher query: {cypher_query}")
        result = self._execute_cypher_query(cypher_query)
        return result

    def execute_query(self, query: str):
        logger.info(f"Executing cypher query: {query}")
        result = self._execute_cypher_query(query)
        return result

    def _execute_cypher_query(self, cypher_query: str):
        with self.driver.session() as session:
            result = session.run(cypher_query)
            if not result:
                return None

            # TODO: remove hardcodes
            results_data = []
            for record in result:
                # Extract nodes and relationships from the record
                a1 = record['a1']
                t1 = record['t1']
                a2 = record['a2']
                s1 = record['s1']
                s2 = record['s2']

                results_data.append({
                    'a1': a1,
                    't1': t1,
                    'a2': a2,
                    's1': dict(s1),
                    's2': dict(s2)
                })

            return results_data
