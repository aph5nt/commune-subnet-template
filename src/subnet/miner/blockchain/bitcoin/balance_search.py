from loguru import logger
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from src.subnet.validator.database import db_manager
from .. import BaseBalanceSearch


class BitcoinBalanceSearch(BaseBalanceSearch):
    def __init__(self):
        pass

    async def execute_query(self, query: str):
        # Basic check to disallow DDL queries
        ddl_keywords = ["CREATE", "ALTER", "DROP", "TRUNCATE", "INSERT", "UPDATE", "DELETE"]

        if any(keyword in query.upper() for keyword in ddl_keywords):
            raise ValueError("DDL queries are not allowed. Only data selection queries are permitted.")

        try:
            logger.info(f"Executing sql query: {query}")
            async with db_manager.session() as session:
                result = await session.execute(text(query))
                columns = result.keys()
                rows = result.fetchall()
                result = [dict(zip(columns, row)) for row in rows]
                return result

        except SQLAlchemyError as e:
            logger.error(f"An error occurred: {str(e)}")
            return []

    async def solve_challenge(self, block_heights: list[int]):
        try:
            logger.info(f"Executing balance sum query for block heights: {block_heights}")
            async with db_manager.session() as session:
                query = "SELECT SUM(d_balance) FROM balance_changes WHERE block IN :block_heights"
                result = await session.execute(text(query), {'block_heights': tuple(block_heights)})
                sum_d_balance = result.scalar()
                return sum_d_balance

        except SQLAlchemyError as e:
            logger.error(f"An error occurred: {str(e)}")
            return None