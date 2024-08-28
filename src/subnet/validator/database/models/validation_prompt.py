from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, update, insert
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import delete
from datetime import datetime

from src.subnet.validator.database import OrmBase
from src.subnet.validator.database.base_model import to_dict
from src.subnet.validator.database.session_manager import DatabaseSessionManager

import random

Base = declarative_base()


class ValidationPrompt(OrmBase):
    __tablename__ = 'validation_prompt'
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(String, nullable=False)
    block = Column(String, nullable=False)

class ValidationPromptManager:
    def __init__(self, session_manager: DatabaseSessionManager):
        self.session_manager = session_manager

    async def store_prompt(self, prompt: str, block: str):
        async with self.session_manager.session() as session:
            async with session.begin():
                stmt = insert(ValidationPrompt).values(
                    prompt=prompt,
                    block=block
                )
                await session.execute(stmt)

    async def get_prompt_by_id(self, prompt_id: int):
        async with self.session_manager.session() as session:
            result = await session.execute(
                select(ValidationPrompt).where(ValidationPrompt.id == prompt_id)
            )
            return to_dict(result.scalars().first())

    async def get_random_prompt(self):
        async with self.session_manager.session() as session:
            # First, get the min and max ID in the table
            min_id_result = await session.execute(
                select(ValidationPrompt.id).order_by(ValidationPrompt.id.asc()).limit(1)
            )
            min_id = min_id_result.scalar()

            max_id_result = await session.execute(
                select(ValidationPrompt.id).order_by(ValidationPrompt.id.desc()).limit(1)
            )
            max_id = max_id_result.scalar()

            if min_id is None or max_id is None:
                return None  # No records found

            # Generate a random ID within the range
            random_id = random.randint(min_id, max_id)

            # Retrieve the prompt with the random ID
            result = await session.execute(
                select(ValidationPrompt).where(ValidationPrompt.id == random_id)
            )
            return to_dict(result.scalars().first())


