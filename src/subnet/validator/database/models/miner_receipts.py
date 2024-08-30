from dataclasses import dataclass

from pydantic import BaseModel
from sqlalchemy import Column, String, DateTime, update, insert, BigInteger, Boolean, UniqueConstraint, Text, select, \
    func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import case
from datetime import datetime, timedelta
from src.subnet.validator.database import OrmBase
from src.subnet.validator.database.session_manager import DatabaseSessionManager

Base = declarative_base()


class MinerReceipt(OrmBase):
    __tablename__ = 'miner_receipts'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String, nullable=False)
    miner_key = Column(String, nullable=False)
    prompt_hash = Column(Text, nullable=False)
    accepted = Column(Boolean, nullable=False, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('miner_key', 'request_id', name='uq_miner_key_request_id'),
    )


class StatsData(BaseModel):
    accepted_count: int
    not_accepted_count: int


class ReceiptStats(BaseModel):
    last_day: StatsData
    last_week: StatsData
    last_month: StatsData


class MinerReceiptManager:
    def __init__(self, session_manager: DatabaseSessionManager):
        self.session_manager = session_manager

    async def store_miner_receipt(self, request_id: str, miner_key: str, prompt_hash: str, timestamp: datetime):
        async with self.session_manager.session() as session:
            async with session.begin():
                stmt = insert(MinerReceipt).values(
                    request_id=request_id,
                    miner_key=miner_key,
                    prompt_hash=prompt_hash,
                    accepted=False,
                    timestamp=timestamp
                )
                await session.execute(stmt)

    async def accept_miner_receipt(self, request_id: str, miner_key: str):
        async with self.session_manager.session() as session:
            async with session.begin():
                stmt = update(MinerReceipt).where(
                    (MinerReceipt.request_id == request_id) & (MinerReceipt.miner_key == miner_key)
                ).values(accepted=True)
                await session.execute(stmt)

    async def get_receipts_by_miner_key(self, miner_key: str, page: int = 1, page_size: int = 10):
        async with self.session_manager.session() as session:
            # Calculate offset
            offset = (page - 1) * page_size

            # Query total number of receipts
            total_items_result = await session.execute(
                select(func.count(MinerReceipt.id))
                .where(MinerReceipt.miner_key == miner_key)
            )
            total_items = total_items_result.scalar()

            # Calculate total pages
            total_pages = (total_items + page_size - 1) // page_size

            # Query paginated receipts
            result = await session.execute(
                select(MinerReceipt)
                .where(MinerReceipt.miner_key == miner_key)
                .order_by(MinerReceipt.timestamp.desc())
                .limit(page_size)
                .offset(offset)
            )
            receipts = result.scalars().all()

            return {
                "receipts": receipts,
                "total_pages": total_pages,
                "total_items": total_items
            }

    async def get_receipts_stats_by_miner_key(self, miner_key: str) -> Stats:
        async with self.session_manager.session() as session:
            now = datetime.utcnow()
            last_day = now - timedelta(days=1)
            last_week = now - timedelta(weeks=1)
            last_month = now - timedelta(days=30)

            # Query for last day
            query_day = select(
                func.sum(case((MinerReceipt.accepted == True, 1), else_=0)).label('accepted_count'),
                func.sum(case((MinerReceipt.accepted == False, 1), else_=0)).label('not_accepted_count')
            ).where(
                MinerReceipt.miner_key == miner_key,
                MinerReceipt.timestamp >= last_day
            )

            # Query for last week
            query_week = select(
                func.sum(case((MinerReceipt.accepted == True, 1), else_=0)).label('accepted_count'),
                func.sum(case((MinerReceipt.accepted == False, 1), else_=0)).label('not_accepted_count')
            ).where(
                MinerReceipt.miner_key == miner_key,
                MinerReceipt.timestamp >= last_week
            )

            # Query for last month
            query_month = select(
                func.sum(case((MinerReceipt.accepted == True, 1), else_=0)).label('accepted_count'),
                func.sum(case((MinerReceipt.accepted == False, 1), else_=0)).label('not_accepted_count')
            ).where(
                MinerReceipt.miner_key == miner_key,
                MinerReceipt.timestamp >= last_month
            )

            result_day = await session.execute(query_day)
            result_week = await session.execute(query_week)
            result_month = await session.execute(query_month)

            stats = {
                'last_day': result_day.fetchone(),
                'last_week': result_week.fetchone(),
                'last_month': result_month.fetchone()
            }

            stats_result = Stats(
                last_day=ReceiptStats(
                    accepted_count=stats['last_day'].accepted_count,
                    not_accepted_count=stats['last_day'].not_accepted_count
                ),
                last_week=ReceiptStats(
                    accepted_count=stats['last_week'].accepted_count,
                    not_accepted_count=stats['last_week'].not_accepted_count
                ),
                last_month=ReceiptStats(
                    accepted_count=stats['last_month'].accepted_count,
                    not_accepted_count=stats['last_month'].not_accepted_count
                )
            )

            return stats_result
