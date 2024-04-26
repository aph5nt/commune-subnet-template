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

Base = declarative_base()


class MinerDiscovery(OrmBase):
    __tablename__ = 'miner_discovery'
    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(Integer, nullable=False)
    miner_key = Column(String, nullable=False, unique=True)
    miner_address = Column(String, nullable=False, default='0.0.0.0')
    miner_ip_port = Column(String, nullable=False, default='0')
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    network = Column(String, nullable=False)
    rank = Column(Float, nullable=False, default=0.0)


class MinerDiscoveryManager:
    def __init__(self, session_manager: DatabaseSessionManager):
        self.session_manager = session_manager

    async def store_miner_metadata(self, uid: int, miner_key: str, miner_address: str, miner_ip_port: str, network: str,
                                   rank: float = 0.0):
        async with self.session_manager.session() as session:
            async with session.begin():
                stmt = insert(MinerDiscovery).values(
                    uid=uid,
                    miner_key=miner_key,
                    miner_address=miner_address,
                    miner_ip_port=miner_ip_port,
                    network=network,
                    rank=rank,
                    timestamp=datetime.utcnow()
                ).on_conflict_do_update(
                    index_elements=['miner_key'],
                    set_={
                        'uid': uid,
                        'miner_address': miner_address,
                        'miner_ip_port': miner_ip_port,
                        'network': network,
                        'rank': rank,
                        'timestamp': datetime.utcnow()
                    }
                )
                await session.execute(stmt)

    async def get_miner_by_key(self, miner_key: str):
        async with self.session_manager.session() as session:
            result = await session.execute(
                select(MinerDiscovery).where(MinerDiscovery.miner_key == miner_key)
            )
            return to_dict(result.scalars().first())

    async def get_miners_by_network(self, network: Optional[str]):
        if not network:
            async with self.session_manager.session() as session:
                result = await session.execute(
                    select(MinerDiscovery)
                    .order_by(MinerDiscovery.timestamp, MinerDiscovery.rank)
                )
                return [to_dict(result) for result in result.scalars().all()]
        else:
            async with self.session_manager.session() as session:
                result = await session.execute(
                    select(MinerDiscovery)
                    .where(MinerDiscovery.network == network)
                    .order_by(MinerDiscovery.timestamp, MinerDiscovery.rank)
                )
                return [to_dict(result) for result in result.scalars().all()]

    async def update_miner_rank(self, miner_key: str, new_rank: float):
        async with self.session_manager.session() as session:
            async with session.begin():
                stmt = update(MinerDiscovery).where(
                    MinerDiscovery.miner_key == miner_key
                ).values(rank=new_rank)
                await session.execute(stmt)

    async def remove_all_records(self):
        async with self.session_manager.session() as session:
            async with session.begin():
                await session.execute(delete(MinerDiscovery))

    async def remove_miner_by_key(self, miner_key: str):
        async with self.session_manager.session() as session:
            async with session.begin():
                await session.execute(
                    delete(MinerDiscovery).where(MinerDiscovery.miner_key == miner_key)
                )
