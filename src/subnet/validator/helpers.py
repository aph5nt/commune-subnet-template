from typing import Any, cast

from communex.client import CommuneClient
from communex.misc import get_map_modules


def raise_exception_if_not_registered(validator_key, modules):
    val_ss58 = validator_key.ss58_address
    if val_ss58 not in modules.keys():
        raise RuntimeError(f"key {val_ss58} is not registered in subnet")


def get_miners(client: CommuneClient, netuid: int) -> dict[str, dict[str, Any]]:
    modules = cast(dict[str, Any], get_map_modules(client, netuid=netuid, include_balances=False))
    for miner_key, miner_metadata in modules.items():
        if miner_metadata['stake'] < 100:
            yield miner_key, miner_metadata
