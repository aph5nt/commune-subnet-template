import random
from src.subnet.validator.llm.factory import LLMFactory
from src.subnet.validator._config import ValidatorSettings, load_environment
from src.subnet.validator.nodes.bitcoin.node import BitcoinNode
from src.subnet.validator.database.session_manager import DatabaseSessionManager
from src.subnet.validator.database.models.validation_prompt import ValidationPromptManager


async def main(wallet_address: str, network: str):
    # Ensure environment is loaded
    env = 'testnet'  # or 'mainnet' depending on your test
    load_environment(env)

    settings = ValidatorSettings()
    llm = LLMFactory.create_llm(settings)

    # Initialize session manager and ValidationPromptManager
    session_manager = DatabaseSessionManager()
    session_manager.init(settings.DATABASE_URL)
    validation_prompt_manager = ValidationPromptManager(session_manager)

    try:
        btc = BitcoinNode()

        # Get the current highest block height
        highest_block_height = btc.get_current_block_height()

        # Set the lowest block height (typically block 0 is the genesis block)
        lowest_block_height = 0

        # Calculate a random block height between the lowest and highest
        random_block_height = random.randint(lowest_block_height, highest_block_height)

        # Get a random vin or vout address from the random block and also get block data
        random_vin_or_vout = btc.get_random_vin_or_vout(random_block_height)
        print(f"Random Vin or Vout: {random_vin_or_vout}")
        print(f"Block Data: {random_vin_or_vout['block_data']}")  # Print block data if needed

        # Use the address in the LLM prompt generation
        prompt = llm.build_prompt_from_wallet_address(random_vin_or_vout["address"], network)
        print(f"Generated Prompt: {prompt}")

        # Store the generated prompt and block information in the database
        await validation_prompt_manager.store_prompt(prompt, random_vin_or_vout['block_data'])
        print(f"Prompt stored in the database successfully.")

    except Exception as e:
        print(f"An error occurred while generating or storing the prompt: {e}")


if __name__ == "__main__":
    import sys
    import asyncio

    if len(sys.argv) != 3:
        print("Usage: python llm_prompt_utility.py <wallet_address> <network>")
        sys.exit(1)

    wallet_address = sys.argv[1]
    network = sys.argv[2]

    # Run the async main function
    asyncio.run(main(wallet_address, network))
