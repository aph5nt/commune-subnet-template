from src.subnet.validator.llm.factory import LLMFactory
from src.subnet.validator._config import ValidatorSettings, load_environment


def main(wallet_address: str, network: str):
    # Ensure environment is loaded
    env = 'testnet'  # or 'mainnet' depending on your test
    load_environment(env)

    settings = ValidatorSettings()
    llm = LLMFactory.create_llm(settings)

    try:
        prompt = llm.build_prompt_from_wallet_address(wallet_address, network)
        print(f"Generated Prompt: {prompt}")
    except Exception as e:
        print(f"An error occurred while generating the prompt: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python llm_test_utility.py <wallet_address> <network>")
        sys.exit(1)

    wallet_address = sys.argv[1]
    network = sys.argv[2]

    main(wallet_address, network)
