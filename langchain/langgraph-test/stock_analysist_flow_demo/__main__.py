
import os

from dotenv import load_dotenv

from .logging_config import setup_logging

# from .workflow_1 import run_workflow
from .workflow import run_workflow


def main():
    """Main entry point for the scientific research agent."""

    # Load environment variables
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it in your .env file or environment."
        )

    setup_logging()

    # Run the workflow
    run_workflow()


if __name__ == "__main__":
    main()
