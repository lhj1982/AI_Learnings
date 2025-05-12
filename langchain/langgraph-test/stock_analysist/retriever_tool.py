import logging

import requests
from langchain_core.tools import tool

# Setup module logger
logger = logging.getLogger(__name__)


@tool
def index_files():
  """Index files if under predefined file path

  Args:

  Returns:
      Updated prompt
  """
  return []
