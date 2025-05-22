import logging

import csv
from langchain_core.tools import tool

# Setup module logger
logger = logging.getLogger(__name__)


@tool
def save_csv_file(content: str):
  """Save content into file

  Args: content which contains csv data

  Return: csv file name
  """
  # content = 'Date,stock name/code,company overview,recent financials,valuation metrics,technical analysis,catalysts,risks,recommendation \
  #   2023-10-05,SAAB,"SAAB AB is a Swedish aerospace and defense company that provides technology and services for military and civil markets. Its key revenue streams include defense, security solutions, and aerospace products.","Revenue: SEK 11 billion (QoQ +5%, YoY +12%); EPS: SEK 4.5 (QoQ +3%, YoY +10%); Margins: Gross 25%, Operating 15%.","P/E Ratio: 20x; P/S Ratio: 2.5x; EV/EBITDA: 15x","Key Support: SEK 200; Key Resistance: SEK 230; 50-day MA: SEK 210.","Increased government spending on defense, new contracts with NATO allies, potential expansions into emerging markets.","Geopolitical tensions affecting defense procurement, delays in contract signings, currency fluctuations impacting profitability.","Buy, Price Target: SEK 250, Rationale: Strong recent performance, favorable defense spending trends, robust contract backlog."'
  logger.info(content)
  with open("./output/result.txt", "w") as text_file:
    text_file.write(content)
    text_file.close()
  return "Success"
