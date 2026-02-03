#!/bin/bash
# Quick test script for Cosmos LLM API setup

echo "=================================================="
echo "Testing Cosmos LLM API Configuration"
echo "=================================================="
echo ""

# Check environment
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Create .env from .env.example and configure LLM_API_KEY and LLM_BASE_URL"
    exit 1
fi

# Check if LLM_API_KEY is set
if ! grep -q "^LLM_API_KEY=" .env; then
    echo "❌ Error: LLM_API_KEY not set in .env"
    echo "Add: LLM_API_KEY=your-cosmos-api-key"
    exit 1
fi

# Check if LLM_BASE_URL is set
if ! grep -q "^LLM_BASE_URL=" .env; then
    echo "❌ Error: LLM_BASE_URL not set in .env"
    echo "Add: LLM_BASE_URL=https://your-cosmos-endpoint.com/v1"
    exit 1
fi

echo "✓ Environment configured"
echo ""

# Run basic test
echo "Running basic LLM connection test..."
python test_llm_config.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Basic test failed. Check your LLM_API_KEY and LLM_BASE_URL"
    exit 1
fi

echo ""
echo "Running Cosmos agent test..."
python test_cosmos_agent.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Agent test failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ All tests passed!"
echo "You can now run: python main_cosmos.py"
echo "=================================================="
