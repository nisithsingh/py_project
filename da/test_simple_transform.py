#!/usr/bin/env python3
"""
Simple test script to test LLM transformation without full preprocessor
Tests just the core transformation logic
"""
import os
import sys
import asyncio
import json
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.llm_transformer import LLMTransformer


async def simple_transform_test():
    """Simple test of transformation logic"""
    print("Simple LLM Transformation Test")
    print("="*50)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        print("Set with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Simple employment request
    request = {
        "client_id": "test-123",
        "name": "Test User",
        "claim_category": "employment",
        "cpac_text": "Test employment data",
        "claims": [
            {
                "claim_id": 1,
                "claim_type": "original_cpac_claim",
                "cpac_data": {
                    "employer_name": "Test Corp",
                    "start_date": "2020-01-01T00:00:00Z"
                },
                "supporting_documents": [
                    {"name": "Test Doc", "docId": "doc1", "result_url": "/test"}
                ]
            }
        ]
    }
    
    print("Input request:")
    print(json.dumps(request, indent=2))
    
    transformer = LLMTransformer()
    
    # Test each agent
    for agent_name in ["cpac_consistency_agent", "document_consistency_agent"]:
        print(f"\n🔄 Testing {agent_name}...")
        
        try:
            result = await transformer.transform_request_for_agent(request, agent_name)
            print(f"Success for {agent_name}")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Failed for {agent_name}: {e}")


if __name__ == "__main__":
    asyncio.run(simple_transform_test())