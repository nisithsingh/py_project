#!/usr/bin/env python3
"""
Standalone test script for the LLM-based preprocessor node
Tests the transformation logic without running the full graph
"""
import os
import sys
import asyncio
import json
import logging
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the preprocessor and required utilities
from nodes.preprocessor import preprocess_node
from utils.llm_transformer import LLMTransformer, transform_request_for_agent


async def test_llm_transformer_directly():
    """Test the LLM transformer utility directly"""
    print("\n" + "="*60)
    print("TESTING LLM TRANSFORMER DIRECTLY")
    print("="*60)
    
    # Sample employment request
    employment_request = {
        "client_id": "test-client-123",
        "name": "John Doe",
        "claim_category": "employment",
        "cpac_text": "Employment history shows work at Tech Corp from 2020-2023 with salary of $120,000 annually.",
        "claims": [
            {
                "claim_id": 1,
                "claim_type": "original_cpac_claim",
                "cpac_data": {
                    "employer_name": "Tech Corp",
                    "job_title": "Software Engineer",
                    "start_date": "2020-01-01T00:00:00Z",
                    "end_date": "2023-12-31T00:00:00Z",
                    "annual_compensation": "USD 120000"
                },
                "supporting_documents": [
                    {
                        "name": "Employment Contract",
                        "docId": "doc_001",
                        "result_url": "/path/to/doc_001.json"
                    },
                    {
                        "name": "Payslip",
                        "docId": "doc_002", 
                        "result_url": "/path/to/doc_002.json"
                    }
                ]
            }
        ]
    }
    
    print(f"Original employment request:")
    print(json.dumps(employment_request, indent=2))
    
    # Test transformation for each agent
    agents_to_test = ["cpac_consistency_agent", "document_consistency_agent"]
    
    for agent_name in agents_to_test:
        print(f"\nTesting transformation for: {agent_name}")
        print("-" * 40)
        
        try:
            transformed = await transform_request_for_agent(
                employment_request, 
                agent_name,
                use_fallback=True
            )
            
            print(f"SUCCESS - Transformed request for {agent_name}:")
            print(json.dumps(transformed, indent=2))
            
        except Exception as e:
            print(f"FAILED - Error transforming for {agent_name}: {e}")


async def test_preprocessor_node_directly():
    """Test the preprocessor node directly with mock state"""
    print("\n" + "="*60)
    print("TESTING PREPROCESSOR NODE DIRECTLY")
    print("="*60)
    
    # Mock state that would normally come from the graph
    mock_state = {
        "client_id": "test-client-456",
        "client_name": "Jane Smith",
        "thread_id": "test-thread-789",
        "request_data": {
            "claim_category": "employment",
            "cpac_text": "Jane worked at Finance LLC from 2019-2022 earning $95,000 per year.",
            "claims": [
                {
                    "claim_id": 2,
                    "claim_type": "original_cpac_claim",
                    "cpac_data": {
                        "employer_name": "Finance LLC",
                        "job_title": "Financial Analyst",
                        "start_date": "2019-06-01T00:00:00Z",
                        "end_date": "2022-05-31T00:00:00Z",
                        "annual_compensation": "USD 95000"
                    },
                    "supporting_documents": [
                        {
                            "name": "Offer Letter",
                            "docId": "doc_003",
                            "result_url": "/path/to/doc_003.json"
                        }
                    ]
                }
            ]
        },
        "active_agents": ["cpac_consistency_reviewer", "Document Consistency Agent"],
        "debug": {
            "print_results": True
        }
    }
    
    print(f"Mock state input:")
    print(json.dumps(mock_state, indent=2))
    
    try:
        print(f"\nRunning preprocessor node...")
        result = await preprocess_node(mock_state)
        
        print(f"\nSUCCESS - Preprocessor node result:")
        print(json.dumps(result, indent=2))
        
        # Analyze the transformation logs
        if "transformation_logs" in result:
            print(f"\nTransformation Summary:")
            for agent, status in result["transformation_logs"].items():
                status_icon = "CORRECT" if status == "SUCCESS" else "IGNORE"
                print(f"  {status_icon} {agent}: {status}")
        
    except Exception as e:
        print(f"FAILED - Error in preprocessor node: {e}")
        import traceback
        traceback.print_exc()


async def test_agent_card_loading():
    """Test that agent cards are loaded correctly"""
    print("\n" + "="*60)
    print("TESTING AGENT CARD LOADING")
    print("="*60)
    
    transformer = LLMTransformer()
    
    print(f"Agent cards directory: {transformer.agent_cards_dir}")
    print(f"Loaded agent cards: {list(transformer.agent_cards.keys())}")
    
    for agent_name, card in transformer.agent_cards.items():
        print(f"\nAgent: {agent_name}")
        print(f"   Description: {card.get('description', 'N/A')}")
        print(f"   Request Schema Required Fields: {card.get('requestSchema', {}).get('required', [])}")


def test_environment_setup():
    """Test that required environment variables are set"""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT SETUP")
    print("="*60)
    
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["OPENAI_MODEL"]
    
    print("Required Environment Variables:")
    for var in required_vars:
        value = os.getenv(var)
        status = "SET" if value else "❌ MISSING"
        print(f"   {var}: {status}")
    
    print("\n🔧 Optional Environment Variables:")
    for var in optional_vars:
        value = os.getenv(var, "default")
        print(f"   {var}: {value}")


async def run_all_tests():
    """Run all preprocessor tests"""
    print("STARTING PREPROCESSOR TESTS")
    print("="*80)
    
    # Test environment first
    test_environment_setup()
    
    # Test agent card loading
    await test_agent_card_loading()
    
    # Test LLM transformer directly (requires API key)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        await test_llm_transformer_directly()
        await test_preprocessor_node_directly()
    else:
        print("\nSKIPPING LLM TESTS - OPENAI_API_KEY not set")
        print("   Set OPENAI_API_KEY environment variable to test LLM functionality")
    
    print("\n" + "="*80)
    print("PREPROCESSOR TESTS COMPLETE")


if __name__ == "__main__":
    asyncio.run(run_all_tests())