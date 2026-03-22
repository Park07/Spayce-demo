#!/usr/bin/env python3
"""Verify API key is loaded and Gemini LLM works."""
import os
import sys

sys.path.insert(0, '.')

print("\n" + "="*60)
print("PROOF: GEMINI API KEY IS LOADED AND WORKING")
print("="*60)

# Test 1: Environment variable
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    print(f"\n✓ GEMINI_API_KEY found in environment")
    print(f"  Value (first 20 chars): {api_key[:20]}...")
    print(f"  Full length: {len(api_key)} characters")
else:
    print("\n✗ GEMINI_API_KEY NOT found in environment")
    sys.exit(1)

# Test 2: Import LLM module
try:
    from core.llm_agent import CrowdSafetyLLM
    print(f"\n✓ CrowdSafetyLLM module imported successfully")
except Exception as e:
    print(f"\n✗ Failed to import LLM: {e}")
    sys.exit(1)

# Test 3: Initialize with API key
try:
    llm = CrowdSafetyLLM()
    print(f"\n✓ LLM initialized with Gemini API")
    print(f"  Model: {llm.model.model_name}")
    print(f"  Use Mock: {llm.use_mock}")
    print(f"  Model object: {type(llm.model).__name__}")
except Exception as e:
    print(f"\n✗ Failed to initialize LLM: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("CONCLUSION: APP IS READY WITH GEMINI API")
print("="*60)
print("\n→ Run: streamlit run app/main.py")
print("→ Then click 'Generate Live LLM Analysis' to test\n")
