#!/usr/bin/env python3
"""
Configuration Checker for OpenAI Client

This script helps diagnose and fix configuration issues for OpenAI API.
"""

import os
import sys
from pathlib import Path

import dotenv
dotenv.load_dotenv()

def check_openai_config():
    """Check OpenAI configuration"""
    print("=" * 50)
    print("OPENAI CONFIGURATION")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    print(f"OPENAI_API_KEY: {'✓ Set' if api_key else '✗ Not set'}")
    if api_key:
        print(f"  Value: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else api_key}")
    
    print(f"OPENAI_BASE_URL: {'✓ Set' if base_url else '✗ Not set (using default)'}")
    if base_url:
        print(f"  Value: {base_url}")
    else:
        print(f"  Default: https://api.openai.com/v1")
    
    return bool(api_key)

def check_model_name(model_name):
    """Check OpenAI model configuration"""
    print("\n" + "=" * 50)
    print("MODEL ANALYSIS")
    print("=" * 50)
    
    print(f"Model name: {model_name}")
    print(f"Will use: OpenAI client")
    
    print("Required environment variables:")
    print("  - OPENAI_API_KEY")
    print("Optional environment variables:")
    print("  - OPENAI_BASE_URL (if using custom endpoint)")
    
    return check_openai_config()

def suggest_fixes():
    """Suggest fixes for common issues"""
    print("\n" + "=" * 50)
    print("TROUBLESHOOTING SUGGESTIONS")
    print("=" * 50)
    
    print("1. Create a .env file in the current directory with:")
    print("   OPENAI_API_KEY=your_openai_api_key_here")
    print("")
    print("   # Optional: If using custom endpoint")
    print("   OPENAI_BASE_URL=https://your-custom-endpoint.com/v1")
    print("")
    print("2. Get your OpenAI API key from:")
    print("   https://platform.openai.com/api-keys")
    print("")
    print("3. Supported OpenAI models include:")
    print("   - gpt-4o")
    print("   - gpt-4o-mini")
    print("   - gpt-4-turbo")
    print("   - gpt-4")
    print("   - gpt-3.5-turbo")
    print("")
    print("4. If you're getting errors, check:")
    print("   - API key is valid and not expired")
    print("   - You have sufficient credits/quota")
    print("   - Model name is correct and you have access to it")

def main():
    """Main configuration checker"""
    print("OPENAI CLIENT CONFIGURATION CHECKER")
    print("This script helps diagnose OpenAI API configuration errors\n")
    
    # Check current directory for .env file
    env_file = Path(".env")
    print(f".env file: {'✓ Found' if env_file.exists() else '✗ Not found'}")
    if env_file.exists():
        print(f"  Location: {env_file.absolute()}")
    
    # Get model name from command line or use default
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
    
    # Check configuration for the model
    config_ok = check_model_name(model_name)
    
    # Show overall status
    print("\n" + "=" * 50)
    print("OVERALL STATUS")
    print("=" * 50)
    
    if config_ok:
        print("✓ Configuration looks good!")
        print("If you're still getting errors, check:")
        print("  1. API key validity")
        print("  2. Account has sufficient credits/quota")
        print("  3. Model name is correct")
        print("  4. You have access to the specified model")
    else:
        print("✗ Configuration issues detected")
        suggest_fixes()

if __name__ == "__main__":
    main() 