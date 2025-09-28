#!/usr/bin/env python3
"""
NEXUS AI Setup Verification Script
Checks if all requirements are properly installed
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} - Installed")
        return True
    except ImportError:
        print(f"❌ {package_name} - Missing")
        return False

def check_ollama():
    """Check if Ollama is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama - Running with {len(models)} models")
            
            # Check for required models
            model_names = [model['name'] for model in models]
            if 'llama3.2:latest' in model_names:
                print("  ✅ llama3.2:latest - Available")
            else:
                print("  ⚠️  llama3.2:latest - Missing (optional)")
            
            if 'deepseek-r1:1.5b' in model_names:
                print("  ✅ deepseek-r1:1.5b - Available")
            else:
                print("  ⚠️  deepseek-r1:1.5b - Missing (optional)")
            
            return True
        else:
            print("⚠️  Ollama - Not responding (optional for basic demo)")
            return False
    except Exception:
        print("⚠️  Ollama - Not available (optional for basic demo)")
        return False

def main():
    print("🧠 NEXUS AI - Setup Verification")
    print("=" * 40)
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    print("\n📦 Checking Required Packages:")
    required_packages = [
        ('streamlit', 'streamlit'),
        ('numpy', 'numpy'),
        ('plotly', 'plotly'),
        ('aiohttp', 'aiohttp'),
        ('requests', 'requests')
    ]
    
    for package, import_name in required_packages:
        if not check_package(package, import_name):
            all_good = False
    
    print("\n🤖 Checking Optional LLM Setup:")
    check_ollama()
    
    print("\n" + "=" * 40)
    if all_good:
        print("🎉 NEXUS AI is ready to launch!")
        print("\nRun: python launch_nexus.py")
    else:
        print("⚠️  Some requirements missing. Install with:")
        print("pip install -r requirements_minimal.txt")
    
    print("=" * 40)

if __name__ == "__main__":
    main()
