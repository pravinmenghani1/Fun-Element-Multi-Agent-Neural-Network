#!/usr/bin/env python3
"""
NEXUS AI - Multi-Agent Travel Intelligence System
Simple launcher script
"""

import subprocess
import sys
import os

def main():
    print("🧠 NEXUS AI - Multi-Agent Travel Intelligence")
    print("=" * 50)
    
    # Check if required packages are installed
    try:
        import streamlit
        import numpy
        import plotly
        import aiohttp
        import requests
        print("✅ All required packages found")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_minimal.txt"])
    
    # Launch NEXUS AI
    print("\n🚀 Launching NEXUS AI...")
    print("🌐 Open your browser to: http://localhost:8501")
    print("=" * 50)
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "spectacular_ui.py"])

if __name__ == "__main__":
    main()
