#!/usr/bin/env python3
"""
ARIA - Autonomous Reasoning Intelligence Assistant
Enhanced launcher script with WOW factors
"""

import subprocess
import sys
import os

def main():
    print("🌟 ARIA - Autonomous Reasoning Intelligence Assistant")
    print("=" * 60)
    print("✨ Multi-Agent Neural Architecture • Real-time Learning • Intelligent Recommendations ✨")
    print()
    
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
    
    # Launch ARIA
    print("\n🚀 Launching ARIA with Enhanced WOW Factors...")
    print("🌟 New Features:")
    print("   • AI Personality-based Recommendations")
    print("   • Neural Network Learning Animation")
    print("   • Interactive AI Decision Tree")
    print("   • Personalized Day-by-Day Itineraries")
    print("   • Real-time Intelligence Insights")
    print()
    print("🌐 Open your browser to: http://localhost:8501")
    print("=" * 60)
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "spectacular_ui.py"])

if __name__ == "__main__":
    main()
