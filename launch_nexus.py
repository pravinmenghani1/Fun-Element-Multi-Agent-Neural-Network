#!/usr/bin/env python3
"""
Cloudcognoscente Voyager - Where Intelligence Meets Exploration
Enhanced launcher script with local attraction APIs
"""

import subprocess
import sys
import os

def main():
    print("🌟 Cloudcognoscente Voyager")
    print("🧠 Where Intelligence Meets Exploration")
    print("=" * 70)
    print("✨ Multi-Agent Neural Architecture • Real-time Learning • Intelligent Discovery ✨")
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
    
    # Launch Cloudcognoscente Voyager
    print("\n🚀 Launching Cloudcognoscente Voyager with Enhanced Features...")
    print("🌟 New Features:")
    print("   • AI Personality-based Recommendations")
    print("   • Neural Network Learning Animation")
    print("   • Interactive AI Decision Tree")
    print("   • Local Attraction API Integration")
    print("   • Tourism Board + Places + TripAdvisor APIs")
    print("   • Personalized Day-by-Day Itineraries")
    print("   • Real-time Intelligence Insights")
    print()
    print("🌐 Open your browser to: http://localhost:8501")
    print("=" * 70)
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "spectacular_ui.py"])

if __name__ == "__main__":
    main()
