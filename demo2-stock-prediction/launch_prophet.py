#!/usr/bin/env python3
"""
🚀 Neural Stock Market Prophet Launcher
Where AI Meets Wall Street - Built by Pravin Menghani
"""

import subprocess
import sys
import os

def main():
    print("🚀 Neural Stock Market Prophet")
    print("📈 Where AI Meets Wall Street")
    print("=" * 70)
    print("✨ Multi-Agent Neural Architecture • Real-time Market Analysis • Predictive Intelligence ✨")
    print()
    print("Built with ❤️ by Pravin Menghani - In love with Neural Networks!!")
    print()
    
    # Check if required packages are installed
    try:
        import streamlit
        import numpy
        import plotly
        import pandas
        import aiohttp
        print("✅ All required packages found")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Launch Neural Stock Market Prophet
    print("\n🚀 Launching Neural Stock Market Prophet...")
    print("🌟 Features:")
    print("   • 📈 LSTM: Long-term Market Memory")
    print("   • 🌊 GRU: Efficient Trend Capture")
    print("   • 🎯 CNN: Chart Pattern Recognition")
    print("   • 🔄 Transformer: Multi-timeframe Attention")
    print("   • 🎭 GAN: Market Scenario Simulation")
    print("   • 🧠 Ensemble: Combined Neural Intelligence")
    print("   • 📊 Real-time Prediction Engine")
    print("   • 🌐 3D Neural Network Topology")
    print()
    print("🌐 Open your browser to: http://localhost:8501")
    print("=" * 70)
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "neural_prophet.py"])

if __name__ == "__main__":
    main()
