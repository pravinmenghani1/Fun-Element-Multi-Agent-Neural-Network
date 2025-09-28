#!/usr/bin/env python3
"""
🏥 Smart Health Diagnosis AI Launcher
Where AI Meets Medicine - Built by Pravin Menghani
"""

import subprocess
import sys
import os

def main():
    print("🏥 Smart Health Diagnosis AI")
    print("🩺 Where AI Meets Medicine")
    print("=" * 70)
    print("✨ Multi-Agent Neural Architecture • Real-time Health Analysis • Intelligent Diagnosis ✨")
    print()
    print("Built with ❤️ by Pravin Menghani - In love with Neural Networks!!")
    print()
    
    # Check if required packages are installed
    try:
        import streamlit
        import numpy
        import plotly
        import pandas
        print("✅ All required packages found")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "numpy", "plotly", "pandas"])
    
    # Launch Smart Health Diagnosis AI
    print("\n🚀 Launching Smart Health Diagnosis AI...")
    print("🌟 Features:")
    print("   • 🧠 FNN: Basic symptom analysis")
    print("   • 🔄 RNN: Medical history patterns")
    print("   • 👁️ CNN: Medical image analysis")
    print("   • 🎯 Decision Tree: Rule-based diagnosis")
    print("   • 🤖 Ensemble: Combined medical expertise")
    print("   • 📊 Real-time diagnosis engine")
    print("   • 🌐 3D medical AI network")
    print("   • 🏥 Interactive patient consultation")
    print()
    print("⚠️  EDUCATIONAL DEMO ONLY - NOT FOR ACTUAL MEDICAL DIAGNOSIS")
    print()
    print("🌐 Open your browser to: http://localhost:8501")
    print("=" * 70)
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "health_ai.py"])

if __name__ == "__main__":
    main()
