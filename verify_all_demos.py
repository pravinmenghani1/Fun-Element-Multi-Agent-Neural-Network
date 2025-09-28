#!/usr/bin/env python3
"""
🤖 Fun Element Multi-Agent Neural Network - Demo Verification Script

This script verifies that all 3 demos are properly set up and can be launched.
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_directory_structure():
    """Verify the 3 demo directories exist"""
    demos = [
        "demo1-travel-agent",
        "demo2-stock-prediction", 
        "demo3-medical-ai"
    ]
    
    print("\n📁 Checking directory structure...")
    for demo in demos:
        if os.path.exists(demo):
            print(f"✅ {demo}/")
        else:
            print(f"❌ {demo}/ - Missing!")
            return False
    return True

def check_demo_files(demo_dir, required_files):
    """Check if required files exist in demo directory"""
    print(f"\n🔍 Checking {demo_dir} files...")
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(demo_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing!")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_dependencies(demo_dir):
    """Check if dependencies can be imported"""
    requirements_file = os.path.join(demo_dir, "requirements.txt")
    if not os.path.exists(requirements_file):
        print(f"❌ {demo_dir}/requirements.txt not found")
        return False
    
    print(f"📦 Checking {demo_dir} dependencies...")
    
    # Basic dependencies that should be available
    basic_deps = ["streamlit", "numpy", "plotly"]
    
    for dep in basic_deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - Not installed")
            return False
    
    return True

def main():
    """Main verification function"""
    print("🤖 Fun Element Multi-Agent Neural Network - Demo Verification")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check directory structure
    if not check_directory_structure():
        return False
    
    # Define required files for each demo
    demo_files = {
        "demo1-travel-agent": [
            "README.md",
            "launch_travel_agent.py",
            "travel_agent_ui.py",
            "travel_agent_system.py",
            "requirements.txt"
        ],
        "demo2-stock-prediction": [
            "README.md", 
            "launch_prophet.py",
            "neural_prophet.py",
            "prediction_engine.py",
            "requirements.txt"
        ],
        "demo3-medical-ai": [
            "README.md",
            "launch_health_ai.py", 
            "health_ai.py",
            "medical_diagnosis_engine.py",
            "requirements.txt"
        ]
    }
    
    # Check each demo
    all_good = True
    for demo_dir, files in demo_files.items():
        if not check_demo_files(demo_dir, files):
            all_good = False
        
        if not check_dependencies(demo_dir):
            all_good = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 All demos are properly set up!")
        print("\n🚀 Quick Start Commands:")
        print("cd demo1-travel-agent && python launch_travel_agent.py")
        print("cd demo2-stock-prediction && python launch_prophet.py") 
        print("cd demo3-medical-ai && python launch_health_ai.py")
    else:
        print("❌ Some issues found. Please fix them before running demos.")
        print("\n💡 To install missing dependencies:")
        print("cd <demo-directory> && pip install -r requirements.txt")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
