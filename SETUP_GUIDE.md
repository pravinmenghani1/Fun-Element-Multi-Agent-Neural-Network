# 🧠 NEXUS AI - Complete Setup Guide

## 🚀 Quick Start (3 Steps)

```bash
# 1. Clone the repository
git clone https://github.com/pravinmenghani1/Fun-Element-Multi-Agent-Neural-Network.git
cd Fun-Element-Multi-Agent-Neural-Network

# 2. Install requirements
pip install -r requirements_minimal.txt

# 3. Launch NEXUS AI
python launch_nexus.py
```

**That's it! NEXUS AI will launch at http://localhost:8501** 🎉

## 🔧 Detailed Setup Instructions

### Prerequisites
- **Python 3.8+** (Check with `python --version`)
- **Internet connection** (for API calls)
- **Modern web browser**

### Step 1: Clone Repository
```bash
git clone https://github.com/pravinmenghani1/Fun-Element-Multi-Agent-Neural-Network.git
cd Fun-Element-Multi-Agent-Neural-Network
```

### Step 2: Verify Setup
```bash
python verify_setup.py
```
This will check all requirements and show what's missing.

### Step 3: Install Dependencies
```bash
# Install minimal requirements
pip install -r requirements_minimal.txt

# Or install individually
pip install streamlit numpy plotly aiohttp requests
```

### Step 4: Launch NEXUS AI
```bash
# Recommended launch method
python launch_nexus.py

# Alternative launch
streamlit run spectacular_ui.py
```

## 🤖 Optional: Enhanced LLM Experience

For the full AI experience with local LLM integration:

### Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai

### Setup LLM Models
```bash
# Start Ollama service
ollama serve

# Pull required models (in separate terminal)
ollama pull llama3.2:latest
ollama pull deepseek-r1:1.5b

# Verify installation
ollama list
```

### Test LLM Connection
```bash
curl http://localhost:11434/api/tags
```

## 🎯 Launch Options

### Option 1: Simple Launch (Recommended)
```bash
python launch_nexus.py
```

### Option 2: Direct Streamlit
```bash
streamlit run spectacular_ui.py
```

### Option 3: Custom Port
```bash
streamlit run spectacular_ui.py --server.port 8502
```

## 🔍 Troubleshooting

### Common Issues & Solutions

#### "Python not found"
```bash
# Try python3 instead
python3 --version
python3 launch_nexus.py
```

#### "pip not found"
```bash
# Install pip
python -m ensurepip --upgrade
# or
python3 -m ensurepip --upgrade
```

#### "Streamlit not found"
```bash
# Install streamlit
pip install streamlit
# or
python -m pip install streamlit
```

#### "Port 8501 already in use"
```bash
# Use different port
streamlit run spectacular_ui.py --server.port 8502
```

#### "Module not found" errors
```bash
# Reinstall all requirements
pip install -r requirements_minimal.txt --force-reinstall
```

#### LLM connection issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Pull missing models
ollama pull llama3.2:latest
ollama pull deepseek-r1:1.5b
```

## 🎓 What You'll Experience

### Without LLM (Basic Demo):
- ✅ Multi-agent travel booking system
- ✅ Real API integration
- ✅ Interactive visualizations
- ✅ AI concept demonstrations
- ✅ Complete booking workflow

### With LLM (Enhanced Experience):
- ✅ All basic features PLUS:
- ✅ Intelligent travel analysis
- ✅ Personalized recommendations
- ✅ Advanced AI insights
- ✅ Dynamic itinerary optimization

## 📊 System Requirements

### Minimum:
- Python 3.8+
- 2GB RAM
- Internet connection
- Modern browser

### Recommended:
- Python 3.9+
- 4GB RAM
- Ollama with LLM models
- Chrome/Firefox browser

## 🎉 Success Indicators

When everything is working, you should see:
1. **Terminal:** "NEXUS AI launching..."
2. **Browser:** Opens automatically to http://localhost:8501
3. **Interface:** Beautiful NEXUS AI dashboard loads
4. **Functionality:** Can enter travel details and see AI agents work

## 📞 Need Help?

1. **Run verification:** `python verify_setup.py`
2. **Check issues:** [GitHub Issues](https://github.com/pravinmenghani1/Fun-Element-Multi-Agent-Neural-Network/issues)
3. **Read troubleshooting** section above
4. **Create new issue** if problem persists

## 🚀 Ready to Launch!

Once setup is complete:
```bash
python launch_nexus.py
```

**Welcome to the future of AI-powered travel intelligence!** 🧠✈️
