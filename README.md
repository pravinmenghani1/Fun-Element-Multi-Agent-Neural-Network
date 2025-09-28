# 🧠 NEXUS AI - Multi-Agent Travel Intelligence System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced AI demonstration showcasing multiple neural network architectures working together to solve complex travel booking problems through real-time API integration.

## 🎯 What is NEXUS AI?

NEXUS AI is a sophisticated multi-agent system that demonstrates cutting-edge AI concepts:

- **🔮 RNN (Recurrent Neural Networks)** - Long-term price trend analysis
- **👁️ Temporal CNN** - Short-term volatility detection  
- **🎯 Transformer Attention** - Hotel ranking optimization
- **🔄 Variational Autoencoders (VAE)** - Attraction recommendation generation
- **📊 Autoencoders** - Weather data compression and insights
- **🤖 Ensemble Learning** - Coordinated multi-agent decision making
- **🌐 Real API Integration** - Live travel booking platforms

## 🏗️ System Architecture

```
                    🧠 NEXUS AI COORDINATOR
                         (Ensemble Learning)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   🔮 RNN Agent          👁️ CNN Agent         🎯 Transformer
   (Long-term            (Short-term          (Attention
    Trends)               Patterns)            Mechanism)
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                   🔄 VAE + 📊 Autoencoder
                   (Generation + Compression)
                              │
                    🤖 LLM Enhancement Layer
                  (llama3.2 + deepseek-r1:1.5b)
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8 or higher**
2. **Git** (for cloning)
3. **Internet connection** (for API calls)

### Installation

```bash
# Clone the repository
git clone https://github.com/pravinmenghani1/Fun-Element-Multi-Agent-Neural-Network.git
cd Fun-Element-Multi-Agent-Neural-Network

# Install dependencies
pip install -r requirements_minimal.txt

# Launch NEXUS AI
python launch_nexus.py
```

### Alternative Launch
```bash
streamlit run spectacular_ui.py
```

## 🛠️ Complete Setup Guide

### Step 1: System Requirements
```bash
# Check Python version
python --version  # Should be 3.8+

# Install pip if not available
python -m ensurepip --upgrade
```

### Step 2: Install Dependencies
```bash
# Core dependencies
pip install streamlit>=1.28.0
pip install numpy>=1.24.0
pip install plotly>=5.15.0
pip install aiohttp>=3.8.0
pip install requests>=2.31.0

# Or install all at once
pip install -r requirements_minimal.txt
```

### Step 3: Optional - Local LLM Setup (Enhanced Experience)

For the full AI experience with local LLM integration:

#### Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - Download from https://ollama.ai
```

#### Start Ollama Service
```bash
ollama serve
```

#### Pull Required Models
```bash
ollama pull llama3.2:latest
ollama pull deepseek-r1:1.5b
```

#### Verify LLM Setup
```bash
# Test connection
curl http://localhost:11434/api/tags

# Should show your installed models
```

### Step 4: Launch NEXUS AI
```bash
# Simple launch (recommended)
python launch_nexus.py

# Direct launch
streamlit run spectacular_ui.py

# Custom port
streamlit run spectacular_ui.py --server.port 8502
```

### Step 5: Access the System
Open your browser and navigate to:
- **Local:** http://localhost:8501
- **Custom port:** http://localhost:8502

## 🌟 Features & Capabilities

### 🔄 Real-time API Integration
- **Flight APIs:** Amadeus, IndiGo, Air India, SpiceJet
- **Hotel APIs:** Booking.com integration
- **Weather APIs:** OpenWeatherMap
- **Attraction APIs:** Foursquare integration

### 🧠 AI Concepts Demonstrated
1. **Feedforward Neural Networks** - Base agent architecture
2. **RNN** - Sequential price pattern analysis
3. **Temporal CNN** - Time-series volatility detection
4. **Transformer Attention** - Multi-head feature ranking
5. **VAE** - Latent space attraction generation
6. **Autoencoders** - Weather data compression
7. **Ensemble Learning** - Multi-agent coordination

### 📊 Interactive Visualizations
- **3D Agent Network** - Multi-agent collaboration
- **Real-time API Monitor** - Live response tracking
- **RNN Trend Analysis** - Price prediction charts
- **CNN Pattern Recognition** - Volatility detection
- **Attention Heatmaps** - Feature importance
- **VAE Latent Space** - Attraction clustering
- **Live Sentiment Analysis** - Destination insights

## 🎯 How to Use

1. **Enter Travel Details**
   - Origin and destination cities
   - Travel dates and budget
   - Number of travelers
   - Preferences (accommodation, activity level)

2. **Launch AI Agents**
   - Click "🚀 Launch NEXUS AI Agents"
   - Watch real-time agent coordination
   - Explore AI concept visualizations

3. **Review AI Results**
   - RNN-optimized flight recommendations
   - Attention-ranked hotel selections
   - VAE-generated attraction suggestions
   - Autoencoder weather analysis
   - LLM-enhanced insights

4. **Book Your Trip**
   - Select preferred options
   - Confirm complete itinerary
   - Get AI confidence scores

## 🔧 Troubleshooting

### Common Issues

#### Streamlit Not Found
```bash
pip install streamlit
# or
python -m pip install streamlit
```

#### Port Already in Use
```bash
streamlit run spectacular_ui.py --server.port 8502
```

#### LLM Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Pull missing models
ollama pull llama3.2:latest
ollama pull deepseek-r1:1.5b
```

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements_minimal.txt --force-reinstall
```

#### API Timeout Issues
- Check internet connection
- Some APIs use demo data if real endpoints are unavailable
- System gracefully handles API failures

## 📁 Project Structure

```
NEXUS-AI/
├── spectacular_ui.py           # Main NEXUS AI interface
├── travel_agent_system.py      # Multi-agent architecture
├── enhanced_flight_agent.py    # RNN/CNN flight analysis
├── llm_integration.py          # Local LLM integration
├── real_api_integration.py     # Real API calls
├── launch_nexus.py            # Simple launcher
├── requirements_minimal.txt    # Dependencies
└── README.md                  # This file
```

## 🎓 Educational Value

### For Students
- **Practical AI Learning** - See theoretical concepts in action
- **Multi-Agent Systems** - Understand agent coordination
- **Neural Networks** - Visual understanding of different architectures
- **Real-world Applications** - AI solving actual problems

### For Educators
- **Interactive Demonstrations** - Engage students with hands-on examples
- **Modular Teaching** - Focus on individual AI concepts
- **Visual Learning** - Charts and graphs explain complex topics
- **Complete Curriculum** - 7+ AI concepts in one system

### For Developers
- **System Architecture** - Learn multi-agent design patterns
- **API Integration** - Real-world API handling
- **Async Programming** - Concurrent agent execution
- **UI/UX Design** - Professional interface development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama Team** - Local LLM infrastructure
- **Streamlit** - Amazing web app framework
- **Travel APIs** - Real-world data integration
- **AI Research Community** - Foundational concepts

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/pravinmenghani1/Fun-Element-Multi-Agent-Neural-Network/issues)
- **Discussions:** [GitHub Discussions](https://github.com/pravinmenghani1/Fun-Element-Multi-Agent-Neural-Network/discussions)

---

**🧠 Experience the future of AI-powered travel intelligence with NEXUS AI!** ✈️🤖
