#!/usr/bin/env python3
"""
🚀 Neural Stock Market Prophet - Where AI Meets Wall Street
Multi-Agent Neural Network System for Stock Market Prediction
Built by Pravin Menghani, in love ❤️ with Neural Networks!!
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import time
import random

# Import our custom modules
from neural_visualizations import *
from prediction_engine import *
from llm_assistant import StockLLMAssistant

def main():
    st.set_page_config(
        page_title="🚀 Neural Stock Market Prophet - Where AI Meets Wall Street",
        page_icon="📈🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize LLM Assistant
    if 'llm_assistant' not in st.session_state:
        st.session_state.llm_assistant = StockLLMAssistant(use_local=True)
    
    # Ensure llm_available attribute exists (for backward compatibility)
    if not hasattr(st.session_state.llm_assistant, 'llm_available'):
        st.session_state.llm_assistant.llm_available = st.session_state.llm_assistant._check_llm_availability()
    
    # Spectacular CSS styling with tooltips
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(45deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: pulse-glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes pulse-glow {
        from { box-shadow: 0 10px 30px rgba(255, 107, 107, 0.5); }
        to { box-shadow: 0 10px 40px rgba(69, 183, 209, 0.8); }
    }
    
    .problem-statement {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dotted #999;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(50,50,50,0.95));
        color: #fff;
        text-align: left;
        border-radius: 10px;
        padding: 15px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
        line-height: 1.4;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: rgba(0,0,0,0.9) transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .prediction-card {
        background: linear-gradient(45deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        animation: shimmer 2s ease-in-out infinite alternate;
    }
    
    @keyframes shimmer {
        from { background-position: 0% 50%; }
        to { background-position: 100% 50%; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Spectacular header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Neural Stock Market Prophet</h1>
        <h2>📈 Where AI Meets Wall Street</h2>
        <p>✨ Multi-Agent Neural Architecture • Real-time Market Analysis • Predictive Intelligence ✨</p>
        <p style="font-size: 14px; margin-top: 10px; opacity: 0.9;">
            Built by <strong>Pravin Menghani</strong>, in love ❤️ with Neural Networks!!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clear Problem Statement
    st.markdown("""
    <div class="problem-statement">
        <h3>🎯 THE CHALLENGE: Can AI Predict Stock Prices Better Than Humans?</h3>
        <p><strong>Problem:</strong> Stock market prediction is extremely difficult. Professional traders use complex analysis, but still struggle with accuracy.</p>
        <p><strong>Solution:</strong> We'll use 6 different AI neural networks, each with unique strengths, to analyze real stock data and make predictions.</p>
        <p><strong>Goal:</strong> See which AI technique works best and learn how different neural networks "think" about financial markets.</p>
        <p><strong>Why This Matters:</strong> Understanding AI in finance helps you learn cutting-edge technology used by Wall Street professionals!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with AI concepts
    with st.sidebar:
        st.header("🧠 Neural Network Arsenal")
        
        # Add expandable concepts guide
        with st.expander("📚 Stock Market AI Guide", expanded=False):
            st.markdown("""
            **📈 LSTM:** Long-term market memory  
            **🌊 GRU:** Efficient trend capture  
            **🎯 CNN:** Pattern recognition in charts  
            **🔄 Transformer:** Multi-timeframe attention  
            **🎭 GAN:** Generate realistic market scenarios  
            **🤖 Reinforcement Learning:** Adaptive trading strategy  
            **🧠 Ensemble:** Combine all predictions  
            """)
        
        concept_tab = st.selectbox("Select Neural Network:", [
            "🌐 3D Neural Network Topology",
            "📈 LSTM: Market Memory",
            "🌊 GRU: Trend Capture", 
            "🎯 CNN: Chart Patterns",
            "🔄 Transformer: Multi-Attention",
            "🎭 GAN: Market Simulation",
            "🤖 Reinforcement Learning: Trading Strategy",
            "🧠 Neural Trading Battle",
            "📊 Real-time Prediction Engine"
        ])
        
        if concept_tab == "🌐 3D Neural Network Topology":
            st.plotly_chart(create_3d_neural_topology(), use_container_width=True)
            st.markdown("""
            **🌐 Neural Network Battle Arena**
            
            **What you're seeing:**
            - 6 different neural networks competing
            - Real-time connections and data flow
            - Color-coded by prediction accuracy
            
            **Why this matters:**
            - Each network has unique strengths
            - Ensemble combines all for best results
            - Visual understanding of AI architecture
            
            **🚀 Watch the networks battle for prediction supremacy!**
            """)
        
        elif concept_tab == "📈 LSTM: Market Memory":
            st.plotly_chart(create_lstm_memory_visualization(), use_container_width=True)
            st.markdown("""
            **📈 LSTM: The Market's Memory Master**
            
            **What LSTM does:**
            - Remembers long-term market patterns
            - Forgets irrelevant noise automatically
            - Learns from historical bull/bear cycles
            
            **Why perfect for stocks:**
            - Markets have long-term memory effects
            - Past trends influence future movements
            - Can remember crashes, booms, patterns
            
            **🧠 LSTM = Long Short-Term Memory - the AI that never forgets important market lessons!**
            """)
        
        elif concept_tab == "🌊 GRU: Trend Capture":
            st.plotly_chart(create_gru_trend_visualization(), use_container_width=True)
            st.markdown("""
            **🌊 GRU: The Efficient Trend Hunter**
            
            **What GRU does:**
            - Captures trends with fewer parameters than LSTM
            - Faster training, efficient processing
            - Focuses on recent important patterns
            
            **Why great for trading:**
            - Markets change quickly - need fast adaptation
            - Simpler than LSTM but still powerful
            - Perfect for real-time trading decisions
            
            **⚡ GRU = Gated Recurrent Unit - the streamlined trend master!**
            """)
        
        elif concept_tab == "🎯 CNN: Chart Patterns":
            st.plotly_chart(create_cnn_pattern_visualization(), use_container_width=True)
            st.markdown("""
            **🎯 CNN: The Chart Pattern Detective**
            
            **What CNN does:**
            - Recognizes visual patterns in price charts
            - Detects head & shoulders, triangles, flags
            - Learns from candlestick formations
            
            **Why amazing for stocks:**
            - Technical analysis is all about visual patterns
            - Can spot patterns humans miss
            - Works like a trader's trained eye
            
            **👁️ CNN sees the market like a master technical analyst!**
            """)
        
        elif concept_tab == "🔄 Transformer: Multi-Attention":
            st.plotly_chart(create_transformer_attention_viz(), use_container_width=True)
            st.markdown("""
            **🔄 Transformer: The Multi-Timeframe Master**
            
            **What Transformer does:**
            - Pays attention to multiple timeframes simultaneously
            - Weighs importance of different market factors
            - Processes all data in parallel
            
            **Why revolutionary for trading:**
            - Markets operate on multiple timeframes
            - Need to consider news, technicals, sentiment
            - Attention mechanism focuses on what matters
            
            **🎯 Transformer attention = Having multiple expert traders analyzing simultaneously!**
            """)
        
        elif concept_tab == "🎭 GAN: Market Simulation":
            st.plotly_chart(create_gan_simulation_viz(), use_container_width=True)
            st.markdown("""
            **🎭 GAN: The Market Reality Generator**
            
            **What GAN does:**
            - Generator creates fake market scenarios
            - Discriminator tries to detect fake vs real
            - They compete until fake becomes indistinguishable
            
            **Why mind-blowing for trading:**
            - Can simulate thousands of market scenarios
            - Tests strategies on realistic but fake data
            - Prepares for situations that haven't happened yet
            
            **🎪 GAN = Two AIs competing to create perfect market simulations!**
            """)
        
        elif concept_tab == "🤖 Reinforcement Learning: Trading Strategy":
            st.plotly_chart(create_rl_strategy_viz(), use_container_width=True)
            st.markdown("""
            **🤖 Reinforcement Learning: The Adaptive Trader**
            
            **What RL does:**
            - Learns optimal trading actions through trial and error
            - Gets rewards for profitable trades, penalties for losses
            - Adapts strategy based on market feedback
            
            **Why revolutionary for trading:**
            - No need for labeled training data
            - Learns from actual trading results
            - Continuously adapts to changing markets
            - Can discover novel trading strategies
            
            **🎯 RL Agent = AI trader that learns from wins and losses like a human!**
            """)
        
        elif concept_tab == "🧠 Neural Trading Battle":
            st.plotly_chart(create_neural_battle_viz(), use_container_width=True)
            st.markdown("""
            **🧠 Watch Neural Networks Battle for Trading Supremacy!**
            
            **The Competition:**
            - Each network makes predictions
            - Real-time accuracy tracking
            - Winner takes all approach
            
            **What you learn:**
            - Different networks excel at different times
            - Ensemble combines strengths
            - AI competition drives innovation
            
            **🏆 May the best neural network win!**
            """)
        
        elif concept_tab == "📊 Real-time Prediction Engine":
            st.plotly_chart(create_realtime_prediction_engine(), use_container_width=True)
            st.info("📊 Live prediction engine showing all networks working together")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📈 Stock Selection")
        
        # Clear explanation box
        st.info("""
        💡 **How This Works:**
        1. **Pick a stock** - Choose which company to analyze
        2. **Set prediction days** - How far into future (1-30 days)
        3. **Choose risk level** - Conservative/Moderate/Aggressive
        4. **Select AI networks** - Each has different strengths:
           - 📈 **LSTM**: Remembers long-term patterns
           - 🌊 **GRU**: Captures recent trends  
           - 🎯 **CNN**: Recognizes chart patterns
           - 🔄 **Transformer**: Multi-factor analysis
           - 🎭 **GAN**: Creates market scenarios
           - 🤖 **RL**: Learns trading strategies
        """)
        
        # CRYSTAL CLEAR EXPLANATION
        st.success("""
        🎯 **YES - ALL OF THESE PREDICT STOCK PRICES!** Here's exactly what each one does:
        
        📈 **LSTM Analysis** → Predicts price by remembering long-term market patterns (like "stocks usually go up after crashes")
        
        🌊 **GRU Trends** → Predicts price by focusing on recent momentum (like "stock has been rising for 5 days")
        
        🎯 **CNN Patterns** → Predicts price by recognizing chart shapes (like "this looks like a breakout pattern")
        
        🔄 **Transformer Insights** → Predicts price by analyzing multiple factors at once (price + volume + news)
        
        🎭 **GAN Simulation** → Creates multiple "what-if" price scenarios to test risk
        
        🤖 **RL Strategy** → Learns the best BUY/SELL/HOLD action and predicts based on that strategy
        
        🧠 **Ensemble Decision** → Combines ALL predictions into one final, more accurate price prediction
        """)
        
        with st.form("stock_prophet_form"):
            stock_symbol = st.selectbox("🏢 Select Stock", [
                "AAPL - Apple Inc.",
                "GOOGL - Alphabet Inc.", 
                "TSLA - Tesla Inc.",
                "MSFT - Microsoft Corp.",
                "AMZN - Amazon.com Inc.",
                "NVDA - NVIDIA Corp.",
                "META - Meta Platforms"
            ])
            
            prediction_days = st.slider("🔮 Prediction Horizon (Days)", 1, 30, 7)
            investment_amount = st.number_input("💰 Investment Amount ($)", 1000, 1000000, 10000, step=1000)
            risk_tolerance = st.selectbox("⚡ Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
            
            st.markdown("**🎯 Neural Network Selection:**")
            use_lstm = st.checkbox("📈 LSTM (Market Memory)", value=True)
            use_gru = st.checkbox("🌊 GRU (Trend Capture)", value=True)
            use_cnn = st.checkbox("🎯 CNN (Pattern Recognition)", value=True)
            use_transformer = st.checkbox("🔄 Transformer (Multi-Attention)", value=True)
            use_gan = st.checkbox("🎭 GAN (Market Simulation)", value=True)
            use_rl = st.checkbox("🤖 Reinforcement Learning (Trading Strategy)", value=True)
            
            submitted = st.form_submit_button("🚀 Launch Neural Prophet", type="primary")
        
        if submitted:
            st.session_state.stock_request = {
                'symbol': stock_symbol.split(' - ')[0],
                'company': stock_symbol.split(' - ')[1],
                'days': prediction_days,
                'investment': investment_amount,
                'risk': risk_tolerance,
                'networks': {
                    'lstm': use_lstm,
                    'gru': use_gru,
                    'cnn': use_cnn,
                    'transformer': use_transformer,
                    'gan': use_gan,
                    'rl': use_rl
                }
            }
            st.session_state.neural_analysis = True
    
    with col2:
        if hasattr(st.session_state, 'neural_analysis') and st.session_state.neural_analysis:
            
            # Neural Network Analysis in Progress
            st.markdown("### 🤖 Neural Networks in Action")
            
            # Real-time neural network status
            request = st.session_state.stock_request
            selected_networks = [k for k, v in request['networks'].items() if v]
            
            # Create dynamic columns based on selected networks
            if len(selected_networks) >= 3:
                neural_cols = st.columns(3)
            else:
                neural_cols = st.columns(len(selected_networks))
            
            network_info = {
                'lstm': ("📈 LSTM", "Analyzing long-term market memory..."),
                'gru': ("🌊 GRU", "Capturing recent trend patterns..."),
                'cnn': ("🎯 CNN", "Recognizing chart formations..."),
                'transformer': ("🔄 Transformer", "Multi-factor market analysis..."),
                'gan': ("🎭 GAN", "Generating market scenarios..."),
                'rl': ("🤖 RL", "Learning trading strategies...")
            }
            
            for i, network in enumerate(selected_networks[:len(neural_cols)]):
                if network in network_info:
                    name, status = network_info[network]
                    with neural_cols[i]:
                        st.markdown(f"""
                        <div class="agent-card">
                            <h4>{name}</h4>
                            <div>{status}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Progress with spectacular effects
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Run neural analysis
            async def run_neural_analysis():
                request = st.session_state.stock_request
                
                # Step 1: Data collection with real API calls
                status_text.text("📊 Fetching REAL-TIME stock price from Yahoo Finance...")
                progress_bar.progress(10)
                
                # Show real-time price fetching
                with st.spinner(f"Getting live price for {request['symbol']}..."):
                    from stock_data_fetcher import stock_fetcher
                    current_price = await stock_fetcher.get_current_price(request['symbol'])
                    st.success(f"✅ Live Price: {request['symbol']} = ${current_price:.2f}")
                
                await asyncio.sleep(0.5)
                
                status_text.text("📈 Collecting historical market data...")
                progress_bar.progress(25)
                await asyncio.sleep(0.5)
                
                # Step 2: Neural network processing
                status_text.text("🧠 Neural networks analyzing real market patterns...")
                progress_bar.progress(50)
                predictions = await generate_neural_predictions(request)
                
                # Step 3: Ensemble combination
                status_text.text("🎯 Combining neural network insights...")
                progress_bar.progress(80)
                ensemble_result = combine_predictions(predictions)
                
                # Step 4: Final analysis
                status_text.text("✨ Generating investment recommendations...")
                progress_bar.progress(100)
                
                return predictions, ensemble_result
            
            # Execute neural analysis
            if 'neural_results' not in st.session_state:
                try:
                    predictions, ensemble = asyncio.run(run_neural_analysis())
                    st.session_state.neural_results = (predictions, ensemble)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.neural_results = None
            
            # Display spectacular results
            if hasattr(st.session_state, 'neural_results') and st.session_state.neural_results:
                predictions, ensemble = st.session_state.neural_results
                request = st.session_state.stock_request
                
                # CRYSTAL CLEAR EXPLANATION OF RESULTS
                st.markdown("### 🎯 What Just Happened?")
                st.info(f"""
                **Each AI analyzed {request['symbol']} stock and made its own price prediction:**
                
                🧠 **Different AIs, Different Approaches:**
                - Some focus on long-term patterns, others on recent trends
                - Some look at chart shapes, others at multiple factors
                - Each gives a different predicted price for {request['days']} days from now
                
                🎯 **The Final Answer:** All predictions are combined into one "Ensemble" result - this is usually the most accurate!
                """)
                
                # Spectacular prediction display
                st.markdown("### 🎯 Neural Prophet Predictions")
                
                # Main prediction card
                from datetime import datetime
                current_time = datetime.now().strftime("%H:%M:%S")
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>🚀 {request['company']} ({request['symbol']})</h3>
                    <h2>Predicted Price: ${ensemble['predicted_price']:.2f}</h2>
                    <p><strong>Current Price:</strong> ${ensemble['current_price']:.2f} 📊 (Live at {current_time})</p>
                    <p><strong>Expected Return:</strong> {ensemble['expected_return']:.1f}%</p>
                    <p><strong>Confidence:</strong> {ensemble['confidence']:.1f}%</p>
                    <p><strong>Risk Level:</strong> {ensemble['risk_level']}</p>
                    <p><strong>Data Source:</strong> {ensemble.get('data_source', 'Real market analysis')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 🤖 LLM INTELLIGENT EXPLANATION
                st.markdown("### 🤖 AI Financial Advisor Explanation")
                with st.spinner("🧠 AI is analyzing the predictions..."):
                    prediction_data = {
                        'price': ensemble['predicted_price'],
                        'confidence': ensemble['confidence'] / 100,
                        'trend': 'bullish' if ensemble['expected_return'] > 0 else 'bearish' if ensemble['expected_return'] < -2 else 'neutral',
                        'volatility': 'high' if abs(ensemble['expected_return']) > 10 else 'moderate' if abs(ensemble['expected_return']) > 5 else 'low',
                        'risk': request['risk'].lower(),
                        'expected_return': ensemble['expected_return']
                    }
                    
                    llm_explanation = st.session_state.llm_assistant.explain_prediction(prediction_data)
                    st.info(llm_explanation)
                
                # Neural network breakdown
                available_tabs = []
                if 'lstm' in predictions:
                    available_tabs.append("📈 LSTM Analysis")
                if 'gru' in predictions:
                    available_tabs.append("🌊 GRU Trends")
                if 'cnn' in predictions:
                    available_tabs.append("🎯 CNN Patterns")
                if 'transformer' in predictions:
                    available_tabs.append("🔄 Transformer Insights")
                if 'gan' in predictions:
                    available_tabs.append("🎭 GAN Simulation")
                if 'rl' in predictions:
                    available_tabs.append("🤖 RL Strategy")
                available_tabs.append("🧠 Ensemble Decision")
                
                tabs = st.tabs(available_tabs)
                tab_idx = 0
                
                if 'lstm' in predictions:
                    with tabs[tab_idx]:
                        st.markdown("**📈 LSTM: Long-Term Market Memory Analysis**")
                        st.plotly_chart(create_lstm_prediction_chart(predictions['lstm']), use_container_width=True)
                        st.write(f"**LSTM Prediction:** ${predictions['lstm']['price']:.2f}")
                        st.write(f"**Memory Confidence:** {predictions['lstm']['confidence']:.1f}%")
                        st.write(f"**Key Pattern:** {predictions['lstm']['pattern']}")
                    tab_idx += 1
                
                if 'gru' in predictions:
                    with tabs[tab_idx]:
                        st.markdown("**🌊 GRU: Efficient Trend Capture**")
                        st.plotly_chart(create_gru_trend_chart(predictions['gru']), use_container_width=True)
                        st.write(f"**GRU Prediction:** ${predictions['gru']['price']:.2f}")
                        st.write(f"**Trend Strength:** {predictions['gru']['trend_strength']:.1f}%")
                        st.write(f"**Direction:** {predictions['gru']['direction']}")
                    tab_idx += 1
                
                if 'cnn' in predictions:
                    with tabs[tab_idx]:
                        st.markdown("**🎯 CNN: Chart Pattern Recognition**")
                        st.plotly_chart(create_cnn_pattern_chart(predictions['cnn']), use_container_width=True)
                        st.write(f"**CNN Prediction:** ${predictions['cnn']['price']:.2f}")
                        st.write(f"**Pattern Match:** {predictions['cnn']['pattern_match']:.1f}%")
                        st.write(f"**Detected Pattern:** {predictions['cnn']['detected_pattern']}")
                    tab_idx += 1
                
                if 'transformer' in predictions:
                    with tabs[tab_idx]:
                        st.markdown("**🔄 Transformer: Multi-Timeframe Attention**")
                        st.plotly_chart(create_transformer_attention_chart(predictions['transformer']), use_container_width=True)
                        st.write(f"**Transformer Prediction:** ${predictions['transformer']['price']:.2f}")
                        st.write(f"**Attention Score:** {predictions['transformer']['attention_score']:.1f}%")
                        st.write(f"**Key Focus:** {predictions['transformer']['focus_area']}")
                    tab_idx += 1
                
                if 'gan' in predictions:
                    with tabs[tab_idx]:
                        st.markdown("**🎭 GAN: Market Scenario Simulation**")
                        st.write(f"**GAN Prediction:** ${predictions['gan']['price']:.2f}")
                        st.write(f"**Scenario Range:** ${predictions['gan']['scenario_range']:.2f}")
                        st.write(f"**Risk Assessment:** {predictions['gan']['risk_assessment']}")
                        st.write(f"**Simulation Quality:** {predictions['gan']['simulation_quality']:.2f}")
                        
                        # Show scenario distribution
                        scenarios = predictions['gan']['scenarios']
                        st.write("**Generated Scenarios:**")
                        for i, scenario in enumerate(scenarios):
                            st.write(f"  Scenario {i+1}: ${scenario:.2f}")
                    tab_idx += 1
                
                if 'rl' in predictions:
                    with tabs[tab_idx]:
                        st.markdown("**🤖 Reinforcement Learning: Trading Strategy**")
                        st.write(f"**RL Prediction:** ${predictions['rl']['price']:.2f}")
                        st.write(f"**Strategy Action:** {predictions['rl']['strategy_action']}")
                        st.write(f"**Strategy Confidence:** {predictions['rl']['strategy_confidence']:.1f}%")
                        st.write(f"**Episodes Trained:** {predictions['rl']['episodes_trained']:,}")
                        st.write(f"**Learned Pattern:** {predictions['rl']['learned_pattern']}")
                        st.write(f"**Reward Score:** {predictions['rl']['reward_score']:.2f}")
                    tab_idx += 1
                
                with tabs[tab_idx]:
                    st.markdown("**🧠 Ensemble: Combined Neural Intelligence**")
                    st.plotly_chart(create_ensemble_decision_chart(ensemble), use_container_width=True)
                    
                    st.markdown("### 🎯 Investment Recommendation")
                    
                    # Dynamic recommendation based on return AND confidence
                    return_pct = ensemble['expected_return']
                    confidence = ensemble['confidence']
                    
                    # Calculate recommendation strength
                    if return_pct > 15 and confidence > 85:
                        st.success(f"🚀 STRONG BUY: Exceptional {return_pct:.1f}% return with {confidence:.0f}% confidence!")
                    elif return_pct > 10 and confidence > 75:
                        st.success(f"📈 BUY: Strong {return_pct:.1f}% return expected (High confidence: {confidence:.0f}%)")
                    elif return_pct > 7 and confidence > 70:
                        st.info(f"💰 BUY: Good {return_pct:.1f}% return potential (Confidence: {confidence:.0f}%)")
                    elif return_pct > 3 and confidence > 60:
                        st.info(f"📊 MODERATE BUY: Decent {return_pct:.1f}% return expected (Confidence: {confidence:.0f}%)")
                    elif return_pct > 0 and confidence > 50:
                        st.warning(f"⚖️ HOLD: Small {return_pct:.1f}% return, moderate confidence ({confidence:.0f}%)")
                    elif return_pct > 0 and confidence <= 50:
                        st.warning(f"⚠️ HOLD: {return_pct:.1f}% return but low confidence ({confidence:.0f}%) - Wait for better signals")
                    elif return_pct > -3 and confidence > 60:
                        st.warning(f"⚖️ HOLD: Minor loss expected ({return_pct:.1f}%), consider waiting")
                    elif return_pct > -7:
                        st.error(f"📉 SELL: Moderate loss predicted ({return_pct:.1f}%)")
                    else:
                        st.error(f"🔴 STRONG SELL: Significant loss expected ({return_pct:.1f}%)")
                    
                    # Add confidence indicator
                    if confidence > 80:
                        st.success(f"🎯 High Confidence Signal: {confidence:.0f}%")
                    elif confidence > 60:
                        st.info(f"⚖️ Moderate Confidence: {confidence:.0f}%")
                    else:
                        st.warning(f"⚠️ Low Confidence: {confidence:.0f}% - Use caution")
                
                # Final action button
                st.markdown("---")
                if st.button("💰 EXECUTE NEURAL PROPHET STRATEGY", type="primary", use_container_width=True):
                    st.balloons()
                    st.success("🎉 Neural Prophet strategy activated!")
                    
                    st.markdown(f"""
                    **🚀 Neural Stock Market Prophet Summary:**
                    - **📈 LSTM:** Analyzed {request['days']} days of market memory
                    - **🌊 GRU:** Captured efficient trend patterns  
                    - **🎯 CNN:** Recognized technical chart formations
                    - **🔄 Transformer:** Applied multi-timeframe attention
                    - **🎭 GAN:** Generated realistic market scenarios
                    - **🤖 Reinforcement Learning:** Learned adaptive trading strategy
                    - **🧠 Ensemble:** Combined all neural insights
                    - **💰 Investment:** ${request['investment']:,} optimally allocated
                    - **🎯 Confidence:** {ensemble['confidence']:.1f}% prediction accuracy
                    - **✨ Where AI Meets Wall Street!**
                    """)
                
                # 💬 INTERACTIVE CHAT WITH AI ADVISOR
                st.markdown("---")
                st.markdown("### 💬 Chat with AI Financial Advisor")
                
                # LLM Status Indicator
                col_status, col_refresh = st.columns([3, 1])
                
                with col_status:
                    llm_status = st.session_state.llm_assistant.llm_available
                    if llm_status:
                        st.success("🤖 AI Advisor Online - Enhanced responses available!")
                    else:
                        st.warning("⚠️ AI Advisor in Basic Mode - Install Ollama for enhanced responses")
                
                with col_refresh:
                    if st.button("🔄 Refresh Status"):
                        st.session_state.llm_assistant.llm_available = st.session_state.llm_assistant._check_llm_availability()
                        st.rerun()
                
                if not llm_status:
                    with st.expander("🛠️ How to enable enhanced AI responses"):
                        st.code("""
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download AI model
ollama pull llama3.1:8b

# Start service
ollama serve
                        """)
                
                st.info("Ask questions about the predictions, investment strategies, or market analysis!")
                
                # Chat interface
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                user_question = st.text_input("💭 Ask your question:", placeholder="e.g., What's the risk if I invest $50k? Should I buy now?")
                
                if st.button("🤖 Ask AI Advisor") and user_question:
                    with st.spinner("🧠 AI is thinking..."):
                        market_data = {
                            'symbol': request['symbol'],
                            'company': request['company'],
                            'predicted_price': ensemble['predicted_price'],
                            'current_price': ensemble['current_price'],
                            'expected_return': ensemble['expected_return'],
                            'confidence': ensemble['confidence'],
                            'investment_amount': request['investment'],
                            'risk_tolerance': request['risk'],
                            'prediction_days': request['days']
                        }
                        
                        ai_response = st.session_state.llm_assistant.chat_response(user_question, market_data)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({"user": user_question, "ai": ai_response})
                
                # Display chat history
                if st.session_state.chat_history:
                    st.markdown("#### 💬 Conversation History")
                    for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):  # Show last 3 exchanges
                        st.markdown(f"**You:** {chat['user']}")
                        st.markdown(f"**🤖 AI Advisor:** {chat['ai']}")
                        if i < len(st.session_state.chat_history[-3:]) - 1:
                            st.markdown("---")
    
    # Footer with signature
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); 
                border-radius: 15px; color: white; margin-top: 2rem;">
        <h3>🚀 Neural Stock Market Prophet</h3>
        <p><strong>Where AI Meets Wall Street</strong></p>
        <p style="margin-top: 1rem; font-size: 16px;">
            Built with ❤️ by <strong>Pravin Menghani</strong><br>
            In love with Neural Networks!!
        </p>
        <p style="font-size: 14px; opacity: 0.9; margin-top: 1rem;">
            📈 Multi-Agent AI • 🧠 Neural Predictions • 💰 Smart Trading
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
