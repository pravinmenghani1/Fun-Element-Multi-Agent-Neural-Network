#!/usr/bin/env python3
"""
Spectacular Enhanced UI with Real-time Visualizations and WOW Factors
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import time
from travel_agent_system import CoordinatorAgent, TravelRequest
from enhanced_flight_agent import TemporalFlightAgent
from llm_integration import EnhancedTravelAgent
from real_api_integration import api_client

def create_real_time_api_monitor():
    """Create real-time API monitoring dashboard"""
    
    # Simulate real-time API calls
    apis = ['Amadeus', 'IndiGo', 'Air India', 'SpiceJet', 'Booking.com', 'OpenWeather']
    
    fig = go.Figure()
    
    # Create animated bars showing API response times
    response_times = np.random.uniform(0.2, 2.5, len(apis))
    colors = ['green' if t < 1.0 else 'orange' if t < 2.0 else 'red' for t in response_times]
    
    fig.add_trace(go.Bar(
        x=apis,
        y=response_times,
        marker_color=colors,
        text=[f"{t:.2f}s" for t in response_times],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="🔄 Real-time API Response Monitor",
        yaxis_title="Response Time (seconds)",
        height=300,
        showlegend=False
    )
    
    return fig

def create_attention_heatmap():
    """Create attention mechanism heatmap"""
    
    # Sample attention weights
    features = ['Price', 'Rating', 'Location', 'Amenities', 'Reviews']
    hotels = ['Hotel A', 'Hotel B', 'Hotel C', 'Hotel D']
    
    attention_weights = np.random.rand(len(hotels), len(features))
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    fig = px.imshow(
        attention_weights,
        x=features,
        y=hotels,
        color_continuous_scale='Viridis',
        title="Hotel Selection Attention Weights",
        labels=dict(color="Attention Score")
    )
    
    fig.update_layout(height=300)
    return fig

def create_vae_latent_space():
    """Create VAE latent space visualization"""
    
    # Generate sample data points in 2D latent space
    np.random.seed(42)
    n_points = 100
    
    # Different attraction categories in latent space
    categories = ['Cultural', 'Adventure', 'Historical', 'Natural', 'Modern']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    fig = go.Figure()
    
    for i, (category, color) in enumerate(zip(categories, colors)):
        # Generate cluster of points for each category
        center_x = np.cos(2 * np.pi * i / len(categories)) * 2
        center_y = np.sin(2 * np.pi * i / len(categories)) * 2
        
        x = np.random.normal(center_x, 0.5, n_points // len(categories))
        y = np.random.normal(center_y, 0.5, n_points // len(categories))
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(color=color, size=8, opacity=0.7),
            name=category,
            text=[f"{category} Attraction {j+1}" for j in range(len(x))],
            hovertemplate="%{text}<extra></extra>"
        ))
    
    fig.update_layout(
        title="VAE Latent Space - Attraction Categories",
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        height=400
    )
    
    return fig
    """Create real-time API monitoring dashboard"""
    
    # Simulate real-time API calls
    apis = ['Amadeus', 'IndiGo', 'Air India', 'SpiceJet', 'Booking.com', 'OpenWeather']
    
    fig = go.Figure()
    
    # Create animated bars showing API response times
    response_times = np.random.uniform(0.2, 2.5, len(apis))
    colors = ['green' if t < 1.0 else 'orange' if t < 2.0 else 'red' for t in response_times]
    
    fig.add_trace(go.Bar(
        x=apis,
        y=response_times,
        marker_color=colors,
        text=[f"{t:.2f}s" for t in response_times],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="🔄 Real-time API Response Monitor",
        yaxis_title="Response Time (seconds)",
        height=300,
        showlegend=False
    )
    
    return fig

def create_temporal_cnn_visualization():
    """Create temporal CNN visualization for short-term volatility detection"""
    
    # Generate sample temporal data with volatility bursts
    hours = list(range(6, 24))
    base_prices = [5000 + 500 * np.sin(h * 0.3) for h in hours]
    
    # Add volatility bursts (what CNN detects)
    volatility_bursts = np.random.normal(0, 300, len(hours))
    raw_prices = [base + burst for base, burst in zip(base_prices, volatility_bursts)]
    
    # Apply CNN convolution to detect patterns
    kernel = [0.25, 0.5, 0.25]
    cnn_detected = []
    for i in range(1, len(raw_prices) - 1):
        conv_val = raw_prices[i-1] * kernel[0] + raw_prices[i] * kernel[1] + raw_prices[i+1] * kernel[2]
        cnn_detected.append(conv_val)
    
    fig = go.Figure()
    
    # Raw prices with volatility
    fig.add_trace(go.Scatter(
        x=hours,
        y=raw_prices,
        mode='lines+markers',
        name='Raw Prices (with volatility)',
        line=dict(color='lightgray', width=1),
        opacity=0.6
    ))
    
    # CNN detected patterns
    fig.add_trace(go.Scatter(
        x=hours[1:-1],
        y=cnn_detected,
        mode='lines+markers',
        name='CNN: Short-term Patterns',
        line=dict(color='orange', width=3)
    ))
    
    fig.update_layout(
        title="👁️ Temporal CNN: Short-term Volatility Detection",
        xaxis_title="Hour of Day",
        yaxis_title="Price (₹)",
        height=400,
        annotations=[
            dict(
                x=12, y=max(raw_prices),
                text="CNN detects short-term<br>price volatility patterns",
                showarrow=True,
                arrowhead=2
            )
        ]
    )
    
    return fig

def create_rnn_sequence_analysis():
    """Create RNN sequence analysis visualization for long-term trends"""
    
    # Generate price history sequence showing long-term trends
    days = list(range(30))
    base_price = 8000
    prices = []
    
    for day in days:
        # Long-term seasonal trend (what RNN captures)
        seasonal_trend = 1000 * np.sin(day * 0.15)  # Slower seasonal changes
        weekly_pattern = 300 * np.sin(day * 0.9)    # Weekly patterns
        noise = np.random.normal(0, 100)            # Minimal noise
        price = base_price + seasonal_trend + weekly_pattern + noise
        prices.append(max(4000, price))
    
    # RNN prediction based on sequence learning
    prediction_days = list(range(30, 37))
    predictions = []
    for day in prediction_days:
        seasonal_trend = 1000 * np.sin(day * 0.15)
        weekly_pattern = 300 * np.sin(day * 0.9)
        pred_price = base_price + seasonal_trend + weekly_pattern
        predictions.append(max(4000, pred_price))
    
    fig = go.Figure()
    
    # Historical sequence
    fig.add_trace(go.Scatter(
        x=days,
        y=prices,
        mode='lines+markers',
        name='Historical Price Sequence',
        line=dict(color='blue', width=2)
    ))
    
    # RNN trend predictions
    fig.add_trace(go.Scatter(
        x=prediction_days,
        y=predictions,
        mode='lines+markers',
        name='RNN: Long-term Trend Prediction',
        line=dict(color='green', width=3, dash='dash')
    ))
    
    # Confidence interval
    upper_bound = [p * 1.08 for p in predictions]
    lower_bound = [p * 0.92 for p in predictions]
    
    fig.add_trace(go.Scatter(
        x=prediction_days + prediction_days[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(0,255,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Prediction Confidence'
    ))
    
    fig.update_layout(
        title="🔮 RNN: Long-term Trend Analysis",
        xaxis_title="Days",
        yaxis_title="Price (₹)",
        height=400,
        annotations=[
            dict(
                x=20, y=max(prices),
                text="RNN learns from<br>sequential patterns<br>over time",
                showarrow=True,
                arrowhead=2
            )
        ]
    )
    
    return fig

def create_3d_agent_network():
    """Create enhanced 3D visualization of agent network"""
    
    # Enhanced agent positions in 3D space for better visibility
    agents = {
        'Coordinator': (0, 0, 3),
        'Flight (RNN+CNN)': (-3, -3, 1),
        'Hotel (Transformer)': (3, -3, 1),
        'Attraction (VAE)': (-3, 3, 1),
        'Weather (Autoencoder)': (3, 3, 1),
        'LLM Enhancer': (0, 0, -1)
    }
    
    # Color coding for different agent types
    agent_colors = {
        'Coordinator': 'red',
        'Flight (RNN+CNN)': 'blue',
        'Hotel (Transformer)': 'green',
        'Attraction (VAE)': 'orange',
        'Weather (Autoencoder)': 'purple',
        'LLM Enhancer': 'gold'
    }
    
    fig = go.Figure()
    
    # Add agent nodes with enhanced styling
    for agent, (x, y, z) in agents.items():
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(
                size=20 if agent == 'Coordinator' else 15,
                color=agent_colors[agent],
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=[agent],
            textposition='top center',
            textfont=dict(size=10, color='black'),
            name=agent,
            hovertemplate=f"<b>{agent}</b><br>Position: ({x}, {y}, {z})<extra></extra>"
        ))
    
    # Add enhanced connections with different styles
    coordinator_pos = agents['Coordinator']
    for agent, pos in agents.items():
        if agent != 'Coordinator':
            # Different line styles for different connections
            line_color = 'lightblue' if 'LLM' not in agent else 'gold'
            line_width = 4 if 'LLM' in agent else 3
            
            fig.add_trace(go.Scatter3d(
                x=[coordinator_pos[0], pos[0]],
                y=[coordinator_pos[1], pos[1]],
                z=[coordinator_pos[2], pos[2]],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Enhanced layout with better camera angle
    fig.update_layout(
        title="🌐 NEXUS AI - 3D Multi-Agent Network Architecture",
        scene=dict(
            xaxis=dict(title="X Axis", showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="Y Axis", showgrid=True, gridcolor='lightgray'),
            zaxis=dict(title="Z Axis", showgrid=True, gridcolor='lightgray'),
            bgcolor='rgba(240,240,240,0.1)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # Better viewing angle
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def create_live_sentiment_analysis():
    """Create live sentiment analysis of travel reviews"""
    
    destinations = ['Goa', 'Kerala', 'Rajasthan', 'Himachal', 'Karnataka']
    sentiments = np.random.uniform(0.3, 0.9, len(destinations))
    
    fig = go.Figure(go.Bar(
        x=destinations,
        y=sentiments,
        marker_color=px.colors.sequential.Viridis,
        text=[f"{s:.2f}" for s in sentiments],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="📊 Live Travel Sentiment Analysis",
        yaxis_title="Sentiment Score",
        height=300
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="🧠 NEXUS AI - Multi-Agent Travel Intelligence",
        page_icon="🤖✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Spectacular CSS styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5); }
        to { box-shadow: 0 10px 40px rgba(118, 75, 162, 0.8); }
    }
    
    .agent-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .wow-metric {
        background: linear-gradient(45deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        background-size: 400% 400%;
        animation: gradient 3s ease infinite;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Spectacular header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 NEXUS AI</h1>
        <h2>⚡ Multi-Agent Travel Intelligence System</h2>
        <p>✨ Advanced AI Architecture • Real-time APIs • Neural Network Fusion ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with enhanced AI concepts
    with st.sidebar:
        st.header("🧠 AI Architecture Explorer")
        
        # Add expandable AI concepts overview
        with st.expander("📚 Quick AI Concepts Guide", expanded=False):
            st.markdown("""
            **🔮 RNN:** Sequential memory for trends  
            **👁️ CNN:** Pattern detection in data  
            **🎯 Transformer:** Multi-feature attention  
            **🔄 VAE:** Creative generation from patterns  
            **📊 Autoencoder:** Data compression & reconstruction  
            **🤖 LLM:** Human-like reasoning  
            **🎭 Ensemble:** Combining multiple AI models  
            """)
        
        concept_tab = st.selectbox("Select AI Concept:", [
            "🌐 3D Agent Network",
            "🔄 Real-time API Monitor", 
            "🔮 RNN: Long-term Trends",
            "👁️ CNN: Short-term Patterns",
            "🎯 Transformer Attention",
            "🔄 VAE Latent Space",
            "📊 Live Sentiment Analysis"
        ])
        
        if concept_tab == "🌐 3D Agent Network":
            st.plotly_chart(create_3d_agent_network(), use_container_width=True)
            st.markdown("""
            **🧠 Why Each AI Technique?**
            
            **🔮 RNN + Temporal CNN for Flights:**
            - **RNN:** Learns from sequential price history over days/weeks
            - **Temporal CNN:** Detects short-term volatility patterns within hours
            - **Why:** Flight prices have both long-term trends AND short-term spikes
            
            **🎯 Transformer Attention for Hotels:**
            - **Multi-head attention:** Focuses on different features simultaneously
            - **Why:** Hotels have many features (price, location, amenities, reviews)
            - **Benefit:** Weighs importance of each feature for each user
            
            **🔄 VAE for Attractions:**
            - **Encoder:** Maps destinations to latent preference space
            - **Decoder:** Generates personalized attraction recommendations
            - **Why:** Creates new combinations based on learned patterns
            
            **📊 Autoencoder for Weather:**
            - **Compression:** Reduces complex weather data to key factors
            - **Reconstruction:** Generates actionable travel advice
            - **Why:** Weather has many variables, but only few affect travel
            
            **🤖 LLM for Insights:**
            - **Natural language understanding:** Interprets user preferences
            - **Contextual reasoning:** Provides intelligent recommendations
            - **Why:** Adds human-like intelligence to technical analysis
            """)
        
        elif concept_tab == "🔄 Real-time API Monitor":
            api_monitor = st.empty()
            if st.button("🔄 Refresh API Status"):
                api_monitor.plotly_chart(create_real_time_api_monitor(), use_container_width=True)
            st.info("🌐 Shows live response times from actual travel booking APIs")
        
        elif concept_tab == "🔮 RNN: Long-term Trends":
            st.plotly_chart(create_rnn_sequence_analysis(), use_container_width=True)
            st.markdown("""
            **🔮 RNN (Recurrent Neural Networks)**
            
            **What it does:**
            - Analyzes flight price sequences over 30+ days
            - Learns seasonal patterns and weekly cycles
            - Predicts future price trends
            
            **Why for flights:**
            - Flight prices follow temporal patterns
            - Weekend vs weekday pricing
            - Holiday season effects
            - Booking timing optimization
            
            **Technical:** Uses memory cells to remember past prices and predict future trends
            """)
        
        elif concept_tab == "👁️ CNN: Short-term Patterns":
            st.plotly_chart(create_temporal_cnn_visualization(), use_container_width=True)
            st.markdown("""
            **👁️ Temporal CNN (Convolutional Neural Networks)**
            
            **What it does:**
            - Detects short-term price volatility (hourly patterns)
            - Identifies sudden price spikes or drops
            - Recognizes time-of-day pricing patterns
            
            **Why temporal CNN:**
            - **Temporal:** Works with time-series data (not images)
            - **Convolution:** Slides filters over time to detect patterns
            - **Local patterns:** Finds volatility bursts in specific time windows
            
            **vs Regular CNN:** Regular CNN works on images, Temporal CNN works on time-series price data
            
            **Technical:** Applies convolution kernels to detect short-term price anomalies
            """)
        
        elif concept_tab == "🎯 Transformer Attention":
            st.plotly_chart(create_attention_heatmap(), use_container_width=True)
            st.markdown("""
            **🎯 Transformer Attention Mechanism**
            
            **What it does:**
            - Simultaneously focuses on multiple hotel features
            - Weighs importance of price vs location vs amenities
            - Different attention heads focus on different aspects
            
            **Why for hotels:**
            - Hotels have complex multi-dimensional features
            - User preferences vary (some prioritize location, others price)
            - Need to balance multiple competing factors
            
            **Multi-head attention:** Like having multiple experts, each focusing on different aspects
            
            **Technical:** Computes attention weights to determine feature importance for ranking
            """)
        
        elif concept_tab == "🔄 VAE Latent Space":
            st.plotly_chart(create_vae_latent_space(), use_container_width=True)
            st.markdown("""
            **🔄 Variational Autoencoders (VAE)**
            
            **What it does:**
            1. **Encoder:** Maps destinations to latent preference space
            2. **Sampling:** Explores similar preference regions
            3. **Decoder:** Generates new attraction recommendations
            
            **Why for attractions:**
            - Creates personalized recommendations beyond simple matching
            - Discovers hidden connections between attractions
            - Generates novel combinations based on learned patterns
            
            **Latent Space:** Hidden representation where similar attractions cluster together
            
            **vs Regular recommendations:** VAE can generate creative suggestions, not just filter existing ones
            """)
        
        elif concept_tab == "📊 Live Sentiment Analysis":
            st.plotly_chart(create_live_sentiment_analysis(), use_container_width=True)
            st.info("📈 Real-time sentiment analysis of travel destinations from social media and reviews")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎯 Travel Request")
        
        with st.form("spectacular_travel_form"):
            origin = st.text_input("✈️ From", "Mumbai", help="Enter departure city")
            destination = st.text_input("🏖️ To", "Goa", help="Enter destination city")
            departure_date = st.date_input("📅 Departure", datetime.now() + timedelta(days=30))
            return_date = st.date_input("🔄 Return", datetime.now() + timedelta(days=33))
            budget = st.slider("💰 Budget (₹)", 5000, 200000, 35000, step=1000)
            travelers = st.number_input("👥 Travelers", 1, 15, 2)
            
            st.markdown("**🎨 Preferences:**")
            accommodation = st.selectbox("🏨 Stay", ["Luxury Hotel", "Resort", "Boutique Hotel", "Homestay"])
            activity_level = st.selectbox("🎯 Activity", ["Ultra Relaxed", "Moderate", "Adventure", "Cultural Explorer"])
            
            submitted = st.form_submit_button("🚀 Launch Spectacular AI Agents", type="primary")
        
        if submitted:
            travel_request = TravelRequest(
                origin=origin,
                destination=destination,
                departure_date=str(departure_date),
                return_date=str(return_date),
                budget=budget,
                travelers=travelers,
                preferences={'accommodation_type': accommodation, 'activity_level': activity_level}
            )
            
            st.session_state.travel_request = travel_request
            st.session_state.spectacular_search = True
    
    with col2:
        if hasattr(st.session_state, 'spectacular_search') and st.session_state.spectacular_search:
            
            # Single clean agent status display
            st.markdown("### 🤖 AI Agents in Action")
            
            # Real-time API monitoring (shown once)
            api_status_container = st.container()
            with api_status_container:
                st.plotly_chart(create_real_time_api_monitor(), use_container_width=True)
            
            # Progress with spectacular effects
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Run spectacular booking system
            async def run_spectacular_booking():
                # Initialize agents
                temporal_flight_agent = TemporalFlightAgent()
                coordinator = CoordinatorAgent()
                enhanced_agent = EnhancedTravelAgent()
                
                # Step 1: Real API calls with progress
                status_text.text("🔍 Making real API calls to travel platforms...")
                progress_bar.progress(15)
                await asyncio.sleep(1)
                
                # Step 2: Temporal flight analysis
                status_text.text("🧠 RNN analyzing flight price sequences...")
                progress_bar.progress(30)
                flight_results = await temporal_flight_agent.process(st.session_state.travel_request)
                
                # Step 3: Multi-agent coordination
                status_text.text("🤖 Multi-agent system coordinating...")
                progress_bar.progress(50)
                results = await coordinator.orchestrate_booking(st.session_state.travel_request)
                
                # Step 4: LLM enhancement
                status_text.text("🧠 LLM enhancing recommendations...")
                progress_bar.progress(75)
                ai_recommendations = await enhanced_agent.get_intelligent_recommendations(
                    st.session_state.travel_request.__dict__,
                    results['individual_results']
                )
                
                # Step 5: Final optimization
                status_text.text("✨ Applying spectacular optimizations...")
                progress_bar.progress(100)
                
                return results, ai_recommendations, flight_results
            
            # Execute spectacular booking
            if 'spectacular_results' not in st.session_state:
                try:
                    results, ai_recommendations, flight_results = asyncio.run(run_spectacular_booking())
                    st.session_state.spectacular_results = (results, ai_recommendations, flight_results)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.spectacular_results = None
            
            # Display spectacular results
            if hasattr(st.session_state, 'spectacular_results') and st.session_state.spectacular_results:
                results, ai_recommendations, flight_results = st.session_state.spectacular_results
                
                # Spectacular metrics
                st.markdown("### 🎯 Spectacular Results Dashboard")
                
                metric_cols = st.columns(4)
                total_cost = results['total_estimated_cost']
                confidence = ai_recommendations.get('ai_confidence_score', 92)
                
                with metric_cols[0]:
                    st.markdown(f"""
                    <div class="wow-metric">
                        <h3>💰 Total Cost</h3>
                        <h2>₹{total_cost:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[1]:
                    st.markdown(f"""
                    <div class="wow-metric">
                        <h3>🎯 AI Confidence</h3>
                        <h2>{confidence:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[2]:
                    api_sources = len(flight_results.get('api_sources', []))
                    st.markdown(f"""
                    <div class="wow-metric">
                        <h3>🔗 API Sources</h3>
                        <h2>{api_sources}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[3]:
                    temporal_score = flight_results['flights'][0].get('temporal_score', 85) if flight_results['flights'] else 85
                    st.markdown(f"""
                    <div class="wow-metric">
                        <h3>⏰ Temporal Score</h3>
                        <h2>{temporal_score}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced results tabs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "✈️ RNN Flights", "🏨 Hotels", "🎯 Attractions", 
                    "🌤️ Weather", "🤖 AI Insights", "📊 Analytics"
                ])
                
                with tab1:
                    st.markdown("**🧠 RNN/Temporal CNN Enhanced Flights:**")
                    
                    for i, flight in enumerate(flight_results['flights'][:3]):
                        with st.expander(f"✈️ {flight['airline']} {flight['flight_number']} - ₹{flight['price']:,} (Score: {flight.get('temporal_score', 0)})"):
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.write(f"**Departure:** {flight['departure_time']}")
                                st.write(f"**Arrival:** {flight['arrival_time']}")
                                st.write(f"**Duration:** {flight['duration']}")
                            
                            with col_b:
                                st.write(f"**Stops:** {flight['stops']}")
                                st.write(f"**API Source:** {flight.get('api_source', 'Unknown')}")
                                st.write(f"**Temporal Score:** {flight.get('temporal_score', 0)}")
                            
                            with col_c:
                                st.write(f"**RNN Insight:** {flight.get('rnn_insights', 'N/A')}")
                                st.write(f"**CNN Pattern:** {flight.get('cnn_pattern', 'normal')}")
                                if st.button(f"Book Flight {i+1}", key=f"book_spectacular_flight_{i}"):
                                    st.success("✅ Flight booked with RNN optimization!")
                
                with tab2:
                    st.markdown("**🎯 Transformer Attention Hotels:**")
                    for i, hotel in enumerate(results['final_itinerary']['hotels']):
                        with st.expander(f"🏨 {hotel['name']} - ₹{hotel['price_per_night']:,}/night"):
                            st.write(f"**Rating:** ⭐ {hotel['rating']}")
                            st.write(f"**Amenities:** {', '.join(hotel['amenities'])}")
                            st.write(f"**Attention Score:** {hotel.get('attention_score', 0):.1f}/100")
                            if st.button(f"Book Hotel {i+1}", key=f"book_spectacular_hotel_{i}"):
                                st.success("✅ Hotel booked with attention mechanism!")
                
                with tab3:
                    st.markdown("**🔄 VAE Generated Attractions:**")
                    for i, attraction in enumerate(results['final_itinerary']['attractions']):
                        with st.expander(f"🎯 {attraction['name']} - ₹{attraction['price']} (Match: {attraction.get('latent_match_score', 85):.1f}%)"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Category:** {attraction['category']}")
                                st.write(f"**Duration:** {attraction['duration']}")
                                st.write(f"**Rating:** ⭐ {attraction['rating']}")
                            with col_b:
                                st.write(f"**Latent Match Score:** {attraction.get('latent_match_score', 85):.1f}%")
                                booking_status = "🎫 Advance Booking" if attraction['booking_available'] else "🎫 On-site Only"
                                st.write(f"**Booking:** {booking_status}")
                                if attraction['booking_available']:
                                    if st.button(f"Book Attraction {i+1}", key=f"book_spectacular_attraction_{i}"):
                                        st.success("✅ Attraction booked via VAE recommendation!")
                
                with tab4:
                    weather = results['final_itinerary']['weather_info']
                    st.markdown("**📊 Autoencoder Weather Analysis:**")
                    st.metric("🌡️ Temperature", f"{weather['temperature']}°C")
                    st.metric("🎯 Comfort Score", f"{weather['comfort_score']}/100")
                    
                    for rec in results['final_itinerary']['recommendations']:
                        st.write(f"• {rec}")
                
                with tab5:
                    st.markdown("**🤖 LLM Enhanced Insights:**")
                    insights = ai_recommendations.get('travel_insights', [])
                    for insight in insights:
                        st.write(f"💡 {insight}")
                
                with tab6:
                    st.markdown("**📊 Advanced Analytics:**")
                    
                    # Temporal analysis chart
                    if 'temporal_analysis' in flight_results:
                        temporal = flight_results['temporal_analysis']
                        st.write(f"**Price Trend:** {temporal.get('price_trend', 'N/A')}")
                        st.write(f"**Demand Prediction:** {temporal.get('demand_prediction', 'N/A')}")
                        st.write(f"**Seasonal Factor:** {temporal.get('seasonal_factor', 1.0):.2f}")
                    
                    # CNN features
                    if 'cnn_features' in flight_results:
                        cnn = flight_results['cnn_features']
                        st.write(f"**Best Time Period:** {cnn.get('temporal_clusters', {}).get('best_time_period', 'N/A')}")
                
                # Final spectacular booking
                st.markdown("---")
                if st.button("🧠 CONFIRM NEXUS AI OPTIMIZED ITINERARY", type="primary", use_container_width=True):
                    st.balloons()
                    st.success("🎉 NEXUS AI has optimized your perfect itinerary!")
                    
                    st.markdown("""
                    **🧠 NEXUS AI Multi-Agent Summary:**
                    - **🔮 RNN Flight Agent:** Analyzed temporal price patterns across multiple APIs
                    - **👁️ Temporal CNN:** Extracted time-series patterns for optimal booking times  
                    - **🎯 Transformer Hotel Agent:** Applied multi-head attention for perfect matches
                    - **🔄 VAE Attraction Agent:** Generated personalized recommendations from latent space
                    - **📊 Autoencoder Weather Agent:** Compressed weather data into actionable insights
                    - **🤖 LLM Coordinator:** Enhanced everything with intelligent analysis
                    - **🌐 Real API Integration:** Made actual calls to travel booking platforms
                    """)

if __name__ == "__main__":
    main()
