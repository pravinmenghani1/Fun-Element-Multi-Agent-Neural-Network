#!/usr/bin/env python3
"""
Neural Network Visualizations for Stock Market Prophet
Amazing visualizations that make neural networks stick to memory forever!
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_3d_neural_topology():
    """Create 3D visualization of neural network battle arena"""
    
    # Neural network positions in 3D battle arena
    networks = {
        'LSTM': (0, 0, 3),
        'GRU': (-3, -2, 1),
        'CNN': (3, -2, 1),
        'Transformer': (-3, 2, 1),
        'GAN': (3, 2, 1),
        'Ensemble': (0, 0, -1)
    }
    
    # Color coding by network type
    network_colors = {
        'LSTM': 'red',
        'GRU': 'blue', 
        'CNN': 'green',
        'Transformer': 'orange',
        'GAN': 'purple',
        'Ensemble': 'gold'
    }
    
    fig = go.Figure()
    
    # Add network nodes with enhanced styling
    for network, (x, y, z) in networks.items():
        size = 25 if network == 'Ensemble' else 18
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(
                size=size,
                color=network_colors[network],
                opacity=0.8,
                line=dict(width=3, color='white')
            ),
            text=[network],
            textposition='top center',
            textfont=dict(size=10, color='black', family='Arial Black'),
            name=network,
            hovertemplate=f"<b>{network}</b><br>Neural Network<br>Position: ({x}, {y}, {z})<extra></extra>"
        ))
    
    # Add battle connections
    ensemble_pos = networks['Ensemble']
    for network, pos in networks.items():
        if network != 'Ensemble':
            # Different line styles for different connections
            line_color = network_colors[network]
            line_width = 4
            
            fig.add_trace(go.Scatter3d(
                x=[ensemble_pos[0], pos[0]],
                y=[ensemble_pos[1], pos[1]],
                z=[ensemble_pos[2], pos[2]],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title="🌐 Neural Network Battle Arena - Stock Market Prediction",
        scene=dict(
            xaxis=dict(title="Market Volatility", showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="Time Horizon", showgrid=True, gridcolor='lightgray'),
            zaxis=dict(title="Prediction Accuracy", showgrid=True, gridcolor='lightgray'),
            bgcolor='rgba(240,240,240,0.1)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def create_lstm_memory_visualization():
    """Create LSTM memory cell visualization"""
    
    # Generate memory states over time
    time_steps = list(range(1, 21))
    
    # LSTM gates
    forget_gate = [0.3 + 0.4 * np.sin(t * 0.3) + np.random.normal(0, 0.1) for t in time_steps]
    input_gate = [0.6 + 0.3 * np.cos(t * 0.2) + np.random.normal(0, 0.1) for t in time_steps]
    output_gate = [0.5 + 0.4 * np.sin(t * 0.4) + np.random.normal(0, 0.1) for t in time_steps]
    cell_state = [0.7 + 0.2 * np.sin(t * 0.1) + np.random.normal(0, 0.05) for t in time_steps]
    
    fig = go.Figure()
    
    # Add LSTM gates
    fig.add_trace(go.Scatter(
        x=time_steps, y=forget_gate,
        mode='lines+markers',
        name='🚪 Forget Gate',
        line=dict(color='red', width=3),
        hovertemplate="<b>Forget Gate</b><br>Time: %{x}<br>Activation: %{y:.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=time_steps, y=input_gate,
        mode='lines+markers',
        name='📥 Input Gate',
        line=dict(color='blue', width=3),
        hovertemplate="<b>Input Gate</b><br>Time: %{x}<br>Activation: %{y:.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=time_steps, y=output_gate,
        mode='lines+markers',
        name='📤 Output Gate',
        line=dict(color='green', width=3),
        hovertemplate="<b>Output Gate</b><br>Time: %{x}<br>Activation: %{y:.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=time_steps, y=cell_state,
        mode='lines+markers',
        name='🧠 Cell State (Memory)',
        line=dict(color='purple', width=4),
        hovertemplate="<b>Cell State</b><br>Time: %{x}<br>Memory: %{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="📈 LSTM Memory Gates - How AI Remembers Market Patterns",
        xaxis_title="Time Steps (Market Days)",
        yaxis_title="Gate Activation (0-1)",
        height=400,
        annotations=[
            dict(
                x=15, y=0.8,
                text="🧠 Cell State stores<br>long-term market memory",
                showarrow=True,
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="purple"
            )
        ]
    )
    
    return fig

def create_gru_trend_visualization():
    """Create GRU trend capture visualization"""
    
    # Generate trend data
    days = list(range(1, 31))
    
    # Market trend with GRU gates
    market_trend = [100 + 2*d + 10*np.sin(d*0.2) + np.random.normal(0, 3) for d in days]
    reset_gate = [0.4 + 0.3 * np.sin(d * 0.3) + np.random.normal(0, 0.1) for d in days]
    update_gate = [0.6 + 0.2 * np.cos(d * 0.25) + np.random.normal(0, 0.1) for d in days]
    
    fig = go.Figure()
    
    # Market price trend
    fig.add_trace(go.Scatter(
        x=days, y=market_trend,
        mode='lines+markers',
        name='📈 Stock Price',
        line=dict(color='black', width=3),
        yaxis='y1'
    ))
    
    # GRU gates
    fig.add_trace(go.Scatter(
        x=days, y=reset_gate,
        mode='lines',
        name='🔄 Reset Gate',
        line=dict(color='red', width=2),
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=days, y=update_gate,
        mode='lines',
        name='⬆️ Update Gate',
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="🌊 GRU Trend Capture - Efficient Market Memory",
        xaxis_title="Trading Days",
        yaxis=dict(title="Stock Price ($)", side='left'),
        yaxis2=dict(title="Gate Activation", side='right', overlaying='y', range=[0, 1]),
        height=400
    )
    
    return fig

def create_cnn_pattern_visualization():
    """Create CNN pattern recognition visualization"""
    
    # Generate candlestick-like pattern data
    days = list(range(1, 21))
    
    # Create different chart patterns
    head_shoulders = [50, 55, 60, 58, 65, 70, 68, 60, 55, 50, 45, 48, 52, 50, 45, 40, 38, 35, 32, 30]
    
    # CNN filters detecting patterns
    filter1 = np.convolve(head_shoulders, [0.25, 0.5, 0.25], mode='same')  # Smoothing filter
    filter2 = np.convolve(head_shoulders, [-1, 0, 1], mode='same')  # Edge detection
    filter3 = np.convolve(head_shoulders, [1, -2, 1], mode='same')  # Peak detection
    
    fig = go.Figure()
    
    # Original price pattern
    fig.add_trace(go.Scatter(
        x=days, y=head_shoulders,
        mode='lines+markers',
        name='📊 Price Pattern',
        line=dict(color='black', width=3),
        fill='tonexty'
    ))
    
    # CNN filters
    fig.add_trace(go.Scatter(
        x=days, y=filter1,
        mode='lines',
        name='🔍 Smoothing Filter',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=days, y=filter2 + 50,  # Offset for visibility
        mode='lines',
        name='⚡ Edge Detection',
        line=dict(color='red', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=days, y=filter3 + 50,  # Offset for visibility
        mode='lines',
        name='🎯 Peak Detection',
        line=dict(color='green', width=2, dash='dashdot')
    ))
    
    fig.update_layout(
        title="🎯 CNN Pattern Recognition - Head & Shoulders Detection",
        xaxis_title="Trading Days",
        yaxis_title="Price / Filter Response",
        height=400,
        annotations=[
            dict(
                x=10, y=70,
                text="📈 Head & Shoulders<br>Pattern Detected!",
                showarrow=True,
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red"
            )
        ]
    )
    
    return fig

def create_transformer_attention_viz():
    """Create transformer attention mechanism visualization"""
    
    # Market factors that transformer pays attention to
    factors = ['Price', 'Volume', 'News', 'Sentiment', 'Technical', 'Macro']
    timeframes = ['1D', '1W', '1M', '3M', '6M', '1Y']
    
    # Generate attention weights
    np.random.seed(42)
    attention_weights = np.random.rand(len(timeframes), len(factors))
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    fig = px.imshow(
        attention_weights,
        x=factors,
        y=timeframes,
        color_continuous_scale='Viridis',
        title="🔄 Transformer Attention - Multi-Timeframe Market Analysis",
        labels=dict(color="Attention Weight")
    )
    
    fig.update_layout(height=400)
    return fig

def create_gan_simulation_viz():
    """Create GAN market simulation visualization"""
    
    days = list(range(1, 101))
    
    # Real market data (simplified)
    real_data = [100 + 0.5*d + 10*np.sin(d*0.1) + np.random.normal(0, 2) for d in days]
    
    # GAN generated fake data (getting better over epochs)
    epochs = [1, 10, 50, 100]
    colors = ['red', 'orange', 'yellow', 'green']
    
    fig = go.Figure()
    
    # Real data
    fig.add_trace(go.Scatter(
        x=days, y=real_data,
        mode='lines',
        name='📊 Real Market Data',
        line=dict(color='black', width=3)
    ))
    
    # GAN generated data at different epochs
    for epoch, color in zip(epochs, colors):
        noise_level = max(0.1, 5 - epoch/20)  # Less noise as training progresses
        fake_data = [100 + 0.5*d + 10*np.sin(d*0.1) + np.random.normal(0, noise_level) for d in days]
        
        fig.add_trace(go.Scatter(
            x=days, y=fake_data,
            mode='lines',
            name=f'🎭 GAN Epoch {epoch}',
            line=dict(color=color, width=2, dash='dash'),
            opacity=0.7
        ))
    
    fig.update_layout(
        title="🎭 GAN Market Simulation - Fake vs Real Data",
        xaxis_title="Trading Days",
        yaxis_title="Stock Price ($)",
        height=400,
        annotations=[
            dict(
                x=80, y=max(real_data),
                text="🎯 GAN learns to generate<br>realistic market scenarios",
                showarrow=True,
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="green"
            )
        ]
    )
    
    return fig

def create_neural_battle_viz():
    """Create neural network battle visualization"""
    
    epochs = list(range(1, 21))
    
    # Different networks' performance over time
    lstm_acc = [60 + 2*e + np.random.normal(0, 2) for e in epochs]
    gru_acc = [55 + 2.5*e + np.random.normal(0, 2) for e in epochs]
    cnn_acc = [50 + 3*e + np.random.normal(0, 2) for e in epochs]
    transformer_acc = [45 + 3.5*e + np.random.normal(0, 2) for e in epochs]
    
    fig = go.Figure()
    
    networks = [
        ('📈 LSTM', lstm_acc, 'red'),
        ('🌊 GRU', gru_acc, 'blue'),
        ('🎯 CNN', cnn_acc, 'green'),
        ('🔄 Transformer', transformer_acc, 'orange')
    ]
    
    for name, accuracy, color in networks:
        fig.add_trace(go.Scatter(
            x=epochs, y=accuracy,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=3),
            marker=dict(size=8)
        ))
    
    # Add ensemble (best of all)
    ensemble_acc = [max(lstm_acc[i], gru_acc[i], cnn_acc[i], transformer_acc[i]) + 5 for i in range(len(epochs))]
    fig.add_trace(go.Scatter(
        x=epochs, y=ensemble_acc,
        mode='lines+markers',
        name='🏆 Ensemble Winner',
        line=dict(color='gold', width=4),
        marker=dict(size=10, symbol='star')
    ))
    
    fig.update_layout(
        title="🧠 Neural Network Battle - Trading Accuracy Competition",
        xaxis_title="Training Epochs",
        yaxis_title="Prediction Accuracy (%)",
        height=400,
        yaxis=dict(range=[40, 110])
    )
    
    return fig

def create_rl_strategy_viz():
    """Create reinforcement learning strategy visualization"""
    
    episodes = list(range(1, 101))
    
    # RL learning curve - starts poor, gets better
    cumulative_reward = []
    total_reward = 0
    
    for episode in episodes:
        # Early episodes: poor performance, later: much better
        if episode < 20:
            episode_reward = np.random.normal(-50, 30)  # Losing money initially
        elif episode < 50:
            episode_reward = np.random.normal(10, 40)   # Learning phase
        else:
            episode_reward = np.random.normal(80, 25)   # Profitable phase
        
        total_reward += episode_reward
        cumulative_reward.append(total_reward)
    
    # Trading actions over time
    actions = ['BUY', 'SELL', 'HOLD']
    action_counts = [40, 35, 25]  # Distribution of actions
    
    fig = go.Figure()
    
    # Cumulative reward curve
    fig.add_trace(go.Scatter(
        x=episodes, y=cumulative_reward,
        mode='lines+markers',
        name='💰 Cumulative Profit/Loss',
        line=dict(color='green', width=3),
        fill='tonexty'
    ))
    
    # Add learning phases
    fig.add_vrect(x0=1, x1=20, fillcolor="red", opacity=0.2, 
                  annotation_text="Learning Phase", annotation_position="top left")
    fig.add_vrect(x0=20, x1=50, fillcolor="yellow", opacity=0.2,
                  annotation_text="Adaptation Phase", annotation_position="top left")
    fig.add_vrect(x0=50, x1=100, fillcolor="green", opacity=0.2,
                  annotation_text="Profitable Phase", annotation_position="top left")
    
    fig.update_layout(
        title="🤖 Reinforcement Learning: AI Trader Learning Curve",
        xaxis_title="Trading Episodes",
        yaxis_title="Cumulative Profit ($)",
        height=400,
        annotations=[
            dict(
                x=75, y=max(cumulative_reward),
                text="🎯 RL learns optimal<br>trading strategy!",
                showarrow=True,
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="green"
            )
        ]
    )
    
    return fig

def create_realtime_prediction_engine():
    """Create real-time prediction engine visualization"""
    
    # Simulate real-time predictions
    time_points = pd.date_range(start='2024-01-01', periods=100, freq='H')
    
    # Multiple predictions updating in real-time
    predictions = []
    for i, time_point in enumerate(time_points):
        base_price = 150 + 10 * np.sin(i * 0.1)
        lstm_pred = base_price + np.random.normal(0, 2)
        gru_pred = base_price + np.random.normal(0, 1.5)
        cnn_pred = base_price + np.random.normal(0, 2.5)
        
        predictions.append({
            'time': time_point,
            'LSTM': lstm_pred,
            'GRU': gru_pred,
            'CNN': cnn_pred,
            'Ensemble': (lstm_pred + gru_pred + cnn_pred) / 3
        })
    
    df = pd.DataFrame(predictions)
    
    fig = go.Figure()
    
    for column in ['LSTM', 'GRU', 'CNN', 'Ensemble']:
        color = {'LSTM': 'red', 'GRU': 'blue', 'CNN': 'green', 'Ensemble': 'gold'}[column]
        width = 4 if column == 'Ensemble' else 2
        
        fig.add_trace(go.Scatter(
            x=df['time'], y=df[column],
            mode='lines',
            name=f'🤖 {column}',
            line=dict(color=color, width=width)
        ))
    
    fig.update_layout(
        title="📊 Real-time Neural Prediction Engine",
        xaxis_title="Time",
        yaxis_title="Predicted Price ($)",
        height=400
    )
    
    return fig
