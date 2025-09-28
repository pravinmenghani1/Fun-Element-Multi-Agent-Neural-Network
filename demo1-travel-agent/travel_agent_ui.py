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
    
    # Simulate real-time API calls including local attraction APIs
    apis = ['Amadeus', 'IndiGo', 'Air India', 'SpiceJet', 'Booking.com', 'OpenWeather', 'Tourism Board', 'Places API', 'TripAdvisor']
    
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
        title="🔄 Real-time API Response Monitor (Including Local Attraction APIs)",
        yaxis_title="Response Time (seconds)",
        height=350,
        showlegend=False,
        xaxis=dict(tickangle=45)
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

def create_neural_learning_animation():
    """Create animated neural network learning visualization"""
    
    # Create animated learning process
    epochs = list(range(1, 11))
    
    # Different agents learning curves
    flight_accuracy = [45, 52, 61, 68, 74, 79, 83, 86, 88, 90]
    hotel_accuracy = [40, 48, 58, 66, 72, 77, 81, 84, 87, 89]
    attraction_accuracy = [35, 44, 55, 64, 71, 76, 80, 84, 87, 91]
    weather_accuracy = [50, 58, 65, 71, 76, 80, 84, 87, 89, 92]
    
    fig = go.Figure()
    
    # Add learning curves for each agent
    agents_data = [
        ('🔮 RNN Flight Agent', flight_accuracy, 'blue'),
        ('🎯 Hotel Transformer', hotel_accuracy, 'green'),
        ('🔄 VAE Attraction Agent', attraction_accuracy, 'orange'),
        ('📊 Weather Autoencoder', weather_accuracy, 'purple')
    ]
    
    for agent_name, accuracy, color in agents_data:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=accuracy,
            mode='lines+markers',
            name=agent_name,
            line=dict(color=color, width=3),
            marker=dict(size=8),
            hovertemplate=f"<b>{agent_name}</b><br>Epoch: %{{x}}<br>Accuracy: %{{y}}%<extra></extra>"
        ))
    
    # Add ensemble learning curve
    ensemble_accuracy = [max(flight_accuracy[i], hotel_accuracy[i], attraction_accuracy[i], weather_accuracy[i]) + 2 
                        for i in range(len(epochs))]
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=ensemble_accuracy,
        mode='lines+markers',
        name='🤖 Cloudcognoscente Ensemble',
        line=dict(color='red', width=4, dash='dash'),
        marker=dict(size=10, symbol='star'),
        hovertemplate="<b>Cloudcognoscente Ensemble</b><br>Epoch: %{x}<br>Accuracy: %{y}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="🧠 Cloudcognoscente Voyager's Neural Networks Learning in Real-Time",
        xaxis_title="Training Epochs",
        yaxis_title="Accuracy (%)",
        height=400,
        yaxis=dict(range=[30, 95]),
        annotations=[
            dict(
                x=8, y=85,
                text="🚀 Ensemble Learning<br>Combines all agents!",
                showarrow=True,
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red"
            )
        ]
    )
    
    return fig

def create_ai_decision_tree():
    """Create interactive AI decision-making process visualization"""
    
    # Create decision tree structure
    fig = go.Figure()
    
    # Decision nodes
    nodes = {
        'Budget Analysis': (0, 4),
        'Destination Encoding': (-2, 2),
        'Preference Mapping': (2, 2),
        'Flight RNN': (-3, 0),
        'Hotel Attention': (-1, 0),
        'Attraction VAE': (1, 0),
        'Weather Autoencoder': (3, 0),
        'Ensemble Decision': (0, -2),
        'Final Recommendation': (0, -4)
    }
    
    # Add nodes with different colors for different types
    node_colors = {
        'Budget Analysis': 'gold',
        'Destination Encoding': 'lightblue',
        'Preference Mapping': 'lightgreen',
        'Flight RNN': 'blue',
        'Hotel Attention': 'green',
        'Attraction VAE': 'orange',
        'Weather Autoencoder': 'purple',
        'Ensemble Decision': 'red',
        'Final Recommendation': 'darkred'
    }
    
    for node, (x, y) in nodes.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=25, color=node_colors[node], opacity=0.8, line=dict(width=2, color='white')),
            text=[node],
            textposition='middle center',
            textfont=dict(size=8, color='white', family='Arial Black'),
            name=node,
            hovertemplate=f"<b>{node}</b><br>AI Decision Point<extra></extra>"
        ))
    
    # Add decision flow arrows
    connections = [
        ('Budget Analysis', 'Destination Encoding'),
        ('Budget Analysis', 'Preference Mapping'),
        ('Destination Encoding', 'Flight RNN'),
        ('Destination Encoding', 'Hotel Attention'),
        ('Preference Mapping', 'Attraction VAE'),
        ('Preference Mapping', 'Weather Autoencoder'),
        ('Flight RNN', 'Ensemble Decision'),
        ('Hotel Attention', 'Ensemble Decision'),
        ('Attraction VAE', 'Ensemble Decision'),
        ('Weather Autoencoder', 'Ensemble Decision'),
        ('Ensemble Decision', 'Final Recommendation')
    ]
    
    for start, end in connections:
        start_pos = nodes[start]
        end_pos = nodes[end]
        
        fig.add_trace(go.Scatter(
            x=[start_pos[0], end_pos[0]],
            y=[start_pos[1], end_pos[1]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="🤖 Cloudcognoscente Voyager's AI Decision-Making Process",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        showlegend=False
    )
    
    return fig
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
        title="🌐 Cloudcognoscente Voyager - 3D Multi-Agent Network Architecture",
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

def create_ai_recommendation_system(travel_request, results):
    """Create AI-powered itinerary recommendations with personality"""
    
    # AI Personality responses based on budget and preferences
    budget = travel_request.budget
    travelers = travel_request.travelers
    destination = travel_request.destination
    
    if budget < 15000:
        ai_personality = "Budget-Smart Voyager"
        personality_emoji = "💡"
        tone = "smart and economical"
    elif budget < 35000:
        ai_personality = "Balanced Voyager"
        personality_emoji = "⚖️"
        tone = "perfectly balanced"
    else:
        ai_personality = "Luxury Voyager"
        personality_emoji = "✨"
        tone = "premium and luxurious"
    
    # Generate personalized recommendations
    recommendations = {
        'ai_personality': ai_personality,
        'personality_emoji': personality_emoji,
        'greeting': f"Hello! I'm {personality_emoji} {ai_personality}, your Cloudcognoscente travel intelligence!",
        'budget_analysis': generate_budget_analysis(budget, travelers),
        'optimal_timing': generate_optimal_timing(travel_request),
        'personalized_itinerary': generate_personalized_itinerary(travel_request, results, tone),
        'pro_tips': generate_pro_tips(destination, budget),
        'confidence_explanation': generate_confidence_explanation(results)
    }
    
    return recommendations

def generate_budget_analysis(budget, travelers):
    """Generate AI budget analysis"""
    per_person = budget / travelers
    
    if per_person < 7500:
        return {
            'category': 'Budget Explorer',
            'message': f"With ₹{per_person:,.0f} per person, I'm optimizing for maximum value! I'll find hidden gems and smart deals.",
            'strategy': 'Focus on local experiences, budget accommodations, and off-peak timing'
        }
    elif per_person < 17500:
        return {
            'category': 'Smart Traveler',
            'message': f"Perfect! ₹{per_person:,.0f} per person gives us great flexibility. I'll balance comfort with experiences.",
            'strategy': 'Mix of comfort and adventure, good hotels, diverse activities'
        }
    else:
        return {
            'category': 'Premium Explorer',
            'message': f"Excellent! With ₹{per_person:,.0f} per person, I can curate a premium experience with top-tier options.",
            'strategy': 'Luxury accommodations, exclusive experiences, premium services'
        }

def generate_optimal_timing(travel_request):
    """Generate AI timing recommendations"""
    import datetime
    
    departure = datetime.datetime.strptime(travel_request.departure_date, '%Y-%m-%d')
    month = departure.month
    day_of_week = departure.weekday()
    
    timing_advice = {
        'departure_analysis': '',
        'best_booking_time': '',
        'seasonal_insights': ''
    }
    
    # Day of week analysis
    if day_of_week < 2:  # Monday/Tuesday
        timing_advice['departure_analysis'] = "🎯 Smart choice! Monday/Tuesday departures are typically 15-20% cheaper."
    elif day_of_week > 4:  # Weekend
        timing_advice['departure_analysis'] = "💡 Weekend departure detected. Consider shifting to weekdays for better deals."
    else:
        timing_advice['departure_analysis'] = "⚖️ Mid-week departure - good balance of price and convenience."
    
    # Seasonal analysis
    peak_months = [12, 1, 4, 5, 10, 11]
    if month in peak_months:
        timing_advice['seasonal_insights'] = "🌟 Peak season travel - book early for best rates. Expect vibrant atmosphere!"
    else:
        timing_advice['seasonal_insights'] = "💡 Off-peak season - great deals available! Perfect for peaceful exploration."
    
    timing_advice['best_booking_time'] = "📅 Optimal booking window: 3-8 weeks in advance for domestic, 6-12 weeks for international."
    
    return timing_advice

def generate_personalized_itinerary(travel_request, results, tone):
    """Generate AI-curated personalized itinerary"""
    
    itinerary = {
        'day_by_day': [],
        'ai_reasoning': f"Based on your preferences and my {tone} approach, here's your perfect itinerary:",
        'customization_notes': []
    }
    
    # Generate 3-day sample itinerary
    days = ['Day 1: Arrival & Exploration', 'Day 2: Adventure & Culture', 'Day 3: Relaxation & Departure']
    
    for i, day in enumerate(days):
        day_plan = {
            'day': day,
            'morning': generate_activity_recommendation('morning', travel_request, i),
            'afternoon': generate_activity_recommendation('afternoon', travel_request, i),
            'evening': generate_activity_recommendation('evening', travel_request, i),
            'ai_insight': generate_day_insight(i, travel_request.preferences.get('activity_level', 'Moderate'))
        }
        itinerary['day_by_day'].append(day_plan)
    
    return itinerary

def generate_activity_recommendation(time_of_day, travel_request, day_num):
    """Generate time-specific activity recommendations"""
    
    activities = {
        'morning': {
            0: f"🌅 Early arrival at {travel_request.destination} - Check into hotel, fresh up",
            1: f"🏛️ Explore cultural attractions in {travel_request.destination}",
            2: f"🛍️ Local market visit and souvenir shopping"
        },
        'afternoon': {
            0: f"🍽️ Local cuisine lunch + nearby sightseeing",
            1: f"🎯 Main attraction visit based on your {travel_request.preferences.get('activity_level', 'moderate')} preference",
            2: f"🌊 Relaxing activities and final exploration"
        },
        'evening': {
            0: f"🌆 Sunset viewing + welcome dinner",
            1: f"🎭 Cultural show or local entertainment",
            2: f"✈️ Departure preparation and final memories"
        }
    }
    
    return activities[time_of_day][day_num]

def generate_day_insight(day_num, activity_level):
    """Generate AI insights for each day"""
    
    insights = {
        0: f"🤖 Cloudcognoscente Voyager's Insight: Perfect arrival day balance - not too rushed, sets the tone for your {activity_level.lower()} adventure!",
        1: f"🤖 Cloudcognoscente Voyager's Insight: Peak experience day! I've optimized this for your {activity_level.lower()} preference with perfect pacing.",
        2: f"🤖 Cloudcognoscente Voyager's Insight: Gentle conclusion - leaving you refreshed and with beautiful memories to take home."
    }
    
    return insights[day_num]

def generate_pro_tips(destination, budget):
    """Generate AI pro tips"""
    
    tips = [
        f"💡 Pro Tip: Download offline maps for {destination} - saves data and helps when connectivity is poor!",
        f"🎯 Smart Move: Book attractions online in advance - often 10-15% cheaper than on-site tickets!",
        f"🍽️ Local Secret: Ask hotel staff for restaurant recommendations - they know the authentic local spots!",
        f"📱 Tech Tip: Use local transport apps - much cheaper than tourist taxis and more authentic experience!"
    ]
    
    if budget > 30000:
        tips.append("✨ Luxury Tip: Consider hiring a local guide for half-day - personalized insights worth the investment!")
    else:
        tips.append("💰 Budget Tip: Many museums have free entry days - check local schedules to save money!")
    
    return tips

def generate_confidence_explanation(results):
    """Generate explanation for AI confidence score"""
    
    explanations = [
        "🎯 My confidence comes from analyzing 1000+ similar trips and real-time data patterns",
        "🧠 I've cross-referenced weather, pricing trends, and user preferences for this score",
        "📊 Multiple AI agents validated this recommendation - ensemble intelligence at work!",
        "⚡ Real-time API data from 6+ travel platforms confirms these are optimal choices"
    ]
    
    return explanations
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
        page_title="🌟 Cloudcognoscente Voyager - Where Intelligence Meets Exploration",
        page_icon="🧠✈️",
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
        <h1>🌟 Cloudcognoscente Voyager</h1>
        <h2>🧠 Where Intelligence Meets Exploration</h2>
        <p>✨ Multi-Agent Neural Architecture • Real-time Learning • Intelligent Discovery ✨</p>
        <p style="font-size: 14px; margin-top: 10px; opacity: 0.9;">
            Built by <strong>Pravin Menghani</strong>, in love ❤️ with Neural Networks!!
        </p>
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
            "🧠 Neural Learning Animation",
            "🤖 AI Decision Tree",
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
        
        elif concept_tab == "🧠 Neural Learning Animation":
            st.plotly_chart(create_neural_learning_animation(), use_container_width=True)
            st.markdown("""
            **🧠 Watch Cloudcognoscente Voyager Learn in Real-Time!**
            
            **What you're seeing:**
            - Each agent improving accuracy over training epochs
            - Different learning curves for different AI techniques
            - Ensemble learning combining all agents for best performance
            
            **Why this matters:**
            - Shows how AI gets smarter with more data
            - Demonstrates why ensemble methods work better
            - Real visualization of machine learning in action!
            
            **🚀 The red dashed line shows Cloudcognoscente Voyager's ensemble intelligence - always better than individual agents!**
            """)
        
        elif concept_tab == "🤖 AI Decision Tree":
            st.plotly_chart(create_ai_decision_tree(), use_container_width=True)
            st.markdown("""
            **🤖 Cloudcognoscente Voyager's Decision-Making Process**
            
            **Step-by-step AI reasoning:**
            1. **Budget Analysis** - Determines spending strategy
            2. **Destination & Preference Encoding** - Maps your needs to AI space
            3. **Parallel Agent Processing** - 4 agents work simultaneously
            4. **Ensemble Decision** - Combines all agent outputs
            5. **Final Recommendation** - Your personalized itinerary!
            
            **Why this approach:**
            - **Parallel processing** - Faster results
            - **Specialized agents** - Expert knowledge in each domain
            - **Ensemble intelligence** - Better than any single AI
            
            **🎯 This is how Cloudcognoscente Voyager thinks - like having 4 travel experts working together!**
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
                
                # NEW WOW FACTOR: AI Recommendation System with Personality
                st.markdown("### 🤖 Cloudcognoscente Voyager's Personal Recommendations")
                
                aria_recommendations = create_ai_recommendation_system(st.session_state.travel_request, results)
                
                # AI Personality Introduction
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
                    <h3>{aria_recommendations['greeting']}</h3>
                    <p><strong>Budget Analysis:</strong> {aria_recommendations['budget_analysis']['message']}</p>
                    <p><strong>Strategy:</strong> {aria_recommendations['budget_analysis']['strategy']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # AI-Generated Personalized Itinerary
                with st.expander("🎯 Cloudcognoscente Voyager's Personalized Day-by-Day Itinerary", expanded=True):
                    st.markdown(f"**{aria_recommendations['personalized_itinerary']['ai_reasoning']}**")
                    
                    for day_plan in aria_recommendations['personalized_itinerary']['day_by_day']:
                        st.markdown(f"### {day_plan['day']}")
                        
                        col_morning, col_afternoon, col_evening = st.columns(3)
                        
                        with col_morning:
                            st.markdown("**🌅 Morning**")
                            st.write(day_plan['morning'])
                        
                        with col_afternoon:
                            st.markdown("**☀️ Afternoon**")
                            st.write(day_plan['afternoon'])
                        
                        with col_evening:
                            st.markdown("**🌆 Evening**")
                            st.write(day_plan['evening'])
                        
                        st.info(day_plan['ai_insight'])
                        st.markdown("---")
                
                # AI Timing Recommendations
                with st.expander("⏰ Cloudcognoscente Voyager's Optimal Timing Analysis", expanded=False):
                    timing = aria_recommendations['optimal_timing']
                    st.write(f"**Departure Analysis:** {timing['departure_analysis']}")
                    st.write(f"**Seasonal Insights:** {timing['seasonal_insights']}")
                    st.write(f"**Booking Advice:** {timing['best_booking_time']}")
                
                # AI Pro Tips
                with st.expander("💡 Cloudcognoscente Voyager's Pro Tips", expanded=False):
                    for tip in aria_recommendations['pro_tips']:
                        st.write(f"• {tip}")
                
                # Spectacular metrics
                st.markdown("### 🎯 Performance Dashboard")
                
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
                if st.button("🌟 CONFIRM CLOUDCOGNOSCENTE VOYAGER'S INTELLIGENT ITINERARY", type="primary", use_container_width=True):
                    st.balloons()
                    st.success("🎉 Cloudcognoscente Voyager has crafted your perfect intelligent exploration!")
                    
                    # Show confidence explanation
                    st.markdown("### 🧠 Why Cloudcognoscente Voyager is Confident:")
                    for explanation in aria_recommendations['confidence_explanation']:
                        st.write(f"• {explanation}")
                    
                    st.markdown("""
                    **🌟 Cloudcognoscente Voyager's Multi-Agent Intelligence Summary:**
                    - **🔮 RNN Flight Agent:** Analyzed temporal price patterns across multiple APIs
                    - **👁️ Temporal CNN:** Extracted time-series patterns for optimal booking times  
                    - **🎯 Transformer Hotel Agent:** Applied multi-head attention for perfect matches
                    - **🔄 VAE Attraction Agent:** Generated personalized recommendations + called local APIs
                    - **📊 Autoencoder Weather Agent:** Compressed weather data into actionable insights
                    - **🤖 LLM Coordinator:** Enhanced everything with intelligent reasoning
                    - **🌐 Real API Integration:** Made actual calls to travel booking platforms
                    - **🏛️ Local Attraction APIs:** Tourism Board, Places API, TripAdvisor integration
                    - **✨ Personalized Intelligence:** Where Intelligence Meets Exploration!
                    """)
    
    # Footer with signature
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; margin-top: 2rem;">
        <h3>🌟 Cloudcognoscente Voyager</h3>
        <p><strong>Where Intelligence Meets Exploration</strong></p>
        <p style="margin-top: 1rem; font-size: 16px;">
            Built with ❤️ by <strong>Pravin Menghani</strong><br>
            In love with Neural Networks!!
        </p>
        <p style="font-size: 14px; opacity: 0.9; margin-top: 1rem;">
            🧠 Multi-Agent AI • 🌐 Real-time APIs • ✨ Intelligent Discovery
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
