#!/usr/bin/env python3
"""
Medical AI Visualizations for Smart Health Diagnosis AI
Amazing visualizations that make medical AI concepts crystal clear!
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_3d_medical_network():
    """Create 3D visualization of medical AI network"""
    
    # Medical AI positions in 3D space
    medical_ais = {
        'FNN': (0, 0, 3),
        'RNN': (-3, -2, 1),
        'CNN': (3, -2, 1),
        'Decision Tree': (-3, 2, 1),
        'Ensemble': (0, 0, -1)
    }
    
    # Color coding by medical specialty
    ai_colors = {
        'FNN': 'blue',
        'RNN': 'green', 
        'CNN': 'red',
        'Decision Tree': 'orange',
        'Ensemble': 'gold'
    }
    
    fig = go.Figure()
    
    # Add AI nodes
    for ai, (x, y, z) in medical_ais.items():
        size = 25 if ai == 'Ensemble' else 18
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(
                size=size,
                color=ai_colors[ai],
                opacity=0.8,
                line=dict(width=3, color='white')
            ),
            text=[ai],
            textposition='top center',
            textfont=dict(size=10, color='black', family='Arial Black'),
            name=ai,
            hovertemplate=f"<b>{ai}</b><br>Medical AI Specialist<br>Position: ({x}, {y}, {z})<extra></extra>"
        ))
    
    # Add connections to ensemble
    ensemble_pos = medical_ais['Ensemble']
    for ai, pos in medical_ais.items():
        if ai != 'Ensemble':
            fig.add_trace(go.Scatter3d(
                x=[ensemble_pos[0], pos[0]],
                y=[ensemble_pos[1], pos[1]],
                z=[ensemble_pos[2], pos[2]],
                mode='lines',
                line=dict(color=ai_colors[ai], width=4),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title="🌐 Medical AI Network - Smart Health Diagnosis System",
        scene=dict(
            xaxis=dict(title="Symptom Complexity", showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="Medical History Depth", showgrid=True, gridcolor='lightgray'),
            zaxis=dict(title="Diagnostic Accuracy", showgrid=True, gridcolor='lightgray'),
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

def create_fnn_symptom_analysis():
    """Create FNN symptom analysis visualization"""
    
    # Simulate FNN layers processing symptoms
    layers = ['Input\n(Symptoms)', 'Hidden Layer 1\n(Pattern Detection)', 'Hidden Layer 2\n(Disease Mapping)', 'Output\n(Diagnosis)']
    
    # Symptom processing through layers
    symptoms = ['Fever', 'Cough', 'Fatigue', 'Headache']
    layer_activations = []
    
    for i, layer in enumerate(layers):
        if i == 0:  # Input layer
            activations = [0.8, 0.6, 0.9, 0.4]  # Symptom presence
        elif i == 1:  # Hidden layer 1
            activations = [0.7, 0.8, 0.5, 0.9, 0.6]  # Pattern detection
        elif i == 2:  # Hidden layer 2
            activations = [0.6, 0.8, 0.7]  # Disease mapping
        else:  # Output layer
            activations = [0.85]  # Final diagnosis confidence
        
        layer_activations.append(activations)
    
    fig = go.Figure()
    
    # Create network visualization
    x_positions = [0, 1, 2, 3]
    
    for i, (layer, activations) in enumerate(zip(layers, layer_activations)):
        y_positions = np.linspace(-len(activations)/2, len(activations)/2, len(activations))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=[x_positions[i]] * len(activations),
            y=y_positions,
            mode='markers+text',
            marker=dict(
                size=[a * 30 + 10 for a in activations],
                color=[a for a in activations],
                colorscale='Viridis',
                showscale=True if i == 0 else False,
                colorbar=dict(title="Activation Level")
            ),
            text=[f"{a:.2f}" for a in activations],
            textposition='middle center',
            name=layer,
            hovertemplate=f"<b>{layer}</b><br>Activation: %{{text}}<extra></extra>"
        ))
    
    # Add connections between layers
    for i in range(len(layers) - 1):
        for j in range(len(layer_activations[i])):
            for k in range(len(layer_activations[i + 1])):
                y1 = np.linspace(-len(layer_activations[i])/2, len(layer_activations[i])/2, len(layer_activations[i]))[j]
                y2 = np.linspace(-len(layer_activations[i + 1])/2, len(layer_activations[i + 1])/2, len(layer_activations[i + 1]))[k]
                
                fig.add_trace(go.Scatter(
                    x=[x_positions[i], x_positions[i + 1]],
                    y=[y1, y2],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        title="🧠 FNN: Feedforward Neural Network - Symptom Analysis",
        xaxis=dict(tickvals=x_positions, ticktext=layers, title="Network Layers"),
        yaxis=dict(title="Neurons", showticklabels=False),
        height=400,
        showlegend=False
    )
    
    return fig

def create_rnn_history_analysis():
    """Create RNN medical history analysis visualization"""
    
    # Patient visits over time
    visits = list(range(1, 11))
    
    # Symptoms progression
    fever_history = [0, 0.2, 0.5, 0.8, 0.9, 0.7, 0.4, 0.2, 0, 0]
    cough_history = [0, 0, 0.3, 0.6, 0.8, 0.9, 0.8, 0.6, 0.3, 0.1]
    fatigue_history = [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1]
    
    # RNN memory state
    memory_state = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.8, 0.6, 0.4, 0.3]
    
    fig = go.Figure()
    
    # Add symptom histories
    fig.add_trace(go.Scatter(
        x=visits, y=fever_history,
        mode='lines+markers',
        name='🌡️ Fever History',
        line=dict(color='red', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=visits, y=cough_history,
        mode='lines+markers',
        name='😷 Cough History',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=visits, y=fatigue_history,
        mode='lines+markers',
        name='😴 Fatigue History',
        line=dict(color='green', width=3)
    ))
    
    # Add RNN memory state
    fig.add_trace(go.Scatter(
        x=visits, y=memory_state,
        mode='lines+markers',
        name='🧠 RNN Memory State',
        line=dict(color='purple', width=4, dash='dash'),
        marker=dict(size=10, symbol='diamond')
    ))
    
    fig.update_layout(
        title="🔄 RNN: Medical History Pattern Analysis",
        xaxis_title="Patient Visits Over Time",
        yaxis_title="Symptom Severity / Memory Activation",
        height=400,
        annotations=[
            dict(
                x=6, y=0.9,
                text="🧠 RNN remembers<br>symptom patterns!",
                showarrow=True,
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="purple"
            )
        ]
    )
    
    return fig

def create_cnn_imaging_analysis():
    """Create CNN medical imaging analysis visualization"""
    
    # Simulate medical image analysis
    image_features = ['Edges', 'Textures', 'Shapes', 'Abnormalities']
    detection_layers = ['Conv Layer 1', 'Conv Layer 2', 'Conv Layer 3', 'Classification']
    
    # Feature detection strength across layers
    detection_strength = np.random.rand(len(detection_layers), len(image_features))
    detection_strength = detection_strength / detection_strength.max() * 100
    
    fig = px.imshow(
        detection_strength,
        x=image_features,
        y=detection_layers,
        color_continuous_scale='Viridis',
        title="👁️ CNN: Medical Image Feature Detection",
        labels=dict(color="Detection Strength (%)")
    )
    
    fig.update_layout(height=400)
    return fig

def create_decision_tree_diagnosis():
    """Create decision tree medical diagnosis visualization"""
    
    # Decision tree nodes
    fig = go.Figure()
    
    # Tree structure
    nodes = {
        'Root': (0, 0, 'Fever > 38°C?'),
        'Yes_Fever': (-2, -1, 'Cough present?'),
        'No_Fever': (2, -1, 'Fatigue severe?'),
        'Yes_Cough': (-3, -2, 'Flu'),
        'No_Cough': (-1, -2, 'Infection'),
        'Yes_Fatigue': (1, -2, 'Chronic condition'),
        'No_Fatigue': (3, -2, 'Minor illness')
    }
    
    # Add nodes
    for node, (x, y, text) in nodes.items():
        color = 'lightblue' if 'Yes' in node or 'No' in node else 'lightgreen' if node == 'Root' else 'lightcoral'
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=40, color=color, line=dict(width=2, color='black')),
            text=[text],
            textposition='middle center',
            textfont=dict(size=10),
            name=node,
            showlegend=False,
            hovertemplate=f"<b>{text}</b><extra></extra>"
        ))
    
    # Add connections
    connections = [
        ('Root', 'Yes_Fever'),
        ('Root', 'No_Fever'),
        ('Yes_Fever', 'Yes_Cough'),
        ('Yes_Fever', 'No_Cough'),
        ('No_Fever', 'Yes_Fatigue'),
        ('No_Fever', 'No_Fatigue')
    ]
    
    for start, end in connections:
        x1, y1, _ = nodes[start]
        x2, y2, _ = nodes[end]
        
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="🎯 Decision Tree: Rule-based Medical Diagnosis",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )
    
    return fig

def create_ai_doctor_consultation():
    """Create AI doctor consultation visualization"""
    
    visits = list(range(1, 21))
    
    # Different AI doctors' diagnostic accuracy over time
    fnn_accuracy = [60 + 1.5*v + np.random.normal(0, 3) for v in visits]
    rnn_accuracy = [55 + 2*v + np.random.normal(0, 3) for v in visits]
    cnn_accuracy = [50 + 2.5*v + np.random.normal(0, 3) for v in visits]
    decision_tree_accuracy = [70 + 1*v + np.random.normal(0, 2) for v in visits]
    
    fig = go.Figure()
    
    doctors = [
        ('🧠 FNN Doctor', fnn_accuracy, 'blue'),
        ('🔄 RNN Doctor', rnn_accuracy, 'green'),
        ('👁️ CNN Doctor', cnn_accuracy, 'red'),
        ('🎯 Decision Tree Doctor', decision_tree_accuracy, 'orange')
    ]
    
    for name, accuracy, color in doctors:
        fig.add_trace(go.Scatter(
            x=visits, y=accuracy,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=3),
            marker=dict(size=8)
        ))
    
    # Add ensemble (best combination)
    ensemble_accuracy = [max(fnn_accuracy[i], rnn_accuracy[i], cnn_accuracy[i], decision_tree_accuracy[i]) + 5 for i in range(len(visits))]
    fig.add_trace(go.Scatter(
        x=visits, y=ensemble_accuracy,
        mode='lines+markers',
        name='🏥 Medical Team Consensus',
        line=dict(color='gold', width=4),
        marker=dict(size=10, symbol='star')
    ))
    
    fig.update_layout(
        title="🤖 AI Medical Team - Diagnostic Accuracy Comparison",
        xaxis_title="Patient Cases",
        yaxis_title="Diagnostic Accuracy (%)",
        height=400,
        yaxis=dict(range=[40, 110])
    )
    
    return fig

def create_realtime_diagnosis_engine():
    """Create real-time diagnosis engine visualization"""
    
    # Simulate real-time medical analysis
    time_points = pd.date_range(start='2024-01-01', periods=50, freq='H')
    
    # Multiple AI diagnoses updating in real-time
    diagnoses = []
    for i, time_point in enumerate(time_points):
        base_confidence = 70 + 10 * np.sin(i * 0.1)
        fnn_conf = base_confidence + np.random.normal(0, 5)
        rnn_conf = base_confidence + np.random.normal(0, 4)
        cnn_conf = base_confidence + np.random.normal(0, 6)
        
        diagnoses.append({
            'time': time_point,
            'FNN': max(0, min(100, fnn_conf)),
            'RNN': max(0, min(100, rnn_conf)),
            'CNN': max(0, min(100, cnn_conf)),
            'Ensemble': max(0, min(100, (fnn_conf + rnn_conf + cnn_conf) / 3))
        })
    
    df = pd.DataFrame(diagnoses)
    
    fig = go.Figure()
    
    for column in ['FNN', 'RNN', 'CNN', 'Ensemble']:
        color = {'FNN': 'blue', 'RNN': 'green', 'CNN': 'red', 'Ensemble': 'gold'}[column]
        width = 4 if column == 'Ensemble' else 2
        
        fig.add_trace(go.Scatter(
            x=df['time'], y=df[column],
            mode='lines',
            name=f'🤖 {column}',
            line=dict(color=color, width=width)
        ))
    
    fig.update_layout(
        title="📊 Real-time Medical Diagnosis Engine",
        xaxis_title="Time",
        yaxis_title="Diagnostic Confidence (%)",
        height=400
    )
    
    return fig
