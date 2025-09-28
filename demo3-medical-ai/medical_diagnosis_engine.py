#!/usr/bin/env python3
"""
Medical Diagnosis Engine for Smart Health Diagnosis AI
Generates realistic medical diagnoses using different AI architectures
"""

import numpy as np
import pandas as pd
from datetime import datetime
import asyncio
import random
import plotly.graph_objects as go

async def generate_medical_diagnoses(request):
    """Generate medical diagnoses from multiple AI doctors"""
    
    symptoms = request['symptoms']
    age = request['age']
    gender = request['gender']
    severity = request['severity']
    history = request['history']
    duration = request['duration']
    ai_doctors = request['ai_doctors']
    
    diagnoses = {}
    
    if ai_doctors['fnn']:
        diagnoses['fnn'] = await generate_fnn_diagnosis(symptoms, age, severity)
    
    if ai_doctors['rnn']:
        diagnoses['rnn'] = await generate_rnn_diagnosis(symptoms, history, duration)
    
    if ai_doctors['cnn']:
        diagnoses['cnn'] = await generate_cnn_diagnosis(symptoms, request.get('image'))
    
    if ai_doctors['decision_tree']:
        diagnoses['decision_tree'] = await generate_decision_tree_diagnosis(symptoms, age, severity)
    
    return diagnoses

async def generate_fnn_diagnosis(symptoms, age, severity):
    """Generate FNN (Feedforward Neural Network) diagnosis"""
    
    await asyncio.sleep(0.5)  # Simulate processing time
    
    # FNN analyzes symptoms in a straightforward manner
    symptom_weights = {
        'Fever': 0.8,
        'Cough': 0.7,
        'Fatigue': 0.6,
        'Headache': 0.5,
        'Nausea': 0.6,
        'Chest Pain': 0.9,
        'Shortness of Breath': 0.9,
        'Dizziness': 0.4,
        'Joint Pain': 0.5,
        'Skin Rash': 0.6,
        'Abdominal Pain': 0.7,
        'Back Pain': 0.4,
        'Sore Throat': 0.6,
        'Runny Nose': 0.3
    }
    
    # Calculate symptom score
    total_score = sum(symptom_weights.get(symptom, 0.5) for symptom in symptoms)
    symptom_match = min(95, max(60, total_score * 10 + np.random.normal(0, 5)))
    
    # Age factor
    age_factor = 1.0
    if age > 60:
        age_factor = 1.2
    elif age < 18:
        age_factor = 0.8
    
    # Severity factor
    severity_multiplier = {'Mild': 0.8, 'Moderate': 1.0, 'Severe': 1.3}[severity]
    
    # FNN diagnosis based on symptom patterns
    if 'Fever' in symptoms and 'Cough' in symptoms:
        if 'Fatigue' in symptoms:
            diagnosis = 'Viral Infection (Flu-like)'
        else:
            diagnosis = 'Upper Respiratory Infection'
    elif 'Chest Pain' in symptoms or 'Shortness of Breath' in symptoms:
        diagnosis = 'Respiratory Condition'
    elif 'Abdominal Pain' in symptoms and 'Nausea' in symptoms:
        diagnosis = 'Gastrointestinal Issue'
    elif 'Headache' in symptoms and 'Fatigue' in symptoms:
        diagnosis = 'Stress-related Condition'
    elif 'Joint Pain' in symptoms:
        diagnosis = 'Musculoskeletal Condition'
    else:
        diagnosis = 'General Malaise'
    
    # Adjust confidence based on factors
    final_confidence = symptom_match * age_factor * severity_multiplier
    final_confidence = min(95, max(65, final_confidence))
    
    return {
        'diagnosis': diagnosis,
        'symptom_match': symptom_match,
        'confidence': final_confidence,
        'key_symptoms': symptoms[:3],  # Top 3 symptoms
        'analysis_method': 'Feedforward pattern matching'
    }

async def generate_rnn_diagnosis(symptoms, history, duration):
    """Generate RNN (Recurrent Neural Network) diagnosis considering history"""
    
    await asyncio.sleep(0.6)  # Simulate processing time
    
    # RNN considers medical history and temporal patterns
    history_risk_factors = {
        'Diabetes': ['Fatigue', 'Dizziness'],
        'Hypertension': ['Headache', 'Dizziness'],
        'Heart Disease': ['Chest Pain', 'Shortness of Breath'],
        'Asthma': ['Cough', 'Shortness of Breath'],
        'Allergies': ['Runny Nose', 'Skin Rash'],
        'Cancer': ['Fatigue', 'Nausea'],
        'Kidney Disease': ['Fatigue', 'Back Pain'],
        'Liver Disease': ['Fatigue', 'Abdominal Pain']
    }
    
    # Calculate history relevance
    history_score = 0
    relevant_conditions = []
    
    for condition in history:
        if condition in history_risk_factors:
            risk_symptoms = history_risk_factors[condition]
            matches = len(set(symptoms) & set(risk_symptoms))
            if matches > 0:
                history_score += matches * 20
                relevant_conditions.append(condition)
    
    history_relevance = min(95, max(40, history_score + np.random.normal(0, 10)))
    
    # Duration analysis
    duration_factor = {
        'Less than 1 day': 'Acute onset',
        '1-3 days': 'Acute condition',
        '4-7 days': 'Subacute condition',
        '1-2 weeks': 'Prolonged condition',
        'More than 2 weeks': 'Chronic condition'
    }[duration]
    
    # RNN diagnosis based on history patterns
    if relevant_conditions:
        if 'Diabetes' in relevant_conditions:
            diagnosis = 'Diabetic Complication'
        elif 'Heart Disease' in relevant_conditions:
            diagnosis = 'Cardiac-related Symptoms'
        elif 'Asthma' in relevant_conditions:
            diagnosis = 'Asthma Exacerbation'
        else:
            diagnosis = f'{relevant_conditions[0]} Complication'
    else:
        # No relevant history
        if 'More than 2 weeks' in duration:
            diagnosis = 'Chronic Condition (New)'
        else:
            diagnosis = 'Acute Medical Condition'
    
    return {
        'diagnosis': diagnosis,
        'history_relevance': history_relevance,
        'confidence': min(90, max(70, history_relevance)),
        'risk_factors': relevant_conditions if relevant_conditions else ['No significant history'],
        'temporal_pattern': duration_factor,
        'analysis_method': 'Sequential pattern analysis with memory'
    }

async def generate_cnn_diagnosis(symptoms, image):
    """Generate CNN diagnosis based on visual analysis"""
    
    await asyncio.sleep(0.7)  # Simulate processing time
    
    # CNN analyzes visual symptoms and uploaded images
    visual_symptoms = {
        'Skin Rash': 'Dermatological condition',
        'Joint Pain': 'Musculoskeletal swelling',
        'Chest Pain': 'Chest imaging analysis',
        'Abdominal Pain': 'Abdominal imaging analysis',
        'Back Pain': 'Spinal imaging analysis'
    }
    
    # Check for visual symptoms
    visual_indicators = []
    for symptom in symptoms:
        if symptom in visual_symptoms:
            visual_indicators.append(visual_symptoms[symptom])
    
    # Image analysis simulation
    if image is not None:
        image_analysis = "Medical image uploaded - AI analyzing visual patterns"
        # Simulate image processing
        image_confidence = np.random.uniform(75, 92)
    else:
        image_analysis = "No medical image provided - analyzing symptom descriptions"
        image_confidence = np.random.uniform(60, 80)
    
    # CNN diagnosis based on visual patterns
    if visual_indicators:
        if 'Dermatological' in visual_indicators[0]:
            diagnosis = 'Skin Condition'
        elif 'Musculoskeletal' in visual_indicators[0]:
            diagnosis = 'Joint/Muscle Disorder'
        elif 'Chest' in visual_indicators[0]:
            diagnosis = 'Chest Abnormality'
        elif 'Abdominal' in visual_indicators[0]:
            diagnosis = 'Abdominal Condition'
        else:
            diagnosis = 'Structural Abnormality'
    else:
        diagnosis = 'No Visual Abnormalities Detected'
    
    return {
        'diagnosis': diagnosis,
        'image_analysis': image_analysis,
        'confidence': image_confidence,
        'visual_indicators': visual_indicators if visual_indicators else ['No visual symptoms'],
        'pattern_recognition': 'Convolutional feature extraction',
        'analysis_method': 'Visual pattern recognition'
    }

async def generate_decision_tree_diagnosis(symptoms, age, severity):
    """Generate Decision Tree diagnosis using medical rules"""
    
    await asyncio.sleep(0.4)  # Simulate processing time
    
    # Decision tree follows logical medical rules
    decision_path = []
    
    # Rule 1: Check for emergency symptoms
    emergency_symptoms = ['Chest Pain', 'Shortness of Breath']
    if any(symptom in symptoms for symptom in emergency_symptoms):
        decision_path.append("Emergency symptoms detected")
        if age > 50:
            decision_path.append("Age > 50: High risk")
            diagnosis = 'Potential Cardiac Emergency'
            rule_confidence = 90
        else:
            decision_path.append("Age ≤ 50: Moderate risk")
            diagnosis = 'Respiratory/Cardiac Evaluation Needed'
            rule_confidence = 80
    
    # Rule 2: Check for infection symptoms
    elif 'Fever' in symptoms:
        decision_path.append("Fever present")
        if 'Cough' in symptoms or 'Sore Throat' in symptoms:
            decision_path.append("Respiratory symptoms present")
            diagnosis = 'Upper Respiratory Infection'
            rule_confidence = 85
        elif 'Abdominal Pain' in symptoms:
            decision_path.append("Abdominal symptoms present")
            diagnosis = 'Gastrointestinal Infection'
            rule_confidence = 80
        else:
            decision_path.append("Systemic symptoms only")
            diagnosis = 'Viral Syndrome'
            rule_confidence = 75
    
    # Rule 3: Check for pain-related conditions
    elif any(pain in symptoms for pain in ['Headache', 'Joint Pain', 'Back Pain', 'Abdominal Pain']):
        decision_path.append("Pain symptoms detected")
        if severity == 'Severe':
            decision_path.append("Severe pain: Urgent evaluation")
            diagnosis = 'Acute Pain Syndrome'
            rule_confidence = 85
        else:
            decision_path.append("Mild-moderate pain: Conservative management")
            diagnosis = 'Chronic Pain Condition'
            rule_confidence = 70
    
    # Rule 4: Default case
    else:
        decision_path.append("Non-specific symptoms")
        diagnosis = 'General Medical Evaluation Needed'
        rule_confidence = 65
    
    return {
        'diagnosis': diagnosis,
        'rule_confidence': rule_confidence,
        'confidence': rule_confidence,
        'decision_path': ' → '.join(decision_path),
        'medical_rules_applied': len(decision_path),
        'analysis_method': 'Evidence-based decision rules'
    }

def combine_medical_diagnoses(diagnoses):
    """Combine all AI medical diagnoses into ensemble result"""
    
    if not diagnoses:
        return {
            'primary_diagnosis': 'Insufficient Data',
            'confidence': 50.0,
            'severity': 'Unknown',
            'recommendation': 'Consult healthcare provider',
            'consensus': 'No AI analysis available'
        }
    
    # Extract diagnoses and confidences
    ai_diagnoses = []
    confidences = []
    
    for ai_type, diagnosis in diagnoses.items():
        ai_diagnoses.append(diagnosis['diagnosis'])
        confidences.append(diagnosis['confidence'])
    
    # Find most common diagnosis or create ensemble diagnosis
    if len(set(ai_diagnoses)) == 1:
        # All AIs agree
        primary_diagnosis = ai_diagnoses[0]
        consensus = "Strong consensus among all AI doctors"
    else:
        # Different opinions - create ensemble diagnosis
        if any('Infection' in diag for diag in ai_diagnoses):
            primary_diagnosis = 'Infectious Process'
        elif any('Cardiac' in diag or 'Heart' in diag for diag in ai_diagnoses):
            primary_diagnosis = 'Cardiovascular Concern'
        elif any('Respiratory' in diag for diag in ai_diagnoses):
            primary_diagnosis = 'Respiratory Condition'
        elif any('Pain' in diag for diag in ai_diagnoses):
            primary_diagnosis = 'Pain-related Condition'
        else:
            primary_diagnosis = 'Multi-system Condition'
        
        consensus = f"Mixed opinions - {len(set(ai_diagnoses))} different diagnoses considered"
    
    # Calculate ensemble confidence
    ensemble_confidence = np.mean(confidences)
    
    # Determine severity
    if ensemble_confidence > 85:
        severity = 'High'
        recommendation = 'Seek immediate medical attention'
    elif ensemble_confidence > 70:
        severity = 'Moderate'
        recommendation = 'Schedule appointment with healthcare provider'
    else:
        severity = 'Low'
        recommendation = 'Monitor symptoms and consult if worsening'
    
    return {
        'primary_diagnosis': primary_diagnosis,
        'confidence': round(ensemble_confidence, 1),
        'severity': severity,
        'recommendation': recommendation,
        'consensus': consensus,
        'ai_opinions': len(diagnoses),
        'agreement_level': 'High' if len(set(ai_diagnoses)) <= 2 else 'Moderate' if len(set(ai_diagnoses)) <= 3 else 'Low'
    }

def create_fnn_diagnosis_chart(fnn_diagnosis):
    """Create FNN diagnosis visualization"""
    
    # Symptom analysis breakdown
    symptoms = fnn_diagnosis['key_symptoms']
    symptom_weights = [0.8, 0.6, 0.4]  # Decreasing importance
    
    fig = go.Figure(data=[
        go.Bar(
            x=symptoms,
            y=symptom_weights,
            marker_color=['red', 'orange', 'yellow'],
            text=[f"{w:.1f}" for w in symptom_weights],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"🧠 FNN Analysis: {fnn_diagnosis['diagnosis']}",
        xaxis_title="Key Symptoms",
        yaxis_title="Symptom Weight",
        height=300
    )
    
    return fig

def create_rnn_diagnosis_chart(rnn_diagnosis):
    """Create RNN diagnosis visualization"""
    
    # History relevance over time
    time_points = ['Past', 'Recent', 'Current']
    relevance_scores = [
        rnn_diagnosis['history_relevance'] * 0.6,
        rnn_diagnosis['history_relevance'] * 0.8,
        rnn_diagnosis['history_relevance']
    ]
    
    fig = go.Figure(data=[
        go.Scatter(
            x=time_points,
            y=relevance_scores,
            mode='lines+markers',
            line=dict(color='green', width=4),
            marker=dict(size=12)
        )
    ])
    
    fig.update_layout(
        title=f"🔄 RNN Analysis: {rnn_diagnosis['diagnosis']}",
        xaxis_title="Time Period",
        yaxis_title="Medical History Relevance (%)",
        height=300
    )
    
    return fig

def create_cnn_diagnosis_chart(cnn_diagnosis):
    """Create CNN diagnosis visualization"""
    
    # Visual analysis components
    components = ['Pattern Recognition', 'Feature Extraction', 'Abnormality Detection']
    scores = [
        cnn_diagnosis['confidence'] * 0.9,
        cnn_diagnosis['confidence'] * 0.8,
        cnn_diagnosis['confidence'] * 0.7
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=components,
            y=scores,
            marker_color=['blue', 'cyan', 'lightblue'],
            text=[f"{s:.1f}%" for s in scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"👁️ CNN Analysis: {cnn_diagnosis['diagnosis']}",
        xaxis_title="Visual Analysis Components",
        yaxis_title="Analysis Score (%)",
        height=300
    )
    
    return fig

def create_decision_tree_chart(dt_diagnosis):
    """Create Decision Tree diagnosis visualization"""
    
    # Decision path visualization
    steps = dt_diagnosis['decision_path'].split(' → ')
    step_confidence = [dt_diagnosis['rule_confidence'] * (1 - i*0.1) for i in range(len(steps))]
    
    fig = go.Figure(data=[
        go.Scatter(
            x=list(range(len(steps))),
            y=step_confidence,
            mode='lines+markers',
            line=dict(color='orange', width=4),
            marker=dict(size=15, symbol='diamond'),
            text=steps,
            textposition='top center'
        )
    ])
    
    fig.update_layout(
        title=f"🎯 Decision Tree: {dt_diagnosis['diagnosis']}",
        xaxis_title="Decision Steps",
        yaxis_title="Rule Confidence (%)",
        height=300,
        xaxis=dict(tickvals=list(range(len(steps))), ticktext=[f"Step {i+1}" for i in range(len(steps))])
    )
    
    return fig

def create_ensemble_diagnosis_chart(ensemble):
    """Create ensemble diagnosis visualization"""
    
    # AI team consensus
    metrics = ['Diagnostic Confidence', 'Severity Assessment', 'Recommendation Strength']
    scores = [
        ensemble['confidence'],
        {'Low': 60, 'Moderate': 80, 'High': 95}[ensemble['severity']],
        85  # Recommendation strength
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=scores,
            marker_color=['gold', 'orange', 'red'],
            text=[f"{s:.1f}%" for s in scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"🏥 Medical Team Consensus: {ensemble['primary_diagnosis']}",
        xaxis_title="Assessment Metrics",
        yaxis_title="Score (%)",
        height=300
    )
    
    return fig
