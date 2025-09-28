#!/usr/bin/env python3
"""
🏥 Smart Health Diagnosis AI - Where AI Meets Medicine
Multi-Agent Neural Network System for Medical Diagnosis
Built by Pravin Menghani, in love ❤️ with Neural Networks!!
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio
import random

# Import our custom modules
from medical_visualizations import *
from medical_diagnosis_engine import *
from medical_llm_assistant import MedicalLLMAssistant

def main():
    st.set_page_config(
        page_title="🏥 Smart Health Diagnosis AI - Where AI Meets Medicine",
        page_icon="🩺🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize Medical LLM Assistant
    if 'medical_llm_assistant' not in st.session_state:
        st.session_state.medical_llm_assistant = MedicalLLMAssistant(use_local=True)
    
    # Ensure llm_available attribute exists (for backward compatibility)
    if not hasattr(st.session_state.medical_llm_assistant, 'llm_available'):
        st.session_state.medical_llm_assistant.llm_available = st.session_state.medical_llm_assistant._check_llm_availability()
    
    # Spectacular CSS styling with medical theme
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
    
    .diagnosis-card {
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
    
    .medical-warning {
        background: linear-gradient(135deg, #ff7b7b 0%, #ff9999 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #ff0000;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Spectacular header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Smart Health Diagnosis AI</h1>
        <h2>🩺 Where AI Meets Medicine</h2>
        <p>✨ Multi-Agent Neural Architecture • Real-time Health Analysis • Intelligent Diagnosis ✨</p>
        <p style="font-size: 14px; margin-top: 10px; opacity: 0.9;">
            Built by <strong>Pravin Menghani</strong>, in love ❤️ with Neural Networks!!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clear Problem Statement
    st.markdown("""
    <div class="problem-statement">
        <h3>🎯 THE CHALLENGE: Can AI Help Doctors Diagnose Diseases More Accurately?</h3>
        <p><strong>Problem:</strong> Medical diagnosis is complex and requires analyzing many symptoms, test results, and patient history. Even experienced doctors can miss patterns.</p>
        <p><strong>Solution:</strong> We'll use 5 different AI neural networks, each specialized for different aspects of medical diagnosis.</p>
        <p><strong>Goal:</strong> See how different AI techniques can assist doctors and learn how neural networks process medical information.</p>
        <p><strong>Why This Matters:</strong> AI is revolutionizing healthcare - understanding these concepts prepares you for the future of medicine!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.markdown("""
    <div class="medical-warning">
        <h4>⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
        <p><strong>This is an EDUCATIONAL DEMO only!</strong> This AI system is for learning purposes and should NEVER be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with AI concepts
    with st.sidebar:
        st.header("🧠 Medical AI Arsenal")
        
        with st.expander("📚 Medical AI Guide", expanded=False):
            st.markdown("""
            **🧠 FNN:** Basic symptom analysis  
            **🔄 RNN:** Patient history patterns  
            **👁️ CNN:** Medical image analysis  
            **🎯 Decision Tree:** Rule-based diagnosis  
            **🤖 Ensemble:** Combines all AI doctors  
            """)
        
        concept_tab = st.selectbox("Select Medical AI:", [
            "🌐 3D Medical AI Network",
            "🧠 FNN: Symptom Analysis",
            "🔄 RNN: Patient History", 
            "👁️ CNN: Medical Imaging",
            "🎯 Decision Tree: Rule-based Diagnosis",
            "🤖 AI Doctor Consultation",
            "📊 Real-time Diagnosis Engine"
        ])
        
        if concept_tab == "🌐 3D Medical AI Network":
            st.plotly_chart(create_3d_medical_network(), use_container_width=True)
            st.markdown("""
            **🌐 Medical AI Network Topology**
            
            **What you're seeing:**
            - 5 different AI "doctors" working together
            - Each specializes in different medical analysis
            - Color-coded by diagnostic accuracy
            
            **Why this matters:**
            - Each AI has unique medical expertise
            - Ensemble combines all for best diagnosis
            - Visual understanding of AI collaboration
            
            **🩺 Watch the AI doctors collaborate for accurate diagnosis!**
            """)
        
        elif concept_tab == "🧠 FNN: Symptom Analysis":
            st.plotly_chart(create_fnn_symptom_analysis(), use_container_width=True)
            st.markdown("""
            **🧠 FNN: The Basic Medical AI**
            
            **What FNN does:**
            - Feedforward Neural Network - simplest AI type
            - Analyzes symptoms in straightforward way
            - Maps symptoms directly to possible diseases
            
            **Why perfect for symptoms:**
            - Symptoms have direct relationships to diseases
            - Fast and efficient analysis
            - Good starting point for diagnosis
            
            **🔍 FNN = The AI that thinks step-by-step through symptoms!**
            """)
        
        elif concept_tab == "🔄 RNN: Patient History":
            st.plotly_chart(create_rnn_history_analysis(), use_container_width=True)
            st.markdown("""
            **🔄 RNN: The Medical Memory Expert**
            
            **What RNN does:**
            - Recurrent Neural Network - has memory
            - Analyzes patient history over time
            - Remembers previous symptoms and treatments
            
            **Why great for medical history:**
            - Diseases develop over time
            - Past symptoms affect current diagnosis
            - Can spot patterns across visits
            
            **📚 RNN = AI doctor with perfect memory of your medical history!**
            """)
        
        elif concept_tab == "👁️ CNN: Medical Imaging":
            st.plotly_chart(create_cnn_imaging_analysis(), use_container_width=True)
            st.markdown("""
            **👁️ CNN: The Medical Image Detective**
            
            **What CNN does:**
            - Convolutional Neural Network - sees patterns
            - Analyzes X-rays, MRIs, CT scans
            - Detects abnormalities in medical images
            
            **Why amazing for imaging:**
            - Can spot tiny details humans might miss
            - Trained on thousands of medical images
            - Works like a radiologist's trained eye
            
            **🔍 CNN sees medical images like an expert radiologist!**
            """)
        
        elif concept_tab == "🎯 Decision Tree: Rule-based Diagnosis":
            st.plotly_chart(create_decision_tree_diagnosis(), use_container_width=True)
            st.markdown("""
            **🎯 Decision Tree: The Logical Medical AI**
            
            **What Decision Tree does:**
            - Follows logical medical rules
            - "If symptom A and B, then likely disease X"
            - Transparent reasoning process
            
            **Why valuable for diagnosis:**
            - Doctors can understand the reasoning
            - Based on established medical knowledge
            - Provides clear explanation paths
            
            **🌳 Decision Tree = AI that thinks like a medical textbook!**
            """)
        
        elif concept_tab == "🤖 AI Doctor Consultation":
            st.plotly_chart(create_ai_doctor_consultation(), use_container_width=True)
            st.markdown("""
            **🤖 Watch AI Doctors Collaborate in Real-time!**
            
            **The Medical Team:**
            - Each AI specializes in different aspects
            - Real-time accuracy comparison
            - Collaborative diagnosis approach
            
            **What you learn:**
            - Different AIs excel at different medical tasks
            - Ensemble combines all expertise
            - AI teamwork in healthcare
            
            **👨‍⚕️ Like having multiple specialists consult on your case!**
            """)
        
        elif concept_tab == "📊 Real-time Diagnosis Engine":
            st.plotly_chart(create_realtime_diagnosis_engine(), use_container_width=True)
            st.info("📊 Live diagnosis engine showing all AI doctors working together")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🩺 Patient Information")
        
        # Clear explanation box
        st.info("""
        💡 **How This Works:**
        1. **Enter symptoms** - Describe what you're experiencing
        2. **Provide history** - Any relevant medical background
        3. **Upload image** - X-ray, scan, or photo (optional)
        4. **Select AI doctors** - Choose which AI specialists to consult
        5. **Get diagnosis** - See what each AI thinks and final recommendation
        """)
        
        # CRYSTAL CLEAR EXPLANATION
        st.success("""
        🎯 **YES - ALL AI DOCTORS ANALYZE YOUR CASE!** Here's what each does:
        
        🧠 **FNN Analysis** → Analyzes your symptoms using basic neural network
        
        🔄 **RNN History** → Considers your medical history and patterns over time
        
        👁️ **CNN Imaging** → Examines any medical images for abnormalities
        
        🎯 **Decision Tree** → Follows logical medical rules for diagnosis
        
        🤖 **Ensemble Diagnosis** → Combines all AI opinions for final recommendation
        """)
        
        with st.form("health_diagnosis_form"):
            # Patient Demographics
            st.markdown("**👤 Basic Information:**")
            age = st.slider("Age", 1, 100, 30, help="Patient's age affects disease probability")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Some diseases are more common in certain genders")
            
            # Symptoms Input
            st.markdown("**🤒 Current Symptoms:**")
            symptoms = st.multiselect("Select symptoms you're experiencing:", [
                "Fever", "Headache", "Cough", "Fatigue", "Nausea", "Chest Pain",
                "Shortness of Breath", "Dizziness", "Joint Pain", "Skin Rash",
                "Abdominal Pain", "Back Pain", "Sore Throat", "Runny Nose"
            ], help="Select all symptoms you're currently experiencing")
            
            symptom_severity = st.selectbox("Symptom Severity", ["Mild", "Moderate", "Severe"], 
                                          help="How severe are your symptoms overall?")
            
            # Medical History
            st.markdown("**📋 Medical History:**")
            medical_history = st.multiselect("Previous conditions:", [
                "Diabetes", "Hypertension", "Heart Disease", "Asthma", "Allergies",
                "Cancer", "Kidney Disease", "Liver Disease", "None"
            ], help="Select any previous medical conditions")
            
            # Duration
            symptom_duration = st.selectbox("How long have you had these symptoms?", [
                "Less than 1 day", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks"
            ], help="Duration helps determine if condition is acute or chronic")
            
            # AI Selection
            st.markdown("**🤖 AI Medical Team Selection:**")
            st.caption("Each AI doctor has different medical expertise - select multiple for comprehensive analysis!")
            
            use_fnn = st.checkbox("🧠 FNN (Symptom Analysis)", value=True, 
                                help="Basic neural network that analyzes symptom patterns")
            use_rnn = st.checkbox("🔄 RNN (Medical History)", value=True,
                                help="AI with memory that considers your medical history over time")
            use_cnn = st.checkbox("👁️ CNN (Medical Imaging)", value=True,
                                help="AI that analyzes medical images and visual symptoms")
            use_decision_tree = st.checkbox("🎯 Decision Tree (Rule-based)", value=True,
                                          help="AI that follows logical medical rules and guidelines")
            
            # Optional image upload
            st.markdown("**📸 Medical Image (Optional):**")
            uploaded_image = st.file_uploader("Upload medical image", type=['png', 'jpg', 'jpeg'],
                                            help="Upload X-ray, scan, or photo of affected area (optional)")
            
            submitted = st.form_submit_button("🩺 Consult AI Medical Team", type="primary")
        
        if submitted:
            if not symptoms:
                st.error("Please select at least one symptom to analyze!")
            else:
                st.session_state.health_request = {
                    'age': age,
                    'gender': gender,
                    'symptoms': symptoms,
                    'severity': symptom_severity,
                    'history': medical_history,
                    'duration': symptom_duration,
                    'image': uploaded_image,
                    'ai_doctors': {
                        'fnn': use_fnn,
                        'rnn': use_rnn,
                        'cnn': use_cnn,
                        'decision_tree': use_decision_tree
                    }
                }
                st.session_state.medical_analysis = True
    
    with col2:
        if hasattr(st.session_state, 'medical_analysis') and st.session_state.medical_analysis:
            
            # AI Medical Team in Action
            st.markdown("### 🤖 AI Medical Team Analyzing...")
            
            # Real-time AI doctor status
            request = st.session_state.health_request
            selected_doctors = [k for k, v in request['ai_doctors'].items() if v]
            
            # Create dynamic columns based on selected AI doctors
            if len(selected_doctors) >= 3:
                doctor_cols = st.columns(3)
            else:
                doctor_cols = st.columns(len(selected_doctors))
            
            doctor_info = {
                'fnn': ("🧠 FNN", "Analyzing symptom patterns..."),
                'rnn': ("🔄 RNN", "Reviewing medical history..."),
                'cnn': ("👁️ CNN", "Examining medical images..."),
                'decision_tree': ("🎯 Decision Tree", "Following medical protocols...")
            }
            
            for i, doctor in enumerate(selected_doctors[:len(doctor_cols)]):
                if doctor in doctor_info:
                    name, status = doctor_info[doctor]
                    with doctor_cols[i]:
                        st.markdown(f"""
                        <div class="agent-card">
                            <h4>{name}</h4>
                            <div>{status}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Progress with medical analysis
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Run medical analysis
            async def run_medical_analysis():
                request = st.session_state.health_request
                
                # Step 1: Symptom analysis
                status_text.text("🔍 Analyzing symptoms and patient information...")
                progress_bar.progress(20)
                await asyncio.sleep(1)
                
                # Step 2: AI processing
                status_text.text("🧠 AI medical team consulting on your case...")
                progress_bar.progress(50)
                diagnoses = await generate_medical_diagnoses(request)
                
                # Step 3: Cross-referencing
                status_text.text("📋 Cross-referencing with medical databases...")
                progress_bar.progress(80)
                ensemble_result = combine_medical_diagnoses(diagnoses)
                
                # Step 4: Final consultation
                status_text.text("✨ Preparing comprehensive medical assessment...")
                progress_bar.progress(100)
                
                return diagnoses, ensemble_result
            
            # Execute medical analysis
            if 'medical_results' not in st.session_state:
                try:
                    diagnoses, ensemble = asyncio.run(run_medical_analysis())
                    st.session_state.medical_results = (diagnoses, ensemble)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.medical_results = None
            
            # Display medical results
            if hasattr(st.session_state, 'medical_results') and st.session_state.medical_results:
                diagnoses, ensemble = st.session_state.medical_results
                request = st.session_state.health_request
                
                # CRYSTAL CLEAR EXPLANATION OF RESULTS
                st.markdown("### 🎯 What Just Happened?")
                st.info(f"""
                **Each AI doctor analyzed your case using different medical expertise:**
                
                🧠 **Different AI Doctors, Different Specialties:**
                - Some focus on symptom patterns, others on medical history
                - Some analyze images, others follow medical protocols
                - Each provides different diagnostic insights
                
                🎯 **The Final Diagnosis:** All AI opinions are combined into one comprehensive assessment!
                """)
                
                # Main diagnosis display
                st.markdown("### 🩺 AI Medical Team Diagnosis")
                
                st.markdown(f"""
                <div class="diagnosis-card">
                    <h3>🏥 Medical Assessment Results</h3>
                    <h2>Primary Diagnosis: {ensemble['primary_diagnosis']}</h2>
                    <p><strong>Confidence Level:</strong> {ensemble['confidence']:.1f}%</p>
                    <p><strong>Severity Assessment:</strong> {ensemble['severity']}</p>
                    <p><strong>Recommended Action:</strong> {ensemble['recommendation']}</p>
                    <p><strong>AI Team Consensus:</strong> {ensemble['consensus']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 🏥 MEDICAL LLM INTELLIGENT EXPLANATION
                st.markdown("### 🤖 AI Medical Assistant Explanation")
                
                # LLM Status Indicator
                col_status, col_refresh = st.columns([3, 1])
                
                with col_status:
                    llm_status = st.session_state.medical_llm_assistant.llm_available
                    if llm_status:
                        st.success("🤖 Medical AI Online - Enhanced explanations available!")
                    else:
                        st.warning("⚠️ Medical AI in Basic Mode - Enhanced analysis available with local LLM")
                
                with col_refresh:
                    if st.button("🔄 Refresh Medical AI Status"):
                        st.session_state.medical_llm_assistant.llm_available = st.session_state.medical_llm_assistant._check_llm_availability()
                        st.rerun()
                
                with st.spinner("🧠 Medical AI is analyzing the diagnosis..."):
                    diagnosis_data = {
                        'condition': ensemble['primary_diagnosis'],
                        'confidence': ensemble['confidence'],
                        'severity': ensemble['severity'],
                        'location': 'General assessment',
                        'risk_level': ensemble['severity'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    medical_explanation = st.session_state.medical_llm_assistant.explain_diagnosis(diagnosis_data)
                    st.info(medical_explanation)
                
                # AI doctor breakdown
                available_tabs = []
                if 'fnn' in diagnoses:
                    available_tabs.append("🧠 FNN Analysis")
                if 'rnn' in diagnoses:
                    available_tabs.append("🔄 RNN History")
                if 'cnn' in diagnoses:
                    available_tabs.append("👁️ CNN Imaging")
                if 'decision_tree' in diagnoses:
                    available_tabs.append("🎯 Decision Tree")
                available_tabs.append("🤖 Final Diagnosis")
                
                tabs = st.tabs(available_tabs)
                tab_idx = 0
                
                if 'fnn' in diagnoses:
                    with tabs[tab_idx]:
                        st.markdown("**🧠 FNN: Basic Symptom Analysis**")
                        st.write(f"**FNN Diagnosis:** {diagnoses['fnn']['diagnosis']}")
                        st.write(f"**Symptom Match:** {diagnoses['fnn']['symptom_match']:.1f}%")
                        st.write(f"**Key Symptoms:** {', '.join(diagnoses['fnn']['key_symptoms'])}")
                        st.plotly_chart(create_fnn_diagnosis_chart(diagnoses['fnn']), use_container_width=True)
                    tab_idx += 1
                
                if 'rnn' in diagnoses:
                    with tabs[tab_idx]:
                        st.markdown("**🔄 RNN: Medical History Analysis**")
                        st.write(f"**RNN Diagnosis:** {diagnoses['rnn']['diagnosis']}")
                        st.write(f"**History Relevance:** {diagnoses['rnn']['history_relevance']:.1f}%")
                        st.write(f"**Risk Factors:** {', '.join(diagnoses['rnn']['risk_factors'])}")
                        st.plotly_chart(create_rnn_diagnosis_chart(diagnoses['rnn']), use_container_width=True)
                    tab_idx += 1
                
                if 'cnn' in diagnoses:
                    with tabs[tab_idx]:
                        st.markdown("**👁️ CNN: Medical Image Analysis**")
                        st.write(f"**CNN Diagnosis:** {diagnoses['cnn']['diagnosis']}")
                        st.write(f"**Image Analysis:** {diagnoses['cnn']['image_analysis']}")
                        st.write(f"**Visual Indicators:** {', '.join(diagnoses['cnn']['visual_indicators'])}")
                        st.plotly_chart(create_cnn_diagnosis_chart(diagnoses['cnn']), use_container_width=True)
                    tab_idx += 1
                
                if 'decision_tree' in diagnoses:
                    with tabs[tab_idx]:
                        st.markdown("**🎯 Decision Tree: Rule-based Analysis**")
                        st.write(f"**Decision Tree Diagnosis:** {diagnoses['decision_tree']['diagnosis']}")
                        st.write(f"**Rule Confidence:** {diagnoses['decision_tree']['rule_confidence']:.1f}%")
                        st.write(f"**Decision Path:** {diagnoses['decision_tree']['decision_path']}")
                        st.plotly_chart(create_decision_tree_chart(diagnoses['decision_tree']), use_container_width=True)
                    tab_idx += 1
                
                with tabs[tab_idx]:
                    st.markdown("**🤖 Final Medical Team Consensus**")
                    st.plotly_chart(create_ensemble_diagnosis_chart(ensemble), use_container_width=True)
                    
                    st.markdown("### 🏥 Medical Recommendations")
                    if ensemble['severity'] == 'High':
                        st.error(f"🚨 URGENT: {ensemble['recommendation']}")
                    elif ensemble['severity'] == 'Moderate':
                        st.warning(f"⚠️ ATTENTION: {ensemble['recommendation']}")
                    else:
                        st.info(f"ℹ️ ADVICE: {ensemble['recommendation']}")
                
                # Final medical advice
                st.markdown("---")
                if st.button("📋 ACCEPT AI MEDICAL TEAM ASSESSMENT", type="primary", use_container_width=True):
                    st.balloons()
                    st.success("🎉 AI Medical Team has completed comprehensive analysis!")
                    
                    st.markdown(f"""
                    **🏥 Smart Health Diagnosis AI Summary:**
                    - **🧠 FNN:** Analyzed symptom patterns and correlations
                    - **🔄 RNN:** Considered medical history and temporal patterns
                    - **👁️ CNN:** Examined visual symptoms and medical imaging
                    - **🎯 Decision Tree:** Applied evidence-based medical protocols
                    - **🤖 Ensemble:** Combined all AI medical expertise
                    - **🩺 Assessment:** {ensemble['primary_diagnosis']} with {ensemble['confidence']:.1f}% confidence
                    - **⚠️ Remember:** This is educational only - consult real doctors!
                    - **✨ Where AI Meets Medicine!**
                    """)
                
                # 💬 INTERACTIVE MEDICAL CHAT
                st.markdown("---")
                st.markdown("### 💬 Chat with Medical AI Assistant")
                st.info("Ask questions about the diagnosis, symptoms, or general health information!")
                
                # Medical chat interface
                if 'medical_chat_history' not in st.session_state:
                    st.session_state.medical_chat_history = []
                
                user_medical_question = st.text_input("💭 Ask your medical question:", 
                                                    placeholder="e.g., What does this diagnosis mean? What should I do next?")
                
                if st.button("🩺 Ask Medical AI") and user_medical_question:
                    with st.spinner("🧠 Medical AI is analyzing your question..."):
                        medical_context = {
                            'diagnosis': ensemble['primary_diagnosis'],
                            'confidence': ensemble['confidence'],
                            'severity': ensemble['severity'],
                            'symptoms': symptoms_list if 'symptoms_list' in locals() else []
                        }
                        
                        ai_medical_response = st.session_state.medical_llm_assistant.chat_response(user_medical_question, medical_context)
                        
                        # Add to medical chat history
                        st.session_state.medical_chat_history.append({
                            "user": user_medical_question, 
                            "ai": ai_medical_response
                        })
                
                # Display medical chat history
                if st.session_state.medical_chat_history:
                    st.markdown("#### 💬 Medical Consultation History")
                    for i, chat in enumerate(reversed(st.session_state.medical_chat_history[-3:])):  # Show last 3 exchanges
                        st.markdown(f"**You:** {chat['user']}")
                        st.markdown(f"**🩺 Medical AI:** {chat['ai']}")
                        if i < len(st.session_state.medical_chat_history[-3:]) - 1:
                            st.markdown("---")
                
                # Medical disclaimer for chat
                st.warning("⚠️ **Medical Disclaimer:** This AI chat is for educational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment.")
    
    # Footer with signature
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); 
                border-radius: 15px; color: white; margin-top: 2rem;">
        <h3>🏥 Smart Health Diagnosis AI</h3>
        <p><strong>Where AI Meets Medicine</strong></p>
        <p style="margin-top: 1rem; font-size: 16px;">
            Built with ❤️ by <strong>Pravin Menghani</strong><br>
            In love with Neural Networks!!
        </p>
        <p style="font-size: 14px; opacity: 0.9; margin-top: 1rem;">
            🧠 Multi-Agent AI • 🩺 Medical Analysis • 🏥 Smart Diagnosis
        </p>
        <p style="font-size: 12px; margin-top: 1rem; background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 5px;">
            ⚠️ EDUCATIONAL DEMO ONLY - NOT FOR ACTUAL MEDICAL DIAGNOSIS
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
