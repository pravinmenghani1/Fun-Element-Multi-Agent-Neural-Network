"""
🏥 Medical LLM Assistant for Health AI
Adds intelligent explanation layer to medical diagnoses
"""

import requests
import json
from typing import Dict, Any, List

class MedicalLLMAssistant:
    def __init__(self, use_local=True):
        self.use_local = use_local
        self.ollama_url = "http://localhost:11434/api/generate"
        self.active_model = "llama3.2:latest"
        self.llm_available = self._check_llm_availability()
    
    def _check_llm_availability(self) -> bool:
        """Check if medical LLM is available and working"""
        if not self.use_local:
            return False
        try:
            models_to_test = ["llama3.2:latest", "llama3.2", "mistral:latest"]
            
            for model in models_to_test:
                test_payload = {
                    "model": model,
                    "prompt": "Hello",
                    "stream": False
                }
                
                test_response = requests.post(self.ollama_url, json=test_payload, timeout=10)
                if test_response.status_code == 200 and test_response.json().get('response'):
                    self.active_model = model
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    def explain_diagnosis(self, diagnosis_data: Dict[str, Any]) -> str:
        """Convert medical diagnosis to patient-friendly explanation"""
        
        if self.llm_available:
            prompt = f"""
You are a medical AI assistant. Explain this diagnosis in clear, patient-friendly language:

Diagnosis: {diagnosis_data.get('condition', 'Unknown')}
Confidence: {diagnosis_data.get('confidence', 0):.1f}%
Severity: {diagnosis_data.get('severity', 'Unknown')}
Affected Area: {diagnosis_data.get('location', 'Not specified')}
Risk Level: {diagnosis_data.get('risk_level', 'Unknown')}

Provide:
1. What this condition means in simple terms
2. Typical symptoms and causes
3. Recommended next steps
4. When to seek immediate medical attention

Keep it under 200 words, use simple language, and include appropriate medical disclaimers.
"""
            
            llm_response = self._query_ollama(prompt)
            if llm_response and "LLM unavailable" not in llm_response:
                return f"🏥 **Medical AI Explanation:**\n\n{llm_response}"
        
        return self._medical_fallback_explanation(diagnosis_data)
    
    def generate_medical_report(self, diagnosis_data: Dict[str, Any], symptoms: List[str] = None) -> str:
        """Generate comprehensive medical report"""
        
        if self.llm_available:
            symptoms_text = ", ".join(symptoms) if symptoms else "Not provided"
            
            prompt = f"""
Generate a medical report based on this AI analysis:

Primary Diagnosis: {diagnosis_data.get('condition', 'Unknown')}
Confidence Level: {diagnosis_data.get('confidence', 0):.1f}%
Severity Assessment: {diagnosis_data.get('severity', 'Unknown')}
Location/Area: {diagnosis_data.get('location', 'Not specified')}
Patient Symptoms: {symptoms_text}

Include:
1. Clinical Summary
2. Diagnostic Confidence Assessment
3. Recommended Treatment Options
4. Follow-up Care Instructions
5. Warning Signs to Watch For

Format as a professional medical report but keep language accessible.
"""
            
            llm_response = self._query_ollama(prompt)
            if llm_response and "LLM unavailable" not in llm_response:
                return llm_response
        
        return self._generate_fallback_report(diagnosis_data, symptoms)
    
    def symptom_analysis(self, symptoms: List[str], patient_info: Dict = None) -> str:
        """Analyze symptoms and provide preliminary assessment"""
        
        if self.llm_available and symptoms:
            symptoms_text = ", ".join(symptoms)
            age = patient_info.get('age', 'Not specified') if patient_info else 'Not specified'
            gender = patient_info.get('gender', 'Not specified') if patient_info else 'Not specified'
            
            prompt = f"""
As a medical AI, analyze these symptoms and provide preliminary assessment:

Symptoms: {symptoms_text}
Patient Age: {age}
Patient Gender: {gender}

Provide:
1. Possible conditions that match these symptoms
2. Urgency level (Low/Medium/High)
3. Recommended immediate actions
4. Questions to ask the patient for better diagnosis

Important: Always recommend consulting a healthcare professional.
Keep response under 250 words.
"""
            
            llm_response = self._query_ollama(prompt)
            if llm_response and "LLM unavailable" not in llm_response:
                return llm_response
        
        return self._symptom_fallback_analysis(symptoms)
    
    def chat_response(self, user_question: str, medical_context: Dict) -> str:
        """Handle medical questions with context"""
        
        if self.llm_available:
            prompt = f"""
You are a medical AI assistant. Answer this patient question based on the medical context:

Patient Question: "{user_question}"

Medical Context:
- Diagnosis: {medical_context.get('diagnosis', 'Not available')}
- Confidence: {medical_context.get('confidence', 0):.1f}%
- Severity: {medical_context.get('severity', 'Unknown')}
- Symptoms: {medical_context.get('symptoms', 'Not provided')}

Provide a helpful, accurate response that:
1. Addresses their specific question
2. Uses the medical context appropriately
3. Includes appropriate medical disclaimers
4. Recommends consulting healthcare professionals when needed

Keep it conversational and under 200 words.
"""
            
            llm_response = self._query_ollama(prompt)
            if llm_response and "LLM unavailable" not in llm_response:
                return llm_response
        
        return self._medical_chat_fallback(user_question, medical_context)
    
    def _query_ollama(self, prompt: str) -> str:
        """Query local Ollama LLM for medical analysis"""
        try:
            payload = {
                "model": self.active_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for medical accuracy
                    "top_p": 0.8
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                if result:
                    return result
            
            return "LLM unavailable"
                
        except Exception as e:
            return "LLM unavailable"
    
    def _medical_fallback_explanation(self, data: Dict[str, Any]) -> str:
        """Enhanced fallback for medical explanations"""
        condition = data.get('condition', 'Unknown condition')
        confidence = data.get('confidence', 0)
        severity = data.get('severity', 'Unknown')
        
        if confidence > 80:
            conf_text = "high confidence"
            conf_emoji = "🎯"
        elif confidence > 60:
            conf_text = "moderate confidence"
            conf_emoji = "⚖️"
        else:
            conf_text = "low confidence"
            conf_emoji = "⚠️"
        
        return f"""
🏥 **Medical AI Analysis:**

{conf_emoji} **Diagnosis:** {condition} ({conf_text}: {confidence:.1f}%)
📊 **Severity Level:** {severity}

**What this means:**
• The AI has detected patterns consistent with {condition.lower()}
• Confidence level indicates {"strong" if confidence > 75 else "moderate" if confidence > 50 else "preliminary"} diagnostic certainty
• {"Immediate medical attention recommended" if severity == "High" else "Medical consultation advised" if severity == "Medium" else "Monitor symptoms and consult if worsening"}

⚠️ **Important:** This is AI analysis for educational purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment.

💡 **Note:** For detailed medical explanations, enhanced AI analysis is available with local LLM setup.
"""
    
    def _generate_fallback_report(self, data: Dict[str, Any], symptoms: List[str]) -> str:
        """Generate basic medical report without LLM"""
        condition = data.get('condition', 'Unknown')
        confidence = data.get('confidence', 0)
        
        return f"""
📋 **MEDICAL AI REPORT**

**Primary Finding:** {condition}
**Diagnostic Confidence:** {confidence:.1f}%
**Analysis Date:** {data.get('timestamp', 'Current session')}

**Clinical Summary:**
The AI analysis indicates patterns consistent with {condition.lower()}. 

**Symptoms Analyzed:** {', '.join(symptoms) if symptoms else 'Image-based analysis'}

**Recommendations:**
• Consult healthcare professional for confirmation
• Monitor symptoms for any changes
• Follow standard medical protocols for {condition.lower()}

**Disclaimer:** This AI analysis is for educational and screening purposes only.
"""
    
    def _symptom_fallback_analysis(self, symptoms: List[str]) -> str:
        """Basic symptom analysis without LLM"""
        if not symptoms:
            return "No symptoms provided for analysis."
        
        return f"""
🔍 **Symptom Analysis:**

**Reported Symptoms:** {', '.join(symptoms)}

**Preliminary Assessment:**
• Multiple symptoms detected requiring medical evaluation
• Recommend comprehensive medical examination
• Urgency level: Moderate (consult healthcare provider)

**Next Steps:**
1. Document all symptoms with timeline
2. Schedule appointment with healthcare provider
3. Monitor for any worsening symptoms

⚠️ **Important:** Symptom analysis requires professional medical evaluation.
"""
    
    def _medical_chat_fallback(self, question: str, context: Dict) -> str:
        """Fallback for medical chat questions"""
        diagnosis = context.get('diagnosis', 'your condition')
        
        return f"""
🏥 **Medical AI Response:**

Regarding your question about {diagnosis}:

Based on the AI analysis, I can provide general information, but specific medical questions require professional healthcare consultation.

**General Guidance:**
• Follow standard medical protocols for {diagnosis.lower()}
• Monitor symptoms as advised by healthcare providers
• Seek immediate medical attention for any concerning changes

**For detailed medical advice:** Please consult with qualified healthcare professionals who can provide personalized medical guidance.

💡 **Enhanced responses available with local medical LLM setup.**
"""
