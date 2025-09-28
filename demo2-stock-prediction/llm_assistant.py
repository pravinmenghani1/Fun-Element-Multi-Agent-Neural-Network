"""
🤖 LLM Assistant for Neural Prophet
Adds intelligent explanation layer to stock predictions
"""

import requests
import json
from typing import Dict, Any

class StockLLMAssistant:
    def __init__(self, use_local=True):
        self.use_local = use_local
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def explain_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Convert numerical prediction to human explanation"""
        
        prompt = f"""
You are a financial advisor AI. Explain this stock prediction in simple terms:

Stock Price Prediction: ₹{prediction_data.get('price', 'N/A')}
Confidence Level: {prediction_data.get('confidence', 0)*100:.1f}%
Trend Direction: {prediction_data.get('trend', 'neutral')}
Volatility: {prediction_data.get('volatility', 'moderate')}
Risk Level: {prediction_data.get('risk', 'medium')}

Provide:
1. What this means for investors
2. Key risks to consider
3. Recommended action (buy/hold/sell)

Keep it under 150 words, use simple language.
"""
        
        if self.use_local:
            return self._query_ollama(prompt)
        else:
            return self._fallback_explanation(prediction_data)
    
    def _query_ollama(self, prompt: str) -> str:
        """Query local Ollama LLM"""
        try:
            payload = {
                "model": "llama3.1:8b",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get('response', 'Analysis unavailable')
            else:
                return self._fallback_explanation({})
                
        except Exception as e:
            return f"LLM unavailable. Error: {str(e)}"
    
    def _fallback_explanation(self, data: Dict[str, Any]) -> str:
        """Fallback explanation when LLM unavailable"""
        confidence = data.get('confidence', 0) * 100
        trend = data.get('trend', 'neutral')
        
        if confidence > 80:
            conf_text = "high confidence"
        elif confidence > 60:
            conf_text = "moderate confidence"
        else:
            conf_text = "low confidence"
        
        return f"""
📊 Prediction Summary:
The model shows {conf_text} in a {trend} trend. 

💡 Investment Insight:
- {trend.title()} signals suggest {"buying opportunity" if trend == "bullish" else "caution advised" if trend == "bearish" else "hold position"}
- Confidence: {confidence:.1f}% - {"Strong signal" if confidence > 70 else "Moderate signal"}

⚠️ Always do your own research before investing.
"""

    def chat_response(self, user_question: str, market_data: Dict) -> str:
        """Handle user questions about predictions"""
        
        prompt = f"""
User Question: {user_question}
Market Data: {json.dumps(market_data, indent=2)}

Answer the user's question about stock predictions based on the data.
Be helpful, accurate, and include appropriate disclaimers.
"""
        
        if self.use_local:
            return self._query_ollama(prompt)
        else:
            return "Chat feature requires local LLM setup. Please install Ollama."
