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
        self.llm_available = self._check_llm_availability()
    
    def _check_llm_availability(self) -> bool:
        """Check if Llama 3.1 8B is available and working"""
        if not self.use_local:
            return False
        try:
            # Test actual query with Llama 3.1 8B
            test_payload = {
                "model": "llama3.1:8b",
                "prompt": "Hello",
                "stream": False
            }
            
            test_response = requests.post(self.ollama_url, json=test_payload, timeout=10)
            return test_response.status_code == 200 and test_response.json().get('response')
            
        except Exception as e:
            return False
    
    def explain_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Convert numerical prediction to human explanation"""
        
        if self.llm_available:
            prompt = f"""
You are a financial advisor AI. Explain this stock prediction in simple terms:

Stock Price Prediction: ₹{prediction_data.get('price', 'N/A')}
Confidence Level: {prediction_data.get('confidence', 0)*100:.1f}%
Trend Direction: {prediction_data.get('trend', 'neutral')}
Expected Return: {prediction_data.get('expected_return', 0):.1f}%
Risk Level: {prediction_data.get('risk', 'medium')}

Provide:
1. What this means for investors
2. Key risks to consider  
3. Recommended action (buy/hold/sell)

Keep it under 150 words, use simple language.
"""
            
            llm_response = self._query_ollama(prompt)
            if llm_response and "LLM unavailable" not in llm_response:
                return f"🤖 **AI Financial Advisor:**\n\n{llm_response}"
        
        # Enhanced fallback explanation
        return self._enhanced_fallback_explanation(prediction_data)
    
    def chat_response(self, user_question: str, market_data: Dict) -> str:
        """Handle user questions about predictions"""
        
        if self.llm_available:
            prompt = f"""
You are a helpful financial advisor AI. Answer this user question about stock predictions:

User Question: "{user_question}"

Market Analysis Data:
- Stock: {market_data.get('symbol')} ({market_data.get('company')})
- Current Price: ${market_data.get('current_price', 0):.2f}
- Predicted Price: ${market_data.get('predicted_price', 0):.2f}
- Expected Return: {market_data.get('expected_return', 0):.1f}%
- Confidence: {market_data.get('confidence', 0):.1f}%
- Investment Amount: ${market_data.get('investment_amount', 0):,}
- Risk Tolerance: {market_data.get('risk_tolerance', 'moderate')}
- Prediction Timeframe: {market_data.get('prediction_days', 7)} days

Provide a helpful, specific answer based on this data. Include:
- Direct answer to their question
- Relevant insights from the prediction data
- Risk considerations
- Actionable advice

Keep it conversational and under 200 words. Always include appropriate disclaimers.
"""
            
            llm_response = self._query_ollama(prompt)
            if llm_response and "LLM unavailable" not in llm_response:
                return llm_response
        
        # Enhanced fallback for chat
        return self._smart_fallback_chat(user_question, market_data)
    
    def _query_ollama(self, prompt: str) -> str:
        """Query local Ollama LLM with Llama 3.1 8B"""
        try:
            payload = {
                "model": "llama3.1:8b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
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
    
    def _enhanced_fallback_explanation(self, data: Dict[str, Any]) -> str:
        """Enhanced fallback explanation when LLM unavailable"""
        confidence = data.get('confidence', 0) * 100
        trend = data.get('trend', 'neutral')
        expected_return = data.get('expected_return', 0)
        price = data.get('price', 0)
        
        # Determine confidence level
        if confidence > 80:
            conf_text = "high confidence"
            conf_emoji = "🎯"
        elif confidence > 60:
            conf_text = "moderate confidence"
            conf_emoji = "⚖️"
        else:
            conf_text = "low confidence"
            conf_emoji = "⚠️"
        
        # Determine action
        if expected_return > 10:
            action = "Strong BUY signal"
            action_emoji = "🚀"
        elif expected_return > 5:
            action = "BUY opportunity"
            action_emoji = "📈"
        elif expected_return > 0:
            action = "HOLD position"
            action_emoji = "⚖️"
        elif expected_return > -5:
            action = "Consider SELLING"
            action_emoji = "📉"
        else:
            action = "SELL signal"
            action_emoji = "🔴"
        
        return f"""
🤖 **AI Financial Advisor Analysis:**

{conf_emoji} **Prediction Confidence:** {confidence:.1f}% - {conf_text}
📊 **Expected Return:** {expected_return:.1f}% 
💰 **Target Price:** ${price:.2f}

{action_emoji} **Recommendation:** {action}

**Key Insights:**
• {trend.title()} trend detected in market patterns
• {"Strong" if confidence > 70 else "Moderate" if confidence > 50 else "Weak"} signal strength
• {"High" if abs(expected_return) > 10 else "Medium" if abs(expected_return) > 5 else "Low"} volatility expected

⚠️ **Risk Warning:** This is AI analysis, not financial advice. Always do your own research and consider consulting a financial advisor.

💡 **Note:** For more detailed analysis, install Ollama LLM for enhanced AI explanations.
"""
    
    def _smart_fallback_chat(self, question: str, market_data: Dict) -> str:
        """Smart fallback for chat when LLM unavailable"""
        question_lower = question.lower()
        symbol = market_data.get('symbol', 'this stock')
        expected_return = market_data.get('expected_return', 0)
        confidence = market_data.get('confidence', 0)
        investment = market_data.get('investment_amount', 0)
        
        # Pattern matching for common questions
        if any(word in question_lower for word in ['invest', 'buy', 'purchase']):
            if expected_return > 5:
                return f"""
📈 **Investment Outlook for {symbol}:**

Based on our AI analysis, there's a **{expected_return:.1f}% expected return** with **{confidence:.1f}% confidence**.

💡 **My Assessment:**
• Positive return signals suggest this could be a good investment opportunity
• With ${investment:,} investment, potential profit could be ${investment * expected_return / 100:,.0f}
• Risk level appears {"low" if confidence > 80 else "moderate" if confidence > 60 else "higher"}

⚠️ **Important:** This is AI analysis, not financial advice. Consider your risk tolerance and consult a financial advisor.
"""
            else:
                return f"""
⚠️ **Investment Caution for {symbol}:**

Our AI shows **{expected_return:.1f}% expected return** - this suggests limited upside potential.

💡 **My Assessment:**
• Current signals don't strongly favor buying
• Consider waiting for better entry points
• {"Low confidence" if confidence < 50 else "Moderate confidence"} in predictions

⚠️ **Important:** This is AI analysis based on current market patterns. Markets can change rapidly.
"""
        
        elif any(word in question_lower for word in ['risk', 'safe', 'danger']):
            risk_level = "High" if confidence < 50 else "Medium" if confidence < 75 else "Low"
            return f"""
🛡️ **Risk Assessment for {symbol}:**

**Risk Level:** {risk_level}
**Confidence:** {confidence:.1f}%
**Volatility:** {"High" if abs(expected_return) > 10 else "Medium" if abs(expected_return) > 5 else "Low"}

💡 **Risk Factors:**
• Prediction confidence is {confidence:.1f}% - {"strong" if confidence > 75 else "moderate" if confidence > 50 else "weak"} signal
• Expected return of {expected_return:.1f}% indicates {"high" if abs(expected_return) > 10 else "moderate"} volatility
• Market conditions can change rapidly

🤖 **For comprehensive risk analysis, install Ollama LLM!**
"""
        
        else:
            return f"""
📊 **Quick Analysis for {symbol}:**

• **Current Prediction:** {expected_return:.1f}% return expected
• **Confidence Level:** {confidence:.1f}%
• **Investment Amount:** ${investment:,}

💡 **General Insights:**
The AI models suggest a {"positive" if expected_return > 0 else "negative"} outlook with {"high" if confidence > 75 else "moderate" if confidence > 50 else "low"} confidence.

🤖 **For detailed answers to your specific questions, please install Ollama LLM for enhanced AI chat capabilities!**

**Setup:** `curl -fsSL https://ollama.ai/install.sh | sh && ollama pull llama3.1:8b`
"""
