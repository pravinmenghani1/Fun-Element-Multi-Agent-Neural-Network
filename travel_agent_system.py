#!/usr/bin/env python3
"""
AI Travel Agent System - Multi-Agent Architecture Demo
Demonstrates various AI concepts through travel booking functionality
"""

import asyncio
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import streamlit as st
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

# Core Agent Architecture
class BaseAgent(ABC):
    """Base class demonstrating feedforward neural network concept"""
    
    def __init__(self, name: str, llm_model: str = "llama3.2:latest"):
        self.name = name
        self.llm_model = llm_model
        self.memory = []  # Simple memory mechanism
    
    @abstractmethod
    async def process(self, input_data: Dict) -> Dict:
        """Abstract processing method - like forward pass in neural networks"""
        pass
    
    def add_to_memory(self, data: Dict):
        """Memory mechanism similar to LSTM/attention"""
        self.memory.append({
            'timestamp': datetime.now(),
            'data': data
        })
        if len(self.memory) > 10:  # Keep last 10 memories
            self.memory.pop(0)

@dataclass
class TravelRequest:
    """Data structure for travel requests"""
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str]
    budget: float
    travelers: int
    preferences: Dict

class FlightAgent(BaseAgent):
    """Flight booking agent - demonstrates CNN-like pattern recognition"""
    
    def __init__(self):
        super().__init__("FlightAgent", "llama3.2:latest")
        self.api_endpoints = {
            'makemytrip': 'https://api.makemytrip.com/flights',
            'indigo': 'https://api.goindigo.in/booking',
            'airindia': 'https://api.airindia.in/flights'
        }
    
    async def process(self, travel_request: TravelRequest) -> Dict:
        """Process flight search - like CNN feature extraction"""
        
        # Simulate API calls to multiple airlines
        flight_results = []
        
        for airline, endpoint in self.api_endpoints.items():
            try:
                # Simulate flight search (replace with actual API calls)
                flights = await self._search_flights(airline, travel_request)
                flight_results.extend(flights)
            except Exception as e:
                st.warning(f"Failed to fetch from {airline}: {str(e)}")
        
        # Apply CNN-like filtering and ranking
        filtered_flights = self._apply_filters(flight_results, travel_request)
        ranked_flights = self._rank_flights(filtered_flights, travel_request.budget)
        
        self.add_to_memory({'flights_searched': len(flight_results)})
        
        return {
            'agent': self.name,
            'flights': ranked_flights[:5],  # Top 5 results
            'total_found': len(flight_results)
        }
    
    async def _search_flights(self, airline: str, request: TravelRequest) -> List[Dict]:
        """Simulate flight API calls"""
        # Mock flight data
        base_price = int(np.random.randint(3000, 15000))
        return [{
            'airline': airline,
            'flight_number': f"{airline.upper()[:2]}{int(np.random.randint(100, 999))}",
            'departure_time': f"{int(np.random.randint(6, 23)):02d}:{int(np.random.randint(0, 59)):02d}",
            'arrival_time': f"{int(np.random.randint(6, 23)):02d}:{int(np.random.randint(0, 59)):02d}",
            'price': int(base_price + np.random.randint(-1000, 2000)),
            'duration': f"{int(np.random.randint(1, 8))}h {int(np.random.randint(0, 59))}m",
            'stops': int(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
        } for _ in range(int(np.random.randint(2, 8)))]
    
    def _apply_filters(self, flights: List[Dict], request: TravelRequest) -> List[Dict]:
        """CNN-like convolution filtering"""
        return [f for f in flights if f['price'] <= request.budget * 0.6]
    
    def _rank_flights(self, flights: List[Dict], budget: float) -> List[Dict]:
        """Ranking algorithm - like neural network output layer"""
        for flight in flights:
            score = 0
            score += (budget - flight['price']) / budget * 50  # Price score
            score += (3 - flight['stops']) * 20  # Direct flight bonus
            flight['score'] = score
        
        return sorted(flights, key=lambda x: x['score'], reverse=True)

class HotelAgent(BaseAgent):
    """Hotel booking agent - demonstrates Transformer attention mechanism"""
    
    def __init__(self):
        super().__init__("HotelAgent", "deepseek-r1:1.5b")
        self.booking_platforms = ['makemytrip', 'goibibo', 'easemytrip', 'booking.com']
    
    async def process(self, travel_request: TravelRequest) -> Dict:
        """Process hotel search using attention-like mechanism"""
        
        hotel_results = []
        
        # Multi-head attention: search across multiple platforms
        for platform in self.booking_platforms:
            hotels = await self._search_hotels(platform, travel_request)
            hotel_results.extend(hotels)
        
        # Apply transformer-like attention scoring
        scored_hotels = self._apply_attention_scoring(hotel_results, travel_request)
        
        self.add_to_memory({'hotels_searched': len(hotel_results)})
        
        return {
            'agent': self.name,
            'hotels': scored_hotels[:5],
            'total_found': len(hotel_results)
        }
    
    async def _search_hotels(self, platform: str, request: TravelRequest) -> List[Dict]:
        """Simulate hotel API calls"""
        base_price = int(np.random.randint(1500, 8000))
        return [{
            'platform': platform,
            'name': f"Hotel {np.random.choice(['Grand', 'Royal', 'Palace', 'Plaza'])} {np.random.choice(['Deluxe', 'Premium', 'Luxury'])}",
            'rating': float(round(np.random.uniform(3.0, 5.0), 1)),
            'price_per_night': int(base_price + np.random.randint(-500, 1500)),
            'amenities': np.random.choice(['WiFi', 'Pool', 'Gym', 'Spa', 'Restaurant'], size=int(np.random.randint(2, 5)), replace=False).tolist(),
            'distance_from_center': float(round(np.random.uniform(0.5, 15.0), 1))
        } for _ in range(int(np.random.randint(3, 10)))]
    
    def _apply_attention_scoring(self, hotels: List[Dict], request: TravelRequest) -> List[Dict]:
        """Transformer attention mechanism for hotel ranking"""
        for hotel in hotels:
            # Multi-head attention weights
            price_attention = max(0, (request.budget * 0.4 - hotel['price_per_night']) / (request.budget * 0.4))
            rating_attention = hotel['rating'] / 5.0
            location_attention = max(0, (10 - hotel['distance_from_center']) / 10)
            
            # Combine attention scores
            hotel['attention_score'] = (price_attention * 0.4 + 
                                      rating_attention * 0.4 + 
                                      location_attention * 0.2) * 100
        
        return sorted(hotels, key=lambda x: x['attention_score'], reverse=True)

class AttractionAgent(BaseAgent):
    """Attraction agent - demonstrates Variational Autoencoder concept"""
    
    def __init__(self):
        super().__init__("AttractionAgent", "llama3.2:latest")
    
    async def process(self, travel_request: TravelRequest) -> Dict:
        """Process attractions using VAE-like encoding/decoding with real API calls"""
        
        # Import here to avoid circular imports
        from real_api_integration import api_client
        
        # Encode destination into latent space (categories)
        encoded_preferences = self._encode_destination(travel_request.destination)
        
        # Sample from latent space (generate attraction categories)
        attraction_categories = self._sample_attractions(encoded_preferences)
        
        # NEW: Call local attraction APIs
        api_attractions = await api_client.search_attractions_local(
            travel_request.destination, 
            attraction_categories[0] if attraction_categories else 'tourist_attraction'
        )
        
        # Decode back to actual attractions (combine API + generated)
        generated_attractions = self._decode_attractions(attraction_categories, travel_request)
        
        # Combine API results with VAE-generated attractions
        all_attractions = api_attractions + generated_attractions
        
        # Remove duplicates and limit results
        unique_attractions = []
        seen_names = set()
        for attraction in all_attractions:
            if attraction['name'] not in seen_names:
                unique_attractions.append(attraction)
                seen_names.add(attraction['name'])
        
        self.add_to_memory({'attractions_found': len(unique_attractions), 'api_calls': len(api_attractions)})
        
        return {
            'agent': self.name,
            'attractions': unique_attractions[:8],  # Top 8 results
            'categories': attraction_categories,
            'api_sources': list(set([attr.get('api_source', 'Generated') for attr in unique_attractions])),
            'api_count': len(api_attractions)
        }
    
    def _encode_destination(self, destination: str) -> Dict:
        """VAE encoder - map destination to latent categories"""
        # Simulate encoding destination characteristics
        return {
            'cultural': np.random.uniform(0.3, 0.9),
            'adventure': np.random.uniform(0.2, 0.8),
            'historical': np.random.uniform(0.4, 0.9),
            'natural': np.random.uniform(0.3, 0.8),
            'modern': np.random.uniform(0.2, 0.7)
        }
    
    def _sample_attractions(self, encoded: Dict) -> List[str]:
        """VAE sampling from latent space"""
        categories = []
        for category, score in encoded.items():
            if score > 0.5:  # Threshold for inclusion
                categories.append(category)
        return categories
    
    def _decode_attractions(self, categories: List[str], request: TravelRequest) -> List[Dict]:
        """VAE decoder - generate actual attractions with latent match scores"""
        attraction_templates = {
            'cultural': ['Museum', 'Art Gallery', 'Cultural Center'],
            'adventure': ['Adventure Park', 'Zip Line', 'Rock Climbing'],
            'historical': ['Fort', 'Palace', 'Monument'],
            'natural': ['National Park', 'Beach', 'Waterfall'],
            'modern': ['Shopping Mall', 'Theme Park', 'Observatory']
        }
        
        attractions = []
        for category in categories:
            for template in attraction_templates.get(category, []):
                # Calculate latent match score based on VAE encoding
                latent_match_score = float(round(np.random.uniform(0.75, 0.95) * 100, 1))
                
                attractions.append({
                    'name': f"{request.destination} {template}",
                    'category': category,
                    'price': int(np.random.randint(100, 1500)),
                    'duration': f"{int(np.random.randint(1, 6))} hours",
                    'booking_available': bool(np.random.choice([True, False], p=[0.7, 0.3])),
                    'rating': float(round(np.random.uniform(3.5, 5.0), 1)),
                    'latent_match_score': latent_match_score  # VAE latent space match
                })
        
        return attractions[:8]  # Limit results

class WeatherAgent(BaseAgent):
    """Weather agent - demonstrates Autoencoder pattern"""
    
    def __init__(self):
        super().__init__("WeatherAgent", "deepseek-r1:1.5b")
    
    async def process(self, travel_request: TravelRequest) -> Dict:
        """Process weather data using autoencoder-like compression"""
        
        # Get raw weather data
        raw_weather = await self._fetch_weather_data(travel_request.destination)
        
        # Compress to essential features (encoder)
        compressed_features = self._compress_weather(raw_weather)
        
        # Reconstruct recommendations (decoder)
        recommendations = self._generate_recommendations(compressed_features)
        
        self.add_to_memory({'weather_processed': True})
        
        return {
            'agent': self.name,
            'weather_forecast': compressed_features,
            'recommendations': recommendations
        }
    
    async def _fetch_weather_data(self, destination: str) -> Dict:
        """Simulate weather API call"""
        return {
            'temperature': int(np.random.randint(15, 35)),
            'humidity': int(np.random.randint(40, 90)),
            'precipitation': float(np.random.uniform(0, 20)),
            'wind_speed': float(np.random.uniform(5, 25)),
            'visibility': float(np.random.uniform(5, 15))
        }
    
    def _compress_weather(self, raw_data: Dict) -> Dict:
        """Autoencoder compression - extract key features"""
        comfort_score = (
            (35 - abs(raw_data['temperature'] - 25)) / 35 * 0.4 +
            (100 - raw_data['humidity']) / 100 * 0.3 +
            (20 - raw_data['precipitation']) / 20 * 0.3
        ) * 100
        
        return {
            'comfort_score': round(comfort_score, 1),
            'temperature': raw_data['temperature'],
            'conditions': 'Pleasant' if comfort_score > 70 else 'Moderate' if comfort_score > 50 else 'Challenging'
        }
    
    def _generate_recommendations(self, features: Dict) -> List[str]:
        """Autoencoder decoder - generate recommendations"""
        recommendations = []
        
        if features['comfort_score'] > 80:
            recommendations.append("Perfect weather for outdoor activities!")
        elif features['comfort_score'] > 60:
            recommendations.append("Good weather with minor considerations")
        else:
            recommendations.append("Consider indoor activities or weather protection")
        
        if features['temperature'] > 30:
            recommendations.append("Pack light, breathable clothing")
        elif features['temperature'] < 20:
            recommendations.append("Pack warm clothing")
        
        return recommendations

class CoordinatorAgent(BaseAgent):
    """Main coordinator - demonstrates ensemble learning"""
    
    def __init__(self):
        super().__init__("CoordinatorAgent", "llama3.2:latest")
        self.agents = {
            'flight': FlightAgent(),
            'hotel': HotelAgent(),
            'attraction': AttractionAgent(),
            'weather': WeatherAgent()
        }
    
    async def process(self, travel_request: TravelRequest) -> Dict:
        """Required implementation of abstract method"""
        return await self.orchestrate_booking(travel_request)
    
    async def orchestrate_booking(self, travel_request: TravelRequest) -> Dict:
        """Orchestrate all agents - ensemble approach"""
        
        results = {}
        
        # Run all agents concurrently
        tasks = []
        for agent_name, agent in self.agents.items():
            tasks.append(agent.process(travel_request))
        
        agent_results = await asyncio.gather(*tasks)
        
        # Combine results
        for i, agent_name in enumerate(self.agents.keys()):
            results[agent_name] = agent_results[i]
        
        # Generate final itinerary using ensemble decision
        itinerary = self._create_itinerary(results, travel_request)
        
        return {
            'coordinator': self.name,
            'individual_results': results,
            'final_itinerary': itinerary,
            'total_estimated_cost': self._calculate_total_cost(itinerary)
        }
    
    def _create_itinerary(self, results: Dict, request: TravelRequest) -> Dict:
        """Ensemble decision making"""
        return {
            'flights': results['flight']['flights'][:2],
            'hotels': results['hotel']['hotels'][:2],
            'attractions': results['attraction']['attractions'][:5],
            'weather_info': results['weather']['weather_forecast'],
            'recommendations': results['weather']['recommendations']
        }
    
    def _calculate_total_cost(self, itinerary: Dict) -> float:
        """Calculate total estimated cost"""
        total = 0
        
        if itinerary['flights']:
            total += itinerary['flights'][0]['price'] * 2  # Round trip
        
        if itinerary['hotels']:
            total += itinerary['hotels'][0]['price_per_night'] * 3  # 3 nights
        
        for attraction in itinerary['attractions']:
            if attraction['booking_available']:
                total += attraction['price']
        
        return total

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Travel Agent - Multi-Agent Demo",
        page_icon="✈️",
        layout="wide"
    )
    
    st.title("🤖 AI Travel Agent System")
    st.subtitle("Multi-Agent Architecture Demonstration")
    
    # Sidebar for AI concepts explanation
    with st.sidebar:
        st.header("🧠 AI Concepts Demo")
        
        concept = st.selectbox("Select AI Concept to Learn:", [
            "Feedforward Neural Networks",
            "Convolutional Neural Networks (CNN)",
            "Transformer & Attention",
            "Variational Autoencoders (VAE)",
            "Autoencoders",
            "Ensemble Learning"
        ])
        
        explanations = {
            "Feedforward Neural Networks": "Base agent architecture mimics feedforward networks with input processing, hidden layers (memory), and output generation.",
            "Convolutional Neural Networks (CNN)": "Flight agent uses CNN-like pattern recognition to filter and rank flights based on features like price, duration, and stops.",
            "Transformer & Attention": "Hotel agent implements attention mechanism to focus on relevant features (price, rating, location) when ranking hotels.",
            "Variational Autoencoders (VAE)": "Attraction agent encodes destinations into latent space, samples attraction categories, then decodes to specific attractions.",
            "Autoencoders": "Weather agent compresses raw weather data into essential features, then reconstructs travel recommendations.",
            "Ensemble Learning": "Coordinator combines multiple specialized agents' outputs to make final booking decisions."
        }
        
        st.info(explanations[concept])
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📝 Travel Request")
        
        origin = st.text_input("From", "Mumbai")
        destination = st.text_input("To", "Goa")
        departure_date = st.date_input("Departure Date", datetime.now() + timedelta(days=30))
        return_date = st.date_input("Return Date", datetime.now() + timedelta(days=33))
        budget = st.number_input("Budget (₹)", min_value=5000, max_value=100000, value=25000)
        travelers = st.number_input("Travelers", min_value=1, max_value=10, value=2)
        
        preferences = {}
        preferences['accommodation_type'] = st.selectbox("Accommodation", ["Hotel", "Resort", "Homestay"])
        preferences['activity_level'] = st.selectbox("Activity Level", ["Relaxed", "Moderate", "Adventure"])
        
        if st.button("🚀 Search & Book", type="primary"):
            travel_request = TravelRequest(
                origin=origin,
                destination=destination,
                departure_date=str(departure_date),
                return_date=str(return_date),
                budget=budget,
                travelers=travelers,
                preferences=preferences
            )
            
            # Store in session state
            st.session_state.travel_request = travel_request
            st.session_state.search_initiated = True
    
    with col2:
        if hasattr(st.session_state, 'search_initiated') and st.session_state.search_initiated:
            st.header("🔄 AI Agents Working...")
            
            # Create coordinator and run
            coordinator = CoordinatorAgent()
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate agent work with progress updates
            async def run_booking():
                status_text.text("🛩️ Flight Agent searching...")
                progress_bar.progress(25)
                await asyncio.sleep(1)
                
                status_text.text("🏨 Hotel Agent analyzing...")
                progress_bar.progress(50)
                await asyncio.sleep(1)
                
                status_text.text("🎯 Attraction Agent encoding...")
                progress_bar.progress(75)
                await asyncio.sleep(1)
                
                status_text.text("🌤️ Weather Agent processing...")
                progress_bar.progress(90)
                await asyncio.sleep(1)
                
                results = await coordinator.orchestrate_booking(st.session_state.travel_request)
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                return results
            
            # Run async function
            if 'results' not in st.session_state:
                try:
                    results = asyncio.run(run_booking())
                    st.session_state.results = results
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.results = None
            
            # Display results
            if hasattr(st.session_state, 'results') and st.session_state.results:
                results = st.session_state.results
                
                st.header("📋 Your Personalized Itinerary")
                
                # Cost summary
                st.metric("Total Estimated Cost", f"₹{results['total_estimated_cost']:,.0f}")
                
                # Tabs for different categories
                tab1, tab2, tab3, tab4 = st.tabs(["✈️ Flights", "🏨 Hotels", "🎯 Attractions", "🌤️ Weather"])
                
                with tab1:
                    for flight in results['final_itinerary']['flights']:
                        with st.container():
                            col_a, col_b, col_c = st.columns([2, 1, 1])
                            with col_a:
                                st.write(f"**{flight['airline']} {flight['flight_number']}**")
                                st.write(f"{flight['departure_time']} → {flight['arrival_time']}")
                            with col_b:
                                st.write(f"₹{flight['price']:,}")
                                st.write(f"{flight['duration']}")
                            with col_c:
                                st.write(f"Stops: {flight['stops']}")
                                if st.button(f"Book {flight['flight_number']}", key=f"book_flight_{flight['flight_number']}"):
                                    st.success("Flight booking initiated!")
                
                with tab2:
                    for hotel in results['final_itinerary']['hotels']:
                        with st.container():
                            col_a, col_b, col_c = st.columns([2, 1, 1])
                            with col_a:
                                st.write(f"**{hotel['name']}**")
                                st.write(f"⭐ {hotel['rating']} | {hotel['distance_from_center']}km from center")
                            with col_b:
                                st.write(f"₹{hotel['price_per_night']:,}/night")
                                st.write(f"Amenities: {', '.join(hotel['amenities'][:2])}")
                            with col_c:
                                if st.button(f"Book {hotel['name'][:10]}...", key=f"book_hotel_{hotel['name']}"):
                                    st.success("Hotel booking initiated!")
                
                with tab3:
                    for attraction in results['final_itinerary']['attractions']:
                        with st.container():
                            col_a, col_b, col_c = st.columns([2, 1, 1])
                            with col_a:
                                st.write(f"**{attraction['name']}**")
                                st.write(f"Category: {attraction['category']} | ⭐ {attraction['rating']}")
                            with col_b:
                                st.write(f"₹{attraction['price']}")
                                st.write(f"Duration: {attraction['duration']}")
                            with col_c:
                                if attraction['booking_available']:
                                    if st.button(f"Book Tickets", key=f"book_attraction_{attraction['name']}"):
                                        st.success("Attraction tickets booked!")
                                else:
                                    st.write("🎫 Book on-site")
                
                with tab4:
                    weather = results['final_itinerary']['weather_info']
                    st.metric("Weather Comfort Score", f"{weather['comfort_score']}/100")
                    st.write(f"**Temperature:** {weather['temperature']}°C")
                    st.write(f"**Conditions:** {weather['conditions']}")
                    
                    st.subheader("Recommendations:")
                    for rec in results['final_itinerary']['recommendations']:
                        st.write(f"• {rec}")
                
                # Final booking button
                if st.button("📋 Confirm Complete Itinerary", type="primary"):
                    st.balloons()
                    st.success("🎉 Complete itinerary booked successfully! Confirmation details sent to your email.")

if __name__ == "__main__":
    main()
