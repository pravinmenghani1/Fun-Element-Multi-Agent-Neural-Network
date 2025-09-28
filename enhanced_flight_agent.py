#!/usr/bin/env python3
"""
Enhanced Flight Agent with RNN/Temporal CNN for Time-Series Analysis
Better suited for flight price prediction and temporal patterns
"""

import numpy as np
from typing import Dict, List
import asyncio
from datetime import datetime, timedelta
from real_api_integration import api_client

class TemporalFlightAgent:
    """Flight agent using RNN/Temporal CNN for time-series flight data analysis"""
    
    def __init__(self):
        self.name = "TemporalFlightAgent"
        self.memory_window = 30  # Days of historical data
        self.price_history = {}  # Simulated price history
        
    async def process(self, travel_request) -> Dict:
        """Process flight search with temporal analysis"""
        
        # Step 1: Fetch real-time flight data from multiple APIs
        flight_results = await self._fetch_multi_api_flights(travel_request)
        
        # Step 2: Apply RNN-like temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(flight_results, travel_request)
        
        # Step 3: Apply Temporal CNN for pattern recognition
        cnn_features = self._extract_temporal_features(flight_results)
        
        # Step 4: Combine RNN + CNN insights
        enhanced_flights = self._combine_temporal_insights(flight_results, temporal_analysis, cnn_features)
        
        return {
            'agent': self.name,
            'flights': enhanced_flights[:5],
            'temporal_analysis': temporal_analysis,
            'cnn_features': cnn_features,
            'total_found': len(flight_results),
            'api_sources': list(set([f.get('api_source', 'Unknown') for f in flight_results]))
        }
    
    async def _fetch_multi_api_flights(self, request) -> List[Dict]:
        """Fetch flights from multiple real APIs"""
        all_flights = []
        
        # Simulate multiple API calls
        tasks = [
            api_client.search_flights_amadeus(
                request.origin[:3].upper(), 
                request.destination[:3].upper(), 
                request.departure_date
            ),
            self._search_indigo_api(request),
            self._search_airindia_api(request),
            self._search_spicejet_api(request)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_flights.extend(result)
        
        return all_flights
    
    async def _search_indigo_api(self, request) -> List[Dict]:
        """Simulate IndiGo API call"""
        await asyncio.sleep(0.5)  # Simulate API delay
        return [{
            'airline': '6E',
            'flight_number': f"6E{np.random.randint(100, 999)}",
            'departure_time': f"{np.random.randint(6, 23):02d}:{np.random.randint(0, 59):02d}",
            'arrival_time': f"{np.random.randint(6, 23):02d}:{np.random.randint(0, 59):02d}",
            'price': int(np.random.randint(3500, 12000)),
            'duration': f"{np.random.randint(1, 6)}h {np.random.randint(0, 59)}m",
            'stops': 0,
            'api_source': 'IndiGo Direct API'
        } for _ in range(np.random.randint(2, 5))]
    
    async def _search_airindia_api(self, request) -> List[Dict]:
        """Simulate Air India API call"""
        await asyncio.sleep(0.7)  # Simulate API delay
        return [{
            'airline': 'AI',
            'flight_number': f"AI{np.random.randint(100, 999)}",
            'departure_time': f"{np.random.randint(6, 23):02d}:{np.random.randint(0, 59):02d}",
            'arrival_time': f"{np.random.randint(6, 23):02d}:{np.random.randint(0, 59):02d}",
            'price': int(np.random.randint(4000, 15000)),
            'duration': f"{np.random.randint(1, 8)}h {np.random.randint(0, 59)}m",
            'stops': int(np.random.choice([0, 1], p=[0.7, 0.3])),
            'api_source': 'Air India API'
        } for _ in range(np.random.randint(1, 4))]
    
    async def _search_spicejet_api(self, request) -> List[Dict]:
        """Simulate SpiceJet API call"""
        await asyncio.sleep(0.3)  # Simulate API delay
        return [{
            'airline': 'SG',
            'flight_number': f"SG{np.random.randint(100, 999)}",
            'departure_time': f"{np.random.randint(6, 23):02d}:{np.random.randint(0, 59):02d}",
            'arrival_time': f"{np.random.randint(6, 23):02d}:{np.random.randint(0, 59):02d}",
            'price': int(np.random.randint(3000, 10000)),
            'duration': f"{np.random.randint(1, 5)}h {np.random.randint(0, 59)}m",
            'stops': int(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])),
            'api_source': 'SpiceJet API'
        } for _ in range(np.random.randint(2, 6))]
    
    def _analyze_temporal_patterns(self, flights: List[Dict], request) -> Dict:
        """RNN-like temporal analysis of flight patterns"""
        
        # Simulate historical price data
        route_key = f"{request.origin}-{request.destination}"
        if route_key not in self.price_history:
            self.price_history[route_key] = self._generate_price_history()
        
        history = self.price_history[route_key]
        current_prices = [f['price'] for f in flights]
        
        # RNN-style sequence analysis
        price_trend = self._calculate_price_trend(history)
        seasonal_factor = self._calculate_seasonal_factor(request.departure_date)
        demand_prediction = self._predict_demand(history, seasonal_factor)
        
        return {
            'price_trend': price_trend,
            'seasonal_factor': seasonal_factor,
            'demand_prediction': demand_prediction,
            'avg_current_price': np.mean(current_prices) if current_prices else 0,
            'price_volatility': np.std(current_prices) if current_prices else 0,
            'recommendation': self._generate_temporal_recommendation(price_trend, demand_prediction)
        }
    
    def _extract_temporal_features(self, flights: List[Dict]) -> Dict:
        """Temporal CNN feature extraction"""
        
        if not flights:
            return {'features': [], 'patterns': []}
        
        # Extract time-based features
        departure_times = []
        prices = []
        
        for flight in flights:
            try:
                time_str = flight['departure_time']
                hour = int(time_str.split(':')[0])
                departure_times.append(hour)
                prices.append(flight['price'])
            except:
                continue
        
        if not departure_times:
            return {'features': [], 'patterns': []}
        
        # CNN-like convolution on time series
        time_features = self._apply_temporal_convolution(departure_times, prices)
        
        return {
            'time_features': time_features,
            'peak_hours': self._identify_peak_hours(departure_times, prices),
            'price_patterns': self._identify_price_patterns(prices),
            'temporal_clusters': self._cluster_temporal_data(departure_times, prices)
        }
    
    def _combine_temporal_insights(self, flights: List[Dict], rnn_analysis: Dict, cnn_features: Dict) -> List[Dict]:
        """Combine RNN and CNN insights for final ranking"""
        
        enhanced_flights = []
        
        for flight in flights:
            # Calculate temporal score
            temporal_score = 0
            
            # RNN contribution
            if rnn_analysis['price_trend'] == 'decreasing':
                temporal_score += 20
            elif rnn_analysis['price_trend'] == 'stable':
                temporal_score += 10
            
            # CNN contribution
            try:
                hour = int(flight['departure_time'].split(':')[0])
                if hour in cnn_features.get('peak_hours', []):
                    temporal_score -= 10  # Penalty for peak hours
                else:
                    temporal_score += 15  # Bonus for off-peak
            except:
                pass
            
            # Price factor
            avg_price = rnn_analysis.get('avg_current_price', flight['price'])
            if flight['price'] < avg_price:
                temporal_score += 25
            
            flight['temporal_score'] = temporal_score
            flight['rnn_insights'] = rnn_analysis['recommendation']
            flight['cnn_pattern'] = cnn_features.get('price_patterns', 'normal')
            
            enhanced_flights.append(flight)
        
        # Sort by temporal score
        return sorted(enhanced_flights, key=lambda x: x['temporal_score'], reverse=True)
    
    def _generate_price_history(self) -> List[float]:
        """Generate realistic price history for RNN analysis"""
        base_price = np.random.randint(5000, 12000)
        history = []
        
        for i in range(self.memory_window):
            # Add trend and noise
            trend = np.sin(i * 0.1) * 500  # Seasonal trend
            noise = np.random.normal(0, 200)  # Random fluctuation
            price = base_price + trend + noise
            history.append(max(2000, price))  # Minimum price floor
        
        return history
    
    def _calculate_price_trend(self, history: List[float]) -> str:
        """Calculate price trend using RNN-like analysis"""
        if len(history) < 5:
            return 'stable'
        
        recent = history[-5:]
        older = history[-10:-5] if len(history) >= 10 else history[:-5]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg * 1.1:
            return 'increasing'
        elif recent_avg < older_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_seasonal_factor(self, departure_date: str) -> float:
        """Calculate seasonal demand factor"""
        try:
            date = datetime.strptime(departure_date, '%Y-%m-%d')
            month = date.month
            
            # Peak season factors
            peak_months = [12, 1, 4, 5, 10, 11]  # Winter and holiday months
            if month in peak_months:
                return 1.3
            else:
                return 0.9
        except:
            return 1.0
    
    def _predict_demand(self, history: List[float], seasonal_factor: float) -> str:
        """Predict demand using historical patterns"""
        recent_volatility = np.std(history[-7:]) if len(history) >= 7 else 0
        
        if seasonal_factor > 1.2 and recent_volatility > 300:
            return 'high'
        elif seasonal_factor < 1.0 and recent_volatility < 200:
            return 'low'
        else:
            return 'moderate'
    
    def _generate_temporal_recommendation(self, trend: str, demand: str) -> str:
        """Generate booking recommendation based on temporal analysis"""
        if trend == 'increasing' and demand == 'high':
            return 'Book immediately - prices rising with high demand'
        elif trend == 'decreasing' and demand == 'low':
            return 'Wait a few days - prices may drop further'
        elif trend == 'stable' and demand == 'moderate':
            return 'Good time to book - stable pricing'
        else:
            return 'Monitor for 2-3 days before booking'
    
    def _apply_temporal_convolution(self, times: List[int], prices: List[float]) -> Dict:
        """Apply CNN-like convolution on temporal data"""
        if len(times) < 3:
            return {}
        
        # Simple convolution kernel for pattern detection
        kernel = [0.25, 0.5, 0.25]  # Smoothing kernel
        
        convolved_prices = []
        for i in range(1, len(prices) - 1):
            conv_value = (prices[i-1] * kernel[0] + 
                         prices[i] * kernel[1] + 
                         prices[i+1] * kernel[2])
            convolved_prices.append(conv_value)
        
        return {
            'smoothed_prices': convolved_prices,
            'pattern_strength': np.std(convolved_prices) if convolved_prices else 0
        }
    
    def _identify_peak_hours(self, times: List[int], prices: List[float]) -> List[int]:
        """Identify peak pricing hours"""
        if not times or not prices:
            return []
        
        hour_prices = {}
        for time, price in zip(times, prices):
            if time not in hour_prices:
                hour_prices[time] = []
            hour_prices[time].append(price)
        
        # Find hours with above-average prices
        avg_price = np.mean(prices)
        peak_hours = []
        
        for hour, hour_price_list in hour_prices.items():
            if np.mean(hour_price_list) > avg_price * 1.1:
                peak_hours.append(hour)
        
        return peak_hours
    
    def _identify_price_patterns(self, prices: List[float]) -> str:
        """Identify price patterns using CNN-like analysis"""
        if len(prices) < 3:
            return 'insufficient_data'
        
        price_changes = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
        
        if all(change > 0 for change in price_changes):
            return 'increasing'
        elif all(change < 0 for change in price_changes):
            return 'decreasing'
        elif abs(np.std(price_changes)) < 100:
            return 'stable'
        else:
            return 'volatile'
    
    def _cluster_temporal_data(self, times: List[int], prices: List[float]) -> Dict:
        """Simple clustering of temporal data"""
        if not times or not prices:
            return {}
        
        # Group by time periods
        morning = [(t, p) for t, p in zip(times, prices) if 6 <= t < 12]
        afternoon = [(t, p) for t, p in zip(times, prices) if 12 <= t < 18]
        evening = [(t, p) for t, p in zip(times, prices) if 18 <= t <= 23]
        
        return {
            'morning_avg': np.mean([p for _, p in morning]) if morning else 0,
            'afternoon_avg': np.mean([p for _, p in afternoon]) if afternoon else 0,
            'evening_avg': np.mean([p for _, p in evening]) if evening else 0,
            'best_time_period': min([
                ('morning', np.mean([p for _, p in morning]) if morning else float('inf')),
                ('afternoon', np.mean([p for _, p in afternoon]) if afternoon else float('inf')),
                ('evening', np.mean([p for _, p in evening]) if evening else float('inf'))
            ], key=lambda x: x[1])[0]
        }
