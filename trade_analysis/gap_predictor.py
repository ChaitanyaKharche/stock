# Add this to your main workflow file or create a new gap_predictor.py
import pandas as pd
import numpy as np


class GapSpecificPredictor:
    """
    Specialized predictor for 3:30-4pm gap trading strategy.
    """
    
    def __init__(self):
        self.gap_thresholds = {
            'significant': 0.5,  # 0.5% gap considered significant
            'major': 1.0,        # 1.0% gap considered major
            'extreme': 2.0       # 2.0% gap considered extreme
        }
        
    def analyze_closing_auction_setup(self, intraday_data: pd.DataFrame, daily_data: pd.DataFrame) -> dict:
        """
        Analyze the setup for potential next-day gap based on closing auction dynamics.
        """
        if intraday_data.empty or daily_data.empty:
            return {"gap_probability": 0.0, "expected_direction": "FLAT", "confidence": "LOW"}
        
        # Get last hour data (3pm-4pm)
        last_hour = intraday_data.tail(4)  # Assuming 15-min intervals
        
        # Calculate closing momentum indicators
        closing_analysis = self._analyze_closing_momentum(last_hour)
        
        # Analyze volume profile
        volume_analysis = self._analyze_volume_profile(last_hour, intraday_data)
        
        # Check for institutional flow patterns
        institutional_flow = self._detect_institutional_activity(last_hour)
        
        # Analyze daily setup
        daily_setup = self._analyze_daily_context(daily_data)
        
        # Combine all factors for gap prediction
        gap_prediction = self._calculate_gap_probability(
            closing_analysis, volume_analysis, institutional_flow, daily_setup
        )
        
        return gap_prediction
    
    def _analyze_closing_momentum(self, last_hour_data: pd.DataFrame) -> dict:
        """Analyze momentum in the last hour of trading."""
        if len(last_hour_data) < 2:
            return {"momentum_score": 0.0, "direction": "FLAT"}
        
        # Calculate price momentum
        first_price = last_hour_data['close'].iloc[0]
        last_price = last_hour_data['close'].iloc[-1]
        price_change = (last_price - first_price) / first_price * 100
        
        # Calculate volume-weighted momentum
        vwap_last_hour = np.average(last_hour_data['close'], weights=last_hour_data['volume'])
        current_vs_vwap = (last_price - vwap_last_hour) / vwap_last_hour * 100
        
        # Momentum score combines price change and VWAP position
        momentum_score = (price_change * 0.7) + (current_vs_vwap * 0.3)
        
        direction = "UP" if momentum_score > 0.1 else ("DOWN" if momentum_score < -0.1 else "FLAT")
        
        return {
            "momentum_score": momentum_score,
            "direction": direction,
            "price_change_pct": price_change,
            "vwap_deviation": current_vs_vwap
        }
    
    def _analyze_volume_profile(self, last_hour_data: pd.DataFrame, full_day_data: pd.DataFrame) -> dict:
        """Analyze volume patterns for gap prediction."""
        if last_hour_data.empty or full_day_data.empty:
            return {"volume_score": 0.0, "unusual_activity": False}
        
        # Calculate average volume for comparison
        avg_volume_last_hour = last_hour_data['volume'].mean()
        avg_volume_full_day = full_day_data['volume'].mean()
        
        # Volume spike detection
        volume_ratio = avg_volume_last_hour / max(avg_volume_full_day, 1)
        
        # Check for unusual activity (volume > 150% of daily average)
        unusual_activity = volume_ratio > 1.5
        
        # Volume trend in last hour
        volume_trend = np.polyfit(range(len(last_hour_data)), last_hour_data['volume'], 1)[0]
        
        # Calculate volume score
        volume_score = min(volume_ratio * 2, 5.0)  # Cap at 5.0
        if volume_trend > 0:
            volume_score *= 1.2  # Boost if volume is increasing
        
        return {
            "volume_score": volume_score,
            "volume_ratio": volume_ratio,
            "unusual_activity": unusual_activity,
            "volume_trend": "INCREASING" if volume_trend > 0 else "DECREASING"
        }
    
    def _detect_institutional_activity(self, last_hour_data: pd.DataFrame) -> dict:
        """Detect potential institutional flow in closing auction."""
        if len(last_hour_data) < 2:
            return {"institutional_score": 0.0, "flow_direction": "NEUTRAL"}
        
        # Large volume bars with small price impact suggest institutional activity
        volume_price_efficiency = []
        
        for i in range(1, len(last_hour_data)):
            volume = last_hour_data['volume'].iloc[i]
            price_change = abs(last_hour_data['close'].iloc[i] - last_hour_data['close'].iloc[i-1])
            
            if price_change > 0:
                efficiency = volume / price_change
                volume_price_efficiency.append(efficiency)
        
        if volume_price_efficiency:
            avg_efficiency = np.mean(volume_price_efficiency)
            # High efficiency (high volume, low price impact) suggests institutional flow
            institutional_score = min(avg_efficiency / 1000000, 3.0)  # Normalize and cap
        else:
            institutional_score = 0.0
        
        # Determine flow direction based on price trend
        price_trend = last_hour_data['close'].iloc[-1] - last_hour_data['close'].iloc[0]
        flow_direction = "BUYING" if price_trend > 0 else ("SELLING" if price_trend < 0 else "NEUTRAL")
        
        return {
            "institutional_score": institutional_score,
            "flow_direction": flow_direction,
            "volume_efficiency": avg_efficiency if volume_price_efficiency else 0
        }
    
    def _analyze_daily_context(self, daily_data: pd.DataFrame) -> dict:
        """Analyze daily context for gap prediction."""
        if len(daily_data) < 5:
            return {"daily_score": 0.0, "trend": "NEUTRAL"}
        
        # Recent trend analysis (5-day)
        recent_closes = daily_data['close'].tail(5)
        trend_slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]
        
        # Volatility analysis
        recent_returns = recent_closes.pct_change().dropna()
        volatility = recent_returns.std()
        
        # Support/Resistance levels
        recent_highs = daily_data['high'].tail(10)
        recent_lows = daily_data['low'].tail(10)
        
        current_price = daily_data['close'].iloc[-1]
        resistance_level = recent_highs.max()
        support_level = recent_lows.min()
        
        # Position relative to S/R
        resistance_distance = (resistance_level - current_price) / current_price * 100
        support_distance = (current_price - support_level) / current_price * 100
        
        # Daily score based on trend and position
        daily_score = 0.0
        if trend_slope > 0 and resistance_distance > 2:  # Uptrend with room to run
            daily_score += 2.0
        elif trend_slope < 0 and support_distance > 2:  # Downtrend with room to fall
            daily_score -= 2.0
        
        # Adjust for volatility
        if volatility > 0.02:  # High volatility increases gap probability
            daily_score *= 1.3
        
        trend = "BULLISH" if trend_slope > 0 else ("BEARISH" if trend_slope < 0 else "NEUTRAL")
        
        return {
            "daily_score": daily_score,
            "trend": trend,
            "volatility": volatility,
            "resistance_distance": resistance_distance,
            "support_distance": support_distance
        }
    
    def _calculate_gap_probability(self, closing_analysis: dict, volume_analysis: dict, 
                                 institutional_flow: dict, daily_setup: dict) -> dict:
        """Calculate overall gap probability and direction."""
        
        # Weighted scoring system
        momentum_weight = 0.3
        volume_weight = 0.25
        institutional_weight = 0.25
        daily_weight = 0.2
        
        # Calculate weighted score
        total_score = (
            closing_analysis["momentum_score"] * momentum_weight +
            volume_analysis["volume_score"] * volume_weight +
            institutional_flow["institutional_score"] * institutional_weight +
            abs(daily_setup["daily_score"]) * daily_weight
        )
        
        # Determine direction
        direction_indicators = [
            closing_analysis["direction"],
            "UP" if volume_analysis["volume_ratio"] > 1.2 else "DOWN",
            institutional_flow["flow_direction"].replace("BUYING", "UP").replace("SELLING", "DOWN"),
            daily_setup["trend"].replace("BULLISH", "UP").replace("BEARISH", "DOWN")
        ]
        
        up_votes = sum(1 for d in direction_indicators if d == "UP")
        down_votes = sum(1 for d in direction_indicators if d == "DOWN")
        
        if up_votes > down_votes:
            expected_direction = "UP"
        elif down_votes > up_votes:
            expected_direction = "DOWN"
        else:
            expected_direction = "FLAT"
        
        # Calculate gap probability (0-100%)
        gap_probability = min(total_score * 10, 100)  # Scale to percentage
        
        # Determine confidence level
        if gap_probability > 70 and abs(up_votes - down_votes) >= 2:
            confidence = "HIGH"
        elif gap_probability > 50 and abs(up_votes - down_votes) >= 1:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return {
            "gap_probability": round(gap_probability, 1),
            "expected_direction": expected_direction,
            "confidence": confidence,
            "total_score": round(total_score, 2),
            "component_scores": {
                "momentum": round(closing_analysis["momentum_score"], 2),
                "volume": round(volume_analysis["volume_score"], 2),
                "institutional": round(institutional_flow["institutional_score"], 2),
                "daily_context": round(daily_setup["daily_score"], 2)
            },
            "direction_consensus": f"{up_votes} UP, {down_votes} DOWN"
        }

# Integration function for your main workflow
def integrate_gap_specific_prediction(symbol: str, multi_tf_data: dict) -> dict:
    """
    Integrate gap-specific prediction into your main workflow.
    """
    gap_predictor = GapSpecificPredictor()
    
    # Get required data
    daily_data = multi_tf_data.get("daily", pd.DataFrame())
    intraday_data = multi_tf_data.get("15m", pd.DataFrame())
    
    if daily_data.empty:
        return {"status": "failed", "reason": "No daily data available"}
    
    # If no intraday data, use daily for basic analysis
    if intraday_data.empty:
        intraday_data = daily_data.tail(4)  # Use last 4 daily bars as proxy
    
    # Perform gap-specific analysis
    gap_analysis = gap_predictor.analyze_closing_auction_setup(intraday_data, daily_data)
    
    return {
        "status": "success",
        "symbol": symbol,
        "gap_prediction": gap_analysis,
        "trading_recommendation": {
            "action": "BUY" if gap_analysis["expected_direction"] == "UP" and gap_analysis["gap_probability"] > 60
                     else "SELL" if gap_analysis["expected_direction"] == "DOWN" and gap_analysis["gap_probability"] > 60
                     else "HOLD",
            "entry_time": "3:30-4:00 PM",
            "exit_strategy": "Next day market open + 30 minutes",
            "risk_level": gap_analysis["confidence"]
        }
    }
