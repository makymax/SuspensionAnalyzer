import pandas as pd
import numpy as np
from typing import Dict, Any

def analyze_suspension_data(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze suspension movement data to extract key metrics.
    
    Args:
        data: DataFrame containing suspension tracking data with columns:
             - time: time in seconds
             - distance: distance between dots in mm
             - velocity: rate of change of distance in mm/s
    
    Returns:
        Dictionary containing analysis results
    """
    # Ensure the DataFrame has the necessary columns
    required_columns = ['time', 'distance', 'velocity']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Data must contain columns: {required_columns}")
    
    # Calculate basic statistics
    avg_travel = data['distance'].mean()
    max_travel = data['distance'].max()
    min_travel = data['distance'].min()
    travel_range = max_travel - min_travel
    
    # Calculate time at which maximum compression occurs
    max_compression_time = data.loc[data['distance'].idxmin(), 'time']
    
    # Calculate speeds (absolute values)
    compression_speeds = data.loc[data['velocity'] < 0, 'velocity'].abs()
    rebound_speeds = data.loc[data['velocity'] > 0, 'velocity']
    
    avg_compression_speed = compression_speeds.mean() if not compression_speeds.empty else 0
    avg_rebound_speed = rebound_speeds.mean() if not rebound_speeds.empty else 0
    max_compression_speed = compression_speeds.max() if not compression_speeds.empty else 0
    max_rebound_speed = rebound_speeds.max() if not rebound_speeds.empty else 0
    
    # Calculate compression/rebound ratio
    if avg_rebound_speed > 0:
        comp_rebound_ratio = avg_compression_speed / avg_rebound_speed
    else:
        comp_rebound_ratio = float('inf')
    
    # Calculate frequency of oscillation (approximate)
    # Find zero-crossings in velocity to identify direction changes
    zero_crossings = np.where(np.diff(np.signbit(data['velocity'])))[0]
    if len(zero_crossings) >= 2:
        avg_time_between_crossings = np.mean(np.diff(data['time'].iloc[zero_crossings]))
        oscillation_frequency = 1 / (2 * avg_time_between_crossings)  # Full cycle requires 2 crossings
    else:
        oscillation_frequency = 0
    
    # Calculate damping ratio (approximate)
    # Using logarithmic decrement method on local maxima
    local_maxima_idx = find_local_maxima(data['distance'])
    if len(local_maxima_idx) >= 2:
        local_maxima = data['distance'].iloc[local_maxima_idx].values
        damping_ratios = []
        
        for i in range(len(local_maxima) - 1):
            if local_maxima[i] > local_maxima[i+1] and local_maxima[i+1] > 0:
                # Calculate logarithmic decrement
                log_dec = np.log(local_maxima[i] / local_maxima[i+1])
                # Convert to damping ratio
                damping_ratio = log_dec / np.sqrt(4 * np.pi**2 + log_dec**2)
                damping_ratios.append(damping_ratio)
        
        damping_ratio = np.mean(damping_ratios) if damping_ratios else 0
    else:
        damping_ratio = 0
    
    # Return analysis results
    return {
        'avg_travel': avg_travel,
        'max_travel': max_travel,
        'min_travel': min_travel,
        'travel_range': travel_range,
        'max_compression_time': max_compression_time,
        'avg_compression_speed': avg_compression_speed,
        'avg_rebound_speed': avg_rebound_speed,
        'max_compression_speed': max_compression_speed,
        'max_rebound_speed': max_rebound_speed,
        'comp_rebound_ratio': comp_rebound_ratio,
        'oscillation_frequency': oscillation_frequency,
        'damping_ratio': damping_ratio
    }

def find_local_maxima(series: pd.Series) -> np.ndarray:
    """
    Find indices of local maxima in a time series.
    
    Args:
        series: Pandas Series containing the data
        
    Returns:
        Array of indices corresponding to local maxima
    """
    # Convert to numpy array for easier manipulation
    data = series.values
    
    # First and last points are not considered
    return np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1
