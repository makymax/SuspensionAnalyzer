from typing import Dict, Any

def get_suspension_recommendations(analysis_results: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Generate suspension setting recommendations based on analysis results.
    
    Args:
        analysis_results: Dictionary containing suspension analysis metrics
        
    Returns:
        Dictionary of recommendations categorized by suspension component
    """
    recommendations = {}
    
    # Damping ratio interpretation
    damping_ratio = analysis_results.get('damping_ratio', 0)
    
    # Compression/rebound ratio
    comp_rebound_ratio = analysis_results.get('comp_rebound_ratio', 1.0)
    
    # Average speeds
    avg_compression_speed = analysis_results.get('avg_compression_speed', 0)
    avg_rebound_speed = analysis_results.get('avg_rebound_speed', 0)
    
    # Travel range utilization (assumes 100mm is typical travel)
    travel_range = analysis_results.get('travel_range', 0)
    travel_utilization = travel_range / 100.0  # Normalized to percentage of typical travel
    
    # Generate recommendation for compression damping
    recommendations['Compression Damping'] = {}
    
    if damping_ratio < 0.3:
        recommendations['Compression Damping']['analysis'] = "Under-damped suspension with excessive oscillation"
        recommendations['Compression Damping']['recommendation'] = "Increase compression damping by 2-3 clicks"
        recommendations['Compression Damping']['improvement'] = "Reduced oscillation and improved stability after impacts"
    elif damping_ratio > 0.7:
        recommendations['Compression Damping']['analysis'] = "Over-damped compression with limited responsiveness"
        recommendations['Compression Damping']['recommendation'] = "Decrease compression damping by 2-3 clicks"
        recommendations['Compression Damping']['improvement'] = "Improved bump absorption and wheel tracking"
    else:
        recommendations['Compression Damping']['analysis'] = "Compression damping is within optimal range"
        recommendations['Compression Damping']['recommendation'] = "Maintain current compression settings"
        recommendations['Compression Damping']['improvement'] = "No changes needed; current setup provides good control"
    
    # Generate recommendation for rebound damping
    recommendations['Rebound Damping'] = {}
    
    if comp_rebound_ratio < 0.7:
        recommendations['Rebound Damping']['analysis'] = "Rebound is relatively slow compared to compression"
        recommendations['Rebound Damping']['recommendation'] = "Decrease rebound damping by 2-3 clicks"
        recommendations['Rebound Damping']['improvement'] = "Quicker recovery from compression, improved successive bump handling"
    elif comp_rebound_ratio > 1.5:
        recommendations['Rebound Damping']['analysis'] = "Rebound is too quick compared to compression"
        recommendations['Rebound Damping']['recommendation'] = "Increase rebound damping by 1-2 clicks"
        recommendations['Rebound Damping']['improvement'] = "Reduced pogo effect and more controlled recovery"
    else:
        recommendations['Rebound Damping']['analysis'] = "Rebound damping is balanced with compression"
        recommendations['Rebound Damping']['recommendation'] = "Maintain current rebound settings"
        recommendations['Rebound Damping']['improvement'] = "Current setup provides good balance between recovery and control"
    
    # Generate recommendation for spring rate / preload
    recommendations['Spring Rate / Preload'] = {}
    
    if travel_utilization < 0.6:
        recommendations['Spring Rate / Preload']['analysis'] = "Suspension travel is under-utilized (too stiff)"
        recommendations['Spring Rate / Preload']['recommendation'] = "Decrease spring preload by 2-3 turns or consider softer springs"
        recommendations['Spring Rate / Preload']['improvement'] = "Better use of available travel and improved comfort"
    elif travel_utilization > 0.9:
        recommendations['Spring Rate / Preload']['analysis'] = "Suspension is using almost all available travel (too soft)"
        recommendations['Spring Rate / Preload']['recommendation'] = "Increase spring preload by 2-3 turns or consider stiffer springs"
        recommendations['Spring Rate / Preload']['improvement'] = "Reduced bottoming out and improved handling on larger impacts"
    else:
        recommendations['Spring Rate / Preload']['analysis'] = "Spring rate is appropriate for the conditions"
        recommendations['Spring Rate / Preload']['recommendation'] = "Maintain current spring settings"
        recommendations['Spring Rate / Preload']['improvement'] = "Current setup provides good travel utilization"
    
    # Generate recommendation for high-speed compression (if available on the shock)
    recommendations['High-Speed Compression'] = {}
    
    if avg_compression_speed > 150:  # Fast compression speeds
        recommendations['High-Speed Compression']['analysis'] = "High compression speeds detected"
        recommendations['High-Speed Compression']['recommendation'] = "If adjustable, increase high-speed compression damping by 1-2 clicks"
        recommendations['High-Speed Compression']['improvement'] = "Better control during fast impacts while maintaining small bump compliance"
    else:
        recommendations['High-Speed Compression']['analysis'] = "Moderate compression speeds detected"
        recommendations['High-Speed Compression']['recommendation'] = "Current high-speed compression setting appears appropriate"
        recommendations['High-Speed Compression']['improvement'] = "No changes needed for current riding conditions"
    
    return recommendations
