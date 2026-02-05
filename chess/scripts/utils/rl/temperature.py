"""
Temperature schedule for exploration control
"""


class TemperatureSchedule:
    """
    Adaptive temperature schedule for exploration
    
    Strategy:
    - Start with HIGH temperature → more exploration
    - Gradually DECREASE → more exploitation
    - Helps model learn diverse strategies early, then refine
    
    Benefits:
    - Better exploration-exploitation balance
    - Faster convergence
    - More robust policies
    """
    
    def __init__(self, start_temp, end_temp, decay_iterations):
        """
        Args:
            start_temp: Initial temperature (high exploration)
            end_temp: Final temperature (low exploration)
            decay_iterations: Number of iterations to decay over
        """
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.decay_iterations = decay_iterations
        
        print(f"🌡️ Temperature Schedule:")
        print(f"   Start: {start_temp} (high exploration)")
        print(f"   End: {end_temp} (low exploration)")
        print(f"   Decay over: {decay_iterations} iterations")
    
    def get_temperature(self, iteration):
        """
        Get temperature for current iteration
        
        Args:
            iteration: Current iteration number
        
        Returns:
            Current temperature value
        """
        if iteration >= self.decay_iterations:
            return self.end_temp
        
        # Linear decay
        progress = iteration / self.decay_iterations
        temp = self.start_temp + (self.end_temp - self.start_temp) * progress
        
        return temp