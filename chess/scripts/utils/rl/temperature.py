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


class AdaptiveTemperatureController:
    """
    Small corrective controller layered on top of the base temperature schedule.

    It reacts to the previous iteration's self-play/eval telemetry without fully
    replacing the main schedule.
    """

    def __init__(self, config):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("adaptive_temperature_enabled", False))
        self.min_temp = float(cfg.get("adaptive_temperature_min", cfg.get("temperature_end", 0.35)))
        self.max_temp = float(cfg.get("adaptive_temperature_max", cfg.get("temperature_start", 0.65)))
        self.max_adjustment = max(0.0, float(cfg.get("adaptive_temperature_max_adjustment", 0.08)))
        self.draw_target = float(cfg.get("adaptive_temperature_draw_target", 0.26))
        self.draw_band = max(1e-6, float(cfg.get("adaptive_temperature_draw_band", 0.05)))
        self.decisive_target = float(cfg.get("adaptive_temperature_decisive_target", 0.70))
        self.decisive_band = max(1e-6, float(cfg.get("adaptive_temperature_decisive_band", 0.08)))
        self.stagnation_eval_gain = float(cfg.get("adaptive_temperature_min_eval_gain", 0.01))
        self.value_guard_step = max(0.0, float(cfg.get("adaptive_temperature_value_guard_step", 0.01)))
        self.adjustment_smoothing = max(0.0, min(1.0, float(cfg.get("adaptive_temperature_smoothing", 0.50))))
        self.threshold_step = max(1, int(cfg.get("adaptive_temperature_threshold_step", 2)))
        self.last_adjustment = 0.0

        if self.enabled:
            print("Adaptive temperature controller:")
            print(
                f"   temp_range=[{self.min_temp:.2f}, {self.max_temp:.2f}], "
                f"max_adjustment=±{self.max_adjustment:.2f}, draw_target={self.draw_target:.0%}"
            )

    def compute(
        self,
        *,
        base_temp,
        base_threshold,
        prev_draw_rate=None,
        prev_decisive_rate=None,
        last_eval_score_rate=None,
        previous_eval_score_rate=None,
        value_guard_streak=0,
    ):
        if not self.enabled:
            return float(base_temp), int(base_threshold), {
                "enabled": False,
                "adjustment": 0.0,
                "reason": "disabled",
            }

        raw_adjustment = 0.0
        reasons = []

        if prev_draw_rate is not None:
            draw_delta = float(prev_draw_rate) - self.draw_target
            if draw_delta > self.draw_band:
                strength = min(1.0, (draw_delta - self.draw_band) / self.draw_band)
                raw_adjustment -= 0.05 * strength
                reasons.append(f"high_draws={float(prev_draw_rate):.1%}")
            elif draw_delta < -self.draw_band:
                strength = min(1.0, ((-draw_delta) - self.draw_band) / self.draw_band)
                raw_adjustment += 0.025 * strength
                reasons.append(f"low_draws={float(prev_draw_rate):.1%}")

        if prev_decisive_rate is not None:
            decisive_delta = self.decisive_target - float(prev_decisive_rate)
            if decisive_delta > self.decisive_band:
                strength = min(1.0, (decisive_delta - self.decisive_band) / self.decisive_band)
                raw_adjustment -= 0.03 * strength
                reasons.append(f"low_decisive={float(prev_decisive_rate):.1%}")
            elif decisive_delta < -self.decisive_band:
                strength = min(1.0, ((-decisive_delta) - self.decisive_band) / self.decisive_band)
                raw_adjustment += 0.015 * strength
                reasons.append(f"high_decisive={float(prev_decisive_rate):.1%}")

        if last_eval_score_rate is not None and previous_eval_score_rate is not None:
            eval_gain = float(last_eval_score_rate) - float(previous_eval_score_rate)
            if eval_gain < self.stagnation_eval_gain:
                raw_adjustment -= 0.02
                reasons.append(f"eval_stagnation={eval_gain:+.1%}")

        if int(value_guard_streak) > 0:
            raw_adjustment -= self.value_guard_step * float(value_guard_streak)
            reasons.append(f"value_guard={int(value_guard_streak)}")

        raw_adjustment = max(-self.max_adjustment, min(self.max_adjustment, raw_adjustment))
        adjustment = (
            self.adjustment_smoothing * self.last_adjustment
            + (1.0 - self.adjustment_smoothing) * raw_adjustment
        )
        adjustment = max(-self.max_adjustment, min(self.max_adjustment, adjustment))
        self.last_adjustment = adjustment

        adjusted_temp = max(self.min_temp, min(self.max_temp, float(base_temp) + adjustment))
        threshold_offset = int(round((adjustment / max(1e-6, self.max_adjustment)) * self.threshold_step))
        adjusted_threshold = max(0, int(base_threshold) + threshold_offset)

        return adjusted_temp, adjusted_threshold, {
            "enabled": True,
            "adjustment": float(adjustment),
            "raw_adjustment": float(raw_adjustment),
            "reason": ", ".join(reasons) if reasons else "stable",
        }
