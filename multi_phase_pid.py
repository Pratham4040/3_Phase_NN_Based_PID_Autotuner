"""
MULTI-PHASE PID CONTROLLER

Works with MultiPhaseAutotuner to provide phase-specific control.
Key features:
- Automatically switches PID gains based on temperature
- Resets integral when switching phases (bumpless transfer)
- Different anti-windup limits per phase
- Phase-aware derivative filtering
"""

import time


class MultiPhasePID:
    """
    PID controller with automatic gain scheduling
    Integrates with MultiPhaseAutotuner for automatic tuning
    """
    
    def __init__(self, autotuner, dt=0.5, verbose=True):
        """
        Args:
            autotuner: MultiPhaseAutotuner instance
            dt: Control loop time step
            verbose: Enable detailed logging
        """
        self.autotuner = autotuner
        self.setpoint = autotuner.setpoint
        self.dt = dt
        self.verbose = verbose
        
        # Current state
        self.current_phase = None
        self.Kp = 0
        self.Ki = 0
        self.Kd = 0
        
        # PID state variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_filtered_derivative = 0.0
        
        # Phase-specific anti-windup limits
        self.integral_limits = {
            "ramp_up": 20.0,
            "approach": 10.0,
            "steady_state": 5.0,
        }
        
        # Derivative filter (α value per phase)
        self.derivative_filter_alpha = {
            "ramp_up": 0.5,      # Less filtering (faster response)
            "approach": 0.3,      # Moderate filtering
            "steady_state": 0.2,  # More filtering (noise rejection)
        }
        
        # Statistics
        self.step_count = 0
        self.phase_switches = 0
        self.last_phase_switch_time = time.time()
        
        print("\n[MultiPhasePID] Initialized")
        print(f"  Setpoint: {self.setpoint:.1f}°C")
        print(f"  Control period: {dt:.2f}s")
    
    def _update_gains_for_phase(self, new_phase, temp):
        """
        Update PID gains when phase changes
        Implements bumpless transfer
        """
        if new_phase == self.current_phase:
            return  # No change
        
        old_phase = self.current_phase
        self.current_phase = new_phase
        
        # Get new gains from autotuner
        gains = self.autotuner.phase_pids[new_phase]
        self.Kp = gains['Kp']
        self.Ki = gains['Ki']
        self.Kd = gains['Kd']
        
        # Bumpless transfer: Reset integral and derivative
        old_integral = self.integral
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_filtered_derivative = 0.0
        
        self.phase_switches += 1
        current_time = time.time()
        time_in_phase = current_time - self.last_phase_switch_time
        self.last_phase_switch_time = current_time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"PHASE TRANSITION: {old_phase.value if old_phase else 'INITIAL'} → {new_phase.value}")
            print(f"{'='*70}")
            print(f"  Time in previous phase: {time_in_phase:.1f}s")
            print(f"  Current temperature: {temp:.3f}°C")
            print(f"  Error: {self.setpoint - temp:+.3f}°C")
            print(f"  New gains: Kp={self.Kp:.3f}, Ki={self.Ki:.3f}, Kd={self.Kd:.3f}")
            print(f"  Integral reset: {old_integral:.3f} → 0.0")
            print(f"{'='*70}\n")
    
    def compute(self, temp):
        """
        Compute PID control output
        
        Args:
            temp: Current temperature
            
        Returns:
            Control output (heater power, 0-1)
        """
        self.step_count += 1
        
        # Determine phase and update gains if needed
        phase = self.autotuner.determine_phase(temp)
        self._update_gains_for_phase(phase, temp)
        
        # Compute error
        error = self.setpoint - temp
        
        # === PROPORTIONAL TERM ===
        p_term = self.Kp * error
        
        # === INTEGRAL TERM with Phase-Specific Anti-Windup ===
        self.integral += error * self.dt
        
        # Get integral limit for current phase
        integral_limit = self.integral_limits[phase.value]
        
        # Clamp integral
        if self.integral > integral_limit:
            self.integral = integral_limit
        elif self.integral < -integral_limit:
            self.integral = -integral_limit
        
        i_term = self.Ki * self.integral
        
        # === DERIVATIVE TERM with Phase-Specific Filtering ===
        raw_derivative = (error - self.prev_error) / self.dt
        
        # Get filter coefficient for current phase
        alpha = self.derivative_filter_alpha[phase.value]
        
        # Apply low-pass filter
        filtered_derivative = (alpha * raw_derivative + 
                               (1 - alpha) * self.prev_filtered_derivative)
        
        d_term = self.Kd * filtered_derivative
        
        # === TOTAL OUTPUT ===
        output = p_term + i_term + d_term
        
        # Clamp to [0, 1]
        clamped_output = max(0.0, min(1.0, output))
        
        # Detailed logging every 50 steps
        if self.verbose and self.step_count % 50 == 0:
            print(f"\n[PID] Step {self.step_count:05d} | Phase: {phase.value}")
            print(f"  Error: {error:+.4f}°C")
            print(f"  P-term: {p_term:+.6f} (Kp={self.Kp:.3f} × e={error:+.4f})")
            print(f"  I-term: {i_term:+.6f} (Ki={self.Ki:.3f} × ∫e={self.integral:+.4f})")
            print(f"  D-term: {d_term:+.6f} (Kd={self.Kd:.3f} × de/dt={filtered_derivative:+.4f})")
            print(f"  Output: {clamped_output:.6f} {'(saturated)' if clamped_output != output else ''}")
        
        # Update state
        self.prev_error = error
        self.prev_filtered_derivative = filtered_derivative
        
        return clamped_output
    
    def get_diagnostics(self):
        """Return current state for debugging"""
        return {
            'phase': self.current_phase.value if self.current_phase else None,
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
            'integral': self.integral,
            'prev_error': self.prev_error,
            'step_count': self.step_count,
            'phase_switches': self.phase_switches,
        }
