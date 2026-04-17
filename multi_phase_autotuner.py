"""
HYBRID MULTI-PHASE NN AUTOTUNING SYSTEM

Key Innovation: Instead of learning global dynamics (which change with environment),
we use gain scheduling with separate NN autotune for each phase.

Architecture:
  - Phase 1 (Ramp): Aggressive heating to reach approach zone
  - Phase 2 (Approach): Moderate control to reach setpoint smoothly  
  - Phase 3 (Steady): Gentle fine control for ±0.1°C precision

Each phase has:
  - Its own NN model learning phase-specific dynamics
  - Its own auto-tuned PID gains
  - Temperature-based switching (environment-independent!)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from enum import Enum


class ControlPhase(Enum):
    """Control phases based on temperature distance from setpoint"""
    RAMP_UP = "ramp_up"          # T < setpoint - 3°C
    APPROACH = "approach"         # setpoint - 3°C < T < setpoint + 0.5°C
    STEADY_STATE = "steady_state" # |error| < 0.2°C


class PhaseNN(nn.Module):
    """
    Lightweight NN for single-phase dynamics
    Smaller than global model since it only learns one operating region
    """
    def __init__(self, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.net(x)


class PhaseNeuralModel:
    """
    Neural model for single control phase
    Key difference from global model: learns RELATIVE dynamics, not absolute
    """
    def __init__(self, phase_name, temp_ref, verbose=True):
        self.phase_name = phase_name
        self.model = PhaseNN(hidden_size=32)  # Smaller than global model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-5
        )
        self.loss_fn = nn.MSELoss()
        
        # Data for this phase only
        self.train_data = deque(maxlen=500)  # Smaller buffer per phase
        self.val_data = deque(maxlen=100)
        
        # Normalization (same as before)
        self.temp_ref = temp_ref
        self.temp_scale = 5.0  # Smaller scale since we're in one region
        
        # Training metrics
        self.train_losses = deque(maxlen=100)
        self.val_losses = deque(maxlen=100)
        self.total_samples = 0
        self.training_steps = 0
        self.verbose = verbose
        
        if verbose:
            print(f"\n[PhaseNN-{phase_name}] Initialized with temp_ref={temp_ref:.1f}°C")
    
    def normalize_temp(self, temp):
        return (temp - self.temp_ref) / self.temp_scale
    
    def denormalize_temp(self, norm_temp):
        return norm_temp * self.temp_scale + self.temp_ref
    
    def add_sample(self, t1, t2, u1, u2, target):
        """Add sample (only when in this phase!)"""
        norm_sample = (
            [self.normalize_temp(t1), self.normalize_temp(t2), u1, u2],
            self.normalize_temp(target)
        )
        
        if self.total_samples % 5 == 0:
            self.val_data.append(norm_sample)
        else:
            self.train_data.append(norm_sample)
        
        self.total_samples += 1
    
    def train_step(self, batch_size=16):
        """Train on phase-specific data"""
        if len(self.train_data) < 30:
            return None
        
        batch_size = min(batch_size, len(self.train_data))
        indices = np.random.choice(len(self.train_data), batch_size, replace=False)
        batch = [self.train_data[i] for i in indices]
        
        x = torch.tensor([d[0] for d in batch], dtype=torch.float32)
        y = torch.tensor([[d[1]] for d in batch], dtype=torch.float32)
        
        self.model.train()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.train_losses.append(loss.item())
        self.training_steps += 1
        
        # Validation every 5 steps
        if self.training_steps % 5 == 0 and len(self.val_data) >= 10:
            self.model.eval()
            with torch.no_grad():
                x_val = torch.tensor([d[0] for d in self.val_data], dtype=torch.float32)
                y_val = torch.tensor([[d[1]] for d in self.val_data], dtype=torch.float32)
                pred_val = self.model(x_val)
                val_loss = self.loss_fn(pred_val, y_val).item()
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
        
        return loss.item()
    
    def predict(self, t1, t2, u1, u2):
        """Predict next temperature"""
        x = torch.tensor([[
            self.normalize_temp(t1),
            self.normalize_temp(t2),
            u1, u2
        ]], dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            norm_pred = self.model(x).item()
            return self.denormalize_temp(norm_pred)
    
    def is_ready(self, loss_threshold=0.03):
        """Check if model is trained enough for tuning"""
        if self.training_steps < 50:
            return False, float('inf'), "Need more training"
        if len(self.val_losses) < 5:
            return False, float('inf'), "Need more validation"
        
        recent_val_loss = np.mean(list(self.val_losses)[-5:])
        if recent_val_loss < loss_threshold:
            return True, recent_val_loss, "Model converged"
        return False, recent_val_loss, f"Loss too high: {recent_val_loss:.6f}"


class MultiPhaseAutotuner:
    """
    Main autotuning system with gain scheduling
    
    Key Features:
    - Three separate NNs for three phases
    - Temperature-based phase switching (robust to environment)
    - Auto-tune PID gains for each phase
    - Performance monitoring and adaptive retuning
    """
    def __init__(self, setpoint=37.0, verbose=True):
        self.setpoint = setpoint
        self.verbose = verbose
        
        # Phase definitions (temperature-based thresholds)
        self.RAMP_THRESHOLD = setpoint - 3.0      # Below this → ramp phase
        self.APPROACH_THRESHOLD = setpoint + 0.5   # Above ramp, below this → approach
        self.STEADY_TOLERANCE = 0.2                # Within this → steady state
        
        # Initialize three neural models (one per phase)
        self.phase_models = {
            ControlPhase.RAMP_UP: PhaseNeuralModel(
                "RAMP", temp_ref=setpoint-5.0, verbose=verbose
            ),
            ControlPhase.APPROACH: PhaseNeuralModel(
                "APPROACH", temp_ref=setpoint-1.0, verbose=verbose
            ),
            ControlPhase.STEADY_STATE: PhaseNeuralModel(
                "STEADY", temp_ref=setpoint, verbose=verbose
            ),
        }
        
        # PID gains for each phase (will be auto-tuned)
        self.phase_pids = {
            ControlPhase.RAMP_UP: {'Kp': 15.0, 'Ki': 2.0, 'Kd': 400.0},
            ControlPhase.APPROACH: {'Kp': 8.0, 'Ki': 1.0, 'Kd': 250.0},
            ControlPhase.STEADY_STATE: {'Kp': 4.0, 'Ki': 0.5, 'Kd': 120.0},
        }
        
        # Performance tracking per phase
        self.phase_errors = {phase: deque(maxlen=50) for phase in ControlPhase}
        self.phase_time_in = {phase: 0.0 for phase in ControlPhase}
        self.last_retune_time = {phase: 0.0 for phase in ControlPhase}
        
        # Training history
        self.temps = []
        self.powers = []
        self.phases = []
        self.current_phase = ControlPhase.RAMP_UP
        
        print("\n" + "="*70)
        print("MULTI-PHASE NN AUTOTUNER INITIALIZED")
        print("="*70)
        print(f"  Setpoint: {setpoint:.1f}°C")
        print(f"  Ramp → Approach: T > {self.RAMP_THRESHOLD:.1f}°C")
        print(f"  Approach → Steady: |error| < {self.STEADY_TOLERANCE:.1f}°C")
        print(f"  Initial gains:")
        for phase, gains in self.phase_pids.items():
            print(f"    {phase.value}: Kp={gains['Kp']:.1f}, Ki={gains['Ki']:.1f}, Kd={gains['Kd']:.1f}")
        print("="*70 + "\n")
    
    def determine_phase(self, temp):
        """
        Determine control phase based on temperature
        THIS IS KEY: Phase based on MEASURED temp, not learned environment!
        """
        error = abs(temp - self.setpoint)
        
        if error < self.STEADY_TOLERANCE:
            return ControlPhase.STEADY_STATE
        elif temp < self.RAMP_THRESHOLD:
            return ControlPhase.RAMP_UP
        else:
            return ControlPhase.APPROACH
    
    def add_sample(self, temp, power):
        """Add data sample and route to appropriate phase model"""
        self.temps.append(temp)
        self.powers.append(power)
        self.phases.append(self.current_phase)
        
        # Only add to phase-specific model if we have history
        if len(self.temps) >= 3:
            phase = self.determine_phase(self.temps[-2])  # Phase of previous state
            
            self.phase_models[phase].add_sample(
                self.temps[-3],
                self.temps[-2],
                self.powers[-3],
                self.powers[-2],
                self.temps[-1]
            )
    
    def train_all_phases(self):
        """Train all phase models"""
        losses = {}
        for phase, model in self.phase_models.items():
            loss = model.train_step(batch_size=16)
            if loss is not None:
                losses[phase.value] = loss
        
        return losses
    
    def attempt_retune_phase(self, phase, current_time, dt=1.0):
        """
        Attempt to retune PID gains for a specific phase
        Only retunes if:
          1. Model is trained enough
          2. Enough time since last retune (avoid thrashing)
          3. Performance is degrading
        """
        # Check if enough time has passed since last retune
        if current_time - self.last_retune_time[phase] < 120:  # 2 minutes minimum
            return False
        
        # Check if model is ready
        model = self.phase_models[phase]
        is_ready, loss, status = model.is_ready(loss_threshold=0.03)
        
        if not is_ready:
            if self.verbose and model.training_steps % 50 == 0:
                print(f"[{phase.value}] Not ready: {status}")
            return False
        
        # Import autotuner functions
        try:
            from autotuner_streamlined import estimate_parameters, compute_tau_K, imc_pid
        except ImportError:
            print(f"❌ Cannot retune {phase.value}: autotuner_streamlined module not found")
            return False
        
        # Use current temperature as operating point for this phase
        if not self.temps:
            return False
        
        current_temp = self.temps[-1]
        current_power = self.powers[-1] if self.powers else 0.5
        
        print(f"\n🔧 Attempting retune for {phase.value} phase")
        
        try:
            # Estimate parameters using phase-specific model
            a, b = estimate_parameters(model, current_temp, current_power, dt, verbose=False)
            
            if a is not None and b is not None:
                tau, K = compute_tau_K(a, b, dt, verbose=False)
                
                if tau is not None and K is not None:
                    # Adjust IMC parameters based on phase
                    if phase == ControlPhase.RAMP_UP:
                        lam, L = 2.0, 0.5  # Aggressive
                    elif phase == ControlPhase.APPROACH:
                        lam, L = 4.0, 1.0  # Moderate
                    else:  # STEADY_STATE
                        lam, L = 8.0, 1.5  # Conservative
                    
                    gains = imc_pid(K, tau, L=L, lam=lam, verbose=False)
                    
                    if gains is not None:
                        kp, ki, kd = gains
                        
                        if all(np.isfinite([kp, ki, kd])):
                            print(f"✅ {phase.value} retuned: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
                            print(f"   (was: Kp={self.phase_pids[phase]['Kp']:.3f}, "
                                  f"Ki={self.phase_pids[phase]['Ki']:.3f}, "
                                  f"Kd={self.phase_pids[phase]['Kd']:.3f})")
                            
                            self.phase_pids[phase] = {'Kp': kp, 'Ki': ki, 'Kd': kd}
                            self.last_retune_time[phase] = current_time
                            return True
        
        except Exception as e:
            if self.verbose:
                print(f"❌ {phase.value} retune failed: {e}")
        
        return False
    
    def get_current_gains(self, temp):
        """Get PID gains for current temperature"""
        self.current_phase = self.determine_phase(temp)
        return self.phase_pids[self.current_phase]
    
    def print_status(self):
        """Print training status for all phases"""
        print("\n" + "="*70)
        print("MULTI-PHASE TRAINING STATUS")
        print("="*70)
        
        for phase, model in self.phase_models.items():
            is_ready, loss, status = model.is_ready()
            print(f"\n{phase.value}:")
            print(f"  Samples: {model.total_samples}")
            print(f"  Training steps: {model.training_steps}")
            print(f"  Status: {status}")
            print(f"  Current gains: Kp={self.phase_pids[phase]['Kp']:.3f}, "
                  f"Ki={self.phase_pids[phase]['Ki']:.3f}, "
                  f"Kd={self.phase_pids[phase]['Kd']:.3f}")
        
        print("="*70 + "\n")
