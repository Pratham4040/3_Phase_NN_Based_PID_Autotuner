import numpy as np


def estimate_parameters(nn_model, T_current, U_current, dt, verbose=True):
    """
    Estimate first-order plant parameters using NN-simulated step response
    
    VALIDATION STRATEGY: Only reject if mathematically impossible or model not ready
    """
    
    if verbose:
        print("\n" + "="*70)
        print("[Parameter Estimation] STARTING")
        print("="*70)
    
    # CHECK 1: Is the neural network trained enough?
    # This is ESSENTIAL - without a good model, everything else is garbage
    is_ready, loss, status = nn_model.get_training_quality(loss_threshold=0.02, min_steps=100)
    
    if verbose:
        print(f"  Model status: {status}")
        print(f"  Validation loss: {loss:.6f}")
    
    if not is_ready:
        if verbose:
            print(f"  ❌ Model not ready for parameter extraction")
            print("="*70 + "\n")
        return None, None
    
    # Simulate step response
    if verbose:
        print(f"\n  Simulating step response from current state:")
        print(f"    Initial temp: {T_current:.3f}°C")
        print(f"    Step input: 0.0 → 0.6 (60% power)")
    
    num_steps = 60
    u_step = 0.6
    
    T_sim = [T_current, T_current]
    U_sim = [U_current, U_current]
    
    for i in range(num_steps):
        T_prev2 = T_sim[-2]
        T_prev = T_sim[-1]
        U_prev2 = U_sim[-2]
        U_prev = U_sim[-1]
        
        T_next = nn_model.predict(T_prev2, T_prev, U_prev2, U_prev)
        T_sim.append(T_next)
        U_sim.append(u_step)
    
    T_sim = np.array(T_sim[2:])
    U_sim = np.array(U_sim[2:])
    
    if verbose:
        print(f"    Simulation complete: {len(T_sim)} points")
        print(f"    Temperature range: {T_sim.min():.3f} to {T_sim.max():.3f}°C")
    
    # Fit first-order model
    if len(T_sim) < 10:
        if verbose:
            print(f"  ❌ Insufficient simulation data")
            print("="*70 + "\n")
        return None, None
    
    T_k = T_sim[:-1]
    T_k1 = T_sim[1:]
    U_k = U_sim[:-1]
    
    A_matrix = np.column_stack([T_k, U_k, np.ones_like(T_k)])
    
    try:
        params, residuals, rank, s = np.linalg.lstsq(A_matrix, T_k1, rcond=None)
        a, b, c = params
        
        fit_error = np.sqrt(residuals[0] / len(T_k1)) if len(residuals) > 0 else 0
        
        if verbose:
            print(f"\n  Least squares fit:")
            print(f"    Parameters: a={a:.6f}, b={b:.6f}, c={c:.6f}")
            print(f"    Fit RMSE: {fit_error:.6f}°C")
        
    except np.linalg.LinAlgError as e:
        if verbose:
            print(f"  ❌ Least squares failed: {e}")
            print("="*70 + "\n")
        return None, None
    
    # CRITICAL VALIDATION ONLY
    # CHECK 2: Stability (MATHEMATICAL REQUIREMENT)
    if not (0 < a < 1):
        if verbose:
            print(f"\n  ❌ CRITICAL: Parameter 'a'={a:.6f} violates stability (must be 0 < a < 1)")
            print(f"     a ≥ 1 → unstable (exponential growth)")
            print(f"     a ≤ 0 → non-physical")
            print("="*70 + "\n")
        return None, None
    
    # CHECK 3: System must respond to input (MATHEMATICAL REQUIREMENT)
    if abs(b) < 1e-6:
        if verbose:
            print(f"\n  ❌ CRITICAL: Parameter 'b'={b:.6f} too small (system doesn't respond to input)")
            print("="*70 + "\n")
        return None, None
    
    # SOFT WARNINGS (don't reject, just inform)
    if verbose:
        print(f"\n  ✓ Parameters pass critical validation")
        if a < 0.85:
            print(f"  ⚠️  NOTE: a={a:.6f} is low (fast dynamics for thermal system)")
        if a > 0.995:
            print(f"  ⚠️  NOTE: a={a:.6f} is very high (very slow dynamics)")
        if abs(b) > 5.0:
            print(f"  ⚠️  NOTE: b={abs(b):.6f} is large (high sensitivity to input)")
        print("="*70 + "\n")
    
    return a, b


def compute_tau_K(a, b, dt, verbose=True):
    """
    Convert discrete parameters to continuous with MINIMAL validation
    """
    
    if verbose:
        print("\n" + "="*70)
        print("[Continuous Parameter Conversion]")
        print("="*70)
    
    if a is None or b is None:
        if verbose:
            print("  ❌ Cannot compute: input parameters are None")
            print("="*70 + "\n")
        return None, None
    
    # Note: stability already checked in estimate_parameters, no need to recheck
    
    tau = -dt / np.log(a)
    K = b / (1 - a)
    
    if verbose:
        print(f"  Input: a={a:.6f}, b={b:.6f}, dt={dt:.3f}s")
        print(f"\n  Computed continuous parameters:")
        print(f"    Time constant τ: {tau:.3f} seconds")
        print(f"    Steady-state gain K: {K:.3f} °C per unit power")
        
        # Informational warnings only (don't reject)
        if tau < dt * 2:
            print(f"  ⚠️  NOTE: τ < 2×dt ({tau:.3f} < {2*dt:.3f}s) - very fast dynamics")
        if tau > 1000:
            print(f"  ⚠️  NOTE: τ > 1000s - very slow dynamics")
        if abs(K) > 100:
            print(f"  ⚠️  NOTE: |K| > 100 - very high gain")
        if abs(K) < 0.1:
            print(f"  ⚠️  NOTE: |K| < 0.1 - very low gain")
        
        print(f"  ✓ Conversion complete (accepting all physically valid values)")
        print("="*70 + "\n")
    
    return tau, K


def imc_pid(K, tau, L=0.7, lam=3.0, verbose=True):
    """
    Compute PID gains with SMART validation - accept wide range, warn on extremes
    """
    
    if verbose:
        print("\n" + "="*70)
        print("[IMC PID Tuning]")
        print("="*70)
    
    if K is None or tau is None:
        if verbose:
            print("  ❌ Cannot compute: K or τ is None")
            print("="*70 + "\n")
        return None
    
    # CRITICAL CHECK: Prevent division by zero
    if abs(K) < 1e-8:
        if verbose:
            print(f"  ❌ CRITICAL: K={K:.8f} too close to zero (would cause division by zero)")
            print("="*70 + "\n")
        return None
    
    if verbose:
        print(f"  Process parameters:")
        print(f"    Gain K: {K:.3f} °C")
        print(f"    Time constant τ: {tau:.3f} s")
        print(f"    Dead time L: {L:.3f} s")
        print(f"  Tuning parameter λ: {lam:.3f} s")
    
    # Adaptive lambda for stability
    adjusted_lam = max(lam, tau * 0.3)
    if adjusted_lam != lam and verbose:
        print(f"    → Adjusted λ: {adjusted_lam:.3f} s (for stability)")
    
    # Compute gains
    kp = tau / (K * (adjusted_lam + L))
    ki = kp / tau
    kd = kp * L
    
    if verbose:
        print(f"\n  Raw computed gains:")
        print(f"    Kp: {kp:.6f}")
        print(f"    Ki: {ki:.6f}")
        print(f"    Kd: {kd:.6f}")
    
    # Auto-correct inverse action
    if kp < 0:
        if verbose:
            print(f"\n  → Detected inverse action, flipping signs")
        kp, ki, kd = -kp, -ki, -kd
    
    # Limit derivative to prevent noise amplification
    max_kd = kp * 2.5
    if kd > max_kd:
        if verbose:
            print(f"  → Limiting Kd from {kd:.6f} to {max_kd:.6f} (prevent noise amplification)")
        kd = max_kd
    
    if verbose:
        print(f"\n  Final PID gains:")
        print(f"    Kp: {kp:.6f}")
        print(f"    Ki: {ki:.6f}")
        print(f"    Kd: {kd:.6f}")
    
    # FINAL SANITY CHECK - only reject truly insane values
    insane_kp = not (1e-6 < kp < 10000)
    insane_ki = not (0 <= ki < 1000)
    insane_kd = not (0 <= kd < 1000)
    
    if insane_kp or insane_ki or insane_kd:
        if verbose:
            print(f"\n  ❌ CRITICAL: Gains outside physically reasonable ranges:")
            if insane_kp:
                print(f"     Kp={kp:.6f} (expect 1e-6 to 10000)")
            if insane_ki:
                print(f"     Ki={ki:.6f} (expect 0 to 1000)")
            if insane_kd:
                print(f"     Kd={kd:.6f} (expect 0 to 1000)")
            print("="*70 + "\n")
        return None
    
    # Warnings for unusual but acceptable values
    if verbose:
        if kp < 0.1 or kp > 200:
            print(f"  ⚠️  NOTE: Kp={kp:.3f} is unusual (typical range: 0.1-200)")
        if ki > 50:
            print(f"  ⚠️  NOTE: Ki={ki:.3f} is high (might cause overshoot)")
        if kd > 20:
            print(f"  ⚠️  NOTE: Kd={kd:.3f} is high (might amplify noise)")
        
        print(f"  ✓ Gains validated and accepted")
        print("="*70 + "\n")
    
    return kp, ki, kd
