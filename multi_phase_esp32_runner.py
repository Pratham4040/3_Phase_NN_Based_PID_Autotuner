"""
COMPLETE INTEGRATION EXAMPLE - Multi-Phase NN Autotuning

This replaces esp32_pid_runner.py with a more robust architecture:
- Gain scheduling (supervisor's approach)
- Automatic tuning per phase (NN-based)
- Environment-independent (temperature-based switching)
- Adaptive retuning when performance degrades

USAGE:
python multi_phase_esp32_runner.py --esp-ip 192.168.137.42 --setpoint 37.0 --autotune
"""

import argparse
import csv
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime

from multi_phase_autotuner import MultiPhaseAutotuner, ControlPhase
from multi_phase_pid import MultiPhasePID


MAX_PWM_CAP = 125


def http_get_text(url, timeout_s):
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8").strip()


def http_post_text(url, body_text, timeout_s):
    request = urllib.request.Request(
        url,
        data=body_text.encode("utf-8"),
        method="POST",
        headers={"Content-Type": "text/plain"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read().decode("utf-8").strip()


def read_temp(esp_ip, timeout_s):
    text = http_get_text(f"http://{esp_ip}/temp", timeout_s)
    safety = False

    if "," in text:
        temp_text, tail = text.split(",", 1)
        safety = "SAFETY" in tail.upper()
    else:
        temp_text = text

    temp_c = float(temp_text)
    return temp_c, safety


def write_pwm(esp_ip, pwm, timeout_s):
    pwm = int(max(0, min(MAX_PWM_CAP, pwm)))
    reply = http_post_text(f"http://{esp_ip}/pwm", str(pwm), timeout_s)
    return reply


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Phase NN Autotuning for ESP32 Temperature Control"
    )
    parser.add_argument("--esp-ip", required=True, help="ESP32 IP address")
    parser.add_argument("--setpoint", type=float, default=37.0, help="Target temperature (°C)")
    parser.add_argument("--dt", type=float, default=0.5, help="Control period (seconds)")
    
    parser.add_argument("--autotune", action="store_true", help="Enable NN-based autotuning")
    parser.add_argument("--retune-interval", type=int, default=100, 
                       help="Attempt retune every N steps (per phase)")
    
    parser.add_argument("--steps", type=int, default=0, help="Max control steps (0=infinite)")
    parser.add_argument("--duration", type=float, default=0.0, help="Max duration seconds (0=infinite)")
    
    parser.add_argument("--host-max-temp", type=float, default=40.0, help="Safety cutoff (°C)")
    parser.add_argument("--request-timeout", type=float, default=1.5, help="HTTP timeout (s)")
    parser.add_argument("--max-failures", type=int, default=8, help="Max consecutive failures")
    
    parser.add_argument("--csv", default="multi_phase_log.csv", help="CSV log path")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    
    # Initialize multi-phase autotuner
    if args.autotune:
        print("\n" + "🤖"*35)
        print("MULTI-PHASE NN AUTOTUNING ENABLED")
        print("🤖"*35)
        
        autotuner = MultiPhaseAutotuner(setpoint=args.setpoint, verbose=True)
        pid = MultiPhasePID(autotuner, dt=args.dt, verbose=True)
        
        print("\n📊 System Architecture:")
        print("  ✓ 3 Neural Networks (one per phase)")
        print("  ✓ Temperature-based phase switching")
        print("  ✓ Automatic PID tuning per phase")
        print("  ✓ Adaptive retuning on performance degradation")
        print("  ✓ Environment-independent operation\n")
    else:
        raise ValueError("Multi-phase system requires --autotune flag")
    
    # Data logging
    temps = []
    powers = []
    phases = []
    
    consecutive_failures = 0
    step = 0
    
    start_t = time.time()
    next_tick = start_t
    
    print("\n" + "="*70)
    print("STARTING MULTI-PHASE TEMPERATURE CONTROL")
    print("="*70)
    print(f"  ESP32 IP:       {args.esp_ip}")
    print(f"  Setpoint:       {args.setpoint:.2f}°C")
    print(f"  Control period: {args.dt:.3f}s")
    print(f"  CSV log:        {args.csv}")
    print(f"  Safety cutoff:  {args.host_max_temp:.2f}°C")
    print("="*70 + "\n")
    
    with open(args.csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "timestamp", "step", "elapsed_s", "temp_c", "error_c",
            "heater_norm", "pwm_cmd", "esp_safety", "host_safety",
            "phase", "kp", "ki", "kd",
            "ramp_loss", "approach_loss", "steady_loss"
        ])
        
        try:
            while True:
                now = time.time()
                if now < next_tick:
                    time.sleep(next_tick - now)
                
                loop_t = time.time()
                elapsed_s = loop_t - start_t
                
                # Check limits
                if args.steps > 0 and step >= args.steps:
                    print("\n" + "="*70)
                    print("STEP LIMIT REACHED")
                    print("="*70)
                    break
                if args.duration > 0 and elapsed_s >= args.duration:
                    print("\n" + "="*70)
                    print("DURATION LIMIT REACHED")
                    print("="*70)
                    break
                
                # Read temperature
                try:
                    temp_c, esp_safety = read_temp(args.esp_ip, args.request_timeout)
                    consecutive_failures = 0
                except (ValueError, urllib.error.URLError, TimeoutError) as exc:
                    consecutive_failures += 1
                    print(f"\n❌ Step {step:05d} | SENSOR READ FAILED ({consecutive_failures}/{args.max_failures})")
                    print(f"   Error: {exc}")
                    
                    try:
                        write_pwm(args.esp_ip, 0, args.request_timeout)
                    except Exception:
                        pass
                    
                    if consecutive_failures >= args.max_failures:
                        print("\n" + "="*70)
                        print("TOO MANY FAILURES - STOPPING")
                        print("="*70)
                        break
                    
                    next_tick += args.dt
                    step += 1
                    continue
                
                # Safety check
                host_safety = temp_c >= args.host_max_temp
                if esp_safety or host_safety:
                    if esp_safety:
                        print(f"\n⚠️  ESP SAFETY TRIGGERED at {temp_c:.2f}°C")
                    if host_safety:
                        print(f"\n⚠️  HOST SAFETY TRIGGERED at {temp_c:.2f}°C")
                    
                    write_pwm(args.esp_ip, 0, args.request_timeout)
                    heater_norm = 0.0
                else:
                    # Compute PID control
                    heater_norm = pid.compute(temp_c)
                
                heater_norm = max(0.0, min(1.0, heater_norm))
                pwm_cmd = int(round(heater_norm * MAX_PWM_CAP))
                
                # Send PWM
                try:
                    write_pwm(args.esp_ip, pwm_cmd, args.request_timeout)
                except (urllib.error.URLError, TimeoutError) as exc:
                    print(f"⚠️  PWM write failed: {exc}")
                
                # Store data
                temps.append(temp_c)
                powers.append(heater_norm)
                current_phase = autotuner.determine_phase(temp_c)
                phases.append(current_phase.value)
                
                # Add sample to autotuner
                autotuner.add_sample(temp_c, heater_norm)
                
                # Train all phase models
                losses = autotuner.train_all_phases()
                
                # Attempt retuning periodically
                if step > 0 and step % args.retune_interval == 0:
                    for phase in ControlPhase:
                        autotuner.attempt_retune_phase(phase, elapsed_s, dt=args.dt)
                
                # Print status
                error_c = args.setpoint - temp_c
                
                # Visual indicators
                if abs(error_c) < 0.1:
                    error_indicator = "✓✓"
                elif abs(error_c) < 0.2:
                    error_indicator = "✓"
                elif abs(error_c) < 0.5:
                    error_indicator = "~"
                else:
                    error_indicator = "!"
                
                pwm_indicator = "⚠️ " if (pwm_cmd == 0 or pwm_cmd == MAX_PWM_CAP) else ""
                
                # Build loss string
                loss_str = ""
                if losses:
                    loss_parts = []
                    for phase_name, loss_val in losses.items():
                        loss_parts.append(f"{phase_name[:4]}={loss_val:.4f}")
                    loss_str = " | " + ", ".join(loss_parts)
                
                print(
                    f"Step {step:05d} | "
                    f"[{current_phase.value.upper()}] | "
                    f"T={temp_c:6.3f}°C | "
                    f"E={error_c:+6.3f}°C {error_indicator} | "
                    f"{pwm_indicator}PWM={pwm_cmd:3d}"
                    f"{loss_str}"
                )
                
                # Log to CSV
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    step,
                    round(elapsed_s, 3),
                    round(temp_c, 4),
                    round(error_c, 4),
                    round(heater_norm, 5),
                    pwm_cmd,
                    int(bool(esp_safety)),
                    int(bool(host_safety)),
                    current_phase.value,
                    round(pid.Kp, 6),
                    round(pid.Ki, 6),
                    round(pid.Kd, 6),
                    losses.get("ramp_up", ""),
                    losses.get("approach", ""),
                    losses.get("steady_state", ""),
                ])
                csv_file.flush()
                
                # Print full status every 200 steps
                if step > 0 and step % 200 == 0:
                    autotuner.print_status()
                
                next_tick += args.dt
                step += 1
        
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("INTERRUPTED BY USER (Ctrl+C)")
            print("="*70)
        
        finally:
            # Safe shutdown
            print(f"\nShutting down safely...")
            try:
                write_pwm(args.esp_ip, 0, args.request_timeout)
                print(f"✅ Sent final PWM=0")
            except Exception as exc:
                print(f"❌ Could not send final PWM=0: {exc}")

            print("\n" + "="*70)
            print("FINAL PID GAINS BY PHASE")
            print("="*70)
            for phase in ControlPhase:
                gains = autotuner.phase_pids.get(phase, {})
                kp = gains.get("Kp", float("nan"))
                ki = gains.get("Ki", float("nan"))
                kd = gains.get("Kd", float("nan"))
                print(f"  {phase.value}: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")
            print("="*70)
            
            # Performance summary
            if temps:
                print(f"\n" + "="*70)
                print(f"PERFORMANCE SUMMARY")
                print(f"="*70)
                print(f"  Total steps: {step}")
                print(f"  Total time:  {elapsed_s:.1f}s")
                
                # Calculate phase-specific performance
                for phase in ControlPhase:
                    phase_indices = [i for i, p in enumerate(phases) if p == phase.value]
                    if phase_indices:
                        phase_temps = [temps[i] for i in phase_indices]
                        phase_errors = [abs(t - args.setpoint) for t in phase_temps]
                        avg_error = sum(phase_errors) / len(phase_errors)
                        max_error = max(phase_errors)
                        in_tolerance = sum(1 for e in phase_errors if e < 0.2)
                        tolerance_pct = 100 * in_tolerance / len(phase_errors)
                        
                        print(f"\n  {phase.value}:")
                        print(f"    Steps in phase: {len(phase_indices)}")
                        print(f"    Avg error: {avg_error:.4f}°C")
                        print(f"    Max error: {max_error:.4f}°C")
                        print(f"    Time in ±0.2°C: {tolerance_pct:.1f}%")
                
                # Overall performance
                all_errors = [abs(t - args.setpoint) for t in temps]
                within_01 = sum(1 for e in all_errors if e < 0.1)
                within_02 = sum(1 for e in all_errors if e < 0.2)
                
                print(f"\n  Overall:")
                print(f"    Avg error: {sum(all_errors)/len(all_errors):.4f}°C")
                print(f"    Max error: {max(all_errors):.4f}°C")
                print(f"    Time in ±0.1°C: {100*within_01/len(all_errors):.1f}%")
                print(f"    Time in ±0.2°C: {100*within_02/len(all_errors):.1f}%")
                
                print(f"\n  CSV saved: {args.csv}")
                print(f"="*70 + "\n")


if __name__ == "__main__":
    main()
