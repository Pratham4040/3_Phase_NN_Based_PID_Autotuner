# 3-Phase NN PID Autotuner

Short, practical project for automatic temperature control on ESP32 using:
- Multi-phase control logic
- Neural-network-based model learning per phase
- Automatic PID gain retuning

This design is made to be robust across different rooms/environments by switching control phase from measured temperature, not room assumptions.

## What This Project Does

Target: hold temperature near setpoint (default 37.0 C) with stable control.

Control phases:
1. Ramp Up: fast heating when far below setpoint
2. Approach: smoother control near target
3. Steady State: fine control around target

Each phase has its own learned dynamics and PID gains.

## Project Files

- multi_phase_esp32_runner.py: main runtime loop, ESP32 HTTP I/O, logging, safety checks
- multi_phase_autotuner.py: phase detection, phase models, auto-retune logic
- multi_phase_pid.py: PID controller with phase-based gain scheduling
- autotuner_streamlined.py: parameter estimation and IMC-style PID gain computation
- requirements.txt: Python dependencies

## Requirements

- Python 3.10+ recommended
- ESP32 firmware exposing:
  - GET /temp -> temperature value (optionally with SAFETY flag)
  - POST /pwm -> heater PWM command
- Network access to ESP32 IP

## Setup

### 1) Create and activate virtual environment

Windows PowerShell:
~~~powershell
python -m venv 3_phase_venv
.\3_phase_venv\Scripts\Activate.ps1
~~~

### 2) Install dependencies

~~~powershell
pip install -r requirements.txt
~~~

### 3) Verify required packages

~~~powershell
python -c "import numpy, torch, matplotlib; print('Dependencies OK')"
~~~

## Run

Basic run:
~~~powershell
python multi_phase_esp32_runner.py --esp-ip 192.168.137.42 --setpoint 37.0 --autotune
~~~

Useful options:
- --dt 0.5                Control period in seconds
- --retune-interval 100   Try retuning every N steps
- --steps 0               Max steps (0 means infinite)
- --duration 0            Max run time in seconds (0 means infinite)
- --host-max-temp 40.0    Host safety cutoff
- --request-timeout 1.5   HTTP timeout
- --max-failures 8        Stop after too many consecutive read failures
- --csv multi_phase_log.csv  CSV output path

Example with limits:
~~~powershell
python multi_phase_esp32_runner.py --esp-ip 192.168.137.42 --setpoint 37.0 --dt 0.5 --autotune --retune-interval 100 --duration 900 --csv multi_phase_log.csv
~~~

## How It Works

1. Runner reads temperature from ESP32 at fixed interval.
2. Autotuner determines current phase from temperature and setpoint.
3. PID controller loads gains for that phase.
4. Control output is computed, clamped, converted to PWM, and sent to ESP32.
5. Sample history is added for model learning.
6. Phase models are trained continuously.
7. At retune interval, each phase may update PID gains if model quality is good and cooldown conditions are met.
8. Status and metrics are printed and appended to CSV.

## Phase Logic

Default thresholds for setpoint S:
- Steady State if absolute error is less than 0.2 C
- Ramp Up if temperature is below S - 3.0 C
- Otherwise Approach

With default S = 37.0 C:
- Ramp Up: T < 34.0 C
- Approach: about 34.0 C to near setpoint
- Steady State: within +-0.2 C of setpoint

## Safety Behavior

- ESP32 safety flag triggers immediate PWM = 0
- Host-side max temperature also triggers PWM = 0
- Repeated sensor failures trigger stop
- On exit (normal or Ctrl+C), runner attempts final PWM = 0

## Output and Logs

Console output includes:
- step number
- active phase
- current temperature
- control error
- PWM command
- training losses per phase (when available)

CSV columns include:
- timestamp, step, elapsed_s
- temp_c, error_c
- heater_norm, pwm_cmd
- esp_safety, host_safety
- phase, kp, ki, kd
- ramp_loss, approach_loss, steady_loss

## Typical Workflow

1. Start with autotune enabled.
2. Let it run through all three phases.
3. Observe error shrinking in steady state.
4. Check CSV summary and tolerance percentages.
5. Re-run in a different room to validate environment independence.

## Troubleshooting

- Cannot connect to ESP32:
  - Check IP address and same network
  - Confirm endpoints /temp and /pwm are reachable
- No phase transitions:
  - Verify heater can physically reach setpoint
  - Increase run duration
- Oscillation near setpoint:
  - Allow more runtime for retuning
  - Increase control period slightly (example: --dt 0.8)
- Early stop due to failures:
  - Increase --request-timeout and/or --max-failures

## Notes

- This repo currently uses autotuner_streamlined.py directly in retuning logic.
- Keep control period and communication timing consistent for stable behavior.
- Use logged CSV data for post-run analysis and tuning audits.
