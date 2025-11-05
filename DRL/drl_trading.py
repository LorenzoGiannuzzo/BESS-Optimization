"""
------------------------------------------------------------------------------------------------------------------------
BATTERY ENERGY STORAGE SYSTEM (BESS) OPTIMIZATION - 4 ERROR TYPES + PDF EXPORT
PSO vs PPO: 4 Error Distributions + Monthly Retraining + PDF Export (DPI=500)
------------------------------------------------------------------------------------------------------------------------
Author: Lorenzo Giannuzzo
Affiliation: Politecnico di Torino - DENERG - Energy Center Lab

FINAL VERSION:
- 4 error distributions: uniform, normal, ornstein-uhlenbeck, fixed-bias (NEW)
- Monthly retraining with ACTUAL forecast errors
- All plots saved in PNG (300 DPI) and PDF (500 DPI)
- NEW: Error types comparison visualization
- Battery degradation model integrated in PSO and PPO (FIXED: cumulative cycles)
- Heatmap visualizations added
- SOH comparison plots added
------------------------------------------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import traceback

# Lorenzo Giannuzzo: ==================== CONFIGURATION ====================
TEST_FIRST_MONTH_ONLY = False
NOISE_TYPES_TO_TEST = ['uniform', 'normal', 'ornstein-uhlenbeck']
FORCE_RETRAIN = False

# Lorenzo Giannuzzo: ==================== BATTERY PARAMETERS ====================
FORECAST_ERROR_MIN = -0.15
FORECAST_ERROR_MAX = 0.15
BATTERY_CAPACITY = 4.0
BATTERY_POWER = 2.0
BATTERY_EFFICIENCY = 0.95
C_RATE = 0.5
SOC_MIN = 0.1
SOC_MAX = 0.9

# Lorenzo Giannuzzo: ==================== DEGRADATION PARAMETERS ====================
DEGRADATION_COST_PER_MWH = 25000  # Lorenzo Giannuzzo: EUR/MWh of throughput

# Lorenzo Giannuzzo: ==================== PSO PARAMETERS ====================
PSO_PARTICLES = 50
PSO_ITERATIONS = 150
PSO_HORIZON = 24
PSO_W_MAX = 0.95
PSO_W_MIN = 0.2

# Lorenzo Giannuzzo: ==================== PPO PARAMETERS ====================
PPO_TOTAL_TIMESTEPS = 500000
PPO_LEARNING_RATE = 1e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_HIDDEN_DIM_1 = 1024
PPO_HIDDEN_DIM_2 = 512
PPO_ENT_COEF = 0.01
PPO_CLIP_RANGE = 0.2
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5

# Lorenzo Giannuzzo: ==================== ACTION SPACE ====================
N_DISCRETE_ACTIONS = 21

# Lorenzo Giannuzzo: ==================== REWARD PARAMETERS ====================
SOC_VIOLATION_PENALTY = 1000.0

# Lorenzo Giannuzzo: ==================== ENVIRONMENT ====================
FORECAST_HORIZON = 24

# Lorenzo Giannuzzo: ==================== MONTHLY RETRAINING PARAMETERS ====================
MONTHLY_RETRAINING_ENABLED = True
MONTHLY_RETRAINING_LOOKBACK = 60
MONTHLY_RETRAINING_TIMESTEPS = 500000
MONTHLY_RETRAINING_FREQUENCY = 30

TRAIN_FILE = '20231101_20231231_PUN.xlsx'
TEST_FILE = '20240101_20241231_PUN.xlsx'

np.random.seed(42)
torch.manual_seed(42)


# Lorenzo Giannuzzo: ==================== DEGRADATION MODEL ====================
def degradation(cycle_num):
    """
    Lorenzo Giannuzzo: Battery degradation function (remaining capacity %)
    Polynomial model based on CUMULATIVE cycle count
    """
    capacity_remaining = (
            -0.00000000000000000000000000000005613 * cycle_num ** 9 +
            0.000000000000000000000000003121 * cycle_num ** 8 -
            0.00000000000000000000006353 * cycle_num ** 7 +
            0.000000000000000000663 * cycle_num ** 6 -
            0.000000000000003987 * cycle_num ** 5 +
            0.00000000001435 * cycle_num ** 4 -
            0.0000000307 * cycle_num ** 3 +
            0.00003746 * cycle_num ** 2 -
            0.0277 * cycle_num + 100
    )
    return max(0, capacity_remaining)


class OrnsteinUhlenbeckNoise:
    """Lorenzo Giannuzzo: Ornstein-Uhlenbeck process for time-correlated forecast errors"""

    def __init__(self, theta=0.15, sigma=0.2, dt=1.0):
        # Lorenzo Giannuzzo: Mean reversion rate
        self.theta = theta
        # Lorenzo Giannuzzo: Volatility
        self.sigma = sigma
        # Lorenzo Giannuzzo: Time step
        self.dt = dt
        # Lorenzo Giannuzzo: Current state
        self.x = 0.0

    def sample(self):
        """Lorenzo Giannuzzo: Generate next correlated error sample"""
        dx = -self.theta * self.x * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.x += dx
        return np.clip(self.x, FORECAST_ERROR_MIN, FORECAST_ERROR_MAX)

    def reset(self):
        """Lorenzo Giannuzzo: Reset process to initial state"""
        self.x = 0.0


class ForecastErrorGenerator:
    """Lorenzo Giannuzzo: Stochastic forecast error generator for price predictions"""

    def __init__(self, error_type='uniform'):
        self.error_type = error_type
        self.ou_noise = OrnsteinUhlenbeckNoise() if error_type == 'ornstein-uhlenbeck' else None
        self.fixed_bias = 0.10  # Lorenzo Giannuzzo: Fixed +10% error for deterministic testing

    def generate_error(self):
        """Lorenzo Giannuzzo: Generate forecast error based on distribution type"""
        if self.error_type == 'uniform':
            return np.random.uniform(FORECAST_ERROR_MIN, FORECAST_ERROR_MAX)
        elif self.error_type == 'normal':
            error = np.random.normal(0, FORECAST_ERROR_MAX / 3)
            return np.clip(error, FORECAST_ERROR_MIN, FORECAST_ERROR_MAX)
        elif self.error_type == 'ornstein-uhlenbeck':
            return self.ou_noise.sample()
        elif self.error_type == 'fixed-bias':
            return self.fixed_bias
        else:
            return 0.0

    def apply_error(self, true_price):
        """Lorenzo Giannuzzo: Apply error to true price to generate forecast"""
        error = self.generate_error()
        forecast_price = true_price * (1 + error)
        return forecast_price, error

    def reset(self):
        """Lorenzo Giannuzzo: Reset error generator state"""
        if self.ou_noise:
            self.ou_noise.reset()


class Battery:
    """Lorenzo Giannuzzo: Battery model with CUMULATIVE degradation tracking"""

    def __init__(self):
        self.capacity = BATTERY_CAPACITY
        self.nominal_capacity = BATTERY_CAPACITY
        self.max_power = min(BATTERY_POWER, BATTERY_CAPACITY * C_RATE)
        self.efficiency = BATTERY_EFFICIENCY
        self.soc = 0.5
        self.soc_min = SOC_MIN
        self.soc_max = SOC_MAX
        # Lorenzo Giannuzzo: CUMULATIVE degradation tracking
        self.cumulative_throughput_kwh = 0.0  # Lorenzo Giannuzzo: FIXED - cumulative energy
        self.equivalent_cycles = 0.0

    def step(self, action):
        """Lorenzo Giannuzzo: Execute battery action and update state"""
        action = np.clip(action, -self.max_power, self.max_power)

        # Lorenzo Giannuzzo: Charging mode
        if action > 0.01:
            if self.soc >= self.soc_max - 1e-6:
                return 0.0

            max_energy_storable = (self.soc_max - self.soc) * self.capacity
            max_power_from_grid = max_energy_storable / self.efficiency
            actual_power_from_grid = min(action, max_power_from_grid)

            if actual_power_from_grid > 0.01:
                energy_stored = actual_power_from_grid * self.efficiency
                self.soc += energy_stored / self.capacity
                self.soc = min(self.soc, self.soc_max)
                # Lorenzo Giannuzzo: FIXED - Add to CUMULATIVE throughput
                self.cumulative_throughput_kwh += energy_stored * 1000
                return actual_power_from_grid
            return 0.0

        # Lorenzo Giannuzzo: Discharging mode
        elif action < -0.01:
            if self.soc <= self.soc_min + 1e-6:
                return 0.0

            max_energy_available = (self.soc - self.soc_min) * self.capacity
            max_power_to_grid = max_energy_available * self.efficiency
            actual_power_to_grid = min(-action, max_power_to_grid)

            if actual_power_to_grid > 0.01:
                energy_from_battery = actual_power_to_grid / self.efficiency
                self.soc -= energy_from_battery / self.capacity
                self.soc = max(self.soc, self.soc_min)
                # Lorenzo Giannuzzo: FIXED - Add to CUMULATIVE throughput
                self.cumulative_throughput_kwh += energy_from_battery * 1000
                return -actual_power_to_grid
            return 0.0

        return 0.0

    def update_degradation(self):
        """
        Lorenzo Giannuzzo: FIXED - Update capacity based on CUMULATIVE cycles
        Now correctly calculates cumulative cycles and applies degradation model
        """
        # Lorenzo Giannuzzo: Calculate CUMULATIVE equivalent cycles
        self.equivalent_cycles = self.cumulative_throughput_kwh / (2 * self.nominal_capacity * 1000)

        # Lorenzo Giannuzzo: Get remaining capacity percentage from degradation model
        capacity_percentage = degradation(self.equivalent_cycles)

        # Lorenzo Giannuzzo: Update actual capacity
        self.capacity = self.nominal_capacity * (capacity_percentage / 100.0)

    def get_soh(self):
        """Lorenzo Giannuzzo: Get State of Health percentage"""
        return (self.capacity / self.nominal_capacity) * 100.0

    def reset(self):
        """Lorenzo Giannuzzo: Reset battery to initial state"""
        self.soc = 0.5

    def copy(self):
        """Lorenzo Giannuzzo: Create deep copy of battery state"""
        b = Battery()
        b.soc = self.soc
        b.capacity = self.capacity
        b.equivalent_cycles = self.equivalent_cycles
        b.cumulative_throughput_kwh = self.cumulative_throughput_kwh  # Lorenzo Giannuzzo: FIXED
        return b


class BatteryTradingEnvClean(gym.Env):
    """Lorenzo Giannuzzo: WORKING VERSION - 26 features, simple reward with degradation cost"""

    def __init__(self, prices, error_generator):
        super(BatteryTradingEnvClean, self).__init__()
        self.prices = prices
        self.error_gen = error_generator
        self.battery = Battery()
        self.current_step = 0
        self.max_steps = len(prices)
        self.forecast_horizon = FORECAST_HORIZON

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.action_values = np.linspace(-BATTERY_POWER, BATTERY_POWER, N_DISCRETE_ACTIONS)

        n_features = 1 + FORECAST_HORIZON + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(n_features,),
            dtype=np.float32
        )

        self.episode_length = 480

    def reset(self, seed=None, options=None):
        """Lorenzo Giannuzzo: Reset environment to initial state"""
        super().reset(seed=seed)
        self.battery.reset()
        self.error_gen.reset()
        self.current_step = 0

        if self.max_steps > 1000:
            self.current_step = np.random.randint(0, min(1000, self.max_steps - 500))

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        """Lorenzo Giannuzzo: Construct observation vector with SOC, forecast prices, and current price"""
        if self.current_step >= len(self.prices):
            return np.zeros(26, dtype=np.float32)

        soc = self.battery.soc
        current_price = self.prices[self.current_step]

        forecast_prices_norm = []
        for h in range(self.forecast_horizon):
            if self.current_step + h < len(self.prices):
                true_future_price = self.prices[self.current_step + h]
                forecast_price, _ = self.error_gen.apply_error(true_future_price)
                forecast_prices_norm.append(forecast_price / 200.0)
            else:
                forecast_prices_norm.append(0.5)

        current_price_norm = current_price / 200.0

        obs = np.array([
            soc,
            *forecast_prices_norm,
            current_price_norm
        ], dtype=np.float32)

        return obs

    def step(self, action_idx):
        """Lorenzo Giannuzzo: Execute action and return reward with degradation cost"""
        action = float(self.action_values[action_idx])
        true_price = self.prices[self.current_step]

        energy = self.battery.step(action)
        profit = -energy * true_price

        # Lorenzo Giannuzzo: Add degradation cost to reward
        degradation_cost = abs(energy) * DEGRADATION_COST_PER_MWH / (2 * 6000)

        soc_penalty = 0.0
        if self.battery.soc > SOC_MAX:
            soc_penalty = -abs(self.battery.soc - SOC_MAX) * SOC_VIOLATION_PENALTY
        elif self.battery.soc < SOC_MIN:
            soc_penalty = -abs(self.battery.soc - SOC_MIN) * SOC_VIOLATION_PENALTY

        # Lorenzo Giannuzzo: Reward includes profit minus degradation cost
        reward = profit - degradation_cost + soc_penalty

        self.current_step += 1
        done = self.current_step >= min(self.max_steps, self.episode_length)
        truncated = False

        if done:
            obs = self._get_observation() if self.current_step < self.max_steps else np.zeros(26, dtype=np.float32)
        else:
            obs = self._get_observation()

        info = {
            'energy': energy,
            'price': true_price,
            'profit': profit,
            'degradation_cost': degradation_cost,
            'soc': self.battery.soc
        }

        return obs, reward, done, truncated, info


class BatteryTradingEnvWithActualForecasts(gym.Env):
    """Lorenzo Giannuzzo: Environment that uses ACTUAL forecast errors from testing history"""

    def __init__(self, true_prices, forecast_history, offset=0):
        super(BatteryTradingEnvWithActualForecasts, self).__init__()
        self.true_prices = true_prices
        self.forecast_history = forecast_history
        self.offset = offset

        self.battery = Battery()
        self.current_step = 0
        self.max_steps = len(true_prices)
        self.forecast_horizon = FORECAST_HORIZON

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.action_values = np.linspace(-BATTERY_POWER, BATTERY_POWER, N_DISCRETE_ACTIONS)

        n_features = 1 + FORECAST_HORIZON + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(n_features,),
            dtype=np.float32
        )

        self.episode_length = 480

        self.forecast_lookup = {f['hour']: f['forecast_price'] for f in forecast_history}

    def reset(self, seed=None, options=None):
        """Lorenzo Giannuzzo: Reset environment to initial state"""
        super().reset(seed=seed)
        self.battery.reset()
        self.current_step = 0

        if self.max_steps > 1000:
            self.current_step = np.random.randint(0, min(1000, self.max_steps - 500))

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        """Lorenzo Giannuzzo: Construct observation using actual forecast history"""
        if self.current_step >= len(self.true_prices):
            return np.zeros(26, dtype=np.float32)

        soc = self.battery.soc
        current_price = self.true_prices[self.current_step]

        forecast_prices_norm = []
        for h in range(self.forecast_horizon):
            if self.current_step + h < len(self.true_prices):
                global_hour = self.current_step + h + self.offset

                if global_hour in self.forecast_lookup:
                    forecast_price = self.forecast_lookup[global_hour]
                else:
                    forecast_price = self.true_prices[self.current_step + h]

                forecast_prices_norm.append(forecast_price / 200.0)
            else:
                forecast_prices_norm.append(0.5)

        current_price_norm = current_price / 200.0

        obs = np.array([
            soc,
            *forecast_prices_norm,
            current_price_norm
        ], dtype=np.float32)

        return obs

    def step(self, action_idx):
        """Lorenzo Giannuzzo: Execute action and return reward with degradation cost"""
        action = float(self.action_values[action_idx])
        true_price = self.true_prices[self.current_step]

        energy = self.battery.step(action)
        profit = -energy * true_price

        # Lorenzo Giannuzzo: Add degradation cost to reward
        degradation_cost = abs(energy) * DEGRADATION_COST_PER_MWH / (2 * 6000)

        soc_penalty = 0.0
        if self.battery.soc > SOC_MAX:
            soc_penalty = -abs(self.battery.soc - SOC_MAX) * SOC_VIOLATION_PENALTY
        elif self.battery.soc < SOC_MIN:
            soc_penalty = -abs(self.battery.soc - SOC_MIN) * SOC_VIOLATION_PENALTY

        # Lorenzo Giannuzzo: Reward includes profit minus degradation cost
        reward = profit - degradation_cost + soc_penalty

        self.current_step += 1
        done = self.current_step >= min(self.max_steps, self.episode_length)
        truncated = False

        if done:
            obs = self._get_observation() if self.current_step < self.max_steps else np.zeros(26, dtype=np.float32)
        else:
            obs = self._get_observation()

        info = {
            'energy': energy,
            'price': true_price,
            'profit': profit,
            'degradation_cost': degradation_cost,
            'soc': self.battery.soc
        }

        return obs, reward, done, truncated, info


class TrainingCallback(BaseCallback):
    """Lorenzo Giannuzzo: Training callback for monitoring PPO progress"""

    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.rollout_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Lorenzo Giannuzzo: Called at end of each rollout to print training statistics"""
        self.rollout_count += 1

        if self.rollout_count % 5 == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            else:
                mean_reward = 0

            reward_per_day = mean_reward / 20
            reward_per_year = reward_per_day * 365

            status = "LOSS" if mean_reward < 0 else "LEARNING" if mean_reward < 2000 else "GOOD" if mean_reward < 5000 else "EXCELLENT"

            print(f"  Rollout {self.rollout_count:3d} | "
                  f"R(20d): EUR {mean_reward:>8,.0f} | "
                  f"~ EUR {reward_per_year:>8,.0f}/yr | "
                  f"{status}")


class PSOOptimizer:
    """Lorenzo Giannuzzo: PSO Optimizer with degradation cost integrated"""

    def __init__(self, n_particles=50, n_iterations=150, w_start=0.95, w_end=0.2, c1=2.0, c2=2.0):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.stagnation_limit = 15

    def optimize(self, battery, forecast_prices, horizon=PSO_HORIZON):
        """Lorenzo Giannuzzo: Run PSO optimization over forecast horizon"""
        n_hours = min(horizon, len(forecast_prices))
        max_power = battery.max_power

        positions = self._smart_initialization(battery, forecast_prices[:n_hours], max_power)
        velocities = np.random.uniform(-1.0, 1.0, (self.n_particles, n_hours))

        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self._evaluate(battery, p, forecast_prices[:n_hours])
                                         for p in positions])

        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        stagnation_counter = 0

        for iteration in range(self.n_iterations):
            # Lorenzo Giannuzzo: Linearly decreasing inertia weight
            w = self.w_start - (self.w_start - self.w_end) * (iteration / self.n_iterations)

            for i in range(self.n_particles):
                r1, r2 = np.random.random(n_hours), np.random.random(n_hours)
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (global_best_position - positions[i])

                velocities[i] = w * velocities[i] + cognitive + social
                max_velocity = max_power * 0.5
                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], -max_power, max_power)

                score = self._evaluate(battery, positions[i], forecast_prices[:n_hours])

                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                    if score > global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()
                        stagnation_counter = 0

            stagnation_counter += 1

            # Lorenzo Giannuzzo: Reinitialize worst particles if stagnated
            if stagnation_counter > self.stagnation_limit:
                n_reinit = self.n_particles // 4
                worst_indices = np.argsort(personal_best_scores)[:n_reinit]
                for idx in worst_indices:
                    noise = np.random.uniform(-max_power * 0.3, max_power * 0.3, n_hours)
                    positions[idx] = np.clip(global_best_position + noise, -max_power, max_power)
                    velocities[idx] = np.random.uniform(-0.5, 0.5, n_hours)
                stagnation_counter = 0

        return global_best_position[0]

    def _smart_initialization(self, battery, prices, max_power):
        """Lorenzo Giannuzzo: Initialize particles with price-based heuristics"""
        n_hours = len(prices)
        positions = np.zeros((self.n_particles, n_hours))

        price_low = np.percentile(prices, 25)
        price_high = np.percentile(prices, 75)

        for i in range(self.n_particles):
            # Lorenzo Giannuzzo: Aggressive strategy (33%)
            if i < self.n_particles // 3:
                for h in range(n_hours):
                    if prices[h] < price_low:
                        positions[i, h] = np.random.uniform(0.5 * max_power, max_power)
                    elif prices[h] > price_high:
                        positions[i, h] = np.random.uniform(-max_power, -0.5 * max_power)
                    else:
                        positions[i, h] = np.random.uniform(-0.3 * max_power, 0.3 * max_power)
            # Lorenzo Giannuzzo: Conservative strategy (33%)
            elif i < 2 * self.n_particles // 3:
                for h in range(n_hours):
                    if prices[h] < price_low:
                        positions[i, h] = np.random.uniform(0, 0.7 * max_power)
                    elif prices[h] > price_high:
                        positions[i, h] = np.random.uniform(-0.7 * max_power, 0)
                    else:
                        positions[i, h] = np.random.uniform(-0.2 * max_power, 0.2 * max_power)
            # Lorenzo Giannuzzo: Random exploration (33%)
            else:
                positions[i] = np.random.uniform(-max_power, max_power, n_hours)

        return positions

    def _evaluate(self, battery, actions, prices):
        """Lorenzo Giannuzzo: Evaluate particle fitness with degradation cost"""
        bat = battery.copy()
        profit = 0.0

        for power, price in zip(actions, prices):
            # Lorenzo Giannuzzo: Charging
            if power > 0.01:
                if bat.soc >= bat.soc_max:
                    continue
                max_energy_storable = (bat.soc_max - bat.soc) * bat.capacity
                max_power_available = max_energy_storable / (1.0 * bat.efficiency)
                actual_power = min(power, max_power_available)

                if actual_power > 0.01:
                    energy_from_grid = actual_power
                    energy_stored = energy_from_grid * bat.efficiency
                    bat.soc += energy_stored / bat.capacity
                    bat.soc = min(bat.soc, bat.soc_max)
                    # Lorenzo Giannuzzo: Cost of buying energy
                    profit -= energy_from_grid * price
                    # Lorenzo Giannuzzo: Degradation cost
                    profit -= energy_from_grid * DEGRADATION_COST_PER_MWH / (2 * 6000)

            # Lorenzo Giannuzzo: Discharging
            elif power < -0.01:
                if bat.soc <= bat.soc_min:
                    continue

                max_energy_available = (bat.soc - bat.soc_min) * bat.capacity
                max_power_available = max_energy_available
                actual_power = min(-power, max_power_available)

                if actual_power > 0.01:
                    energy_to_grid = actual_power
                    energy_from_battery = energy_to_grid
                    bat.soc -= energy_from_battery / bat.capacity
                    bat.soc = max(bat.soc, bat.soc_min)
                    # Lorenzo Giannuzzo: Revenue from selling energy
                    profit += energy_to_grid * price
                    # Lorenzo Giannuzzo: Degradation cost
                    profit -= energy_to_grid * DEGRADATION_COST_PER_MWH / (2 * 6000)

        return profit


def save_plot_as_pdf(fig, filename_base, results_folder, dpi=500):
    """
    Lorenzo Giannuzzo: Helper function to save plots as both PNG and PDF
    """
    # Lorenzo Giannuzzo: PNG with dpi=300
    filename_png = os.path.join(results_folder, f'{filename_base}.png')
    fig.savefig(filename_png, dpi=300, bbox_inches='tight')

    # Lorenzo Giannuzzo: PDF with dpi=500
    filename_pdf = os.path.join(results_folder, f'{filename_base}.pdf')
    fig.savefig(filename_pdf, dpi=dpi, bbox_inches='tight', format='pdf')

    return filename_png, filename_pdf


def plot_error_types_comparison(results_folder):
    """
    Lorenzo Giannuzzo: Plot showing the 4 different error types characteristics
    """
    try:
        print(f"\n  Generating error types comparison plot...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Lorenzo Giannuzzo: Simulate 168 hours (1 week) of price data
        np.random.seed(42)
        hours = 168
        base_price = 100.0

        # Lorenzo Giannuzzo: Simulate realistic price pattern (daily cycle)
        time = np.arange(hours)
        true_prices = base_price + 30 * np.sin(2 * np.pi * time / 24) + 10 * np.sin(2 * np.pi * time / 168)

        # Lorenzo Giannuzzo: Generate 4 types of errors
        error_generators = {
            'uniform': ForecastErrorGenerator('uniform'),
            'normal': ForecastErrorGenerator('normal'),
            'ornstein-uhlenbeck': ForecastErrorGenerator('ornstein-uhlenbeck'),
            'fixed-bias': ForecastErrorGenerator('fixed-bias')
        }

        error_titles = {
            'uniform': 'UNIFORM ERROR\nRandom ±15% (each hour independent)',
            'normal': 'NORMAL ERROR\nGaussian σ=5% (concentrated near 0%)',
            'ornstein-uhlenbeck': 'ORNSTEIN-UHLENBECK ERROR\nTime-correlated (realistic persistence)',
            'fixed-bias': 'FIXED-BIAS ERROR\nDeterministic +10% (always overestimate)'
        }

        error_descriptions = {
            'uniform': 'Completely random errors\nEqual probability for all values in [-15%, +15%]',
            'normal': 'Gaussian distribution\nMost errors near 0%, rare extreme values',
            'ornstein-uhlenbeck': 'Autocorrelated errors\nIf hour N is +5%, hour N+1 likely similar',
            'fixed-bias': 'Constant bias\nAlways predicts 10% higher than true price'
        }

        plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        error_types_list = ['uniform', 'normal', 'ornstein-uhlenbeck', 'fixed-bias']

        for idx, (pos, error_type) in enumerate(zip(plot_positions, error_types_list)):
            ax = axes[pos]

            # Lorenzo Giannuzzo: Generate forecast prices
            error_gen = error_generators[error_type]
            error_gen.reset()

            forecast_prices = []
            errors_pct = []

            for t in range(hours):
                forecast_price, error = error_gen.apply_error(true_prices[t])
                forecast_prices.append(forecast_price)
                errors_pct.append(error * 100)

            # Lorenzo Giannuzzo: Plot prices
            ax.plot(time, true_prices, 'k-', linewidth=2.5, label='True Price', alpha=0.9)
            ax.plot(time, forecast_prices, 'r--', linewidth=2, label='Forecast Price', alpha=0.7)

            ax.fill_between(time, true_prices, forecast_prices, alpha=0.2, color='red')

            ax.set_xlabel('Hour', fontweight='bold', fontsize=11)
            ax.set_ylabel('Price (EUR/MWh)', fontweight='bold', fontsize=11)
            ax.set_title(error_titles[error_type], fontweight='bold', fontsize=12)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)

            # Lorenzo Giannuzzo: Add statistics box
            mean_error = np.mean(errors_pct)
            std_error = np.std(errors_pct)
            min_error = np.min(errors_pct)
            max_error = np.max(errors_pct)

            stats_text = (f'Mean: {mean_error:+.1f}%\n'
                          f'Std: {std_error:.1f}%\n'
                          f'Range: [{min_error:+.1f}%, {max_error:+.1f}%]')

            ax.text(0.02, 0.98, stats_text,
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    fontsize=9, fontweight='bold')

            # Lorenzo Giannuzzo: Add description
            ax.text(0.98, 0.02, error_descriptions[error_type],
                    transform=ax.transAxes, verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                    fontsize=8, style='italic')

        plt.suptitle('Comparison of 4 Forecast Error Distributions\n'
                     'One Week Simulation (168 hours)',
                     fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Lorenzo Giannuzzo: Save both PNG and PDF
        filename_png, filename_pdf = save_plot_as_pdf(fig, 'error_types_comparison', results_folder)

        plt.close()
        print(f"  Saved error types comparison (PNG): {filename_png}")
        print(f"  Saved error types comparison (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR plotting error types comparison: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_heatmaps(results, noise_type, results_folder):
    """
    Lorenzo Giannuzzo: Generate heatmap visualizations for SOC, prices, battery charge/discharge
    """
    try:
        print(f"  Generating heatmaps for {noise_type}...")

        safe_name = noise_type.replace('ornstein-uhlenbeck', 'ou')

        # Lorenzo Giannuzzo: Prepare data - reshape to (days, 24 hours)
        n_hours = len(results)
        n_days = n_hours // 24

        if n_days == 0:
            print(f"  Not enough data for heatmaps")
            return

        # Lorenzo Giannuzzo: Truncate to complete days
        results_truncated = results.iloc[:n_days * 24].copy()

        # Lorenzo Giannuzzo: Reshape data into 2D arrays (days x hours)
        soc_matrix = results_truncated['PSO_SOC'].values.reshape(n_days,
                                                                 24) * 100  # Lorenzo Giannuzzo: Convert to percentage
        price_matrix = results_truncated['True_Price'].values.reshape(n_days, 24)

        # Lorenzo Giannuzzo: Separate charge and discharge
        energy_matrix = results_truncated['PSO_Energy_Actual'].values.reshape(n_days, 24)
        charge_matrix = np.where(energy_matrix > 0, energy_matrix, 0)
        discharge_matrix = np.where(energy_matrix < 0, -energy_matrix, 0)

        # Lorenzo Giannuzzo: Create figure with 4 subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))

        # Lorenzo Giannuzzo: 1. PV Production (placeholder - not applicable for BESS)
        ax = axes[0, 0]
        ax.text(0.5, 0.5, 'N/A\n(BESS only system)',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('PV Production [kW]', fontweight='bold', fontsize=12)
        ax.axis('off')

        # Lorenzo Giannuzzo: 2. Price
        ax = axes[0, 1]
        im = ax.imshow(price_matrix, aspect='auto', cmap='plasma', interpolation='bilinear')
        ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
        ax.set_ylabel('Day of Year', fontweight='bold', fontsize=11)
        ax.set_title('Price [EUR/kWh]', fontweight='bold', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xticklabels(np.arange(0, 24, 3))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('EUR/MWh', fontweight='bold')

        # Lorenzo Giannuzzo: 3. Battery Charge
        ax = axes[1, 0]
        im = ax.imshow(charge_matrix, aspect='auto', cmap='YlOrRd', interpolation='bilinear',
                       vmin=0, vmax=np.max(charge_matrix) if np.max(charge_matrix) > 0 else 1)
        ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
        ax.set_ylabel('Day of Year', fontweight='bold', fontsize=11)
        ax.set_title('Battery Charge [kW]', fontweight='bold', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xticklabels(np.arange(0, 24, 3))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('MW', fontweight='bold')

        # Lorenzo Giannuzzo: 4. Battery Discharge
        ax = axes[1, 1]
        im = ax.imshow(discharge_matrix, aspect='auto', cmap='Blues', interpolation='bilinear',
                       vmin=0, vmax=np.max(discharge_matrix) if np.max(discharge_matrix) > 0 else 1)
        ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
        ax.set_ylabel('Day of Year', fontweight='bold', fontsize=11)
        ax.set_title('Battery Discharge [kW]', fontweight='bold', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xticklabels(np.arange(0, 24, 3))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('MW', fontweight='bold')

        # Lorenzo Giannuzzo: 5. Exported Power (placeholder - same as discharge for BESS)
        ax = axes[2, 0]
        im = ax.imshow(discharge_matrix, aspect='auto', cmap='Greens', interpolation='bilinear',
                       vmin=0, vmax=np.max(discharge_matrix) if np.max(discharge_matrix) > 0 else 1)
        ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
        ax.set_ylabel('Day of Year', fontweight='bold', fontsize=11)
        ax.set_title('Exported Power [kW]', fontweight='bold', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xticklabels(np.arange(0, 24, 3))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('MW', fontweight='bold')

        # Lorenzo Giannuzzo: 6. SOC
        ax = axes[2, 1]
        ax.text(0.5, 0.5, 'Battery SOC\nSee separate plot',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Unused / Notes', fontweight='bold', fontsize=12)
        ax.axis('off')

        plt.suptitle(f'BESS Operation Heatmaps - {noise_type.upper()}\n'
                     f'Year-long Analysis (365 days x 24 hours)',
                     fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Lorenzo Giannuzzo: Save both PNG and PDF
        filename_png, filename_pdf = save_plot_as_pdf(fig, f'heatmaps_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

        # Lorenzo Giannuzzo: Create separate SOC heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(soc_matrix, aspect='auto', cmap='RdYlGn', interpolation='bilinear',
                       vmin=0, vmax=100)
        ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=12)
        ax.set_ylabel('Day of Year', fontweight='bold', fontsize=12)
        ax.set_title(f'Battery State of Charge - {noise_type.upper()}\n'
                     f'(PSO Strategy)',
                     fontweight='bold', fontsize=14)
        ax.set_xticks(np.arange(0, 24, 2))
        ax.set_xticklabels(np.arange(0, 24, 2))

        # Lorenzo Giannuzzo: Add horizontal lines for months (approximate)
        for month in range(1, 12):
            day_of_year = month * 30
            if day_of_year < n_days:
                ax.axhline(y=day_of_year, color='white', linestyle='--', alpha=0.3, linewidth=0.5)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SOC (%)', fontweight='bold', fontsize=12)

        plt.tight_layout()

        # Lorenzo Giannuzzo: Save SOC heatmap
        filename_png, filename_pdf = save_plot_as_pdf(fig, f'heatmap_soc_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved SOC heatmap (PNG): {filename_png}")
        print(f"  Saved SOC heatmap (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR generating heatmaps: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_soh_comparison(results, noise_type, results_folder):
    """
    Lorenzo Giannuzzo: Plot State of Health comparison between PSO and PPO
    """
    try:
        print(f"  Generating SOH comparison plot for {noise_type}...")

        safe_name = noise_type.replace('ornstein-uhlenbeck', 'ou')

        # Lorenzo Giannuzzo: Calculate SOH from throughput
        # Lorenzo Giannuzzo: Track cumulative throughput for both PSO and PPO
        pso_throughput = []
        ppo_throughput = []
        pso_soh = []
        ppo_soh = []

        cumulative_pso_throughput = 0.0
        cumulative_ppo_throughput = 0.0

        for idx in range(len(results)):
            # Lorenzo Giannuzzo: Add absolute energy throughput
            cumulative_pso_throughput += abs(results['PSO_Energy_Actual'].iloc[idx])
            cumulative_ppo_throughput += abs(results['PPO_Energy_Actual'].iloc[idx])

            pso_throughput.append(cumulative_pso_throughput)
            ppo_throughput.append(cumulative_ppo_throughput)

            # Lorenzo Giannuzzo: Calculate equivalent cycles
            pso_cycles = (cumulative_pso_throughput * 1000) / (2 * BATTERY_CAPACITY * 1000)
            ppo_cycles = (cumulative_ppo_throughput * 1000) / (2 * BATTERY_CAPACITY * 1000)

            # Lorenzo Giannuzzo: Calculate SOH using degradation function
            pso_capacity_pct = degradation(pso_cycles)
            ppo_capacity_pct = degradation(ppo_cycles)

            pso_soh.append(pso_capacity_pct)
            ppo_soh.append(ppo_capacity_pct)

        # Lorenzo Giannuzzo: Add SOH to results dataframe
        results['PSO_SOH'] = pso_soh
        results['PPO_SOH'] = ppo_soh
        results['PSO_Throughput_MWh'] = pso_throughput
        results['PPO_Throughput_MWh'] = ppo_throughput

        # Lorenzo Giannuzzo: Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Lorenzo Giannuzzo: 1. SOH over time
        ax = axes[0]
        ax.plot(results['Hour'], results['PSO_SOH'],
                label='PSO', color='darkblue', linewidth=2.5, alpha=0.8)
        ax.plot(results['Hour'], results['PPO_SOH'],
                label='PPO', color='coral', linewidth=2.5, alpha=0.8)

        # Lorenzo Giannuzzo: Add reference lines
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Initial SOH')
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1, label='EOL Threshold (80%)')

        ax.set_xlabel('Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('SOH (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'State of Health Comparison - {noise_type.upper()}',
                     fontweight='bold', fontsize=13)
        ax.legend(fontsize=11, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([75, 101])

        # Lorenzo Giannuzzo: Add statistics box
        final_pso_soh = results['PSO_SOH'].iloc[-1]
        final_ppo_soh = results['PPO_SOH'].iloc[-1]
        soh_diff = final_ppo_soh - final_pso_soh

        stats_text = (f'Final SOH:\n'
                      f'PSO: {final_pso_soh:.2f}%\n'
                      f'PPO: {final_ppo_soh:.2f}%\n'
                      f'Diff: {soh_diff:+.2f}%')

        ax.text(0.98, 0.02, stats_text,
                transform=ax.transAxes, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=10, fontweight='bold')

        # Lorenzo Giannuzzo: 2. Cumulative throughput
        ax = axes[1]
        ax.plot(results['Hour'], results['PSO_Throughput_MWh'],
                label='PSO', color='darkblue', linewidth=2.5, alpha=0.8)
        ax.plot(results['Hour'], results['PPO_Throughput_MWh'],
                label='PPO', color='coral', linewidth=2.5, alpha=0.8)

        ax.set_xlabel('Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cumulative Throughput (MWh)', fontweight='bold', fontsize=12)
        ax.set_title('Cumulative Energy Throughput', fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Lorenzo Giannuzzo: Add statistics box
        final_pso_throughput = results['PSO_Throughput_MWh'].iloc[-1]
        final_ppo_throughput = results['PPO_Throughput_MWh'].iloc[-1]
        throughput_diff = final_ppo_throughput - final_pso_throughput
        throughput_diff_pct = (throughput_diff / final_pso_throughput * 100) if final_pso_throughput > 0 else 0

        stats_text = (f'Final Throughput:\n'
                      f'PSO: {final_pso_throughput:.1f} MWh\n'
                      f'PPO: {final_ppo_throughput:.1f} MWh\n'
                      f'Diff: {throughput_diff:+.1f} MWh ({throughput_diff_pct:+.1f}%)')

        ax.text(0.98, 0.02, stats_text,
                transform=ax.transAxes, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
                fontsize=10, fontweight='bold')

        # Lorenzo Giannuzzo: 3. SOH degradation rate (derivative)
        ax = axes[2]

        # Lorenzo Giannuzzo: Calculate degradation rate (negative derivative of SOH)
        window = 24 * 7  # Lorenzo Giannuzzo: 7-day rolling window
        pso_soh_series = pd.Series(results['PSO_SOH'])
        ppo_soh_series = pd.Series(results['PPO_SOH'])

        # Lorenzo Giannuzzo: Calculate rolling degradation rate
        pso_deg_rate = -pso_soh_series.diff().rolling(window=window).mean() * 24  # Lorenzo Giannuzzo: per day
        ppo_deg_rate = -ppo_soh_series.diff().rolling(window=window).mean() * 24  # Lorenzo Giannuzzo: per day

        ax.plot(results['Hour'], pso_deg_rate,
                label='PSO', color='darkblue', linewidth=2, alpha=0.7)
        ax.plot(results['Hour'], ppo_deg_rate,
                label='PPO', color='coral', linewidth=2, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('Degradation Rate (%/day)', fontweight='bold', fontsize=12)
        ax.set_title('SOH Degradation Rate (7-day rolling average)', fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Lorenzo Giannuzzo: Add average degradation rate
        avg_pso_deg = pso_deg_rate[window:].mean()
        avg_ppo_deg = ppo_deg_rate[window:].mean()

        stats_text = (f'Avg Degradation Rate:\n'
                      f'PSO: {avg_pso_deg:.4f}%/day\n'
                      f'PPO: {avg_ppo_deg:.4f}%/day')

        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8),
                fontsize=10, fontweight='bold')

        plt.tight_layout()

        # Lorenzo Giannuzzo: Save both PNG and PDF
        filename_png, filename_pdf = save_plot_as_pdf(fig, f'soh_comparison_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

        # Lorenzo Giannuzzo: Create additional plot: SOH vs Throughput scatter
        fig, ax = plt.subplots(figsize=(10, 8))

        # Lorenzo Giannuzzo: Sample every 24 hours to reduce clutter
        sample_indices = range(0, len(results), 24)

        ax.scatter(results['PSO_Throughput_MWh'].iloc[sample_indices],
                   results['PSO_SOH'].iloc[sample_indices],
                   c=results['Hour'].iloc[sample_indices], cmap='viridis',
                   s=50, alpha=0.6, edgecolors='black', linewidth=0.5,
                   label='PSO')

        ax.scatter(results['PPO_Throughput_MWh'].iloc[sample_indices],
                   results['PPO_SOH'].iloc[sample_indices],
                   c=results['Hour'].iloc[sample_indices], cmap='plasma',
                   s=50, alpha=0.6, edgecolors='black', linewidth=0.5,
                   marker='^', label='PPO')

        ax.set_xlabel('Cumulative Throughput (MWh)', fontweight='bold', fontsize=12)
        ax.set_ylabel('SOH (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'SOH vs Throughput - {noise_type.upper()}\n'
                     f'(Colored by time progression)',
                     fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Lorenzo Giannuzzo: Add colorbar for time
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(vmin=0, vmax=results['Hour'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Hour', fontweight='bold', fontsize=11)

        plt.tight_layout()

        # Lorenzo Giannuzzo: Save scatter plot
        filename_png, filename_pdf = save_plot_as_pdf(fig, f'soh_vs_throughput_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved SOH vs Throughput (PNG): {filename_png}")
        print(f"  Saved SOH vs Throughput (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR generating SOH comparison: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_soh_summary(all_results, summary_df, results_folder):
    """
    Lorenzo Giannuzzo: Summary plot comparing SOH degradation across all error types
    """
    try:
        print(f"  Generating SOH summary across error types...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        noise_labels = {
            'uniform': 'UNIFORM',
            'normal': 'NORMAL',
            'ornstein-uhlenbeck': 'O-U',
            'fixed-bias': 'FIXED-BIAS'
        }

        # Lorenzo Giannuzzo: Collect final SOH values
        final_soh_data = []

        for noise_type in NOISE_TYPES_TO_TEST:
            if noise_type not in all_results:
                continue

            results = all_results[noise_type]

            # Lorenzo Giannuzzo: Calculate SOH if not already in dataframe
            if 'PSO_SOH' not in results.columns:
                pso_throughput_cumulative = results['PSO_Energy_Actual'].abs().cumsum()
                ppo_throughput_cumulative = results['PPO_Energy_Actual'].abs().cumsum()

                pso_cycles = (pso_throughput_cumulative * 1000) / (2 * BATTERY_CAPACITY * 1000)
                ppo_cycles = (ppo_throughput_cumulative * 1000) / (2 * BATTERY_CAPACITY * 1000)

                results['PSO_SOH'] = pso_cycles.apply(degradation)
                results['PPO_SOH'] = ppo_cycles.apply(degradation)

            final_soh_data.append({
                'noise_type': noise_type,
                'pso_final_soh': results['PSO_SOH'].iloc[-1],
                'ppo_final_soh': results['PPO_SOH'].iloc[-1]
            })

        # Lorenzo Giannuzzo: 1. Final SOH bar chart
        ax = axes[0, 0]
        x = np.arange(len(final_soh_data))
        width = 0.35

        pso_sohs = [d['pso_final_soh'] for d in final_soh_data]
        ppo_sohs = [d['ppo_final_soh'] for d in final_soh_data]
        labels = [noise_labels.get(d['noise_type'], d['noise_type']) for d in final_soh_data]

        bars1 = ax.bar(x - width / 2, pso_sohs, width, label='PSO',
                       color='darkblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width / 2, ppo_sohs, width, label='PPO',
                       color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=2, label='EOL (80%)')
        ax.set_ylabel('Final SOH (%)', fontweight='bold', fontsize=12)
        ax.set_title('Final State of Health by Error Type', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([75, 101])

        # Lorenzo Giannuzzo: Add values on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Lorenzo Giannuzzo: 2. SOH evolution for all error types (PSO)
        ax = axes[0, 1]
        for noise_type in NOISE_TYPES_TO_TEST:
            if noise_type in all_results:
                results = all_results[noise_type]
                # Lorenzo Giannuzzo: Sample every 24 hours for clarity
                sample_indices = range(0, len(results), 24)
                ax.plot(results['Hour'].iloc[sample_indices],
                        results['PSO_SOH'].iloc[sample_indices],
                        label=noise_labels.get(noise_type, noise_type),
                        linewidth=2, alpha=0.8)

        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Hour', fontweight='bold', fontsize=11)
        ax.set_ylabel('SOH (%)', fontweight='bold', fontsize=11)
        ax.set_title('PSO: SOH Evolution by Error Type', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([75, 101])

        # Lorenzo Giannuzzo: 3. SOH evolution for all error types (PPO)
        ax = axes[1, 0]
        for noise_type in NOISE_TYPES_TO_TEST:
            if noise_type in all_results:
                results = all_results[noise_type]
                # Lorenzo Giannuzzo: Sample every 24 hours for clarity
                sample_indices = range(0, len(results), 24)
                ax.plot(results['Hour'].iloc[sample_indices],
                        results['PPO_SOH'].iloc[sample_indices],
                        label=noise_labels.get(noise_type, noise_type),
                        linewidth=2, alpha=0.8)

        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Hour', fontweight='bold', fontsize=11)
        ax.set_ylabel('SOH (%)', fontweight='bold', fontsize=11)
        ax.set_title('PPO: SOH Evolution by Error Type', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([75, 101])

        # Lorenzo Giannuzzo: 4. SOH difference (PPO - PSO)
        ax = axes[1, 1]

        soh_differences = [d['ppo_final_soh'] - d['pso_final_soh'] for d in final_soh_data]
        colors = ['green' if diff > 0 else 'red' for diff in soh_differences]

        bars = ax.bar(x, soh_differences, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.set_ylabel('SOH Difference (PPO - PSO) [%]', fontweight='bold', fontsize=12)
        ax.set_title('SOH Difference: PPO vs PSO', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Lorenzo Giannuzzo: Add values on bars
        for bar, diff in zip(bars, soh_differences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{diff:+.3f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')

        plt.suptitle('State of Health Analysis Across Error Distributions',
                     fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Lorenzo Giannuzzo: Save both PNG and PDF
        filename_png, filename_pdf = save_plot_as_pdf(fig, 'soh_summary_all_errors', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR generating SOH summary: {e}")
        traceback.print_exc()
        plt.close('all')


def train_ppo_clean(prices_df, noise_type, model_path):
    """Lorenzo Giannuzzo: Initial PPO training"""
    error_descriptions = {
        'uniform': 'Uniform distribution [-15%, +15%]',
        'normal': 'Normal distribution (σ=5%, clipped ±15%)',
        'ornstein-uhlenbeck': 'O-U process (time-correlated, ±15%)',
        'fixed-bias': 'Fixed bias +10% (deterministic)'
    }

    print(f"\nINITIAL TRAINING - {noise_type.upper()}")
    print(f"  Error type: {error_descriptions.get(noise_type, noise_type)}")
    print("=" * 70)

    prices = prices_df['EUR/MWh'].values
    error_gen = ForecastErrorGenerator(noise_type)
    env = BatteryTradingEnvClean(prices, error_gen)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[PPO_HIDDEN_DIM_1, PPO_HIDDEN_DIM_2],
            vf=[PPO_HIDDEN_DIM_1, PPO_HIDDEN_DIM_2]
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device='auto'
    )

    callback = TrainingCallback()

    print(f"\nSETUP:")
    print(f"  Features:    26")
    print(f"  Network:     {PPO_HIDDEN_DIM_1}-{PPO_HIDDEN_DIM_2}")
    print(f"  Timesteps:   {PPO_TOTAL_TIMESTEPS:,}")
    print(f"  Degradation: {DEGRADATION_COST_PER_MWH} EUR/MWh")
    print(f"  Device:      {model.device}")
    print("=" * 70)
    print(f"\nStarting initial training...\n")

    model.learn(
        total_timesteps=PPO_TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True
    )

    model.save(model_path)
    print(f"\nInitial training complete. Model saved: {model_path}")

    return model


def monthly_retrain_ppo_corrected(ppo_model, all_prices, current_hour, noise_type, error_gen, forecast_history,
                                  n_test_hours):
    """Lorenzo Giannuzzo: Monthly retraining with ACTUAL forecast errors"""
    lookback_hours = MONTHLY_RETRAINING_LOOKBACK * 24
    start_hour = max(0, current_hour - lookback_hours)

    retrain_true_prices = all_prices[start_hour:current_hour]

    print(f"     Retraining on {len(retrain_true_prices)} hours ({len(retrain_true_prices) / 24:.0f} days)")
    print(f"     Period: hour {start_hour} to {current_hour} (absolute)")

    n_train_hours = len(all_prices) - n_test_hours

    print(f"     Training hours: {n_train_hours}, Test hours: {n_test_hours}")

    if current_hour > n_train_hours:
        test_hour_end = current_hour - n_train_hours
        test_hour_start = max(0, start_hour - n_train_hours)

        print(f"     Need test forecasts from hour {test_hour_start} to {test_hour_end}")
        print(f"     Available forecast history: {len(forecast_history)} entries")

        if len(forecast_history) > 0:
            print(
                f"     Forecast hour range: {min([f['hour'] for f in forecast_history])} to {max([f['hour'] for f in forecast_history])}")

        relevant_forecasts = []
        for f in forecast_history:
            if test_hour_start <= f['hour'] < test_hour_end:
                relevant_forecasts.append(f)

        if len(relevant_forecasts) > 0:
            print(f"     Found {len(relevant_forecasts)} ACTUAL forecast observations")

            temp_env = BatteryTradingEnvWithActualForecasts(
                retrain_true_prices,
                relevant_forecasts,
                offset=test_hour_start
            )
        else:
            print(f"     No forecasts found in required range")
            print(f"     Using random errors as fallback")
            temp_env = BatteryTradingEnvClean(retrain_true_prices, error_gen)
    else:
        print(f"     Retraining window entirely in training period")
        print(f"     Using random errors (no test data yet)")
        temp_env = BatteryTradingEnvClean(retrain_true_prices, error_gen)

    original_env = ppo_model.env
    ppo_model.set_env(temp_env)

    print(f"     Training for {MONTHLY_RETRAINING_TIMESTEPS:,} timesteps...")

    ppo_model.learn(
        total_timesteps=MONTHLY_RETRAINING_TIMESTEPS,
        reset_num_timesteps=False,
        progress_bar=False
    )

    ppo_model.set_env(original_env)

    print(f"     Retraining complete")


def test_with_monthly_retraining(pso, ppo_model, prices_df, all_train_test_prices, noise_type, error_gen):
    """Lorenzo Giannuzzo: Test PSO vs PPO with MONTHLY RETRAINING"""
    prices = prices_df['EUR/MWh'].values
    dates = pd.to_datetime(prices_df['Data'], dayfirst=True)
    n_hours = len(prices)

    print(f"  Testing on {n_hours:,} hours ({n_hours / 24:.0f} days)")

    if MONTHLY_RETRAINING_ENABLED:
        expected_retrains = n_hours // (MONTHLY_RETRAINING_FREQUENCY * 24)
        print(f"  MONTHLY RETRAINING:")
        print(f"     Frequency:  Every {MONTHLY_RETRAINING_FREQUENCY} days")
        print(f"     Lookback:   {MONTHLY_RETRAINING_LOOKBACK} days")
        print(f"     Timesteps:  {MONTHLY_RETRAINING_TIMESTEPS:,} per retrain")
        print(f"     Expected:   ~{expected_retrains} retrainings")
        print(f"     Uses ACTUAL forecast errors seen during testing")

    action_values = np.linspace(-BATTERY_POWER, BATTERY_POWER, N_DISCRETE_ACTIONS)

    battery_pso = Battery()
    pso_energy_actual = []
    pso_socs = []
    pso_instant_profits = []
    pso_profits = []
    pso_degradation_costs = []  # Lorenzo Giannuzzo: Track degradation separately

    battery_ppo = Battery()
    ppo_energy_actual = []
    ppo_socs = []
    ppo_instant_profits = []
    ppo_profits = []
    ppo_degradation_costs = []  # Lorenzo Giannuzzo: Track degradation separately

    forecast_prices_all = []
    actual_forecast_history = []

    cumulative_pso = 0
    cumulative_ppo = 0
    cumulative_pso_degradation = 0  # Lorenzo Giannuzzo: Separate degradation tracking
    cumulative_ppo_degradation = 0  # Lorenzo Giannuzzo: Separate degradation tracking

    monthly_retrains = 0
    retrain_times = []
    performance_before_retrain = []

    progress_markers = [0] + [int(n_hours * i / 20) for i in range(1, 21)]
    start_time = datetime.now()

    test_start_offset = len(all_train_test_prices) - len(prices)

    for t in range(n_hours):
        true_price = prices[t]
        current_day = t // 24

        horizon = min(PSO_HORIZON, n_hours - t)
        forecast_window = []

        for h in range(horizon):
            if t + h < n_hours:
                fp, _ = error_gen.apply_error(prices[t + h])
                forecast_window.append(fp)

        current_forecast = forecast_window[0] if len(forecast_window) > 0 else true_price
        forecast_prices_all.append(current_forecast)

        actual_forecast_history.append({
            'hour': t,
            'true_price': true_price,
            'forecast_price': current_forecast
        })

        # Lorenzo Giannuzzo: Monthly retraining
        if MONTHLY_RETRAINING_ENABLED and t > 0 and current_day > 0 and current_day % MONTHLY_RETRAINING_FREQUENCY == 0:
            if current_day not in [rt // 24 for rt in retrain_times]:
                print(f"\n  MONTHLY RETRAINING #{monthly_retrains + 1} at day {current_day} (hour {t})...")

                recent_window = min(MONTHLY_RETRAINING_FREQUENCY * 24, len(ppo_instant_profits))
                perf_before = sum(ppo_instant_profits[-recent_window:])
                performance_before_retrain.append(perf_before)

                absolute_hour = test_start_offset + t

                monthly_retrain_ppo_corrected(
                    ppo_model,
                    all_train_test_prices,
                    absolute_hour,
                    noise_type,
                    error_gen,
                    actual_forecast_history,
                    len(prices)
                )

                monthly_retrains += 1
                retrain_times.append(t)

                print(f"  Retrained! Total retrains: {monthly_retrains}")
                print()

        # Lorenzo Giannuzzo: PSO optimization (uses degradation cost internally)
        action_pso = pso.optimize(battery_pso, forecast_window, horizon)
        energy_pso = battery_pso.step(action_pso)

        # Lorenzo Giannuzzo: Calculate REAL operational profit (no degradation cost)
        profit_pso = -energy_pso * true_price

        # Lorenzo Giannuzzo: Calculate degradation cost separately (for tracking only)
        degradation_cost_pso = abs(energy_pso) * DEGRADATION_COST_PER_MWH / (2 * 6000)

        cumulative_pso += profit_pso
        cumulative_pso_degradation += degradation_cost_pso

        pso_energy_actual.append(energy_pso)
        pso_socs.append(battery_pso.soc)
        pso_profits.append(cumulative_pso)
        pso_instant_profits.append(profit_pso)
        pso_degradation_costs.append(cumulative_pso_degradation)

        # Lorenzo Giannuzzo: PPO action
        soc = battery_ppo.soc
        forecast_prices_norm = []
        for h in range(FORECAST_HORIZON):
            if h < len(forecast_window):
                forecast_prices_norm.append(forecast_window[h] / 200.0)
            else:
                forecast_prices_norm.append(0.5)

        current_price_norm = true_price / 200.0

        obs = np.array([
            soc,
            *forecast_prices_norm,
            current_price_norm
        ], dtype=np.float32)

        action_idx, _ = ppo_model.predict(obs, deterministic=True)
        action_ppo = float(action_values[action_idx])

        energy_ppo = battery_ppo.step(action_ppo)

        # Lorenzo Giannuzzo: Calculate REAL operational profit (no degradation cost)
        profit_ppo = -energy_ppo * true_price

        # Lorenzo Giannuzzo: Calculate degradation cost separately (for tracking only)
        degradation_cost_ppo = abs(energy_ppo) * DEGRADATION_COST_PER_MWH / (2 * 6000)

        cumulative_ppo += profit_ppo
        cumulative_ppo_degradation += degradation_cost_ppo

        ppo_energy_actual.append(energy_ppo)
        ppo_socs.append(battery_ppo.soc)
        ppo_profits.append(cumulative_ppo)
        ppo_instant_profits.append(profit_ppo)
        ppo_degradation_costs.append(cumulative_ppo_degradation)

        # Lorenzo Giannuzzo: Progress tracking
        if t in progress_markers and t > 0:
            progress_pct = int((t / n_hours * 100))
            elapsed = (datetime.now() - start_time).total_seconds()
            remaining = elapsed * (n_hours - t) / t
            bar_length = 30
            filled = int(bar_length * t / n_hours)
            bar = '#' * filled + '.' * (bar_length - filled)

            info_str = ""
            if MONTHLY_RETRAINING_ENABLED:
                info_str += f" | Retrains:{monthly_retrains}"

            print(f"  [{bar}] {progress_pct:3d}% | "
                  f"PSO: EUR {cumulative_pso:>8,.0f} | PPO: EUR {cumulative_ppo:>8,.0f}"
                  f"{info_str} | ETA: {remaining / 60:.1f}m", end='\r')

    print()

    results = pd.DataFrame({
        'Hour': range(len(pso_energy_actual)),
        'Date': dates[:len(pso_energy_actual)],
        'True_Price': prices[:len(pso_energy_actual)],
        'Forecast_Price': forecast_prices_all[:len(pso_energy_actual)],
        'PSO_Energy_Actual': pso_energy_actual,
        'PSO_SOC': pso_socs,
        'PSO_Instant_Profit': pso_instant_profits,
        'PSO_Cumulative_Profit': pso_profits,
        'PSO_Cumulative_Degradation_Cost': pso_degradation_costs,  # Lorenzo Giannuzzo: Add degradation tracking
        'PPO_Energy_Actual': ppo_energy_actual,
        'PPO_SOC': ppo_socs,
        'PPO_Instant_Profit': ppo_instant_profits,
        'PPO_Cumulative_Profit': ppo_profits,
        'PPO_Cumulative_Degradation_Cost': ppo_degradation_costs  # Lorenzo Giannuzzo: Add degradation tracking
    })

    retraining_stats = {
        'total_retrains': monthly_retrains,
        'retrain_times': retrain_times,
        'performance_before': performance_before_retrain,
        'enabled': MONTHLY_RETRAINING_ENABLED
    }

    # Lorenzo Giannuzzo: Return degradation costs separately
    degradation_info = {
        'pso_total_degradation': cumulative_pso_degradation,
        'ppo_total_degradation': cumulative_ppo_degradation
    }

    return results, cumulative_pso, cumulative_ppo, retraining_stats, degradation_info


def analyze_trading_behavior(results, noise_type):
    """Lorenzo Giannuzzo: Trading behavior analysis"""
    print(f"\n  TRADING BEHAVIOR - {noise_type.upper()}:")
    print(f"  {'-' * 70}")

    for agent in ['PSO', 'PPO']:
        energy_col = f'{agent}_Energy_Actual'

        buys = (results[energy_col] > 0.1).sum()
        sells = (results[energy_col] < -0.1).sum()
        inactive = (results[energy_col].abs() < 0.1).sum()

        avg_buy_price = results[results[energy_col] > 0.1]['True_Price'].mean() if buys > 0 else 0
        avg_sell_price = results[results[energy_col] < -0.1]['True_Price'].mean() if sells > 0 else 0

        soc_min_obs = results[f'{agent}_SOC'].min()
        soc_max_obs = results[f'{agent}_SOC'].max()

        print(f"\n  {agent}:")
        print(f"    Buys:  {buys:>5} hours (avg: EUR {avg_buy_price:>6.2f})")
        print(f"    Sells: {sells:>5} hours (avg: EUR {avg_sell_price:>6.2f})")
        print(f"    Idle:  {inactive:>5} hours ({inactive / len(results) * 100:.1f}%)")

        if buys > 0 and sells > 0:
            spread = avg_sell_price - avg_buy_price
            print(f"    Spread: EUR {spread:>6.2f}", end="")
            if spread > 0:
                print(f" (positive)")
            else:
                print(f" (negative)")

        print(f"    SOC: [{soc_min_obs * 100:.1f}%, {soc_max_obs * 100:.1f}%]")

    print(f"  {'-' * 70}")


def plot_monthly_retraining_performance(results, retrain_stats, noise_type, results_folder):
    """Lorenzo Giannuzzo: Plot with PDF save"""
    safe_name = noise_type.replace('ornstein-uhlenbeck', 'ou')

    try:
        if not retrain_stats['enabled'] or retrain_stats['total_retrains'] == 0:
            print(f"  Skipping retraining plot for {noise_type}")
            return

        print(f"  Generating monthly retraining plot for {noise_type}...")

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        ax = axes[0]
        ax.plot(results['Hour'], results['PSO_Cumulative_Profit'],
                label='PSO (Static)', color='blue', linewidth=2.5, alpha=0.8)
        ax.plot(results['Hour'], results['PPO_Cumulative_Profit'],
                label='PPO (Monthly Retraining)', color='red', linewidth=2.5, alpha=0.8)

        for retrain_time in retrain_stats['retrain_times']:
            ax.axvline(x=retrain_time, color='purple', linestyle='--', alpha=0.6, linewidth=2)

        ax.set_xlabel('Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cumulative Profit (EUR)', fontweight='bold', fontsize=12)
        ax.set_title(f'{noise_type.upper()}: Monthly Retraining Impact\n'
                     f'Purple lines = retraining ({retrain_stats["total_retrains"]} total)',
                     fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        final_pso = results['PSO_Cumulative_Profit'].iloc[-1]
        final_ppo = results['PPO_Cumulative_Profit'].iloc[-1]
        diff_pct = ((final_ppo - final_pso) / abs(final_pso) * 100) if final_pso != 0 else 0

        ax.text(0.02, 0.98,
                f'PSO: EUR {final_pso:,.0f}\n'
                f'PPO: EUR {final_ppo:,.0f}\n'
                f'Diff: {diff_pct:+.1f}%\n'
                f'Retrains: {retrain_stats["total_retrains"]}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=10, fontweight='bold')

        ax = axes[1]
        window = 24 * 7

        pso_rolling = pd.Series(results['PSO_Instant_Profit']).rolling(window=window).mean()
        ppo_rolling = pd.Series(results['PPO_Instant_Profit']).rolling(window=window).mean()

        ax.plot(results['Hour'], pso_rolling, label='PSO (7-day avg)',
                color='blue', linewidth=2, alpha=0.7)
        ax.plot(results['Hour'], ppo_rolling, label='PPO (7-day avg)',
                color='red', linewidth=2, alpha=0.7)

        for retrain_time in retrain_stats['retrain_times']:
            ax.axvline(x=retrain_time, color='purple', linestyle='--', alpha=0.6, linewidth=2)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('Profit/Hour (EUR, 7-day avg)', fontweight='bold', fontsize=12)
        ax.set_title('Rolling Average', fontweight='bold', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Lorenzo Giannuzzo: Save PNG and PDF
        filename_png, filename_pdf = save_plot_as_pdf(fig, f'monthly_retraining_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_daily_detailed_comparison(results, noise_type, results_folder, n_days=5):
    """Lorenzo Giannuzzo: Plot with PDF save"""
    safe_name = noise_type.replace('ornstein-uhlenbeck', 'ou')

    try:
        print(f"  Generating daily detailed plot for {noise_type}...")

        results['Month'] = results['Date'].dt.month
        results['Day'] = results['Date'].dt.day

        unique_days = results.groupby(['Month', 'Day']).size().reset_index()[['Month', 'Day']]
        days_to_plot = min(n_days, len(unique_days))

        if days_to_plot == 0:
            return

        fig, axes = plt.subplots(days_to_plot, 2, figsize=(20, 4 * days_to_plot))
        if days_to_plot == 1:
            axes = axes.reshape(1, -1)

        for idx in range(days_to_plot):
            month = unique_days.iloc[idx]['Month']
            day = unique_days.iloc[idx]['Day']
            day_data = results[(results['Month'] == month) & (results['Day'] == day)].iloc[:24]

            if len(day_data) == 0:
                continue

            ax_energy = axes[idx, 0]
            hours = range(len(day_data))

            ax_price = ax_energy.twinx()
            ax_price.plot(hours, day_data['True_Price'], 'k-', linewidth=2.5,
                          label='True Price', alpha=0.9, zorder=5)
            ax_price.plot(hours, day_data['Forecast_Price'], 'k--', linewidth=1.5,
                          label='Forecast', alpha=0.6, zorder=4)

            width = 0.35

            pso_charge_color = '#006400'
            pso_discharge_color = '#00008B'
            ppo_charge_color = '#90EE90'
            ppo_discharge_color = '#FFB6C1'

            for i, h in enumerate(hours):
                energy_pso = day_data['PSO_Energy_Actual'].iloc[i]
                color_pso = pso_charge_color if energy_pso > 0.01 else pso_discharge_color if energy_pso < -0.01 else 'gray'
                ax_energy.bar(h - width / 2, energy_pso, width,
                              color=color_pso, alpha=0.8, edgecolor='black', linewidth=0.8)

                energy_ppo = day_data['PPO_Energy_Actual'].iloc[i]
                color_ppo = ppo_charge_color if energy_ppo > 0.01 else ppo_discharge_color if energy_ppo < -0.01 else 'lightgray'
                ax_energy.bar(h + width / 2, energy_ppo, width,
                              color=color_ppo, alpha=0.8, edgecolor='black', linewidth=0.8)

            ax_energy.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax_energy.set_xlabel('Hour', fontweight='bold', fontsize=11)
            ax_energy.set_ylabel('Energy (MW)', fontweight='bold', fontsize=10)
            ax_price.set_ylabel('Price (EUR/MWh)', fontweight='bold', fontsize=11)

            date_str = day_data['Date'].iloc[0].strftime('%d %B %Y')
            ax_energy.set_title(f'{date_str} - {noise_type.upper()}',
                                fontweight='bold', fontsize=12)
            ax_energy.grid(True, alpha=0.3)
            ax_energy.set_xlim(-0.5, 23.5)
            ax_energy.set_xticks(range(0, 24, 2))
            ax_energy.set_ylim(-2.5, 2.5)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=pso_charge_color, alpha=0.8, edgecolor='black', label='PSO Charge'),
                Patch(facecolor=pso_discharge_color, alpha=0.8, edgecolor='black', label='PSO Discharge'),
                Patch(facecolor=ppo_charge_color, alpha=0.8, edgecolor='black', label='PPO Charge'),
                Patch(facecolor=ppo_discharge_color, alpha=0.8, edgecolor='black', label='PPO Discharge'),
                plt.Line2D([0], [0], color='k', linewidth=2.5, label='True Price'),
                plt.Line2D([0], [0], color='k', linewidth=1.5, linestyle='--', label='Forecast')
            ]
            ax_energy.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)

            ax_soc = axes[idx, 1]

            ax_soc.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Max')
            ax_soc.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Min')

            width_soc = 0.35

            for i, h in enumerate(hours):
                ax_soc.bar(h - width_soc / 2, day_data['PSO_SOC'].iloc[i] * 100, width_soc,
                           color='#00008B', alpha=0.7, edgecolor='black', linewidth=0.8,
                           label='PSO' if i == 0 else '')

                ax_soc.bar(h + width_soc / 2, day_data['PPO_SOC'].iloc[i] * 100, width_soc,
                           color='#FF7F50', alpha=0.7, edgecolor='black', linewidth=0.8,
                           label='PPO' if i == 0 else '')

            ax_soc.set_xlabel('Hour', fontweight='bold', fontsize=11)
            ax_soc.set_ylabel('SOC (%)', fontweight='bold', fontsize=11)
            ax_soc.set_title(f'{date_str} - SOC', fontweight='bold', fontsize=12)
            ax_soc.grid(True, alpha=0.3)
            ax_soc.set_xlim(-0.5, 23.5)
            ax_soc.set_xticks(range(0, 24, 2))
            ax_soc.set_ylim(0, 100)
            ax_soc.legend(loc='upper right', fontsize=9)

        plt.suptitle(f'Daily Analysis - {noise_type.upper()}',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Lorenzo Giannuzzo: Save PNG and PDF
        filename_png, filename_pdf = save_plot_as_pdf(fig, f'daily_detailed_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_monthly_daily_comparison(results, noise_type, results_folder):
    """Lorenzo Giannuzzo: Plot with PDF save"""
    safe_name = noise_type.replace('ornstein-uhlenbeck', 'ou')

    try:
        print(f"  Generating monthly comparison for {noise_type}...")

        results['Month'] = results['Date'].dt.month
        results['Day'] = results['Date'].dt.day

        months = results['Month'].unique()
        n_months = len(months)

        if n_months == 0:
            return

        fig, axes = plt.subplots(4, 3, figsize=(20, 16))
        axes = axes.flatten()

        for idx, month in enumerate(sorted(months)):
            if idx >= 12:
                break

            month_data = results[results['Month'] == month]
            mid_day = month_data['Day'].median()
            day_data = month_data[month_data['Day'] == int(mid_day)].iloc[:24]

            if len(day_data) == 0:
                day_data = month_data.iloc[:24]

            if len(day_data) == 0:
                continue

            ax = axes[idx]
            hours = range(len(day_data))
            ax2 = ax.twinx()

            ax.plot(hours, day_data['True_Price'], 'k-', linewidth=2, label='Price', alpha=0.8)
            ax.plot(hours, day_data['Forecast_Price'], 'k--', linewidth=1.5, label='Forecast', alpha=0.6)

            ax2.bar([h - 0.2 for h in hours], day_data['PSO_Energy_Actual'], width=0.4,
                    color='darkblue', alpha=0.6, label='PSO', edgecolor='black')
            ax2.bar([h + 0.2 for h in hours], day_data['PPO_Energy_Actual'], width=0.4,
                    color='lightcoral', alpha=0.6, label='PPO', edgecolor='black')

            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

            ax.set_xlabel('Hour', fontweight='bold')
            ax.set_ylabel('Price (EUR/MWh)', fontweight='bold')
            ax2.set_ylabel('Energy (MW)', fontweight='bold')

            month_name = day_data['Date'].iloc[0].strftime('%B %Y')
            ax.set_title(f'{month_name}', fontweight='bold')

            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 23.5)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

        for idx in range(n_months, 12):
            fig.delaxes(axes[idx])

        plt.suptitle(f'Monthly Comparison - {noise_type.upper()}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Lorenzo Giannuzzo: Save PNG and PDF
        filename_png, filename_pdf = save_plot_as_pdf(fig, f'monthly_comparison_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_fair_comparison(all_results, summary_df, results_folder):
    """Lorenzo Giannuzzo: Comparison plot with PDF save"""
    try:
        print(f"  Generating fair comparison...")

        n_noise = len(NOISE_TYPES_TO_TEST)
        fig = plt.figure(figsize=(7 * min(n_noise, 3), 12))
        gs = fig.add_gridspec(3, min(n_noise, 3), hspace=0.3, wspace=0.3)

        noise_labels = {
            'uniform': 'UNIFORM',
            'normal': 'NORMAL',
            'ornstein-uhlenbeck': 'O-U',
            'fixed-bias': 'FIXED-BIAS'
        }

        for idx, noise_type in enumerate(NOISE_TYPES_TO_TEST[:3]):
            if noise_type not in all_results:
                continue

            results = all_results[noise_type]

            # Lorenzo Giannuzzo: Cumulative profits
            ax = fig.add_subplot(gs[0, idx])
            ax.plot(results['Hour'], results['PSO_Cumulative_Profit'],
                    label='PSO', color='darkblue', linewidth=2.5, alpha=0.8)
            ax.plot(results['Hour'], results['PPO_Cumulative_Profit'],
                    label='PPO', color='coral', linewidth=2.5, alpha=0.8)
            ax.set_ylabel('Cumulative Profit (EUR)', fontweight='bold', fontsize=11)
            ax.set_title(noise_labels.get(noise_type, noise_type.upper()), fontweight='bold', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            final_pso = results['PSO_Cumulative_Profit'].iloc[-1]
            final_ppo = results['PPO_Cumulative_Profit'].iloc[-1]
            winner = "PPO" if final_ppo > final_pso else "PSO"
            diff_pct = ((final_ppo - final_pso) / abs(final_pso) * 100) if final_pso != 0 else 0

            ax.text(0.02, 0.98, f'{winner}\nDiff {diff_pct:+.1f}%',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    fontsize=9, fontweight='bold')

            # Lorenzo Giannuzzo: SOC
            ax = fig.add_subplot(gs[1, idx])
            sample_hours = min(168, len(results))
            ax.plot(results['Hour'][:sample_hours],
                    [s * 100 for s in results['PSO_SOC'][:sample_hours]],
                    label='PSO', color='darkblue', linewidth=1.5, alpha=0.7)
            ax.plot(results['Hour'][:sample_hours],
                    [s * 100 for s in results['PPO_SOC'][:sample_hours]],
                    label='PPO', color='coral', linewidth=1.5, alpha=0.7)
            ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel('SOC (%)', fontweight='bold', fontsize=11)
            ax.set_title('SOC (Week 1)', fontweight='bold', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])

            # Lorenzo Giannuzzo: Energy distribution
            ax = fig.add_subplot(gs[2, idx])
            ax.hist(results['PSO_Energy_Actual'], bins=50, color='darkblue',
                    alpha=0.5, edgecolor='black', label='PSO', density=True)
            ax.hist(results['PPO_Energy_Actual'], bins=50, color='coral',
                    alpha=0.5, edgecolor='black', label='PPO', density=True)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('Energy (MW)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Density', fontweight='bold', fontsize=11)
            ax.set_title('Energy Distribution', fontweight='bold', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('PSO vs PPO: Error Distributions + Monthly Retraining',
                     fontsize=15, fontweight='bold', y=0.995)

        # Lorenzo Giannuzzo: Save PNG and PDF
        filename_png, filename_pdf = save_plot_as_pdf(fig, 'comparison_all_errors', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_summary_bar(summary_df, results_folder):
    """Lorenzo Giannuzzo: Summary bar chart with PDF save"""
    try:
        print(f"  Generating summary bar chart...")

        noise_map = {
            'uniform': 'Uniform',
            'normal': 'Normal',
            'ornstein-uhlenbeck': 'O-U',
            'fixed-bias': 'Fixed-Bias'
        }
        noise_labels = [noise_map.get(n, n) for n in summary_df['Noise_Type']]

        x = np.arange(len(summary_df))
        width = 0.35

        # Lorenzo Giannuzzo: Profits
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        pso_profits = summary_df['PSO_Profit'].values
        ppo_profits = summary_df['PPO_Profit'].values

        bars1 = ax.bar(x - width / 2, pso_profits, width, label='PSO',
                       color='darkblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width / 2, ppo_profits, width, label='PPO',
                       color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Annual Profit (EUR)', fontweight='bold', fontsize=13)
        ax.set_title('Final Profits', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(noise_labels, fontsize=11, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=10, fontweight='bold')

        # Lorenzo Giannuzzo: Relative performance
        ax = axes[1]
        differences = summary_df['Diff_Pct'].values
        colors_bars = ['green' if d > -20 else 'orange' if d > -40 else 'red' for d in differences]

        bars = ax.bar(x, differences, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.axhline(y=-20, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Target: -20%')
        ax.set_ylabel('PPO vs PSO (%)', fontweight='bold', fontsize=13)
        ax.set_title('Relative Performance', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(noise_labels, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)

        for bar, diff in zip(bars, differences):
            height = bar.get_height()
            color = 'green' if diff > -20 else 'black'
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{diff:+.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=11, fontweight='bold', color=color)

        plt.tight_layout()

        # Lorenzo Giannuzzo: Save PNG and PDF
        filename_png, filename_pdf = save_plot_as_pdf(fig, 'summary_all_errors', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_error_robustness(summary_df, results_folder):
    """Lorenzo Giannuzzo: Error robustness plot with PDF save"""
    try:
        print(f"  Generating error robustness chart...")

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        noise_map = {
            'uniform': 'Uniform',
            'normal': 'Normal',
            'ornstein-uhlenbeck': 'O-U',
            'fixed-bias': 'Fixed-Bias'
        }

        x = np.arange(len(summary_df))
        width = 0.35

        pso_profits = summary_df['PSO_Profit'].values
        ppo_profits = summary_df['PPO_Profit'].values

        bars1 = ax.bar(x - width / 2, pso_profits, width, label='PSO',
                       color='darkblue', alpha=0.7, edgecolor='black', linewidth=2)
        bars2 = ax.bar(x + width / 2, ppo_profits, width, label='PPO',
                       color='coral', alpha=0.7, edgecolor='black', linewidth=2)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xlabel('Forecast Error Distribution', fontweight='bold', fontsize=14)
        ax.set_ylabel('Annual Profit (EUR)', fontweight='bold', fontsize=14)
        ax.set_title('PPO Robustness Across Error Distributions',
                     fontweight='bold', fontsize=15)

        labels = [noise_map.get(n, n) for n in summary_df['Noise_Type']]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')

        ax.legend(fontsize=13, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        avg_diff = summary_df['Diff_Pct'].mean()
        best_error = summary_df.loc[summary_df['Diff_Pct'].idxmax(), 'Noise_Type']
        worst_error = summary_df.loc[summary_df['Diff_Pct'].idxmin(), 'Noise_Type']

        insight_text = (f'PPO Avg: {avg_diff:+.1f}%\n'
                        f'Best: {best_error}\n'
                        f'Worst: {worst_error}\n'
                        f'Std: {summary_df["Diff_Pct"].std():.1f}%')

        ax.text(0.02, 0.98, insight_text,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=11, fontweight='bold')

        plt.tight_layout()

        # Lorenzo Giannuzzo: Save PNG and PDF
        filename_png, filename_pdf = save_plot_as_pdf(fig, 'error_robustness', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def main():
    print("=" * 80)
    print(" " * 5 + "PSO vs PPO - ERROR TYPES + PDF + CUMULATIVE DEGRADATION + SOH")
    print(" " * 5 + "Monthly Retraining + High-Resolution PDF (DPI=500)")
    print(" " * 5 + "FIXED: Cumulative degradation cycles")
    print("=" * 80)

    print(f"\nSETUP:")
    print(f"   Features:         26")
    print(f"   Network:          {PPO_HIDDEN_DIM_1}-{PPO_HIDDEN_DIM_2}")
    print(f"   Actions:          {N_DISCRETE_ACTIONS}")
    print(f"   Timesteps:        {PPO_TOTAL_TIMESTEPS:,}")
    print(f"   Forecast error:   ±{FORECAST_ERROR_MAX * 100:.0f}%")
    print(f"   Degradation cost: {DEGRADATION_COST_PER_MWH} EUR/MWh")

    print(f"\n   Error types:      {len(NOISE_TYPES_TO_TEST)}")
    for nt in NOISE_TYPES_TO_TEST:
        print(f"      - {nt}")

    print(f"\n   Monthly Retraining: {'ENABLED' if MONTHLY_RETRAINING_ENABLED else 'DISABLED'}")
    if MONTHLY_RETRAINING_ENABLED:
        print(f"      Frequency:     Every {MONTHLY_RETRAINING_FREQUENCY} days")
        print(f"      Lookback:      {MONTHLY_RETRAINING_LOOKBACK} days")
        print(f"      Timesteps:     {MONTHLY_RETRAINING_TIMESTEPS:,}")

    print(f"\n   Export Formats:")
    print(f"      PNG:           300 DPI")
    print(f"      PDF:           500 DPI")

    data_folder = 'data'
    train_path = os.path.join(data_folder, TRAIN_FILE)
    test_path = os.path.join(data_folder, TEST_FILE)

    print(f"\nLoading data...")
    try:
        df_train = pd.read_excel(train_path)
        if 'EUR/MWh' not in df_train.columns and '€/MWh' in df_train.columns:
            df_train.rename(columns={'€/MWh': 'EUR/MWh'}, inplace=True)
        if df_train['EUR/MWh'].dtype == 'object':
            df_train['EUR/MWh'] = df_train['EUR/MWh'].astype(str).str.replace(',', '.').astype(float)
        print(f"  Training: {len(df_train)} rows ({len(df_train) / 24:.0f} days)")

        df_test = pd.read_excel(test_path)
        if 'EUR/MWh' not in df_test.columns and '€/MWh' in df_test.columns:
            df_test.rename(columns={'€/MWh': 'EUR/MWh'}, inplace=True)
        if df_test['EUR/MWh'].dtype == 'object':
            df_test['EUR/MWh'] = df_test['EUR/MWh'].astype(str).str.replace(',', '.').astype(float)

        if TEST_FIRST_MONTH_ONLY:
            df_test = df_test.iloc[:24 * 31]
            print(f"  Test: {len(df_test)} rows ({len(df_test) / 24:.0f} days)")
        else:
            print(f"  Test: {len(df_test)} rows ({len(df_test) / 24:.0f} days)")

        all_train_test_prices = np.concatenate([df_train['EUR/MWh'].values, df_test['EUR/MWh'].values])
        print(f"  Combined: {len(all_train_test_prices)} hours")

    except Exception as e:
        print(f"  Error: {e}")
        return

    pso = PSOOptimizer(n_particles=PSO_PARTICLES, n_iterations=PSO_ITERATIONS)

    all_results = {}
    summary_data = []

    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Lorenzo Giannuzzo: Generate error types comparison FIRST
    print("\n" + "=" * 80)
    print("GENERATING ERROR TYPES COMPARISON")
    print("=" * 80)
    plot_error_types_comparison(results_folder)

    print("\n" + "=" * 80)
    print("TESTING ALL ERROR DISTRIBUTIONS")
    print("=" * 80)

    for noise_idx, noise_type in enumerate(NOISE_TYPES_TO_TEST):
        print("\n" + "+" + "-" * 78 + "+")
        print(f"|{' ' * 10}ERROR TYPE {noise_idx + 1}/{len(NOISE_TYPES_TO_TEST)}: {noise_type.upper():^40}|")
        print("+" + "-" * 78 + "+")

        error_descriptions = {
            'uniform': 'Uniformly distributed random errors in [-15%, +15%]',
            'normal': 'Normally distributed errors (Gaussian) σ=5%, clipped ±15%',
            'ornstein-uhlenbeck': 'Time-correlated errors using Ornstein-Uhlenbeck process',
            'fixed-bias': 'Deterministic fixed bias +10% (always overestimate)'
        }

        if noise_type in error_descriptions:
            print(f"\n  Error: {error_descriptions[noise_type]}")

        try:
            model_path = os.path.join(results_folder, f'ppo_4types_{noise_type}')

            if os.path.exists(model_path + '.zip') and not FORCE_RETRAIN:
                print(f"\n  Found existing model")
                error_gen_dummy = ForecastErrorGenerator(noise_type)
                env_dummy = BatteryTradingEnvClean(df_train['EUR/MWh'].values, error_gen_dummy)
                ppo_model = PPO.load(model_path, env=env_dummy)
                print("  Model loaded")
            else:
                print(f"\n  Training NEW model...")
                ppo_model = train_ppo_clean(df_train, noise_type, model_path)

            print(f"\n  Testing {noise_type.upper()}...")
            print("-" * 78)

            error_gen_test = ForecastErrorGenerator(noise_type)
            error_gen_test.reset()

            start_time = datetime.now()
            results_df, profit_pso, profit_ppo, retrain_stats, degradation_info = test_with_monthly_retraining(
                pso, ppo_model, df_test, all_train_test_prices, noise_type, error_gen_test
            )
            end_time = datetime.now()

            all_results[noise_type] = results_df

            results_path = os.path.join(results_folder, f'results_{noise_type}.csv')
            results_df.to_csv(results_path, index=False)
            print(f"\n  Results saved: {results_path}")

            analyze_trading_behavior(results_df, noise_type)

            # Lorenzo Giannuzzo: Generate plots
            print(f"\n  Generating plots...")
            plot_daily_detailed_comparison(results_df, noise_type, results_folder, n_days=5)
            plot_monthly_daily_comparison(results_df, noise_type, results_folder)

            # Lorenzo Giannuzzo: Generate heatmaps
            plot_heatmaps(results_df, noise_type, results_folder)

            # Lorenzo Giannuzzo: Generate SOH comparison plots
            plot_soh_comparison(results_df, noise_type, results_folder)

            if MONTHLY_RETRAINING_ENABLED:
                plot_monthly_retraining_performance(results_df, retrain_stats, noise_type, results_folder)

            difference = profit_ppo - profit_pso
            pct_diff = (difference / abs(profit_pso) * 100) if profit_pso != 0 else 0
            winner = "PPO" if profit_ppo > profit_pso else "PSO"

            print(f"\n{'-' * 78}")
            print(f"FINAL RESULTS - {noise_type.upper()}")
            print(f"{'-' * 78}")
            print(f"\nOPERATIONAL PROFITS (Real Revenue):")
            print(f"  PSO:        EUR {profit_pso:>15,.2f}")
            print(f"  PPO:        EUR {profit_ppo:>15,.2f}")
            print(f"  Difference: EUR {difference:>15,.2f} ({pct_diff:+.2f}%)")
            print(f"  Winner:     {winner}")

            print(f"\nDEGRADATION COSTS (Tracking Only):")
            print(f"  PSO:        EUR {degradation_info['pso_total_degradation']:>15,.2f}")
            print(f"  PPO:        EUR {degradation_info['ppo_total_degradation']:>15,.2f}")
            print(f"  Difference: EUR {degradation_info['ppo_total_degradation'] - degradation_info['pso_total_degradation']:>15,.2f}")

            print(f"\nNET PROFIT (Operational - Degradation):")
            net_pso = profit_pso - degradation_info['pso_total_degradation']
            net_ppo = profit_ppo - degradation_info['ppo_total_degradation']
            print(f"  PSO:        EUR {net_pso:>15,.2f}")
            print(f"  PPO:        EUR {net_ppo:>15,.2f}")
            print(f"  Difference: EUR {net_ppo - net_pso:>15,.2f}")

            if MONTHLY_RETRAINING_ENABLED and retrain_stats['total_retrains'] > 0:
                print(f"\n  Monthly retrains: {retrain_stats['total_retrains']}")

            if abs(pct_diff) <= 5:
                print(f"\n  OUTSTANDING: Within 5%!")
            elif abs(pct_diff) <= 10:
                print(f"\n  EXCELLENT: Within 10%!")
            elif abs(pct_diff) <= 20:
                print(f"\n  VERY GOOD: Within 20%!")
            else:
                print(f"\n  Good")

            print(f"\n  Time: {(end_time - start_time).total_seconds():.1f}s")
            print(f"{'-' * 78}")

            summary_data.append({
                'Noise_Type': noise_type,
                'PSO_Profit': profit_pso,
                'PPO_Profit': profit_ppo,
                'Difference': difference,
                'Diff_Pct': pct_diff,
                'Winner': winner,
                'Monthly_Retrains': retrain_stats['total_retrains'] if MONTHLY_RETRAINING_ENABLED else 0
            })

        except KeyboardInterrupt:
            print("\n\nInterrupted")
            break
        except Exception as e:
            print(f"\n\nError: {e}")
            traceback.print_exc()
            continue

    # Lorenzo Giannuzzo: Final summary
    if len(summary_data) > 0:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(results_folder, 'summary_4types.csv')
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 80)
        print("FINAL SUMMARY - ERROR TYPES")
        print("=" * 80)

        print(
            f"\n{'Error Type':<25} {'PSO (EUR)':<15} {'PPO (EUR)':<18} {'Diff (%)':<12} {'Retrains':<10} {'Winner':<10}")
        print("-" * 100)

        for _, row in summary_df.iterrows():
            retrains_str = f"{int(row['Monthly_Retrains'])}" if MONTHLY_RETRAINING_ENABLED else "N/A"
            print(f"{row['Noise_Type']:<25} "
                  f"EUR {row['PSO_Profit']:>12,.0f} "
                  f"EUR {row['PPO_Profit']:>15,.0f} "
                  f"{row['Diff_Pct']:>10.2f}% "
                  f"{retrains_str:<10} "
                  f"{row['Winner']:<10}")

        print("-" * 100)

        avg_pso = summary_df['PSO_Profit'].mean()
        avg_ppo = summary_df['PPO_Profit'].mean()
        avg_diff = summary_df['Diff_Pct'].mean()

        print(f"\nAVERAGE:")
        print(f"   PSO average:      EUR {avg_pso:,.2f}")
        print(f"   PPO average:      EUR {avg_ppo:,.2f}")
        print(f"   Avg difference:   {avg_diff:+.2f}%")

        ppo_wins = sum(1 for w in summary_df['Winner'] if w == 'PPO')
        pso_wins = sum(1 for w in summary_df['Winner'] if w == 'PSO')

        print(f"\nVICTORIES:")
        print(f"   PPO: {ppo_wins}/{len(summary_data)}")
        print(f"   PSO: {pso_wins}/{len(summary_data)}")

        if len(summary_df) > 1:
            best_error = summary_df.loc[summary_df['Diff_Pct'].idxmax()]
            worst_error = summary_df.loc[summary_df['Diff_Pct'].idxmin()]

            print(f"\nPERFORMANCE:")
            print(f"   Best:  {best_error['Noise_Type']} ({best_error['Diff_Pct']:+.2f}%)")
            print(f"   Worst: {worst_error['Noise_Type']} ({worst_error['Diff_Pct']:+.2f}%)")
            print(f"   Std:   {summary_df['Diff_Pct'].std():.2f}%")

        print("\n" + "=" * 80)

        # Lorenzo Giannuzzo: Generate comparison plots
        if len(all_results) > 0:
            print(f"\nGenerating comparison plots...")
            plot_fair_comparison(all_results, summary_df, results_folder)
            plot_summary_bar(summary_df, results_folder)
            plot_error_robustness(summary_df, results_folder)
            plot_soh_summary(all_results, summary_df, results_folder)
            print(f"  All plots generated!")

    print("\n" + "=" * 80)
    print("COMPLETE - ERROR TYPES + PDF + CUMULATIVE DEGRADATION + SOH + HEATMAPS")
    print("=" * 80)

    print(f"\nFiles in: {results_folder}/")
    print(f"\nAll plots saved in:")
    print(f"   - PNG (300 DPI)")
    print(f"   - PDF (500 DPI)")

    print(f"\nMain visualizations:")
    print(f"   - error_types_comparison (PNG + PDF)")
    print(f"   - comparison_all_errors (PNG + PDF)")
    print(f"   - summary_all_errors (PNG + PDF)")
    print(f"   - error_robustness (PNG + PDF)")
    print(f"   - soh_summary_all_errors (PNG + PDF)")

    print(f"\nPer error type:")
    for nt in NOISE_TYPES_TO_TEST:
        safe_name = nt.replace('ornstein-uhlenbeck', 'ou')
        print(f"   {nt}:")
        print(f"      - daily_detailed_{safe_name} (PNG + PDF)")
        print(f"      - monthly_comparison_{safe_name} (PNG + PDF)")
        print(f"      - heatmaps_{safe_name} (PNG + PDF)")
        print(f"      - heatmap_soc_{safe_name} (PNG + PDF)")
        print(f"      - soh_comparison_{safe_name} (PNG + PDF)")
        print(f"      - soh_vs_throughput_{safe_name} (PNG + PDF)")
        if MONTHLY_RETRAINING_ENABLED:
            print(f"      - monthly_retraining_{safe_name} (PNG + PDF)")

    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"\nDevice: {device_info}")

    print(f"\nFEATURES:")
    print(f"   - Error types (uniform, normal, o-u)")
    print(f"   - Monthly retraining with actual errors")
    print(f"   - PDF export (500 DPI)")
    print(f"   - Error types comparison visualization")
    print(f"   - Battery degradation model (PSO + PPO) - CUMULATIVE CYCLES")
    print(f"   - Heatmap visualizations")
    print(f"   - SOH tracking and comparison")
    print(f"   - Separate profit and degradation cost tracking")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()