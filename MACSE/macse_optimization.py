"""
------------------------------------------------------------------------------------------------------------------------

BATTERY ENERGY STORAGE SYSTEM (BESS) OPTIMIZATION
Particle Swarm Optimization with Rolling Horizon and MACSE Integration

------------------------------------------------------------------------------------------------------------------------

Author: Lorenzo Giannuzzo
Affiliation: Politecnico di Torino
             Dipartimento Energia (DENERG)
             Energy Center Lab
Email: lorenzo.giannuzzo@polito.it
Description:
    Sistema di ottimizzazione per Battery Energy Storage System (BESS) che
    implementa l'algoritmo Particle Swarm Optimization (PSO) con Rolling
    Horizon per la massimizzazione dei profitti attraverso:
    - Arbitraggio sul mercato elettrico
    - Partecipazione al mercato MACSE (servizi ancillari)
    - Gestione ottimale del degrado della batteria
Version: 1.0.0
Date: October 2025
------------------------------------------------------------------------------------------------------------------------
MIT License
Copyright (c) 2024 Lorenzo Giannuzzo
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch

# ---------------------------------- PARAMETRI CONFIGURABILI -----------------------------------------------------------
SAVE_PLOTS = True                                   # Lorenzo Giannuzzo: Flag per abilitare/disabilitare salvataggio grafici
# Lorenzo Giannuzzo: Parametri Degrado Batteria
DEGRADATION_COST_PER_MWH = 25.0                     # Lorenzo Giannuzzo: €/MWh di throughput (per ottimizzatore)
# Lorenzo Giannuzzo: Parametri MACSE
MACSE_ENABLED = True                                # Lorenzo Giannuzzo: Abilita partecipazione al MACSE
MACSE_CAPACITY_PERCENTAGE = 0.3                     # Lorenzo Giannuzzo: % capacità allocata al MACSE (30%)
MACSE_CONTRACT_YEARS = 1                            # Lorenzo Giannuzzo: Durata contratto MACSE (anni)
MACSE_PRICE_PER_MW_YEAR = 50000                     # Lorenzo Giannuzzo: €/MW/anno (da asta MACSE)
MACSE_AVAILABILITY_TARGET = 0.95                    # Lorenzo Giannuzzo: Target disponibilità (95%)
MACSE_PENALTY_THRESHOLD = 0.92                      # Lorenzo Giannuzzo:  Soglia penale disponibilità (92%)
MACSE_BONUS_THRESHOLD = 0.98                        # Lorenzo Giannuzzo:  Soglia bonus disponibilità (98%)
MACSE_PENALTY_MULTIPLIER = 1.5                      # Lorenzo Giannuzzo:  Moltiplicatore penale
MACSE_BONUS_PERCENTAGE = 0.10                       # Lorenzo Giannuzzo:  Bonus 10% se Avail >= 98%
MACSE_RESPONSE_TIME_MINUTES = 15                    # Lorenzo Giannuzzo:  Tempo di risposta richiesto (minuti)
MACSE_MIN_SOC_FOR_DISCHARGE = 0.20                  # Lorenzo Giannuzzo:  SOC minimo per scarica MACSE
MACSE_MAX_SOC_FOR_CHARGE = 0.80                     # Lorenzo Giannuzzo:  SOC massimo per carica MACSE
np.random.seed(42)                                  # Lorenzo Giannuzzo: to ensure stability on results (avoid PSO randomness exploration)

# ---------------------------------- BESS DEGRADATION FUNCTION ---------------------------------------------------------
def degradation(cycle_num):
    """Funzione di degrado della batteria (capacità rimanente %)"""
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

# ---------------------------------- BATTERY MODEL ---------------------------------------------------------------------
class Battery:
    """Modello di batteria con integrazione MACSE"""
    def __init__(self, capacity_mwh=4.0, efficiency=0.95, max_power_mw=2.0, max_c_rate=0.5):
        self.nominal_capacity = capacity_mwh
        self.capacity = capacity_mwh
        self.efficiency = efficiency
        self.max_power = max_power_mw
        self.max_c_rate = max_c_rate
        self.soc = 0.5
        self.soc_min = 0.1
        self.soc_max = 0.9
        # Lorenzo Giannuzzo: Parametri degrado
        self.equivalent_cycles = 0.0
        self.throughput_kwh = 0.0
        # Lorenzo Giannuzzo: Parametri MACSE
        if MACSE_ENABLED:
            self.macse_capacity = capacity_mwh * MACSE_CAPACITY_PERCENTAGE
            self.trading_capacity = capacity_mwh * (1 - MACSE_CAPACITY_PERCENTAGE)
            self.macse_power = max_power_mw * MACSE_CAPACITY_PERCENTAGE
            self.trading_power = max_power_mw * (1 - MACSE_CAPACITY_PERCENTAGE)
        else:
            self.macse_capacity = 0
            self.trading_capacity = capacity_mwh
            self.macse_power = 0
            self.trading_power = max_power_mw
        # Lorenzo Giannuzzo: Tracking disponibilità MACSE
        self.macse_available_hours = 0
        self.macse_total_hours = 0
    # Lorenzo Giannuzzo: Define Charging mode
    def charge(self, power_mw, dt=1.0):
        """Carica la batteria (solo capacità trading)"""
        max_power_c_rate = self.trading_capacity * self.max_c_rate
        max_power_trading = min(self.trading_power, max_power_c_rate)
        power_mw = min(power_mw, max_power_trading)
        energy = power_mw * dt * self.efficiency
        new_soc = self.soc + (energy / self.capacity)
        new_soc = min(new_soc, self.soc_max)
        actual_energy = (new_soc - self.soc) * self.capacity
        self.soc = new_soc
        self.throughput_kwh += actual_energy * 1000
        return actual_energy / self.efficiency
    # Lorenzo Giannuzzo: Define Discharge mode
    def discharge(self, power_mw, dt=1.0):
        """Scarica la batteria (solo capacità trading)"""
        max_power_c_rate = self.trading_capacity * self.max_c_rate
        max_power_trading = min(self.trading_power, max_power_c_rate)
        power_mw = min(power_mw, max_power_trading)
        energy = power_mw * dt / self.efficiency
        new_soc = self.soc - (energy / self.capacity)
        new_soc = max(new_soc, self.soc_min)
        actual_energy = (self.soc - new_soc) * self.capacity
        self.soc = new_soc
        self.throughput_kwh += actual_energy * 1000
        return actual_energy * self.efficiency
    # Lorenzo Giannuzzo: Checking MACSE availability function
    def check_macse_availability(self):
        """Verifica se la batteria è disponibile per MACSE"""
        if not MACSE_ENABLED:
            return False
        soc_ok_for_discharge = self.soc >= MACSE_MIN_SOC_FOR_DISCHARGE
        soc_ok_for_charge = self.soc <= MACSE_MAX_SOC_FOR_CHARGE
        power_ok = self.macse_power >= 0.9 * (self.max_power * MACSE_CAPACITY_PERCENTAGE)
        return (soc_ok_for_discharge or soc_ok_for_charge) and power_ok
    # Lorenzo Giannuzzo: Updating MACSE availability function
    def update_macse_availability(self, is_available):
        """Aggiorna tracking disponibilità MACSE"""
        self.macse_total_hours += 1
        if is_available:
            self.macse_available_hours += 1
    # Lorenzo Giannuzzo: Getting MACSE availability function
    def get_macse_availability_factor(self):
        """Calcola fattore di disponibilità MACSE"""
        if self.macse_total_hours == 0:
            return 1.0
        delta = 0.0
        avail = (self.macse_available_hours / self.macse_total_hours) * (1 - delta)
        return avail
    # Lorenzo Giannuzzo: Update Degradation Function
    def update_degradation(self):
        """Aggiorna la capacità in base al degrado"""
        self.equivalent_cycles = self.throughput_kwh / (2*self.nominal_capacity * 1000)
        capacity_percentage = degradation(self.equivalent_cycles)
        self.capacity = self.nominal_capacity * (capacity_percentage / 100.0)
        if MACSE_ENABLED:
            self.macse_capacity = self.capacity * MACSE_CAPACITY_PERCENTAGE
            self.trading_capacity = self.capacity * (1 - MACSE_CAPACITY_PERCENTAGE)
    # Lorenzo Giannuzzo: SOH Function
    def get_soh(self):
        return (self.capacity / self.nominal_capacity) * 100.0
    # Lorenzo Giannuzzo: SOC Function
    def get_soc(self):
        return self.soc
    # Lorenzo Giannuzzo: Copy Function
    def copy(self):
        b = Battery(self.nominal_capacity, self.efficiency, self.max_power, self.max_c_rate)
        b.capacity = self.capacity
        b.soc = self.soc
        b.equivalent_cycles = self.equivalent_cycles
        b.throughput_kwh = self.throughput_kwh
        b.macse_capacity = self.macse_capacity
        b.trading_capacity = self.trading_capacity
        b.macse_power = self.macse_power
        b.trading_power = self.trading_power
        return b
    # Lorenzo Giannuzzo: Get max power based on C-rate function
    def get_max_power_by_crate(self):
        """Ritorna la potenza massima per trading consentita dal C-rate"""
        return self.trading_capacity * self.max_c_rate

# ---------------------------------- PSO OPTIMIZER ---------------------------------------------------------------------
class PSOOptimizer:
    """PSO ottimizzato con inizializzazione smart"""
    def __init__(self, n_particles=50, n_iterations=100, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.stagnation_limit = 15
    # Lorenzo Giannuzzo: Define Optimize function
    def optimize(self, battery, prices, horizon_hours=18):
        n_hours = min(horizon_hours, len(prices))
        max_power_limit = min(battery.trading_power, battery.get_max_power_by_crate())
        positions = self._smart_initialization(battery, prices[:n_hours], max_power_limit)
        velocities = np.random.uniform(-1.0, 1.0, (self.n_particles, n_hours))
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self._evaluate(battery, p, prices[:n_hours])
                                         for p in positions])
        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        stagnation_counter = 0
        for iteration in range(self.n_iterations):
            w = self.w_start - (self.w_start - self.w_end) * (iteration / self.n_iterations)
            for i in range(self.n_particles):
                r1, r2 = np.random.random(n_hours), np.random.random(n_hours)
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                max_velocity = max_power_limit * 0.5
                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], -max_power_limit, max_power_limit)
                score = self._evaluate(battery, positions[i], prices[:n_hours])
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                    if score > global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()
                        stagnation_counter = 0
            stagnation_counter += 1
            if stagnation_counter > self.stagnation_limit:
                n_reinit = self.n_particles // 4
                worst_indices = np.argsort(personal_best_scores)[:n_reinit]
                for idx in worst_indices:
                    noise = np.random.uniform(-max_power_limit * 0.3, max_power_limit * 0.3, n_hours)
                    positions[idx] = np.clip(global_best_position + noise, -max_power_limit, max_power_limit)
                    velocities[idx] = np.random.uniform(-0.5, 0.5, n_hours)
                stagnation_counter = 0
        return global_best_position
    # Lorenzo Giannuzzo: Define smart initialization function
    def _smart_initialization(self, battery, prices, max_power):
        n_hours = len(prices)
        positions = np.zeros((self.n_particles, n_hours))
        price_low = np.percentile(prices, 25)
        price_high = np.percentile(prices, 75)
        for i in range(self.n_particles):
            if i < self.n_particles // 3:
                for h in range(n_hours):
                    if prices[h] < price_low:
                        positions[i, h] = np.random.uniform(0.5 * max_power, max_power)
                    elif prices[h] > price_high:
                        positions[i, h] = np.random.uniform(-max_power, -0.5 * max_power)
                    else:
                        positions[i, h] = np.random.uniform(-0.3 * max_power, 0.3 * max_power)
            elif i < 2 * self.n_particles // 3:
                for h in range(n_hours):
                    if prices[h] < price_low:
                        positions[i, h] = np.random.uniform(0, 0.7 * max_power)
                    elif prices[h] > price_high:
                        positions[i, h] = np.random.uniform(-0.7 * max_power, 0)
                    else:
                        positions[i, h] = np.random.uniform(-0.2 * max_power, 0.2 * max_power)
            else:
                positions[i] = np.random.uniform(-max_power, max_power, n_hours)
        return positions
    # Lorenzo Giannuzzo: Define Evaluate function
    def _evaluate(self, battery, actions, prices):
        bat_sim = battery.copy()
        profit = 0.0
        for power, price in zip(actions, prices):
            if power > 0:
                if bat_sim.soc >= bat_sim.soc_max:
                    continue
                max_energy_storable = (bat_sim.soc_max - bat_sim.soc) * bat_sim.trading_capacity
                max_power_available = max_energy_storable / (1.0 * bat_sim.efficiency)
                actual_power = min(power, max_power_available)
                if actual_power > 0.01:
                    energy_from_grid = bat_sim.charge(actual_power, dt=1.0)
                    profit -= energy_from_grid * price
                    profit -= energy_from_grid * DEGRADATION_COST_PER_MWH
            elif power < 0:
                if bat_sim.soc <= bat_sim.soc_min:
                    continue
                max_energy_available = (bat_sim.soc - bat_sim.soc_min) * bat_sim.trading_capacity
                max_power_available = max_energy_available * bat_sim.efficiency / 1.0
                actual_power = min(-power, max_power_available)
                if actual_power > 0.01:
                    energy_to_grid = bat_sim.discharge(actual_power, dt=1.0)
                    profit += energy_to_grid * price
                    profit -= energy_to_grid * DEGRADATION_COST_PER_MWH
        return profit

# ---------------------------------- ROLLING HORIZON SIMULATION --------------------------------------------------------
class RollingHorizonSimulator:
    """Simulatore con MACSE integration"""
    def __init__(self, battery, optimizer, horizon_hours=18, step_hours=1):
        self.battery = battery
        self.optimizer = optimizer
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours
    # Lorenzo Giannuzzo: Define Simulate Function
    def simulate(self, prices_df):
        prices = prices_df['€/MWh'].values
        n_hours = len(prices)
        actions_taken = []
        soc_history = []
        capacity_history = []
        soh_history = []
        profits_history = []
        degradation_costs_history = []
        macse_availability_history = []
        cumulative_profit = 0.0
        cumulative_degradation_cost = 0.0
        print("=" * 60)
        print("SIMULAZIONE BESS CON PSO, ROLLING HORIZON E MACSE")
        print("=" * 60)
        print(f"Capacità nominale: {self.battery.nominal_capacity} MWh")
        print(f"Potenza massima: {self.battery.max_power} MW")
        if MACSE_ENABLED:
            print(f"\n🔵 MACSE ABILITATO:")
            print(f"  - Capacità allocata MACSE: {self.battery.macse_capacity:.2f} MWh ({MACSE_CAPACITY_PERCENTAGE * 100}%)")
            print(f"  - Capacità per trading: {self.battery.trading_capacity:.2f} MWh ({(1 - MACSE_CAPACITY_PERCENTAGE) * 100}%)")
            print(f"  - Potenza MACSE: {self.battery.macse_power:.2f} MW")
            print(f"  - Potenza trading: {self.battery.trading_power:.2f} MW")
            print(f"  - Prezzo contratto: {MACSE_PRICE_PER_MW_YEAR:,.0f} €/MW/anno")
        else:
            print(f"\n🔴 MACSE DISABILITATO")
            print(f"  - L'intera capacità della batteria ({self.battery.capacity:.2f} MWh) è disponibile per trading")
            print(f"  - Potenza totale per trading: {self.battery.max_power:.2f} MW")
            print(f"  - Nessun ricavo da contratti MACSE - solo profitti da arbitraggio")
        print(f"\nC-rate massimo: {self.battery.max_c_rate}C")
        print(f"Costo degrado (ottimizzatore): {DEGRADATION_COST_PER_MWH} €/MWh")
        print(f"Orizzonte: {self.horizon_hours} ore")
        print(f"Ore totali: {n_hours}")
        print("=" * 60)
        print()
        print(f"⏳ Progresso: 0% - Inizio ottimizzazione...")

        current_hour = 0
        last_progress = 0
        degradation_update_interval = 24
        while current_hour < n_hours:
            progress = int((current_hour / n_hours) * 100)
            if progress >= last_progress + 20:
                avail_factor = self.battery.get_macse_availability_factor() if MACSE_ENABLED else 0
                print(
                    f"⏳ {progress}% ({current_hour}/{n_hours} h) - SOH: {self.battery.get_soh():.2f}% - MACSE Avail: {avail_factor * 100:.1f}%")
                last_progress = progress
            if current_hour % degradation_update_interval == 0 and current_hour > 0:
                self.battery.update_degradation()
            end_hour = min(current_hour + self.horizon_hours, n_hours)
            window_prices = prices[current_hour:end_hour]
            optimal_actions = self.optimizer.optimize(
                self.battery, window_prices, self.horizon_hours
            )
            action = optimal_actions[0]
            price = prices[current_hour]
            energy_flow = 0.0
            if action > 0.01:
                if self.battery.soc < self.battery.soc_max:
                    max_energy_storable = (self.battery.soc_max - self.battery.soc) * self.battery.trading_capacity
                    max_power_available = max_energy_storable / (1.0 * self.battery.efficiency)
                    actual_power = min(action, max_power_available)
                    action = actual_power
                    if actual_power > 0.01:
                        energy_from_grid = self.battery.charge(actual_power, dt=1.0)
                        energy_flow = energy_from_grid
                        profit = -energy_from_grid * price
                    else:
                        profit = 0.0
                else:
                    profit = 0.0
                    action = 0.0
            elif action < -0.01:
                if self.battery.soc > self.battery.soc_min:
                    max_energy_available = (self.battery.soc - self.battery.soc_min) * self.battery.trading_capacity
                    max_power_available = max_energy_available * self.battery.efficiency / 1.0
                    actual_power = min(-action, max_power_available)
                    action = -actual_power
                    if actual_power > 0.01:
                        energy_to_grid = self.battery.discharge(actual_power, dt=1.0)
                        energy_flow = energy_to_grid
                        profit = energy_to_grid * price
                    else:
                        profit = 0.0
                else:
                    profit = 0.0
                    action = 0.0
            else:
                profit = 0.0
            macse_available = self.battery.check_macse_availability() if MACSE_ENABLED else False
            if MACSE_ENABLED:
                self.battery.update_macse_availability(macse_available)

            degradation_cost = energy_flow * DEGRADATION_COST_PER_MWH
            cumulative_profit += profit
            cumulative_degradation_cost += degradation_cost
            net_profit = cumulative_profit - cumulative_degradation_cost
            actions_taken.append(action)
            soc_history.append(self.battery.get_soc())
            capacity_history.append(self.battery.capacity)
            soh_history.append(self.battery.get_soh())
            profits_history.append(net_profit)
            degradation_costs_history.append(cumulative_degradation_cost)
            macse_availability_history.append(self.battery.get_macse_availability_factor() if MACSE_ENABLED else 0)
            current_hour += self.step_hours
        self.battery.update_degradation()
        print(f"⏳ Progresso: 100% completato ({n_hours}/{n_hours} ore)")
        print()
        results_df = prices_df.copy()
        pad_length = len(results_df) - len(actions_taken)
        results_df['Azione_MW'] = actions_taken + [0] * pad_length
        results_df['SOC'] = soc_history + [soc_history[-1]] * pad_length
        results_df['Capacità_MWh'] = capacity_history + [capacity_history[-1]] * pad_length
        results_df['SOH_%'] = soh_history + [soh_history[-1]] * pad_length
        results_df['Profitto_Netto_€'] = profits_history + [profits_history[-1]] * pad_length
        results_df['Costo_Degrado_€'] = degradation_costs_history + [degradation_costs_history[-1]] * pad_length
        results_df['MACSE_Availability'] = macse_availability_history + [macse_availability_history[-1]] * pad_length
        return results_df, cumulative_profit, cumulative_degradation_cost

# Lorenzo Giannuzzo: Define MACSE Revenues Evaluation Function
def calculate_macse_revenue(battery):
    """Calcola ricavi e penali MACSE"""
    if not MACSE_ENABLED:
        return 0, 0, 0, 0
    base_revenue = (MACSE_PRICE_PER_MW_YEAR * battery.macse_power * MACSE_CONTRACT_YEARS) / \
                   (365 * 24)
    avail_factor = battery.get_macse_availability_factor()
    penalty = 0
    bonus = 0
    if avail_factor < MACSE_PENALTY_THRESHOLD:
        missing_availability = MACSE_AVAILABILITY_TARGET - avail_factor
        penalty = base_revenue * missing_availability * MACSE_PENALTY_MULTIPLIER * (365 * 24)
    if avail_factor >= MACSE_BONUS_THRESHOLD:
        bonus = base_revenue * MACSE_BONUS_PERCENTAGE * (365 * 24)
    annual_revenue = base_revenue * (365 * 24) - penalty + bonus
    return annual_revenue, base_revenue * (365 * 24), penalty, bonus

# Lorenzo Giannuzzo: Define Plots Function
def create_visualizations(results_df):
    """Crea visualizzazioni incluso MACSE"""
    if not SAVE_PLOTS:
        print("\n⏭️  Salvataggio grafici disabilitato")
        return
    viz_folder = 'visualization'
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    print("\n" + "=" * 60)
    print("GENERAZIONE VISUALIZZAZIONI")
    print("=" * 60)
    if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
        results_df['Data'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')
    results_df['Mese'] = results_df['Data'].dt.month
    if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
        results_df['Data_dt'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')
    else:
        results_df['Data_dt'] = results_df['Data']
    df_daily = results_df.groupby(results_df['Data_dt'].dt.date, as_index=False).last()
    df_daily['Data_dt'] = pd.to_datetime(df_daily['Data_dt'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(df_daily['Data_dt'], df_daily['Profitto_Netto_€'],
             color='#06A77D', linewidth=2)
    ax1.fill_between(df_daily['Data_dt'], 0, df_daily['Profitto_Netto_€'],
                     alpha=0.3, color='#06A77D')
    ax1.set_ylabel('Profitto Trading (€)', fontsize=11, fontweight='bold')
    ax1.set_title('Profitto Trading nel Tempo', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if MACSE_ENABLED:
        ax2.plot(df_daily['Data_dt'], df_daily['MACSE_Availability'] * 100,
                 color='#457B9D', linewidth=2)
        ax2.axhline(y=MACSE_AVAILABILITY_TARGET * 100, color='green', linestyle='--',
                    alpha=0.5, label=f'Target {MACSE_AVAILABILITY_TARGET * 100}%')
        ax2.axhline(y=MACSE_PENALTY_THRESHOLD * 100, color='red', linestyle='--',
                    alpha=0.5, label=f'Soglia penale {MACSE_PENALTY_THRESHOLD * 100}%')
        ax2.set_ylabel('Disponibilità MACSE (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Disponibilità MACSE nel Tempo', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, '01_profitto_macse.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Salvato: 01_profitto_macse.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_daily['Data_dt'], df_daily['SOH_%'], color='#E63946', linewidth=2)
    ax.set_ylabel('SOH (%)', fontsize=11, fontweight='bold')
    ax.set_title('State of Health', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, '02_soh.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Salvato: 02_soh.png")
    print("=" * 60)
    print(f"✓ Visualizzazioni salvate in '{viz_folder}/'")
    print("=" * 60)

# Lorenzo Giannuzzo: Define Technical Report Generation Function
def generate_technical_report(results_df, battery, gross_profit, degradation_cost,
                          macse_revenue, macse_base, macse_penalty, macse_bonus,
                          total_system_profit, battery_investment, simulation_time):
    """Genera report PDF tecnico completo"""
    reports_folder = 'results'
    pdf_path = os.path.join(reports_folder, 'Technical_Report_BESS_Optimization.pdf')
    print("\n" + "=" * 60)
    print("GENERAZIONE REPORT TECNICO PDF")
    print("=" * 60)
    with PdfPages(pdf_path) as pdf:
        # ===================== PAGINA 1: COPERTINA =====================
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')
        # Lorenzo Giannuzzo: Metto il Titolo su due righe
        ax.text(0.5, 0.87, 'BATTERY ENERGY',
                ha='center', va='top', fontsize=24, fontweight='bold', color='black')
        ax.text(0.5, 0.82, 'STORAGE SYSTEM',
                ha='center', va='top', fontsize=24, fontweight='bold', color='black')
        ax.text(0.5, 0.76, 'Optimization Report',
                ha='center', va='top', fontsize=18, style='italic', color='black')

        ax.text(0.5, 0.67, 'Particle Swarm Optimization con Rolling Horizon',
                ha='center', va='top', fontsize=12, color='#333333')
        if MACSE_ENABLED:
            ax.text(0.5, 0.62, 'Integrazione Mercato MACSE',
                    ha='center', va='top', fontsize=11, color='#333333')
        box_lines = [
            f"Capacità Nominale: {battery.nominal_capacity} MWh",
            f"Potenza Massima: {battery.max_power} MW",
            f"Profitto Totale: €{total_system_profit:,.2f}",
            f"SOH Finale: {battery.get_soh():.2f}%",
            f"ROI: {((total_system_profit + battery_investment * (battery.get_soh() / 100) - battery_investment) / battery_investment) * 100:.2f}%"
        ]
        y_box = 0.47
        for line in box_lines:
            ax.text(0.5, y_box, line, ha='center', va='center', fontsize=11, family='monospace')
            y_box -= 0.04
        rect = FancyBboxPatch((0.15, 0.29), 0.7, 0.23, boxstyle="round,pad=0.02",
                              edgecolor='black', facecolor='#f8f8f8', linewidth=1.5,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.5, 0.15, f'Data Generazione: {datetime.now().strftime("%d/%m/%Y %H:%M")}',
                ha='center', va='center', fontsize=10, color='#666666')
        ax.text(0.5, 0.10, 'BESS Optimization Tool v1.0',
                ha='center', va='center', fontsize=9, style='italic', color='#666666')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Pagina 1: Copertina")

        # ===================== PAGINA 2: PARAMETRI =====================
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.95, 'PARAMETRI DI SIMULAZIONE',
                ha='center', va='top', fontsize=16, fontweight='bold', color='black')
        y_pos = 0.87
        # Lorenzo Giannuzzo: Sezione 1 - Batteria
        ax.text(0.05, y_pos, '1. PARAMETRI BATTERIA', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.04
        battery_params = [
            f"Capacità Nominale (E_nom):           {battery.nominal_capacity} MWh",
            f"Potenza Massima (P_max):             {battery.max_power} MW",
            f"Efficienza Round-Trip (η):           {battery.efficiency * 100}%",
            f"C-rate Massimo:                      {battery.max_c_rate}C",
            f"SOC Minimo:                          {battery.soc_min * 100}%",
            f"SOC Massimo:                         {battery.soc_max * 100}%",
            f"SOC Iniziale:                        50%"
        ]
        for param in battery_params:
            ax.text(0.08, y_pos, f"• {param}", fontsize=9, va='top', family='monospace')
            y_pos -= 0.025
        y_pos -= 0.03
        # Lorenzo Giannuzzo: Sezione 2 - PSO
        ax.text(0.05, y_pos, '2. ALGORITMO PSO', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.04
        pso_params = [
            "Numero Particelle (N):               50",
            "Iterazioni per Finestra:             100",
            "Inerzia Iniziale (w_start):          0.9",
            "Inerzia Finale (w_end):              0.4",
            "Coefficiente Cognitivo (c1):         2.0",
            "Coefficiente Sociale (c2):           2.0",
            "Limite Stagnazione:                  15 iterazioni"
        ]
        for param in pso_params:
            ax.text(0.08, y_pos, f"• {param}", fontsize=9, va='top', family='monospace')
            y_pos -= 0.025
        y_pos -= 0.03
        # Lorenzo Giannuzzo: Sezione 3 - Rolling Horizon
        ax.text(0.05, y_pos, '3. ROLLING HORIZON', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.04
        rh_params = [
            "Orizzonte Temporale:                 24 ore",
            "Step Avanzamento:                    1 ora",
            "Aggiornamento Degrado:               Ogni 24 ore"
        ]
        for param in rh_params:
            ax.text(0.08, y_pos, f"• {param}", fontsize=9, va='top', family='monospace')
            y_pos -= 0.025
        y_pos -= 0.03
        # Lorenzo Giannuzzo: Sezione 4 - MACSE (se abilitato)
        if MACSE_ENABLED:
            ax.text(0.05, y_pos, '4. MERCATO MACSE', fontsize=13, fontweight='bold', color='black')
            y_pos -= 0.04
            macse_params = [
                f"Capacità Allocata MACSE:             {MACSE_CAPACITY_PERCENTAGE * 100}% ({battery.macse_capacity:.2f} MWh)",
                f"Capacità Trading:                    {(1 - MACSE_CAPACITY_PERCENTAGE) * 100}% ({battery.trading_capacity:.2f} MWh)",
                f"Prezzo Contratto:                    {MACSE_PRICE_PER_MW_YEAR:,.0f} €/MW/anno",
                f"Target Disponibilità:                {MACSE_AVAILABILITY_TARGET * 100}%",
                f"Soglia Penale:                       {MACSE_PENALTY_THRESHOLD * 100}%",
                f"Soglia Bonus:                        {MACSE_BONUS_THRESHOLD * 100}%"
            ]
            for param in macse_params:
                ax.text(0.08, y_pos, f"• {param}", fontsize=9, va='top', family='monospace')
                y_pos -= 0.025
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Pagina 2: Parametri")

        # ===================== PAGINA 3: MODELLO MATEMATICO =====================
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.95, 'MODELLO MATEMATICO',
                ha='center', va='top', fontsize=16, fontweight='bold', color='black')
        y_pos = 0.87
        ax.text(0.05, y_pos, '1. FUNZIONE OBIETTIVO PSO', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.035
        ax.text(0.08, y_pos, 'Massimizzare il profitto netto su orizzonte H:', fontsize=10)
        y_pos -= 0.04
        formula1 = r'$J = \sum_{t=1}^{H} \left[ E_{sell}(t) \cdot P(t) - E_{buy}(t) \cdot P(t) - (E_{sell}(t) + E_{buy}(t)) \cdot C_{deg} \right]$'
        ax.text(0.5, y_pos, formula1, ha='center', fontsize=10)
        y_pos -= 0.06
        ax.text(0.08, y_pos, f'Con: C_deg = {DEGRADATION_COST_PER_MWH} €/MWh, H = 24 ore',
                fontsize=9, style='italic')
        y_pos -= 0.07
        ax.text(0.05, y_pos, '2. VINCOLI STATE OF CHARGE', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.04
        formula2 = r'$SOC_{min} \leq SOC(t) \leq SOC_{max}$'
        ax.text(0.5, y_pos, formula2, ha='center', fontsize=10)
        y_pos -= 0.055
        formula3 = r'$SOC(t+1) = SOC(t) + \frac{E_{charge}(t) \cdot \eta - E_{discharge}(t)/\eta}{E_{nom}}$'
        ax.text(0.5, y_pos, formula3, ha='center', fontsize=10)
        y_pos -= 0.07
        ax.text(0.05, y_pos, '3. VINCOLI C-RATE', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.04
        formula4 = r'$P(t) \leq \min(P_{max}, E_{nom} \cdot C_{rate})$'
        ax.text(0.5, y_pos, formula4, ha='center', fontsize=10)
        y_pos -= 0.07
        ax.text(0.05, y_pos, '4. MODELLO DEGRADO', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.04
        formula5 = r'$N_{cycles} = \frac{\sum |E(t)|}{E_{nom}}$'
        ax.text(0.5, y_pos, formula5, ha='center', fontsize=10)
        y_pos -= 0.055
        formula6 = r'$C_{remaining}(\%) = \sum_{i=0}^{9} a_i \cdot N_{cycles}^i$'
        ax.text(0.5, y_pos, formula6, ha='center', fontsize=10)
        y_pos -= 0.07
        if MACSE_ENABLED:
            ax.text(0.05, y_pos, '5. SEPARAZIONE MACSE/TRADING', fontsize=13, fontweight='bold', color='black')
            y_pos -= 0.04
            formula7 = r'$E_{MACSE} = E_{nom} \cdot \alpha_{MACSE}$'
            ax.text(0.5, y_pos, formula7, ha='center', fontsize=10)
            y_pos -= 0.05
            formula8 = r'$E_{trading} = E_{nom} \cdot (1 - \alpha_{MACSE})$'
            ax.text(0.5, y_pos, formula8, ha='center', fontsize=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Pagina 3: Modello Matematico")

        # ===================== PAGINA 4: ALGORITMO PSO =====================
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.95, 'ALGORITMO PSO',
                ha='center', va='top', fontsize=16, fontweight='bold', color='black')
        y_pos = 0.87
        ax.text(0.05, y_pos, '1. UPDATE VELOCITÀ', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.04
        formula_v = r'$v_i(t+1) = w(t) \cdot v_i(t) + c_1 r_1 (p_{best,i} - x_i) + c_2 r_2 (g_{best} - x_i)$'
        ax.text(0.5, y_pos, formula_v, ha='center', fontsize=10)
        y_pos -= 0.06
        ax.text(0.08, y_pos, 'Inerzia decrescente: w(t) = 0.9 - 0.5·(t/T_max)', fontsize=9, style='italic')
        y_pos -= 0.07
        ax.text(0.05, y_pos, '2. INIZIALIZZAZIONE INTELLIGENTE', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.04
        init_text = [
            "33% AGGRESSIVA: Carica max se P<P25, Scarica max se P>P75",
            "33% CONSERVATIVA: Azioni moderate",
            "33% RANDOM: Esplorazione completa"
        ]
        for line in init_text:
            ax.text(0.08, y_pos, f"• {line}", fontsize=9, va='top')
            y_pos -= 0.03
        y_pos -= 0.04
        iter_range = np.linspace(0, 100, 100)
        w_range = 0.9 - (0.9 - 0.4) * (iter_range / 100)
        ax_inertia = fig.add_axes([0.15, y_pos - 0.18, 0.7, 0.15])
        ax_inertia.plot(iter_range, w_range, 'k-', linewidth=2)
        ax_inertia.set_xlabel('Iterazione', fontsize=9, fontweight='bold')
        ax_inertia.set_ylabel('Inerzia w', fontsize=9, fontweight='bold')
        ax_inertia.set_title('Decadimento Inerzia PSO', fontsize=10, fontweight='bold')
        ax_inertia.grid(True, alpha=0.3)
        ax_inertia.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax_inertia.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, linewidth=1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Pagina 4: Algoritmo PSO")

        # ===================== PAGINA 5: RISULTATI =====================
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.95, 'RISULTATI ECONOMICI',
                ha='center', va='top', fontsize=16, fontweight='bold', color='black')
        y_pos = 0.86
        # TRADING
        ax.text(0.05, y_pos, 'TRADING', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.05  # Spazio maggiore dopo titolo
        trading_lines = [
            f"Profitto Operativo:              €{gross_profit:>15,.2f}",
            f"Profitto Netto:                  €{gross_profit - degradation_cost:>15,.2f}",
            f"Throughput:                      {battery.throughput_kwh / 1000:>15,.2f} MWh"
        ]
        for line in trading_lines:
            ax.text(0.5, y_pos, line, ha='center', fontsize=9, va='top', family='monospace')
            y_pos -= 0.025
        box_y_trading = y_pos + 0.075
        rect_trading = FancyBboxPatch((0.1, box_y_trading - 0.09), 0.8, 0.10,
                                      boxstyle="round,pad=0.01",
                                      edgecolor='#666666', facecolor='#f0f8ff',
                                      linewidth=1, transform=ax.transAxes)
        ax.add_patch(rect_trading)
        y_pos -= 0.05
        # MACSE
        if MACSE_ENABLED:
            ax.text(0.05, y_pos, 'MACSE', fontsize=13, fontweight='bold', color='black')
            y_pos -= 0.05
            macse_lines = [
                f"Remunerazione Base:              €{macse_base:>15,.2f}",
                f"Penali:                          €{macse_penalty:>15,.2f}",
                f"Bonus:                           €{macse_bonus:>15,.2f}",
                f"Ricavo Netto:                    €{macse_revenue:>15,.2f}",
                f"Disponibilità:                   {battery.get_macse_availability_factor() * 100:>15,.2f}%"
            ]
            for line in macse_lines:
                ax.text(0.5, y_pos, line, ha='center', fontsize=9, va='top', family='monospace')
                y_pos -= 0.025
            box_y_macse = y_pos + 0.125
            rect_macse = FancyBboxPatch((0.1, box_y_macse - 0.14), 0.8, 0.15,
                                        boxstyle="round,pad=0.01",
                                        edgecolor='#666666', facecolor='#f0fff0',
                                        linewidth=1, transform=ax.transAxes)
            ax.add_patch(rect_macse)
            y_pos -= 0.05
        # TOTALE
        ax.text(0.05, y_pos, 'TOTALE', fontsize=13, fontweight='bold', color='black')
        y_pos -= 0.05
        residual_value = battery_investment * (battery.get_soh() / 100.0)
        total_return = total_system_profit + residual_value - battery_investment
        roi = (total_return / battery_investment) * 100
        payback = battery_investment / total_system_profit if total_system_profit > 0 else float('inf')
        total_lines = [
            f"Profitto Totale:                 €{total_system_profit:>15,.2f}",
            f"Investimento:                    €{battery_investment:>15,.2f}",
            f"Valore Residuo:                  €{residual_value:>15,.2f}",
            f"Ritorno Totale:                  €{total_return:>15,.2f}",
            f"ROI:                             {roi:>15,.2f}%",
            f"Payback:                         {payback:>15,.1f} anni"
        ]
        for line in total_lines:
            ax.text(0.5, y_pos, line, ha='center', fontsize=9, va='top', family='monospace')
            y_pos -= 0.025

        box_y_total = y_pos + 0.15
        rect_total = FancyBboxPatch((0.1, box_y_total - 0.17), 0.8, 0.18,
                                    boxstyle="round,pad=0.01",
                                    edgecolor='#666666', facecolor='#fff0f0',
                                    linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect_total)
        y_pos -= 0.06
        # Grafico a torta
        if MACSE_ENABLED and total_system_profit > 0:
            ax_pie = fig.add_axes([0.2, y_pos - 0.25, 0.6, 0.20])
            labels = ['Trading', 'MACSE']
            sizes = [gross_profit, macse_revenue]
            colors = ['#4682B4', '#90EE90']
            ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                       shadow=False, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax_pie.set_title('Composizione Ricavi', fontsize=11, fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Pagina 5: Risultati")

        # ===================== PAGINE 6-7: GRAFICI =====================
        if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
            results_df['Data_dt'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')
        else:
            results_df['Data_dt'] = results_df['Data']
        df_daily = results_df.groupby(results_df['Data_dt'].dt.date, as_index=False).last()
        df_daily['Data_dt'] = pd.to_datetime(df_daily['Data_dt'])
        # PAGINA 6
        fig = plt.figure(figsize=(8.5, 11))
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df_daily['Data_dt'], df_daily['Profitto_Netto_€'], 'k-', linewidth=1.5)
        ax1.fill_between(df_daily['Data_dt'], 0, df_daily['Profitto_Netto_€'], alpha=0.2, color='blue')
        ax1.set_ylabel('Profitto (€)', fontsize=10, fontweight='bold')
        ax1.set_title('Profitto Trading', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(df_daily['Data_dt'], df_daily['SOH_%'], 'k-', linewidth=1.5)
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel('SOH (%)', fontsize=10, fontweight='bold')
        ax2.set_title('State of Health', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax2.set_ylim([0, 105])
        ax3 = plt.subplot(3, 1, 3)
        if MACSE_ENABLED:
            ax3.plot(df_daily['Data_dt'], df_daily['MACSE_Availability'] * 100, 'k-', linewidth=1.5)
            ax3.axhline(y=MACSE_AVAILABILITY_TARGET * 100, color='green', linestyle='--', alpha=0.5)
            ax3.axhline(y=MACSE_PENALTY_THRESHOLD * 100, color='red', linestyle='--', alpha=0.5)
            ax3.set_ylabel('Disponibilità (%)', fontsize=10, fontweight='bold')
            ax3.set_title('Disponibilità MACSE', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        else:
            ax3.text(0.5, 0.5, 'MACSE Non Abilitato', ha='center', va='center',
                     fontsize=12, fontweight='bold', transform=ax3.transAxes)
            ax3.axis('off')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Pagina 6: Grafici Performance")

        # PAGINA 7
        fig = plt.figure(figsize=(8.5, 11))
        ax1 = plt.subplot(2, 2, 1)
        ax1.hist(results_df['SOC'] * 100, bins=30, color='#708090', alpha=0.7, edgecolor='black')
        ax1.axvline(x=results_df['SOC'].mean() * 100, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('SOC (%)', fontsize=9, fontweight='bold')
        ax1.set_ylabel('Frequenza', fontsize=9, fontweight='bold')
        ax1.set_title('Distribuzione SOC', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        results_df['Mese'] = results_df['Data_dt'].dt.month
        energia_carica = results_df[results_df['Azione_MW'] > 0].groupby('Mese')['Azione_MW'].sum()
        energia_scarica = -results_df[results_df['Azione_MW'] < 0].groupby('Mese')['Azione_MW'].sum()
        mesi = range(1, 13)
        x = np.arange(len(mesi))
        width = 0.35
        ax2 = plt.subplot(2, 2, 2)
        ax2.bar(x - width / 2, [energia_carica.get(m, 0) for m in mesi],
                width, label='Carica', color='#4682B4', alpha=0.7)
        ax2.bar(x + width / 2, [energia_scarica.get(m, 0) for m in mesi],
                width, label='Scarica', color='#DC143C', alpha=0.7)
        ax2.set_xlabel('Mese', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Energia (MWh)', fontsize=9, fontweight='bold')
        ax2.set_title('Energia Mensile', fontsize=10, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['G', 'F', 'M', 'A', 'M', 'G', 'L', 'A', 'S', 'O', 'N', 'D'])
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(df_daily['Data_dt'], df_daily['Capacità_MWh'], color='#8B008B', linewidth=1.5)
        ax3.set_ylabel('Capacità (MWh)', fontsize=9, fontweight='bold')
        ax3.set_title('Capacità Batteria', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        results_df['Profitto_Operativo_Orario'] = 0.0
        for idx in range(1, len(results_df)):
            action = results_df.loc[idx, 'Azione_MW']
            price = results_df.loc[idx, '€/MWh']
            if action > 0:  # Carica (costo)
                results_df.loc[idx, 'Profitto_Operativo_Orario'] = -action * price
            elif action < 0:  # Scarica (ricavo)
                results_df.loc[idx, 'Profitto_Operativo_Orario'] = -action * price
        profitto_mensile = results_df.groupby('Mese')['Profitto_Operativo_Orario'].sum()
        colors_profit = ['#228B22' if p > 0 else '#DC143C' for p in profitto_mensile.values]
        ax4 = plt.subplot(2, 2, 4)
        ax4.bar(mesi, [profitto_mensile.get(m, 0) for m in mesi],
                color=colors_profit, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Mese', fontsize=9, fontweight='bold')
        ax4.set_ylabel('Profitto (€)', fontsize=9, fontweight='bold')
        ax4.set_title('Profitto Operativo Mensile', fontsize=10, fontweight='bold')
        ax4.axhline(y=0, color='black', linewidth=1)
        ax4.set_xticks(mesi)
        ax4.set_xticklabels(['G', 'F', 'M', 'A', 'M', 'G', 'L', 'A', 'S', 'O', 'N', 'D'])
        ax4.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Pagina 7: Grafici Operativi")
        d = pdf.infodict()
        d['Title'] = 'BESS Optimization - Technical Report'
        d['Author'] = 'BESS Optimization Tool'
        d['Subject'] = 'Battery Energy Storage System Optimization'
        d['Keywords'] = 'BESS, PSO, MACSE, Battery, Optimization'
        d['CreationDate'] = datetime.now()
    print("=" * 60)
    print(f"✅ Report PDF: {pdf_path}")
    print("=" * 60)

# Lorenzo Giannuzzo: Define Main Function
def main():
    file_path = os.path.join('data', '20240101_20241231_PUN.xlsx')
    print("=" * 60)
    print("BESS OPTIMIZATION WITH MACSE")
    print("=" * 60)
    print("Caricamento dati...")
    try:
        df = pd.read_excel(file_path)
        if df['€/MWh'].dtype == 'object':
            df['€/MWh'] = df['€/MWh'].astype(str).str.replace(',', '.').astype(float)
        print(f"✓ Dati caricati: {len(df)} righe")
        print(f"✓ Range prezzi: {df['€/MWh'].min():.2f} - {df['€/MWh'].max():.2f} €/MWh")
        if 'Data' in df.columns:
            date_range = pd.to_datetime(df['Data'], format='%d/%m/%Y')
            print(f"✓ Periodo: {date_range.min().date()} → {date_range.max().date()}")
            print(f"✓ Durata: {(date_range.max() - date_range.min()).days} giorni")
        print()
    except Exception as e:
        print(f"✗ Errore: {e}")
        return
    battery = Battery(capacity_mwh=4.0, efficiency=0.95, max_power_mw=2.0, max_c_rate=0.5)
    optimizer = PSOOptimizer(n_particles=50, n_iterations=100, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0)
    simulator = RollingHorizonSimulator(battery, optimizer, horizon_hours=24, step_hours=1)
    start_time = datetime.now()
    results_df, gross_profit, degradation_cost = simulator.simulate(df)
    end_time = datetime.now()
    net_profit_trading = gross_profit - degradation_cost
    # Ricavi MACSE
    macse_revenue, macse_base, macse_penalty, macse_bonus = calculate_macse_revenue(battery)
    # Profitto totale sistema
    total_system_profit = gross_profit + macse_revenue
    # Report finale
    print("=" * 60)
    print("RISULTATI SIMULAZIONE")
    print("=" * 60)
    print("\n📊 TRADING (Capacità disponibile per arbitraggio):")
    print(f"  💰 Profitto operativo trading: €{gross_profit:,.2f}")
    print(f"  ⚙️  Costo degrado (tracking): €{degradation_cost:,.2f}")
    print(f"  💵 Profitto netto trading: €{net_profit_trading:,.2f}")
    if MACSE_ENABLED:
        print(f"\n🔵 MACSE (Capacità riservata alla rete):")
        print(f"  📋 Remunerazione base annuale: €{macse_base:,.2f}")
        print(f"  ⚠️  Penali disponibilità: €{macse_penalty:,.2f}")
        print(f"  🎁 Bonus disponibilità: €{macse_bonus:,.2f}")
        print(f"  💰 Ricavo netto MACSE: €{macse_revenue:,.2f}")
        print(f"  📈 Disponibilità finale: {battery.get_macse_availability_factor() * 100:.2f}%")
    print(f"\n💎 SISTEMA TOTALE:")
    print(f"  💰 Profitto totale (Trading + MACSE): €{total_system_profit:,.2f}")
    print(f"\n🔋 STATO BATTERIA:")
    print(f"  📊 SOC finale: {battery.get_soc() * 100:.1f}%")
    print(f"  🔋 Capacità finale: {battery.capacity:.3f} MWh (nominale: {battery.nominal_capacity:.3f} MWh)")
    print(f"  📉 SOH finale: {battery.get_soh():.2f}%")
    print(f"  ♻️  Cicli equivalenti: {battery.equivalent_cycles:.1f}")
    print(f"  ⏱️  Tempo esecuzione: {(end_time - start_time).total_seconds():.1f} secondi")
    print("=" * 60)
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    output_file = os.path.join(results_folder, 'risultati_simulazione_batteria_macse.xlsx')
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Risultati Orari', index=False)
        params_data = {
            'Parametro': [
                'Capacità nominale (MWh)',
                'Potenza massima (MW)',
                'MACSE abilitato',
                'Capacità MACSE (MWh)',
                'Capacità Trading (MWh)',
                'Profitto Trading (€)',
                'Ricavi MACSE (€)',
                'Profitto Totale (€)',
                'SOH finale (%)',
            ],
            'Valore': [
                battery.nominal_capacity,
                battery.max_power,
                'Sì' if MACSE_ENABLED else 'No',
                battery.macse_capacity if MACSE_ENABLED else 0,
                battery.trading_capacity,
                gross_profit,
                macse_revenue,
                total_system_profit,
                battery.get_soh(),
            ]
        }
        params_df = pd.DataFrame(params_data)
        params_df.to_excel(writer, sheet_name='Parametri', index=False)
    print(f"✓ Risultati salvati in: {output_file}")
    create_visualizations(results_df)
    generate_technical_report(
        results_df, battery, gross_profit, degradation_cost,
        macse_revenue, macse_base, macse_penalty, macse_bonus,
        total_system_profit, 600000,
        (end_time - start_time).total_seconds()
    )
    print("\n" + "=" * 60)
    print("✅ SIMULAZIONE COMPLETATA CON SUCCESSO")
    print("=" * 60)

# Lorenzo Giannuzzo: Execute Main Function
if __name__ == "__main__":
    main()