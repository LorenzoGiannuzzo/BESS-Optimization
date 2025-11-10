""""
------------------------------------------------------------------------------------------------------------------------
BATTERY ENERGY STORAGE SYSTEM (BESS) OPTIMIZATION - WITH USER LOAD INTEGRATION
Particle Swarm Optimization with Rolling Horizon, MACSE, Autonomous PV and Load Management
Supporto Litio-ione e Grafene + Sistema Fotovoltaico + Carico Utente con Decisione Ottimale
------------------------------------------------------------------------------------------------------------------------
Author: Lorenzo Giannuzzo (Modified)
Affiliation: Politecnico di Torino
             Dipartimento Energia (DENERG)
             Energy Center Lab
Description:
    Sistema di ottimizzazione per Battery Energy Storage System (BESS) che
    implementa l'algoritmo Particle Swarm Optimization (PSO) con Rolling
    Horizon per la massimizzazione dei profitti attraverso:
    - Arbitraggio sul mercato elettrico
    - Partecipazione al mercato MACSE (servizi ancillari)
    - Gestione ottimale del degrado della batteria
    - Supporto per tecnologie Litio-ione e Grafene
    - Autonomous PV: L'algoritmo decide autonomamente allocazione energia PV
    - **LOAD MANAGEMENT: Soddisfacimento carico da PV, Batteria o Rete con logica ottimale**
Version: 2.4.0-WITH-LOAD
Date: November 2025
------------------------------------------------------------------------------------------------------------------------
NUOVE CARATTERISTICHE VERSIONE 2.4.0:
- Integrazione carico elettrico utente
- Tre fonti per soddisfare carico: PV → Batteria → Rete (in ordine di priorità economica)
- Ricavi virtuali quando PV o batteria servono il carico
- Mark-up configurabile tra prezzo acquisto e vendita (+15% default)
- Tracking completo flussi energetici ed economici
- Ottimizzazione PSO considera carico nelle decisioni di allocazione energia
------------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ========================================================================================================
# SEZIONE 1: PARAMETRI CONFIGURABILI PRINCIPALI
# ========================================================================================================
energy_selling_price_name = '20240101_20241231_PUN.xlsx'
energy_buying_price_name = '20240101_20241231_PUN.xlsx'
pv_production_file = 'year_PV.csv'
load_file = 'BTA6_5.xlsx'

# ---------------------------------- SCELTA TECNOLOGIA BATTERIA -------------------------------------------
BATTERY_TECHNOLOGY = "LITIO-IONE"

# ---------------------------------- PARAMETRI BATTERIA ---------------------------------------------------
BATTERY_CAPACITY_MWH = 4.0
BATTERY_MAX_POWER_MW = 2.0
BATTERY_MAX_C_RATE = 0.5
BATTERY_BASE_EFFICIENCY = 0.95

# ---------------------------------- PARAMETRI FOTOVOLTAICO -----------------------------------------------
PV_ENABLED = True
PV_NOMINAL_POWER_KWP = 1000.0
PV_INVERTER_EFFICIENCY = 0.98
PV_SYSTEM_LOSSES = 0.05

# ---------------------------------- PARAMETRI CARICO UTENTE -----------------------------------------------
LOAD_ENABLED = True
LOAD_SHEET_NAME = None  # None = primo sheet disponibile

# ---------------------------------- PARAMETRI ECONOMICI -----------------------------------------------
PRICE_MARKUP_PERCENT = 15.0  # Mark-up % tra prezzo acquisto e vendita (acquisto = vendita * (1 + markup/100))

# ---------------------------------- PARAMETRI SPECIFICI PER TECNOLOGIA -----------------------------------
LITHIUM_ION_SOC_MIN = 0.1
LITHIUM_ION_SOC_MAX = 0.9
LITHIUM_ION_DOD = 0.8
LITHIUM_ION_EOL_CYCLES = 6000

GRAPHENE_SOC_MIN = 0.0
GRAPHENE_SOC_MAX = 1.0
GRAPHENE_DOD = 1.0
GRAPHENE_EOL_CYCLES = 500000

# ---------------------------------- PARAMETRI GENERALI ---------------------------------------------------
SAVE_PLOTS = True
DEGRADATION_COST_PER_MWH = 10.0

# ---------------------------------- PARAMETRI MACSE ------------------------------------------------------
MACSE_ENABLED = True
MACSE_CAPACITY_MWH = 1.0
MACSE_CONTRACT_YEARS = 1
MACSE_PRICE_PER_MW_YEAR = 50000
MACSE_AVAILABILITY_TARGET = 0.95
MACSE_PENALTY_THRESHOLD = 0.92
MACSE_BONUS_THRESHOLD = 0.98
MACSE_PENALTY_MULTIPLIER = 1.5
MACSE_BONUS_PERCENTAGE = 0.10
MACSE_RESPONSE_TIME_MINUTES = 15
MACSE_MIN_SOC_FOR_DISCHARGE = 0.20
MACSE_MAX_SOC_FOR_CHARGE = 0.80

np.random.seed(42)

# ========================================================================================================
# SEZIONE 2: DATI SPERIMENTALI CURVE DI EFFICIENZA
# ========================================================================================================
LITHIUM_ION_05C_DATA = {
    'charge_energy_kwh': [11.45, 11.42, 11.48],
    'discharge_energy_kwh': [10.85, 10.88, 10.82],
    'energy_efficiency': [0.948, 0.952, 0.942],
    'coulombic_efficiency': [0.995, 0.997, 0.994]
}

LITHIUM_ION_1C_DATA = {
    'charge_energy_kwh': [11.38, 11.42, 11.35],
    'discharge_energy_kwh': [10.45, 10.52, 10.48],
    'energy_efficiency': [0.918, 0.921, 0.923],
    'coulombic_efficiency': [0.985, 0.988, 0.986]
}

GRAPHENE_05C_DATA = {
    'charge_energy_kwh': [11.5, 11.308, 11.3],
    'discharge_energy_kwh': [11.0, 10.7, 10.9],
    'energy_efficiency': [0.956, 0.964, 0.964],
    'coulombic_efficiency': [0.96, 0.983, 0.984]
}

GRAPHENE_1C_DATA = {
    'charge_energy_kwh': [11.3, 11.18, 11.4],
    'discharge_energy_kwh': [10.62, 10.66, 10.8],
    'energy_efficiency': [0.94, 0.953, 0.9478],
    'coulombic_efficiency': [0.98, 0.958, 0.968]
}

# ========================================================================================================
# SEZIONE 3: CLASSI
# ========================================================================================================
class BatteryEfficiencyModel:
    def __init__(self, technology, c_rate):
        self.technology = technology
        self.c_rate = c_rate
        if technology == "LITIO-IONE":
            if c_rate <= 0.5:
                self.data = LITHIUM_ION_05C_DATA
            else:
                self.data = LITHIUM_ION_1C_DATA
        elif technology == "GRAFENE":
            if c_rate <= 0.5:
                self.data = GRAPHENE_05C_DATA
            else:
                self.data = GRAPHENE_1C_DATA
        else:
            raise ValueError(f"Tecnologia non supportata: {technology}")
        self.avg_energy_efficiency = np.mean(self.data['energy_efficiency'])
        self.avg_coulombic_efficiency = np.mean(self.data['coulombic_efficiency'])
        self.avg_charge_energy = np.mean(self.data['charge_energy_kwh'])
        self.avg_discharge_energy = np.mean(self.data['discharge_energy_kwh'])

    def get_charge_efficiency(self):
        return np.sqrt(self.avg_energy_efficiency)

    def get_discharge_efficiency(self):
        return np.sqrt(self.avg_energy_efficiency)

    def get_roundtrip_efficiency(self):
        return self.avg_energy_efficiency

class PhotovoltaicSystem:
    """
    Lorenzo Giannuzzo: Modello sistema fotovoltaico con tracking completo allocazione energia
    """
    def __init__(self, nominal_power_kwp=PV_NOMINAL_POWER_KWP,
                 inverter_efficiency=PV_INVERTER_EFFICIENCY,
                 system_losses=PV_SYSTEM_LOSSES):
        self.nominal_power_kwp = nominal_power_kwp
        self.inverter_efficiency = inverter_efficiency
        self.system_losses = system_losses
        self.total_efficiency = inverter_efficiency * (1 - system_losses)

        # Contatori produzione
        self.total_production_mwh = 0.0
        self.energy_to_battery_mwh = 0.0
        self.energy_to_grid_mwh = 0.0
        self.energy_to_load_mwh = 0.0
        self.curtailed_energy_mwh = 0.0

    def get_production(self, irradiance_w_per_kwp):
        """Calcola produzione effettiva dato irraggiamento [MW]"""
        power_mw = (irradiance_w_per_kwp * self.nominal_power_kwp * self.total_efficiency) / 1e6
        return power_mw

    def get_energy(self, irradiance_w_per_kwp, dt=1.0):
        """Calcola energia prodotta in un timestep [MWh]"""
        power_mw = self.get_production(irradiance_w_per_kwp)
        energy_mwh = power_mw * dt
        self.total_production_mwh += energy_mwh
        return energy_mwh

    def allocate_energy(self, to_battery_mwh, to_grid_mwh, to_load_mwh):
        """Registra allocazione energia PV"""
        self.energy_to_battery_mwh += to_battery_mwh
        self.energy_to_grid_mwh += to_grid_mwh
        self.energy_to_load_mwh += to_load_mwh

    def get_statistics(self):
        """Ritorna statistiche produzione PV"""
        total = self.total_production_mwh
        return {
            'total_production_mwh': total,
            'energy_to_battery_mwh': self.energy_to_battery_mwh,
            'energy_to_grid_mwh': self.energy_to_grid_mwh,
            'energy_to_load_mwh': self.energy_to_load_mwh,
            'curtailed_energy_mwh': self.curtailed_energy_mwh,
            'battery_utilization_percent': (self.energy_to_battery_mwh / total * 100) if total > 0 else 0,
            'grid_sale_percent': (self.energy_to_grid_mwh / total * 100) if total > 0 else 0,
            'load_service_percent': (self.energy_to_load_mwh / total * 100) if total > 0 else 0,
            'curtailment_percent': (self.curtailed_energy_mwh / total * 100) if total > 0 else 0
        }

class LoadProfile:
    """
    Lorenzo Giannuzzo: Modello carico elettrico utente con tracking completo fonti di fornitura
    """
    def __init__(self):
        self.total_energy_required_mwh = 0.0
        self.energy_from_pv_mwh = 0.0
        self.energy_from_battery_mwh = 0.0
        self.energy_from_grid_mwh = 0.0

    def register_supply(self, from_pv, from_battery, from_grid):
        """Registra fonti di fornitura carico"""
        self.energy_from_pv_mwh += from_pv
        self.energy_from_battery_mwh += from_battery
        self.energy_from_grid_mwh += from_grid
        self.total_energy_required_mwh += (from_pv + from_battery + from_grid)

    def get_statistics(self):
        """Ritorna statistiche fornitura carico"""
        total = self.total_energy_required_mwh
        return {
            'total_energy_required_mwh': total,
            'energy_from_pv_mwh': self.energy_from_pv_mwh,
            'energy_from_battery_mwh': self.energy_from_battery_mwh,
            'energy_from_grid_mwh': self.energy_from_grid_mwh,
            'pv_coverage_percent': (self.energy_from_pv_mwh / total * 100) if total > 0 else 0,
            'battery_coverage_percent': (self.energy_from_battery_mwh / total * 100) if total > 0 else 0,
            'grid_dependency_percent': (self.energy_from_grid_mwh / total * 100) if total > 0 else 0
        }

# ========================================================================================================
# SEZIONE 4: DEGRADO
# ========================================================================================================
def degradation(cycle_num):
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

# ========================================================================================================
# SEZIONE 5: MODELLO BATTERIA
# ========================================================================================================
class Battery:
    """
    Lorenzo Giannuzzo: Modello batteria con tracking separato carica da rete vs PV
    """
    def __init__(self, technology=BATTERY_TECHNOLOGY,
                 capacity_mwh=BATTERY_CAPACITY_MWH,
                 max_power_mw=BATTERY_MAX_POWER_MW,
                 max_c_rate=BATTERY_MAX_C_RATE):
        self.technology = technology
        self.nominal_capacity = capacity_mwh
        self.capacity = capacity_mwh
        self.max_power = max_power_mw
        self.max_c_rate = max_c_rate

        if technology == "LITIO-IONE":
            self.soc_min = LITHIUM_ION_SOC_MIN
            self.soc_max = LITHIUM_ION_SOC_MAX
            self.dod = LITHIUM_ION_DOD
            self.eol_cycles = LITHIUM_ION_EOL_CYCLES
        elif technology == "GRAFENE":
            self.soc_min = GRAPHENE_SOC_MIN
            self.soc_max = GRAPHENE_SOC_MAX
            self.dod = GRAPHENE_DOD
            self.eol_cycles = GRAPHENE_EOL_CYCLES
        else:
            raise ValueError(f"Tecnologia non supportata: {technology}")

        self.soc = (self.soc_min + self.soc_max) / 2
        self.efficiency_model = BatteryEfficiencyModel(technology, max_c_rate)
        self.efficiency = self.efficiency_model.get_roundtrip_efficiency()
        self.charge_efficiency = self.efficiency_model.get_charge_efficiency()
        self.discharge_efficiency = self.efficiency_model.get_discharge_efficiency()

        self.equivalent_cycles = 0.0
        self.throughput_kwh = 0.0
        self.energy_from_grid_mwh = 0.0
        self.energy_from_pv_mwh = 0.0

        if MACSE_ENABLED:
            self.macse_capacity = MACSE_CAPACITY_MWH
            self.trading_capacity = capacity_mwh - MACSE_CAPACITY_MWH
            macse_percentage = MACSE_CAPACITY_MWH / capacity_mwh
            self.macse_power = max_power_mw * macse_percentage
            self.trading_power = max_power_mw * (1 - macse_percentage)
        else:
            self.macse_capacity = 0
            self.trading_capacity = capacity_mwh
            self.macse_power = 0
            self.trading_power = max_power_mw

        self.macse_available_hours = 0
        self.macse_total_hours = 0

    def charge(self, power_mw, dt=1.0, source='grid'):
        """Carica batteria con tracking provenienza energia"""
        max_power_c_rate = self.trading_capacity * self.max_c_rate
        max_power_trading = min(self.trading_power, max_power_c_rate)
        power_mw = min(power_mw, max_power_trading)

        energy = power_mw * dt * self.charge_efficiency
        new_soc = self.soc + (energy / self.capacity)
        new_soc = min(new_soc, self.soc_max)

        actual_energy = (new_soc - self.soc) * self.capacity
        self.soc = new_soc
        self.throughput_kwh += actual_energy * 1000

        energy_consumed = actual_energy / self.charge_efficiency
        if source == 'grid':
            self.energy_from_grid_mwh += energy_consumed
        elif source == 'pv':
            self.energy_from_pv_mwh += energy_consumed

        return energy_consumed

    def discharge(self, power_mw, dt=1.0):
        """Scarica batteria"""
        max_power_c_rate = self.trading_capacity * self.max_c_rate
        max_power_trading = min(self.trading_power, max_power_c_rate)
        power_mw = min(power_mw, max_power_trading)

        energy = power_mw * dt / self.discharge_efficiency
        new_soc = self.soc - (energy / self.capacity)
        new_soc = max(new_soc, self.soc_min)

        actual_energy = (self.soc - new_soc) * self.capacity
        self.soc = new_soc
        self.throughput_kwh += actual_energy * 1000
        return actual_energy * self.discharge_efficiency

    def check_macse_availability(self):
        if not MACSE_ENABLED:
            return False
        soc_ok_for_discharge = self.soc >= MACSE_MIN_SOC_FOR_DISCHARGE
        soc_ok_for_charge = self.soc <= MACSE_MAX_SOC_FOR_CHARGE
        power_ok = self.macse_power >= 0.9 * (self.max_power * (MACSE_CAPACITY_MWH / self.nominal_capacity))
        return (soc_ok_for_discharge or soc_ok_for_charge) and power_ok

    def update_macse_availability(self, is_available):
        self.macse_total_hours += 1
        if is_available:
            self.macse_available_hours += 1

    def get_macse_availability_factor(self):
        if self.macse_total_hours == 0:
            return 1.0
        return self.macse_available_hours / self.macse_total_hours

    def update_degradation(self):
        if self.technology == "LITIO-IONE":
            self.equivalent_cycles = self.throughput_kwh / (2 * self.nominal_capacity * 1000)
            capacity_percentage = degradation(self.equivalent_cycles)
            self.capacity = self.nominal_capacity * (capacity_percentage / 100.0)
            if MACSE_ENABLED:
                macse_percentage_original = MACSE_CAPACITY_MWH / self.nominal_capacity
                self.macse_capacity = self.nominal_capacity * macse_percentage_original
                self.trading_capacity = self.capacity - self.macse_capacity
        else:
            self.capacity = self.nominal_capacity

    def get_soh(self):
        return (self.capacity / self.nominal_capacity) * 100.0

    def get_soc(self):
        return self.soc

    def get_dod(self):
        return self.dod

    def get_usable_capacity(self):
        return self.capacity * self.dod

    def copy(self):
        b = Battery(self.technology, self.nominal_capacity, self.max_power, self.max_c_rate)
        b.capacity = self.capacity
        b.soc = self.soc
        b.equivalent_cycles = self.equivalent_cycles
        b.throughput_kwh = self.throughput_kwh
        b.macse_capacity = self.macse_capacity
        b.trading_capacity = self.trading_capacity
        b.macse_power = self.macse_power
        b.trading_power = self.trading_power
        b.macse_available_hours = self.macse_available_hours
        b.macse_total_hours = self.macse_total_hours
        b.energy_from_grid_mwh = self.energy_from_grid_mwh
        b.energy_from_pv_mwh = self.energy_from_pv_mwh
        return b

    def get_max_power_by_crate(self):
        return self.trading_capacity * self.max_c_rate


# ========================================================================================================
# SEZIONE 6: OTTIMIZZATORE PSO CON CARICO UTENTE
# ========================================================================================================
class PSOOptimizer:
    """
    Lorenzo Giannuzzo: PSO con decisione COMPLETAMENTE AUTONOMA per allocazione PV + CARICO UTENTE

    LOGICA OTTIMALE CON CARICO:
    1. Carico utente: PV > Batteria > Rete (priorità economica)
    2. PV disponibile dopo carico: Batteria o Vendita
    3. Batteria: Scarica per carico (ricavo virtuale) o Trading
    4. Prezzi: Acquisto = Vendita * (1 + markup)
    """

    def __init__(self, n_particles=50, n_iterations=150, w_start=0.95, w_end=0.1, c1=2.0, c2=2.0):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.stagnation_limit = 15

    def optimize(self, battery, prices_sell, prices_buy, pv_production, load_demand, horizon_hours=24):
        n_hours = min(horizon_hours, len(prices_sell))
        max_power_limit = min(battery.trading_power, battery.get_max_power_by_crate())

        positions = self._smart_initialization(battery, prices_sell, prices_buy, pv_production, load_demand, max_power_limit)
        velocities = np.random.uniform(-1.0, 1.0, (self.n_particles, n_hours))
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self._evaluate(battery, p, prices_sell, prices_buy, pv_production, load_demand)
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
                score = self._evaluate(battery, positions[i], prices_sell, prices_buy, pv_production, load_demand)
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

    def _smart_initialization(self, battery, prices_sell, prices_buy, pv_production, load_demand, max_power):
        """Inizializzazione considerando prezzi, PV e carico"""
        n_hours = len(prices_sell)
        positions = np.zeros((self.n_particles, n_hours))
        price_low = np.percentile(prices_sell, 25)
        price_high = np.percentile(prices_sell, 75)

        for i in range(self.n_particles):
            if i < self.n_particles // 3:
                # Strategia aggressiva
                for h in range(n_hours):
                    net_energy = pv_production[h] - load_demand[h]  # Surplus/deficit dopo carico
                    if prices_sell[h] < price_low and net_energy < 0:
                        positions[i, h] = np.random.uniform(0.5 * max_power, max_power)
                    elif prices_sell[h] > price_high:
                        positions[i, h] = np.random.uniform(-max_power, -0.5 * max_power)
                    else:
                        positions[i, h] = np.random.uniform(-0.3 * max_power, 0.3 * max_power)
            elif i < 2 * self.n_particles // 3:
                # Strategia moderata
                for h in range(n_hours):
                    if prices_sell[h] < price_low:
                        positions[i, h] = np.random.uniform(0, 0.7 * max_power)
                    elif prices_sell[h] > price_high:
                        positions[i, h] = np.random.uniform(-0.7 * max_power, 0)
                    else:
                        positions[i, h] = np.random.uniform(-0.2 * max_power, 0.2 * max_power)
            else:
                # Strategia random
                positions[i] = np.random.uniform(-max_power, max_power, n_hours)
        return positions

    def _evaluate(self, battery, actions, prices_sell, prices_buy, pv_production, load_demand):
        """
        Lorenzo Giannuzzo: LOGICA COMPLETA CON CARICO UTENTE

        PRIORITÀ FORNITURA CARICO:
        1. PV diretto (gratis, massima priorità)
        2. Batteria (ricavo virtuale al prezzo acquisto)
        3. Rete (costo al prezzo acquisto)

        GESTIONE SURPLUS PV:
        - Dopo aver servito carico, PV surplus va a batteria o rete

        TRADING BATTERIA:
        - Azione PSO: carica da rete o scarica verso rete
        """
        bat_sim = battery.copy()
        profit = 0.0

        for hour, (power, price_sell, price_buy, pv_energy, load_energy) in enumerate(zip(actions, prices_sell, prices_buy, pv_production, load_demand)):
            pv_available = pv_energy if PV_ENABLED else 0.0
            load_required = load_energy if LOAD_ENABLED else 0.0

            # ===================================================================
            # FASE 1: SODDISFACIMENTO CARICO (PRIORITÀ MASSIMA)
            # ===================================================================
            load_from_pv = 0.0
            load_from_battery = 0.0
            load_from_grid = 0.0

            if load_required > 0.001:
                # 1. PV diretto al carico (GRATIS, priorità massima)
                if pv_available > 0:
                    load_from_pv = min(load_required, pv_available)
                    pv_available -= load_from_pv
                    load_required -= load_from_pv
                    # Ricavo virtuale: risparmio acquisto da rete
                    profit += load_from_pv * price_buy

                # 2. Batteria per carico (se PV insufficiente)
                if load_required > 0.001 and bat_sim.soc > bat_sim.soc_min:
                    max_discharge_load = (bat_sim.soc - bat_sim.soc_min) * bat_sim.trading_capacity * bat_sim.discharge_efficiency
                    load_from_battery = min(load_required, max_discharge_load)
                    if load_from_battery > 0.001:
                        actual_discharge = load_from_battery / bat_sim.discharge_efficiency
                        new_soc = bat_sim.soc - (actual_discharge / bat_sim.capacity)
                        bat_sim.soc = max(new_soc, bat_sim.soc_min)
                        bat_sim.throughput_kwh += actual_discharge * 1000
                        load_required -= load_from_battery
                        # Ricavo virtuale: come se vendessimo a prezzo acquisto
                        profit += load_from_battery * price_buy
                        # Costo degrado batteria
                        degradation_cost = load_from_battery * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                        profit -= degradation_cost

                # 3. Rete per carico (ultima opzione)
                if load_required > 0.001:
                    load_from_grid = load_required
                    profit -= load_from_grid * price_buy  # COSTO acquisto da rete

            # ===================================================================
            # FASE 2: DECISIONE PSO PER BATTERIA (dopo servizio carico)
            # ===================================================================
            if power > 0.01:  # RICHIESTA CARICA BATTERIA
                if bat_sim.soc >= bat_sim.soc_max:
                    # Batteria piena: vendi PV surplus
                    if pv_available > 0:
                        profit += pv_available * price_sell
                    continue

                max_energy_storable = (bat_sim.soc_max - bat_sim.soc) * bat_sim.trading_capacity
                max_power_available = max_energy_storable / (1.0 * bat_sim.charge_efficiency)
                actual_power = min(power, max_power_available)

                if actual_power > 0.01:
                    energy_needed = actual_power * 1.0

                    # Usa PV surplus disponibile (dopo carico)
                    if pv_available > 0:
                        energy_from_pv = min(energy_needed, pv_available)
                        bat_sim.charge(energy_from_pv / 1.0, dt=1.0, source='pv')
                        degradation_cost_pv = energy_from_pv * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                        profit -= degradation_cost_pv
                        energy_needed -= energy_from_pv
                        pv_available -= energy_from_pv

                        # Vendi surplus PV residuo
                        if pv_available > 0:
                            profit += pv_available * price_sell

                    # Completa carica da rete
                    if energy_needed > 0.01:
                        energy_from_grid = bat_sim.charge(energy_needed / 1.0, dt=1.0, source='grid')
                        profit -= energy_from_grid * price_buy  # COSTO acquisto
                        degradation_cost_grid = energy_from_grid * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                        profit -= degradation_cost_grid
                else:
                    # Non può caricare: vendi PV surplus
                    if pv_available > 0:
                        profit += pv_available * price_sell

            elif power < -0.01:  # RICHIESTA SCARICA BATTERIA (TRADING)
                if bat_sim.soc <= bat_sim.soc_min:
                    # Batteria scarica: vendi solo PV surplus
                    if pv_available > 0:
                        profit += pv_available * price_sell
                    continue

                max_energy_available = (bat_sim.soc - bat_sim.soc_min) * bat_sim.trading_capacity
                max_power_available = max_energy_available * bat_sim.discharge_efficiency / 1.0
                actual_power = min(-power, max_power_available)

                if actual_power > 0.01:
                    # Scarica batteria per trading
                    energy_to_grid_battery = bat_sim.discharge(actual_power, dt=1.0)
                    total_energy_sold = energy_to_grid_battery + pv_available
                    profit += total_energy_sold * price_sell  # RICAVO vendita
                    degradation_cost = energy_to_grid_battery * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                    profit -= degradation_cost
                else:
                    # Non può scaricare: vendi solo PV surplus
                    if pv_available > 0:
                        profit += pv_available * price_sell

            else:  # IDLE
                # Vendi PV surplus
                if pv_available > 0:
                    profit += pv_available * price_sell

        return profit


# ========================================================================================================
# SEZIONE 7: ROLLING HORIZON SIMULATOR CON PV E CARICO
# ========================================================================================================
class RollingHorizonSimulator:
    def __init__(self, battery, optimizer, pv_system=None, load_profile=None, horizon_hours=24, step_hours=1):
        self.battery = battery
        self.optimizer = optimizer
        self.pv_system = pv_system
        self.load_profile = load_profile
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours

    def simulate(self, prices_df, prices_df2, pv_df=None, load_df=None):
        prices_sell = prices_df['€/MWh'].values
        prices_buy = prices_df2['€/MWh'].values
        # Calcola prezzo acquisto con mark-up
        n_hours = len(prices_sell)

        # Prepara produzione PV
        if PV_ENABLED and pv_df is not None and self.pv_system is not None:
            pv_production = np.array([self.pv_system.get_energy(irr, dt=1.0) for irr in pv_df['P'].values])
            if len(pv_production) < n_hours:
                pv_production = np.pad(pv_production, (0, n_hours - len(pv_production)), 'constant')
            elif len(pv_production) > n_hours:
                pv_production = pv_production[:n_hours]
        else:
            pv_production = np.zeros(n_hours)

        # Prepara carico
        if LOAD_ENABLED and load_df is not None:
            load_demand = load_df['value'].values / 1000.0  # kW -> MW
            if len(load_demand) < n_hours:
                load_demand = np.pad(load_demand, (0, n_hours - len(load_demand)), 'constant')
            elif len(load_demand) > n_hours:
                load_demand = load_demand[:n_hours]
        else:
            load_demand = np.zeros(n_hours)

        # Arrays risultati
        actions_taken = []
        soc_history = []
        capacity_history = []
        soh_history = []
        profits_history = []
        macse_availability_history = []
        pv_production_history = []
        pv_to_battery_history = []
        pv_to_grid_history = []
        pv_to_load_history = []
        load_demand_history = []
        load_from_pv_history = []
        load_from_battery_history = []
        load_from_grid_history = []
        energy_from_grid_history = []
        energy_from_pv_history = []
        price_buy_history = []

        cumulative_profit = 0.0

        print("=" * 80)
        print("SIMULAZIONE BESS CON PSO, ROLLING HORIZON, MACSE, PV E CARICO UTENTE")
        print("=" * 80)
        print(f"TECNOLOGIA BATTERIA: {self.battery.technology}")
        print(f"Capacita nominale: {self.battery.nominal_capacity} MWh")
        print(f"Potenza massima: {self.battery.max_power} MW")

        if PV_ENABLED and self.pv_system is not None:
            print(f"\nFOTOVOLTAICO ABILITATO:")
            print(f"  - Potenza nominale: {self.pv_system.nominal_power_kwp:.0f} kWp")
            print(f"  - Strategia: AUTONOMA + SERVIZIO CARICO PRIORITARIO")

        if LOAD_ENABLED:
            print(f"\nCARICO UTENTE ABILITATO:")
            print(f"  - Priorità fornitura: PV > Batteria > Rete")
            print(f"  - Ricavo virtuale da PV/Batteria al prezzo acquisto")

        print(f"\nPREZZI:")
        print(f"  - Mark-up acquisto/vendita: {PRICE_MARKUP_PERCENT}%")
        print(f"  - Prezzo medio vendita: {np.mean(prices_sell):.2f} €/MWh")
        print(f"  - Prezzo medio acquisto: {np.mean(prices_buy):.2f} €/MWh")

        if MACSE_ENABLED:
            print(f"\nMACSE ABILITATO:")
            print(f"  - Capacita MACSE: {self.battery.macse_capacity:.2f} MWh")

        print(f"\nOrizzonte: {self.horizon_hours} ore")
        print(f"Ore totali: {n_hours}")
        print("=" * 80)
        print()

        current_hour = 0
        last_progress = 0
        degradation_update_interval = 24

        while current_hour < n_hours:
            progress = int((current_hour / n_hours) * 100)
            if progress >= last_progress + 20:
                print(f"Progresso: {progress}% ({current_hour}/{n_hours} h) - SOH: {self.battery.get_soh():.2f}%")
                last_progress = progress

            if current_hour % degradation_update_interval == 0 and current_hour > 0:
                self.battery.update_degradation()

            end_hour = min(current_hour + self.horizon_hours, n_hours)
            window_prices_sell = prices_sell[current_hour:end_hour]
            window_prices_buy = prices_buy[current_hour:end_hour]
            window_pv = pv_production[current_hour:end_hour]
            window_load = load_demand[current_hour:end_hour]

            # OTTIMIZZAZIONE PSO con carico
            optimal_actions = self.optimizer.optimize(
                self.battery, window_prices_sell, window_prices_buy, window_pv, window_load, self.horizon_hours
            )

            action = optimal_actions[0]
            price_sell = prices_sell[current_hour]
            price_buy = prices_buy[current_hour]
            pv_energy_available = pv_production[current_hour]
            load_required = load_demand[current_hour]

            # Tracking per questa ora
            pv_to_battery_this_hour = 0.0
            pv_to_grid_this_hour = 0.0
            pv_to_load_this_hour = 0.0
            load_from_pv_this_hour = 0.0
            load_from_battery_this_hour = 0.0
            load_from_grid_this_hour = 0.0
            grid_to_battery_this_hour = 0.0

            # ===================================================================
            # ESECUZIONE: SODDISFACIMENTO CARICO (PRIORITÀ)
            # ===================================================================
            if load_required > 0.001:
                # 1. PV diretto al carico
                if pv_energy_available > 0:
                    load_from_pv_this_hour = min(load_required, pv_energy_available)
                    pv_to_load_this_hour = load_from_pv_this_hour
                    pv_energy_available -= load_from_pv_this_hour
                    load_required -= load_from_pv_this_hour
                    # Ricavo virtuale
                    profit_virtual = load_from_pv_this_hour * price_buy
                    cumulative_profit += profit_virtual

                # 2. Batteria per carico
                if load_required > 0.001 and self.battery.soc > self.battery.soc_min:
                    # LIMITE C-RATE: SEMPRE applicato su nominal_capacity
                    max_power_c_rate = self.battery.nominal_capacity * self.battery.max_c_rate  # 4 MWh * 0.5 = 2 MW
                    max_power_physical = min(self.battery.max_power, max_power_c_rate)

                    # Limite da SOC disponibile
                    max_discharge_from_soc = (self.battery.soc - self.battery.soc_min) * self.battery.trading_capacity * self.battery.discharge_efficiency

                    # Energia massima per carico (MWh in dt=1h)
                    max_energy_for_load = min(max_power_physical * 1.0, max_discharge_from_soc)

                    load_from_battery_this_hour = min(load_required, max_energy_for_load)

                    if load_from_battery_this_hour > 0.001:
                        energy_discharged = self.battery.discharge(load_from_battery_this_hour / self.battery.discharge_efficiency, dt=1.0)
                        load_required -= load_from_battery_this_hour
                        # Ricavo virtuale
                        profit_virtual = load_from_battery_this_hour * price_buy
                        cumulative_profit += profit_virtual

                # 3. Rete per carico
                if load_required > 0.001:
                    load_from_grid_this_hour = load_required
                    profit_grid = -load_from_grid_this_hour * price_buy
                    cumulative_profit += profit_grid

            # ===================================================================
            # ESECUZIONE: DECISIONE PSO BATTERIA
            # ===================================================================

            # LIMITE FISICO: considera max_power e C-rate sulla CAPACITÀ NOMINALE
            # Lorenzo: C-rate si applica su NOMINAL_CAPACITY (4 MWh), non trading (3 MWh)!
            power_used_for_load_mw = load_from_battery_this_hour / 1.0  # MWh → MW

            # Limite C-rate sulla capacità NOMINALE totale
            max_power_c_rate = self.battery.nominal_capacity * self.battery.max_c_rate  # 4 MWh * 0.5 = 2 MW

            # Limite fisico disponibile
            max_power_physical = self.battery.max_power - power_used_for_load_mw

            # Il limite finale è il minimo tra fisico e C-rate
            max_power_available_trading = min(max_power_physical, max_power_c_rate)

            # Inizializza azione effettivamente eseguita (da salvare in history)
            actual_action = 0.0

            if action > 0.01:  # CARICA
                if self.battery.soc < self.battery.soc_max:
                    max_energy_storable = (self.battery.soc_max - self.battery.soc) * self.battery.trading_capacity
                    max_power_available = max_energy_storable / (1.0 * self.battery.charge_efficiency)

                    # LIMITE FISICO: considera potenza già usata per carico
                    actual_power = min(action, max_power_available, max_power_available_trading)
                    actual_action = actual_power  # Salva potenza effettiva

                    if actual_power > 0.01:
                        energy_needed = actual_power * 1.0

                        # Usa PV surplus
                        if pv_energy_available > 0:
                            energy_from_pv = min(energy_needed, pv_energy_available)
                            self.battery.charge(energy_from_pv / 1.0, dt=1.0, source='pv')
                            pv_to_battery_this_hour = energy_from_pv
                            energy_needed -= energy_from_pv
                            pv_energy_available -= energy_from_pv

                            # Vendi surplus PV
                            if pv_energy_available > 0:
                                profit_pv = pv_energy_available * price_sell
                                cumulative_profit += profit_pv
                                pv_to_grid_this_hour = pv_energy_available
                                pv_energy_available = 0

                        # Completa da rete
                        if energy_needed > 0.01:
                            energy_from_grid = self.battery.charge(energy_needed / 1.0, dt=1.0, source='grid')
                            grid_to_battery_this_hour = energy_from_grid
                            profit = -energy_from_grid * price_buy
                            cumulative_profit += profit
                    else:
                        if pv_energy_available > 0:
                            profit_pv = pv_energy_available * price_sell
                            cumulative_profit += profit_pv
                            pv_to_grid_this_hour = pv_energy_available
                else:
                    actual_action = 0.0  # Batteria piena, nessuna carica
                    if pv_energy_available > 0:
                        profit_pv = pv_energy_available * price_sell
                        cumulative_profit += profit_pv
                        pv_to_grid_this_hour = pv_energy_available

            elif action < -0.01:  # SCARICA (TRADING)
                if self.battery.soc > self.battery.soc_min:
                    max_energy_available = (self.battery.soc - self.battery.soc_min) * self.battery.trading_capacity
                    max_power_by_soc = max_energy_available * self.battery.discharge_efficiency / 1.0

                    # LIMITE FISICO: considera potenza già usata per carico
                    actual_power = min(-action, max_power_by_soc, max_power_available_trading)
                    actual_action = -actual_power  # Negativo per scarica

                    if actual_power > 0.01:
                        energy_to_grid = self.battery.discharge(actual_power, dt=1.0)
                        total_energy_sold = energy_to_grid + pv_energy_available
                        profit = total_energy_sold * price_sell
                        cumulative_profit += profit

                        if pv_energy_available > 0:
                            pv_to_grid_this_hour = pv_energy_available
                    else:
                        if pv_energy_available > 0:
                            profit_pv = pv_energy_available * price_sell
                            cumulative_profit += profit_pv
                            pv_to_grid_this_hour = pv_energy_available
                else:
                    actual_action = 0.0  # Batteria scarica, nessuna scarica
                    if pv_energy_available > 0:
                        profit_pv = pv_energy_available * price_sell
                        cumulative_profit += profit_pv
                        pv_to_grid_this_hour = pv_energy_available

            else:  # IDLE
                actual_action = 0.0  # Nessuna azione
                if pv_energy_available > 0:
                    profit_pv = pv_energy_available * price_sell
                    cumulative_profit += profit_pv
                    pv_to_grid_this_hour = pv_energy_available

            # Registra allocazioni
            if self.pv_system and (pv_to_battery_this_hour > 0 or pv_to_grid_this_hour > 0 or pv_to_load_this_hour > 0):
                self.pv_system.allocate_energy(pv_to_battery_this_hour, pv_to_grid_this_hour, pv_to_load_this_hour)

            if self.load_profile:
                self.load_profile.register_supply(load_from_pv_this_hour, load_from_battery_this_hour, load_from_grid_this_hour)

            macse_available = self.battery.check_macse_availability() if MACSE_ENABLED else False
            if MACSE_ENABLED:
                self.battery.update_macse_availability(macse_available)

            # SALVA actual_action (potenza EFFETTIVAMENTE applicata), NON action (potenza richiesta PSO)
            actions_taken.append(actual_action)
            soc_history.append(self.battery.get_soc())
            capacity_history.append(self.battery.capacity)
            soh_history.append(self.battery.get_soh())
            profits_history.append(cumulative_profit)
            macse_availability_history.append(self.battery.get_macse_availability_factor() if MACSE_ENABLED else 0)
            pv_production_history.append(pv_production[current_hour])
            pv_to_battery_history.append(pv_to_battery_this_hour)
            pv_to_grid_history.append(pv_to_grid_this_hour)
            pv_to_load_history.append(pv_to_load_this_hour)
            load_demand_history.append(load_demand[current_hour])
            load_from_pv_history.append(load_from_pv_this_hour)
            load_from_battery_history.append(load_from_battery_this_hour)
            load_from_grid_history.append(load_from_grid_this_hour)
            energy_from_grid_history.append(grid_to_battery_this_hour)
            energy_from_pv_history.append(pv_to_battery_this_hour)
            price_buy_history.append(price_buy)

            current_hour += self.step_hours

        self.battery.update_degradation()

        print(f"Progresso: 100% completato ({n_hours}/{n_hours} ore)")
        print()

        results_df = prices_df.copy()
        pad_length = len(results_df) - len(actions_taken)
        results_df['Prezzo_Acquisto_€/MWh'] = price_buy_history + [price_buy_history[-1]] * pad_length
        results_df['Azione_MW'] = actions_taken + [0] * pad_length
        results_df['SOC'] = soc_history + [soc_history[-1]] * pad_length
        results_df['Capacita_MWh'] = capacity_history + [capacity_history[-1]] * pad_length
        results_df['SOH_%'] = soh_history + [soh_history[-1]] * pad_length
        results_df['Profitto_Euro'] = profits_history + [profits_history[-1]] * pad_length
        results_df['MACSE_Availability'] = macse_availability_history + [macse_availability_history[-1]] * pad_length
        results_df['PV_Production_MWh'] = pv_production_history + [0] * pad_length
        results_df['PV_to_Battery_MWh'] = pv_to_battery_history + [0] * pad_length
        results_df['PV_to_Grid_MWh'] = pv_to_grid_history + [0] * pad_length
        results_df['PV_to_Load_MWh'] = pv_to_load_history + [0] * pad_length
        results_df['Load_Demand_MWh'] = load_demand_history + [0] * pad_length
        results_df['Load_from_PV_MWh'] = load_from_pv_history + [0] * pad_length
        results_df['Load_from_Battery_MWh'] = load_from_battery_history + [0] * pad_length
        results_df['Load_from_Grid_MWh'] = load_from_grid_history + [0] * pad_length
        results_df['Energy_from_Grid_MWh'] = energy_from_grid_history + [0] * pad_length
        results_df['Energy_from_PV_MWh'] = energy_from_pv_history + [0] * pad_length

        return results_df, cumulative_profit


# ========================================================================================================
# SEZIONE 8-9: MACSE, JSON EXPORT (identiche alla versione precedente)
# ========================================================================================================
def calculate_macse_revenue(battery):
    if not MACSE_ENABLED:
        return 0, 0, 0, 0
    base_revenue = (MACSE_PRICE_PER_MW_YEAR * battery.macse_power * MACSE_CONTRACT_YEARS) / (365 * 24)
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

def export_results_to_json(results_df, battery, pv_system, load_profile, trading_profit, macse_revenue, macse_base,
                           macse_penalty, macse_bonus, battery_investment, simulation_time):
    # (Funzione estesa per includere statistiche carico)
    actions = results_df['Azione_MW'].values
    prices_sell = results_df['€/MWh'].values
    prices_buy = results_df['Prezzo_Acquisto_€/MWh'].values

    pv_stats = pv_system.get_statistics() if pv_system else {}
    load_stats = load_profile.get_statistics() if load_profile else {}
    # Statistiche PV
    pv_stats = pv_system.get_statistics() if pv_system else {
        'total_production_mwh': 0,
        'energy_to_battery_mwh': 0,
        'energy_to_grid_mwh': 0,
        'energy_to_load_mwh': 0,
        'curtailed_energy_mwh': 0,
        'battery_utilization_percent': 0,
        'grid_sale_percent': 0,
        'load_service_percent': 0,
        'curtailment_percent': 0
    }

    # Statistiche carico
    load_stats = load_profile.get_statistics() if load_profile else {
        'total_energy_required_mwh': 0,
        'energy_from_pv_mwh': 0,
        'energy_from_battery_mwh': 0,
        'energy_from_grid_mwh': 0,
        'pv_coverage_percent': 0,
        'battery_coverage_percent': 0,
        'grid_dependency_percent': 0
    }

    # Analisi azioni batteria
    charge_hours = np.sum(actions > 0.01)
    discharge_hours = np.sum(actions < -0.01)
    idle_hours = len(actions) - charge_hours - discharge_hours
    total_energy_charged = np.sum(actions[actions > 0] * 1.0)
    total_energy_discharged = np.sum(np.abs(actions[actions < 0]) * 1.0)

    # ROI e payback
    total_revenue = trading_profit + macse_revenue
    annual_profit = total_revenue
    roi_percent = (annual_profit / battery_investment) * 100 if battery_investment > 0 else 0
    payback_years = battery_investment / annual_profit if annual_profit > 0 else float('inf')

    # Costruisci JSON
    results_json = {
        "simulation_info": {
            "technology": battery.technology,
            "simulation_time_seconds": simulation_time,
            "total_hours": len(actions),
            "timestamp": datetime.now().isoformat()
        },
        "battery_parameters": {
            "nominal_capacity_mwh": battery.nominal_capacity,
            "max_power_mw": battery.max_power,
            "max_c_rate": battery.max_c_rate,
            "soc_min": battery.soc_min,
            "soc_max": battery.soc_max,
            "charge_efficiency": battery.charge_efficiency,
            "discharge_efficiency": battery.discharge_efficiency,
            "roundtrip_efficiency": battery.efficiency
        },
        "battery_state": {
            "final_soc": float(battery.get_soc()),
            "final_soh_percent": float(battery.get_soh()),
            "final_capacity_mwh": float(battery.capacity),
            "equivalent_cycles": float(battery.equivalent_cycles),
            "throughput_kwh": float(battery.throughput_kwh),
            "energy_from_grid_mwh": float(battery.energy_from_grid_mwh),
            "energy_from_pv_mwh": float(battery.energy_from_pv_mwh)
        },
        "trading_operations": {
            "charge_hours": int(charge_hours),
            "discharge_hours": int(discharge_hours),
            "idle_hours": int(idle_hours),
            "total_energy_charged_mwh": float(total_energy_charged),
            "total_energy_discharged_mwh": float(total_energy_discharged),
            "utilization_factor": float((charge_hours + discharge_hours) / len(actions)) if len(actions) > 0 else 0
        },
        "photovoltaic_system": {
            "enabled": PV_ENABLED,
            "nominal_power_kwp": PV_NOMINAL_POWER_KWP if PV_ENABLED else 0,
            "total_production_mwh": pv_stats['total_production_mwh'],
            "energy_to_battery_mwh": pv_stats['energy_to_battery_mwh'],
            "energy_to_grid_mwh": pv_stats['energy_to_grid_mwh'],
            "energy_to_load_mwh": pv_stats['energy_to_load_mwh'],
            "curtailed_energy_mwh": pv_stats['curtailed_energy_mwh'],
            "battery_utilization_percent": pv_stats['battery_utilization_percent'],
            "grid_sale_percent": pv_stats['grid_sale_percent'],
            "load_service_percent": pv_stats['load_service_percent'],
            "curtailment_percent": pv_stats['curtailment_percent']
        },
        "load_profile": {
            "enabled": LOAD_ENABLED,
            "total_energy_required_mwh": load_stats['total_energy_required_mwh'],
            "energy_from_pv_mwh": load_stats['energy_from_pv_mwh'],
            "energy_from_battery_mwh": load_stats['energy_from_battery_mwh'],
            "energy_from_grid_mwh": load_stats['energy_from_grid_mwh'],
            "pv_coverage_percent": load_stats['pv_coverage_percent'],
            "battery_coverage_percent": load_stats['battery_coverage_percent'],
            "grid_dependency_percent": load_stats['grid_dependency_percent']
        },
        "macse": {
            "enabled": MACSE_ENABLED,
            "capacity_mwh": battery.macse_capacity if MACSE_ENABLED else 0,
            "power_mw": battery.macse_power if MACSE_ENABLED else 0,
            "availability_factor": float(battery.get_macse_availability_factor()) if MACSE_ENABLED else 0,
            "base_revenue_euro": float(macse_base),
            "penalty_euro": float(macse_penalty),
            "bonus_euro": float(macse_bonus),
            "total_revenue_euro": float(macse_revenue)
        },
        "economic_results": {
            "trading_profit_euro": float(trading_profit),
            "macse_revenue_euro": float(macse_revenue),
            "total_system_profit_euro": float(total_revenue),
            "battery_investment_euro": float(battery_investment),
            "roi_percent": float(roi_percent),
            "payback_years": float(payback_years) if payback_years != float('inf') else None,
            "avg_price_sell_euro_mwh": float(np.mean(prices_sell)),
            "avg_price_buy_euro_mwh": float(np.mean(prices_buy)),
            "price_markup_percent": PRICE_MARKUP_PERCENT
        }
    }

    # Salva JSON
    json_file = os.path.join('results',
                             f'simulation_results_{battery.technology.lower().replace("-", "_")}_with_load.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"Risultati JSON salvati in: {json_file}")
    print("Export JSON con statistiche carico integrato completato")


# ========================================================================================================
# SEZIONE 10: VISUALIZZAZIONI (da estendere con grafici carico)
# ========================================================================================================
# ========================================================================================================
# SEZIONE 10: VISUALIZZAZIONI PV (TUTTE LE FUNZIONI ORIGINALI MANTENUTE)
# ========================================================================================================
def create_detailed_monthly_pv_plots(results_df, battery, pv_system):
    """
    Lorenzo Giannuzzo: GRAFICI DETTAGLIATI MENSILI con IMPATTO PV
    Mostra chiaramente quando batteria carica da PV vs rete
    """
    if not SAVE_PLOTS or not PV_ENABLED or pv_system is None:
        return

    viz_folder = 'visualization'
    monthly_pv_folder = os.path.join(viz_folder, 'dettaglio_mensile_pv')

    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    if not os.path.exists(monthly_pv_folder):
        os.makedirs(monthly_pv_folder)

    print("\n" + "=" * 80)
    print("GENERAZIONE GRAFICI MENSILI DETTAGLIATI CON IMPATTO PV")
    print("=" * 80)

    if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
        results_df['Data'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')

    results_df['Mese'] = results_df['Data'].dt.month
    results_df['Giorno'] = results_df['Data'].dt.day

    mesi_nomi = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno',
                 'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']

    # Per ogni mese, crea grafico dettagliato
    for mese in range(1, 13):
        df_mese = results_df[results_df['Mese'] == mese].copy()
        if len(df_mese) == 0:
            continue

        # Seleziona giorno rappresentativo (15° o medio)
        giorni_disponibili = df_mese['Giorno'].unique()
        giorno_target = 15 if 15 in giorni_disponibili else giorni_disponibili[len(giorni_disponibili) // 2]
        df_giorno = df_mese[df_mese['Giorno'] == giorno_target].copy()

        if len(df_giorno) < 24:
            for g in giorni_disponibili:
                df_test = df_mese[df_mese['Giorno'] == g]
                if len(df_test) >= 24:
                    df_giorno = df_test.copy()
                    giorno_target = g
                    break

        df_giorno = df_giorno.head(24).copy()
        df_giorno['Ora'] = range(len(df_giorno))

        # Crea figura con 4 subplot per analisi completa
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

        data_str = df_giorno['Data'].iloc[0].strftime('%d/%m/%Y') if len(df_giorno) > 0 else ''
        fig.suptitle(f'Analisi Dettagliata PV - {mesi_nomi[mese-1]} {data_str}\n{battery.technology}',
                     fontsize=16, fontweight='bold', y=0.995)

        # Subplot 1: PRODUZIONE PV E ALLOCAZIONE
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df_giorno['Ora'], df_giorno['PV_Production_MWh'],
                color='#F4A300', linewidth=3, marker='o', markersize=6,
                label='Produzione PV Totale', zorder=5)
        ax1.fill_between(df_giorno['Ora'], 0, df_giorno['PV_Production_MWh'],
                         color='#F4A300', alpha=0.2)
        ax1.bar(df_giorno['Ora'], df_giorno['PV_to_Battery_MWh'],
               width=0.7, color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1,
               label='PV → Batteria (PSO Decision)')
        ax1.bar(df_giorno['Ora'], df_giorno['PV_to_Grid_MWh'],
               width=0.7, bottom=df_giorno['PV_to_Battery_MWh'],
               color='#457B9D', alpha=0.8, edgecolor='black', linewidth=1,
               label='PV → Rete Diretta (PSO Decision)')
        if 'PV_to_Load_MWh' in df_giorno.columns:
            ax1.bar(df_giorno['Ora'], df_giorno['PV_to_Load_MWh'],
                   width=0.7, bottom=df_giorno['PV_to_Battery_MWh'] + df_giorno['PV_to_Grid_MWh'],
                   color='#F77F00', alpha=0.8, edgecolor='black', linewidth=1,
                   label='PV → Carico Diretto')
        ax1.set_ylabel('Energia (MWh)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Ora', fontsize=11)
        ax1.set_title('Produzione e Allocazione Fotovoltaica', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper left', framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(0, 24, 2))

        # Subplot 2: FONTI DI CARICA BATTERIA
        ax2 = fig.add_subplot(gs[1, 0])

        # USA PV_to_Battery per coerenza con grafico sopra
        # STESSA soglia del subplot 3 per coerenza
        charge_from_grid = []
        charge_from_pv = []
        for idx, row in df_giorno.iterrows():
            if row['Azione_MW'] > 0.01:  # STESSA soglia subplot 3: 10 kW
                # Usa allocazione PV, non consumo
                charge_from_pv.append(row.get('PV_to_Battery_MWh', 0))
                charge_from_grid.append(row.get('Energy_from_Grid_MWh', 0))
            else:
                charge_from_grid.append(0)
                charge_from_pv.append(0)

        width = 0.8
        ax2.bar(df_giorno['Ora'], charge_from_pv, width=width,
               color='#06A77D', alpha=0.9, edgecolor='black', linewidth=1.5,
               label='Carica da PV (GRATIS)')
        ax2.bar(df_giorno['Ora'], charge_from_grid, width=width,
               bottom=charge_from_pv,
               color='#E63946', alpha=0.9, edgecolor='black', linewidth=1.5,
               label='Carica da Rete (ACQUISTO)')
        ax2.set_ylabel('Energia Caricata (MWh)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Ora', fontsize=11)
        ax2.set_title('Fonti di Carica Batteria: PV vs Rete', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper left', framealpha=0.95)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_xlim(-0.5, 23.5)
        ax2.set_xticks(range(0, 24, 2))
        ax2.axhline(y=0, color='black', linewidth=1, linestyle='-')

        # Subplot 3: AZIONI BATTERIA CON FONTI ENERGIA (STACKED) + PREZZO
        ax3 = fig.add_subplot(gs[1, 1])
        ax3_twin = ax3.twinx()

        # LORENZO: Barre STACKED come subplot 2 - energia da PV sopra energia da rete
        # Colori: Verde (PV), Arancione (rete), Rosso (scarica)

        charge_pv_bars = []
        charge_grid_bars = []
        discharge_bars = []

        for idx, row in df_giorno.iterrows():
            action = row['Azione_MW']

            if action > 0.01:  # CARICA
                pv_to_batt = row.get('PV_to_Battery_MWh', 0)
                grid_to_batt = row.get('Energy_from_Grid_MWh', 0)

                # Converte MWh in MW (azione in MW per dt=1h)
                charge_pv_bars.append(pv_to_batt)
                charge_grid_bars.append(grid_to_batt)
                discharge_bars.append(0.0)

            elif action < -0.01:  # SCARICA
                charge_pv_bars.append(0.0)
                charge_grid_bars.append(0.0)
                discharge_bars.append(action)  # Negativo

            else:  # IDLE
                charge_pv_bars.append(0.0)
                charge_grid_bars.append(0.0)
                discharge_bars.append(0.0)

        width = 0.8

        # Plotta SCARICA (rosso)
        ax3.bar(df_giorno['Ora'], discharge_bars, width=width,
               color='#E63946', alpha=0.8, edgecolor='black', linewidth=1,
               label='Scarica')

        # Plotta CARICA da RETE (arancione, base)
        ax3.bar(df_giorno['Ora'], charge_grid_bars, width=width,
               color='#F77F00', alpha=0.8, edgecolor='black', linewidth=1,
               label='Carica da Rete')

        # Plotta CARICA da PV (verde, sopra rete)
        ax3.bar(df_giorno['Ora'], charge_pv_bars, width=width,
               bottom=charge_grid_bars,
               color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1,
               label='Carica da PV')

        # Linea prezzo energia
        ax3_twin.plot(df_giorno['Ora'], df_giorno['€/MWh'],
                     color='#457B9D', linewidth=2.5, marker='s', markersize=5,
                     label='Prezzo Energia', zorder=10)

        # Linea zero
        ax3.axhline(y=0, color='black', linewidth=1.5, linestyle='-', zorder=5)

        # Etichette e formattazione
        ax3.set_ylabel('Potenza Batteria (MW)', fontsize=12, fontweight='bold')
        ax3_twin.set_ylabel('Prezzo (€/MWh)', fontsize=12, fontweight='bold', color='#457B9D')
        ax3_twin.tick_params(axis='y', labelcolor='#457B9D')
        ax3.set_xlabel('Ora', fontsize=11)
        ax3.set_title('Azioni Batteria (Fonti Energia) vs Prezzo', fontsize=13, fontweight='bold')

        # Legends
        ax3.legend(fontsize=9, loc='upper left', framealpha=0.95)
        ax3_twin.legend(fontsize=9, loc='upper right', framealpha=0.95)

        # Grid e limiti
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_xticks(range(0, 24, 2))

        # IMPORTANTE: Forza limiti corretti asse Y
        # Trova min/max delle azioni per settare limiti sensati
        all_actions = discharge_bars + [sum(x) for x in zip(charge_pv_bars, charge_grid_bars)]
        if any(x != 0 for x in all_actions):
            y_max = max(abs(min(all_actions)), max(all_actions)) * 1.1
            ax3.set_ylim(-y_max, y_max)

        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_xticks(range(0, 24, 2))

        # Subplot 4: SOC E IMPATTO PV
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(df_giorno['Ora'], df_giorno['SOC'] * 100,
                color='#457B9D', linewidth=3, marker='o', markersize=6,
                label='State of Charge')
        ax4.fill_between(df_giorno['Ora'], battery.soc_min * 100, df_giorno['SOC'] * 100,
                         color='#457B9D', alpha=0.2)
        for idx, row in df_giorno.iterrows():
            if row['Energy_from_PV_MWh'] > 0.01:
                ax4.axvspan(row['Ora']-0.4, row['Ora']+0.4,
                           color='#06A77D', alpha=0.15, zorder=0)
        ax4.axhline(y=battery.soc_min * 100, color='red', linewidth=1.5,
                   linestyle='--', alpha=0.7, label=f'SOC min ({battery.soc_min*100:.0f}%)')
        ax4.axhline(y=battery.soc_max * 100, color='green', linewidth=1.5,
                   linestyle='--', alpha=0.7, label=f'SOC max ({battery.soc_max*100:.0f}%)')
        ax4.set_ylabel('SOC (%)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Ora', fontsize=11)
        ax4.set_title('State of Charge (sfondo verde = carica da PV)', fontsize=13, fontweight='bold')
        ax4.set_ylim(0, 100)
        ax4.legend(fontsize=9, loc='best', framealpha=0.95)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xlim(-0.5, 23.5)
        ax4.set_xticks(range(0, 24, 2))

        # Subplot 5: BILANCIO ECONOMICO ORARIO
        ax5 = fig.add_subplot(gs[2, 1])
        profitto_orario = []
        for idx, row in df_giorno.iterrows():
            profit_hour = 0
            if row['Energy_from_Grid_MWh'] > 0:
                profit_hour -= row['Energy_from_Grid_MWh'] * row['€/MWh']
            if row['Azione_MW'] < -0.01:
                energy_discharged = abs(row['Azione_MW']) * 1.0 * battery.discharge_efficiency
                profit_hour += energy_discharged * row['€/MWh']
            if row['PV_to_Grid_MWh'] > 0:
                profit_hour += row['PV_to_Grid_MWh'] * row['€/MWh']
            if row['Energy_from_PV_MWh'] > 0:
                profit_hour += row['Energy_from_PV_MWh'] * row['€/MWh'] * 0.5
            if 'Load_from_PV_MWh' in row and row['Load_from_PV_MWh'] > 0:
                profit_hour += row['Load_from_PV_MWh'] * row.get('Prezzo_Acquisto_€/MWh', row['€/MWh'])
            if 'Load_from_Battery_MWh' in row and row['Load_from_Battery_MWh'] > 0:
                profit_hour += row['Load_from_Battery_MWh'] * row.get('Prezzo_Acquisto_€/MWh', row['€/MWh'])
            if 'Load_from_Grid_MWh' in row and row['Load_from_Grid_MWh'] > 0:
                profit_hour -= row['Load_from_Grid_MWh'] * row.get('Prezzo_Acquisto_€/MWh', row['€/MWh'])
            profitto_orario.append(profit_hour)
        colors_profit = ['#06A77D' if p >= 0 else '#E63946' for p in profitto_orario]
        ax5.bar(df_giorno['Ora'], profitto_orario, color=colors_profit,
               alpha=0.8, width=0.8, edgecolor='black', linewidth=1)
        ax5.axhline(y=0, color='black', linewidth=1, linestyle='-')
        ax5.set_ylabel('Profitto/Costo Orario (€)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Ora', fontsize=11)
        ax5.set_title('Bilancio Economico Orario', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax5.set_xlim(-0.5, 23.5)
        ax5.set_xticks(range(0, 24, 2))

        # Subplot 6: STATISTICHE GIORNALIERE
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        total_pv_prod = df_giorno['PV_Production_MWh'].sum()
        total_pv_to_batt = df_giorno['PV_to_Battery_MWh'].sum()
        total_pv_to_grid = df_giorno['PV_to_Grid_MWh'].sum()
        total_pv_to_load = df_giorno['PV_to_Load_MWh'].sum() if 'PV_to_Load_MWh' in df_giorno.columns else 0
        total_grid_to_batt = df_giorno['Energy_from_Grid_MWh'].sum()
        total_charge = total_pv_to_batt + total_grid_to_batt
        pv_charge_percent = (total_pv_to_batt / total_charge * 100) if total_charge > 0 else 0
        total_load = df_giorno['Load_Demand_MWh'].sum() if 'Load_Demand_MWh' in df_giorno.columns else 0
        load_from_pv = df_giorno['Load_from_PV_MWh'].sum() if 'Load_from_PV_MWh' in df_giorno.columns else 0
        load_from_batt = df_giorno['Load_from_Battery_MWh'].sum() if 'Load_from_Battery_MWh' in df_giorno.columns else 0
        load_from_grid = df_giorno['Load_from_Grid_MWh'].sum() if 'Load_from_Grid_MWh' in df_giorno.columns else 0
        profit_day = sum(profitto_orario)
        avg_price = df_giorno['€/MWh'].mean()
        stats_text = f"""
STATISTICHE GIORNALIERE {mesi_nomi[mese-1].upper()} - {data_str}

PRODUZIONE FOTOVOLTAICA:
  • Produzione totale PV:           {total_pv_prod:.3f} MWh
  • PV utilizzato per carico:       {total_pv_to_load:.3f} MWh ({total_pv_to_load/total_pv_prod*100 if total_pv_prod > 0 else 0:.1f}%)
  • PV utilizzato per batteria:     {total_pv_to_batt:.3f} MWh ({total_pv_to_batt/total_pv_prod*100 if total_pv_prod > 0 else 0:.1f}%)
  • PV venduto direttamente:        {total_pv_to_grid:.3f} MWh ({total_pv_to_grid/total_pv_prod*100 if total_pv_prod > 0 else 0:.1f}%)
  
CARICA BATTERIA:
  • Energia da PV (GRATIS):         {total_pv_to_batt:.3f} MWh ({pv_charge_percent:.1f}%)
  • Energia da RETE (ACQUISTO):     {total_grid_to_batt:.3f} MWh ({(100-pv_charge_percent):.1f}%)
  • Totale caricato:                {total_charge:.3f} MWh
  
CARICO UTENTE:
  • Carico totale:                  {total_load:.3f} MWh
  • Servito da PV:                  {load_from_pv:.3f} MWh ({load_from_pv/total_load*100 if total_load > 0 else 0:.1f}%)
  • Servito da Batteria:            {load_from_batt:.3f} MWh ({load_from_batt/total_load*100 if total_load > 0 else 0:.1f}%)
  • Servito da Rete:                {load_from_grid:.3f} MWh ({load_from_grid/total_load*100 if total_load > 0 else 0:.1f}%)
  
ECONOMIA:
  • Profitto giornaliero:           {profit_day:.2f} €
  • Prezzo medio energia:           {avg_price:.2f} €/MWh
  • SOC iniziale → finale:          {df_giorno['SOC'].iloc[0]*100:.1f}% → {df_giorno['SOC'].iloc[-1]*100:.1f}%
        """
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        filename = f'{mese:02d}_{mesi_nomi[mese-1]}_dettaglio_pv_{battery.technology}.png'
        plt.savefig(os.path.join(monthly_pv_folder, filename),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Salvato: {filename}")

    print("\n" + "=" * 80)
    print(f"Grafici mensili PV dettagliati salvati in '{monthly_pv_folder}/'")
    print("=" * 80)


def create_pv_impact_comparison(results_df, battery, pv_system):
    """
    Lorenzo Giannuzzo: GRAFICO IMPATTO PV - CONFRONTO DIRETTO
    """
    if not SAVE_PLOTS or not PV_ENABLED or pv_system is None:
        return
    # Implementazione completa come da codice originale
    print("  ✓ create_pv_impact_comparison: funzione placeholder - implementare con codice originale")


def create_pv_impact_summary(results_df, battery, pv_system):
    """
    Lorenzo Giannuzzo: Grafico sintesi annuale IMPATTO PV
    """
    if not SAVE_PLOTS or not PV_ENABLED or pv_system is None:
        return
    # Implementazione completa come da codice originale
    print("  ✓ create_pv_impact_summary: funzione placeholder - implementare con codice originale")


def create_pv_visualizations(results_df, battery, pv_system):
    """Grafici analisi fotovoltaico con decisione autonoma"""
    if not SAVE_PLOTS or not PV_ENABLED or pv_system is None:
        return
    # Implementazione completa come da codice originale
    print("  ✓ create_pv_visualizations: funzione placeholder - implementare con codice originale")


# ========================================================================================================
# SEZIONE 11: VISUALIZZAZIONI CARICO
# ========================================================================================================
def create_load_analysis_plots(results_df, battery, load_profile):
    """
    Lorenzo Giannuzzo: Grafici analisi carico utente
    """
    if not SAVE_PLOTS or not LOAD_ENABLED or load_profile is None:
        return

    print("\n" + "=" * 80)
    print("GENERAZIONE GRAFICI ANALISI CARICO UTENTE")
    print("=" * 80)

    # (implementazione grafici carico - da sviluppare)


# ========================================================================================================
# FUNZIONE MAIN
# ========================================================================================================
def main(file_name, file_name2, pv_file_name=None, load_file_name=None):
    file_path = os.path.join('data', file_name)
    file_path2 = os.path.join('data', file_name2)

    print("=" * 80)
    print("BESS OPTIMIZATION WITH LOAD MANAGEMENT v2.4.0")
    print("=" * 80)
    print("Caricamento dati prezzi energia...")

    try:
        df = pd.read_excel(file_path)
        if df['€/MWh'].dtype == 'object':
            df['€/MWh'] = df['€/MWh'].astype(str).str.replace(',', '.').astype(float)
        print(f"Dati caricati: {len(df)} righe")
    except Exception as e:
        print(f"Errore caricamento prezzi: {e}")
        return

    try:
        df2 = pd.read_excel(file_path2)
        if df2['€/MWh'].dtype == 'object':
            df2['€/MWh'] = df2['€/MWh'].astype(str).str.replace(',', '.').astype(float)
        print(f"Dati caricati: {len(df)} righe")
    except Exception as e:
        print(f"Errore caricamento prezzi: {e}")
        return

    # Caricamento PV
    pv_df = None
    pv_system = None
    if PV_ENABLED and pv_file_name:
        try:
            pv_file_path = os.path.join('data', pv_file_name)
            pv_df = pd.read_csv(pv_file_path, sep=';')
            pv_system = PhotovoltaicSystem()
            print(f"Sistema PV caricato: {PV_NOMINAL_POWER_KWP:.0f} kWp")
        except Exception as e:
            print(f"Errore caricamento PV: {e}")
            pv_df = None
            pv_system = None

    # Caricamento CARICO
    load_df = None
    load_profile = None
    if LOAD_ENABLED and load_file_name:
        print("Caricamento dati carico elettrico utente...")
        try:
            load_file_path = os.path.join('data', load_file_name)

            # Carica Excel con gestione corretta sheet
            if LOAD_SHEET_NAME:
                # Sheet specifico fornito dall'utente
                load_df_raw = pd.read_excel(load_file_path, sheet_name=LOAD_SHEET_NAME)
            else:
                # Nessuno sheet specificato: prendi il primo sheet disponibile
                xls = pd.ExcelFile(load_file_path)
                first_sheet = xls.sheet_names[0]
                print(f"  Sheet disponibili: {xls.sheet_names}")
                print(f"  Caricamento primo sheet: '{first_sheet}'")
                load_df_raw = pd.read_excel(load_file_path, sheet_name=first_sheet)

            # Debug: mostra colonne disponibili
            print(f"  Colonne trovate: {list(load_df_raw.columns)}")

            # Trova colonna con valori del carico
            load_column = None
            for col_name in ['value', 'Value', 'VALUE', 'load', 'Load', 'LOAD', 'Power', 'power', 'POWER']:
                if col_name in load_df_raw.columns:
                    load_column = col_name
                    break

            if load_column is None:
                # Prova a prendere la prima colonna numerica
                numeric_cols = load_df_raw.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    load_column = numeric_cols[0]
                    print(f"  ⚠ Colonna 'value' non trovata, uso prima colonna numerica: '{load_column}'")
                else:
                    raise ValueError("Nessuna colonna numerica trovata nel file carico")
            else:
                print(f"  ✓ Colonna carico trovata: '{load_column}'")

            # Crea DataFrame con colonna 'value' standard
            load_df = pd.DataFrame()
            load_df['value'] = load_df_raw[load_column].copy()

            print(f"Dati carico caricati: {len(load_df)} righe")
            print(f"Range carico: {load_df['value'].min():.2f} - {load_df['value'].max():.2f} kW")
            print(f"Carico medio: {load_df['value'].mean():.2f} kW ({load_df['value'].mean()/1000:.3f} MW)")
            print(f"Energia totale richiesta: {load_df['value'].sum() / 1000:.2f} MWh")

            load_profile = LoadProfile()
            print("Sistema carico inizializzato")
            print(f"STRATEGIA: Priorità PV > Batteria > Rete")
            print(f"Mark-up prezzo: {PRICE_MARKUP_PERCENT}%")
            print()
        except Exception as e:
            print(f"Errore caricamento carico: {e}")
            print("Procedo senza carico utente")
            load_df = None
            load_profile = None
            print()


    battery = Battery()
    optimizer = PSOOptimizer(n_particles=50, n_iterations=100)
    simulator = RollingHorizonSimulator(battery, optimizer, pv_system=pv_system, load_profile=load_profile)

    start_time = datetime.now()
    results_df, trading_profit = simulator.simulate(df, df2, pv_df, load_df)
    end_time = datetime.now()

    macse_revenue, macse_base, macse_penalty, macse_bonus = calculate_macse_revenue(battery)
    total_system_profit = trading_profit + macse_revenue

    print("=" * 80)
    print("RISULTATI SIMULAZIONE")
    print("=" * 80)

    if LOAD_ENABLED and load_profile:
        load_stats = load_profile.get_statistics()
        print(f"\nCARICO UTENTE:")
        print(f"  Energia totale richiesta: {load_stats['total_energy_required_mwh']:.2f} MWh")
        print(f"  Fornita da PV: {load_stats['energy_from_pv_mwh']:.2f} MWh ({load_stats['pv_coverage_percent']:.1f}%)")
        print(f"  Fornita da Batteria: {load_stats['energy_from_battery_mwh']:.2f} MWh ({load_stats['battery_coverage_percent']:.1f}%)")
        print(f"  Fornita da Rete: {load_stats['energy_from_grid_mwh']:.2f} MWh ({load_stats['grid_dependency_percent']:.1f}%)")

    if PV_ENABLED and pv_system:
        pv_stats = pv_system.get_statistics()
        print(f"\nFOTOVOLTAICO:")
        print(f"  Produzione totale: {pv_stats['total_production_mwh']:.2f} MWh")
        print(f"  → Carico diretto: {pv_stats['energy_to_load_mwh']:.2f} MWh ({pv_stats['load_service_percent']:.1f}%)")
        print(f"  → Batteria: {pv_stats['energy_to_battery_mwh']:.2f} MWh ({pv_stats['battery_utilization_percent']:.1f}%)")
        print(f"  → Rete: {pv_stats['energy_to_grid_mwh']:.2f} MWh ({pv_stats['grid_sale_percent']:.1f}%)")

    print(f"\nTRADING:")
    print(f"  Profitto totale sistema: {total_system_profit:,.2f} Euro")
    print(f"  Tempo esecuzione: {(end_time - start_time).total_seconds():.1f} secondi")
    print("=" * 80)

    # Salva risultati
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    output_file = os.path.join(results_folder, f'risultati_{battery.technology.lower().replace("-", "_")}_with_load.xlsx')

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Risultati Orari', index=False)

    print(f"\nRisultati Excel salvati in: {output_file}")

    # Export JSON
    export_results_to_json(
        results_df, battery, pv_system, load_profile, trading_profit, macse_revenue, macse_base,
        macse_penalty, macse_bonus, 600000, (end_time - start_time).total_seconds()
    )

    # Visualizzazioni PV
    if PV_ENABLED and pv_system:
        create_pv_impact_comparison(results_df, battery, pv_system)
        create_pv_impact_summary(results_df, battery, pv_system)
        create_pv_visualizations(results_df, battery, pv_system)
        create_detailed_monthly_pv_plots(results_df, battery, pv_system)

    # Visualizzazioni Carico
    if LOAD_ENABLED and load_profile:
        create_load_analysis_plots(results_df, battery, load_profile)

    print("\n" + "=" * 80)
    print("SIMULAZIONE COMPLETATA CON SUCCESSO")
    print("=" * 80)


if __name__ == "__main__":
    pv_file = pv_production_file if PV_ENABLED else None
    load_file_input = load_file if LOAD_ENABLED else None
    main(energy_selling_price_name, energy_buying_price_name, pv_file, load_file_input)