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
Version: 2.5.0-CORRECTED
Date: November 2025
------------------------------------------------------------------------------------------------------------------------
CORREZIONI VERSIONE 2.5.0:
- FIX: Rimossi virtual revenues sia da PSO che da simulatore (contabilità reale)
- FIX: Aggiunto costo degrado batteria per servizio carico
- FIX: Aggiunto costo degrado batteria per trading
- FIX: Corretto calcolo profitto orario nei grafici
- FIX: Produzione PV ora letta correttamente (era trattata come irradianza)
- FIX: Prezzo acquisto calcolato da vendita con mark-up (non più da file separato)
- FIX: C-rate standardizzato su nominal_capacity
- FIX: Coerenza completa PSO-Simulatore-Report
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
energy_buying_price_name = '20240101_20241231_PUN.xlsx'  # Non più usato, calcolato con mark-up
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
    CORRETTO: Produzione PV letta direttamente in kW, non più moltiplicata per taglia impianto
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

    def load_pv_production(self, pv_value_kw, dt=1.0):
        """
        NUOVO METODO CORRETTO: Carica produzione PV già calcolata in kW
        Utilizzare questo invece di get_energy() quando il file contiene già la produzione
        """
        energy_mwh = (pv_value_kw / 1000.0) * dt
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
    CORRETTO: C-rate sempre su nominal_capacity
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
        """CORRETTO: C-rate sempre su nominal_capacity"""
        return self.nominal_capacity * self.max_c_rate


# ========================================================================================================
# SEZIONE 6: OTTIMIZZATORE PSO - CORRETTO v2.5.0
# ========================================================================================================
class PSOOptimizer:
    """
    CORRETTO v2.5.0: PSO SENZA virtual revenues - contabilità reale
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
        n_hours = len(prices_sell)
        positions = np.zeros((self.n_particles, n_hours))
        price_low = np.percentile(prices_sell, 25)
        price_high = np.percentile(prices_sell, 75)

        for i in range(self.n_particles):
            if i < self.n_particles // 3:
                for h in range(n_hours):
                    net_energy = pv_production[h] - load_demand[h]
                    if prices_sell[h] < price_low and net_energy < 0:
                        positions[i, h] = np.random.uniform(0.5 * max_power, max_power)
                    elif prices_sell[h] > price_high:
                        positions[i, h] = np.random.uniform(-max_power, -0.5 * max_power)
                    else:
                        positions[i, h] = np.random.uniform(-0.3 * max_power, 0.3 * max_power)
            elif i < 2 * self.n_particles // 3:
                for h in range(n_hours):
                    if prices_sell[h] < price_low:
                        positions[i, h] = np.random.uniform(0, 0.7 * max_power)
                    elif prices_sell[h] > price_high:
                        positions[i, h] = np.random.uniform(-0.7 * max_power, 0)
                    else:
                        positions[i, h] = np.random.uniform(-0.2 * max_power, 0.2 * max_power)
            else:
                positions[i] = np.random.uniform(-max_power, max_power, n_hours)
        return positions

    def _evaluate(self, battery, actions, prices_sell, prices_buy, pv_production, load_demand):
        """CORRETTO: NO virtual revenues - solo costi/ricavi reali"""
        bat_sim = battery.copy()
        profit = 0.0

        for hour, (power, price_sell, price_buy, pv_energy, load_energy) in enumerate(
                zip(actions, prices_sell, prices_buy, pv_production, load_demand)
        ):
            pv_available = pv_energy if PV_ENABLED else 0.0
            load_required = load_energy if LOAD_ENABLED else 0.0

            # FASE 1: SODDISFACIMENTO CARICO
            if load_required > 0.001:
                # 1. PV al carico
                if pv_available > 0:
                    load_from_pv = min(load_required, pv_available)
                    pv_available -= load_from_pv
                    load_required -= load_from_pv
                    # NESSUN ricavo virtuale

                # 2. Batteria al carico
                if load_required > 0.001 and bat_sim.soc > bat_sim.soc_min:
                    max_discharge_load = ((bat_sim.soc - bat_sim.soc_min) * bat_sim.trading_capacity * bat_sim.discharge_efficiency)
                    load_from_battery = min(load_required, max_discharge_load)
                    if load_from_battery > 0.001:
                        actual_discharge = load_from_battery / bat_sim.discharge_efficiency
                        new_soc = bat_sim.soc - (actual_discharge / bat_sim.capacity)
                        bat_sim.soc = max(new_soc, bat_sim.soc_min)
                        bat_sim.throughput_kwh += actual_discharge * 1000
                        load_required -= load_from_battery
                        # Solo degrado
                        degradation_cost = (load_from_battery * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles))
                        profit -= degradation_cost

                # 3. Rete al carico
                if load_required > 0.001:
                    load_from_grid = load_required
                    profit -= load_from_grid * price_buy  # COSTO

            # FASE 2: TRADING BATTERIA
            if power > 0.01:  # CARICA
                if bat_sim.soc >= bat_sim.soc_max:
                    if pv_available > 0:
                        profit += pv_available * price_sell
                    continue

                max_energy_storable = (bat_sim.soc_max - bat_sim.soc) * bat_sim.trading_capacity
                max_power_available = max_energy_storable / (1.0 * bat_sim.charge_efficiency)
                actual_power = min(power, max_power_available)

                if actual_power > 0.01:
                    energy_needed = actual_power * 1.0

                    if pv_available > 0:
                        energy_from_pv = min(energy_needed, pv_available)
                        bat_sim.charge(energy_from_pv / 1.0, dt=1.0, source='pv')
                        degradation_cost_pv = (energy_from_pv * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles))
                        profit -= degradation_cost_pv
                        energy_needed -= energy_from_pv
                        pv_available -= energy_from_pv
                        if pv_available > 0:
                            profit += pv_available * price_sell

                    if energy_needed > 0.01:
                        energy_from_grid = bat_sim.charge(energy_needed / 1.0, dt=1.0, source='grid')
                        profit -= energy_from_grid * price_buy  # COSTO
                        degradation_cost_grid = (energy_from_grid * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles))
                        profit -= degradation_cost_grid
                else:
                    if pv_available > 0:
                        profit += pv_available * price_sell

            elif power < -0.01:  # SCARICA
                if bat_sim.soc <= bat_sim.soc_min:
                    if pv_available > 0:
                        profit += pv_available * price_sell
                    continue

                max_energy_available = ((bat_sim.soc - bat_sim.soc_min) * bat_sim.trading_capacity)
                max_power_available = max_energy_available * bat_sim.discharge_efficiency / 1.0
                actual_power = min(-power, max_power_available)

                if actual_power > 0.01:
                    energy_to_grid_battery = bat_sim.discharge(actual_power, dt=1.0)
                    total_energy_sold = energy_to_grid_battery + pv_available
                    profit += total_energy_sold * price_sell  # RICAVO
                    degradation_cost = (energy_to_grid_battery * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles))
                    profit -= degradation_cost
                else:
                    if pv_available > 0:
                        profit += pv_available * price_sell

            else:  # IDLE
                if pv_available > 0:
                    profit += pv_available * price_sell

        return profit


# ========================================================================================================
# SEZIONE 7: ROLLING HORIZON SIMULATOR - CORRETTO v2.5.0
# ========================================================================================================
class RollingHorizonSimulator:
    """CORRETTO v2.5.0: Coerenza completa con PSO"""
    def __init__(self, battery, optimizer, pv_system=None, load_profile=None, horizon_hours=24, step_hours=1):
        self.battery = battery
        self.optimizer = optimizer
        self.pv_system = pv_system
        self.load_profile = load_profile
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours

    def simulate(self, prices_df, pv_df=None, load_df=None):
        prices_sell = prices_df['€/MWh'].values
        # CORRETTO: Calcola prezzo acquisto con mark-up
        prices_buy = prices_sell * (1 + PRICE_MARKUP_PERCENT / 100.0)
        n_hours = len(prices_sell)

        # CORRETTO: Prepara PV (kW -> MWh)
        if PV_ENABLED and pv_df is not None and self.pv_system is not None:
            pv_production = pv_df['P'].values / 1000.0
            self.pv_system.total_production_mwh = np.sum(pv_production)
            if len(pv_production) < n_hours:
                pv_production = np.pad(pv_production, (0, n_hours - len(pv_production)), 'constant')
            elif len(pv_production) > n_hours:
                pv_production = pv_production[:n_hours]
        else:
            pv_production = np.zeros(n_hours)

        if LOAD_ENABLED and load_df is not None:
            load_demand = load_df['value'].values / 1000.0
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
        print("SIMULAZIONE BESS v2.5.0 CORRECTED - CONTABILITÀ REALE")
        print("=" * 80)
        print(f"Tecnologia: {self.battery.technology}")
        print(f"Capacità: {self.battery.nominal_capacity} MWh")
        if PV_ENABLED and self.pv_system:
            print(f"PV: {self.pv_system.nominal_power_kwp:.0f} kWp")
        if LOAD_ENABLED:
            print(f"Carico: {np.sum(load_demand):.2f} MWh totali")
        print(f"Prezzi: vendita {np.mean(prices_sell):.2f}, acquisto {np.mean(prices_buy):.2f} €/MWh")
        print("=" * 80)

        current_hour = 0
        last_progress = 0
        degradation_update_interval = 24

        while current_hour < n_hours:
            progress = int((current_hour / n_hours) * 100)
            if progress >= last_progress + 20:
                print(f"Progresso: {progress}% - SOH: {self.battery.get_soh():.2f}%")
                last_progress = progress

            if current_hour % degradation_update_interval == 0 and current_hour > 0:
                self.battery.update_degradation()

            end_hour = min(current_hour + self.horizon_hours, n_hours)
            optimal_actions = self.optimizer.optimize(
                self.battery,
                prices_sell[current_hour:end_hour],
                prices_buy[current_hour:end_hour],
                pv_production[current_hour:end_hour],
                load_demand[current_hour:end_hour],
                self.horizon_hours
            )

            action = optimal_actions[0]
            price_sell = prices_sell[current_hour]
            price_buy = prices_buy[current_hour]
            pv_energy_available = pv_production[current_hour]
            load_required = load_demand[current_hour]

            pv_to_battery_this_hour = 0.0
            pv_to_grid_this_hour = 0.0
            pv_to_load_this_hour = 0.0
            load_from_pv_this_hour = 0.0
            load_from_battery_this_hour = 0.0
            load_from_grid_this_hour = 0.0
            grid_to_battery_this_hour = 0.0

            # ESECUZIONE: CARICO
            if load_required > 0.001:
                if pv_energy_available > 0:
                    load_from_pv_this_hour = min(load_required, pv_energy_available)
                    pv_to_load_this_hour = load_from_pv_this_hour
                    pv_energy_available -= load_from_pv_this_hour
                    load_required -= load_from_pv_this_hour
                    # CORRETTO: NO ricavo virtuale

                if load_required > 0.001 and self.battery.soc > self.battery.soc_min:
                    max_power_c_rate = self.battery.nominal_capacity * self.battery.max_c_rate
                    max_power_physical = min(self.battery.max_power, max_power_c_rate)
                    max_discharge_from_soc = ((self.battery.soc - self.battery.soc_min) * self.battery.trading_capacity * self.battery.discharge_efficiency)
                    max_energy_for_load = min(max_power_physical * 1.0, max_discharge_from_soc)
                    load_from_battery_this_hour = min(load_required, max_energy_for_load)

                    if load_from_battery_this_hour > 0.001:
                        actual_discharge = load_from_battery_this_hour / self.battery.discharge_efficiency
                        new_soc = self.battery.soc - (actual_discharge / self.battery.capacity)
                        self.battery.soc = max(new_soc, self.battery.soc_min)
                        self.battery.throughput_kwh += actual_discharge * 1000
                        load_required -= load_from_battery_this_hour
                        # CORRETTO: Solo degrado
                        degradation_cost = (load_from_battery_this_hour * DEGRADATION_COST_PER_MWH / (2 * self.battery.eol_cycles))
                        cumulative_profit -= degradation_cost

                if load_required > 0.001:
                    load_from_grid_this_hour = load_required
                    profit_grid = -load_from_grid_this_hour * price_buy
                    cumulative_profit += profit_grid

            # ESECUZIONE: TRADING
            power_used_for_load_mw = load_from_battery_this_hour / 1.0
            max_power_c_rate = self.battery.nominal_capacity * self.battery.max_c_rate
            max_power_physical = self.battery.max_power - power_used_for_load_mw
            max_power_available_trading = min(max_power_physical, max_power_c_rate)
            actual_action = 0.0

            if action > 0.01:  # CARICA
                if self.battery.soc < self.battery.soc_max:
                    max_energy_storable = (self.battery.soc_max - self.battery.soc) * self.battery.trading_capacity
                    max_power_available = max_energy_storable / (1.0 * self.battery.charge_efficiency)
                    actual_power = min(action, max_power_available, max_power_available_trading)
                    actual_action = actual_power

                    if actual_power > 0.01:
                        energy_needed = actual_power * 1.0

                        if pv_energy_available > 0:
                            energy_from_pv = min(energy_needed, pv_energy_available)
                            self.battery.charge(energy_from_pv / 1.0, dt=1.0, source='pv')
                            pv_to_battery_this_hour = energy_from_pv
                            energy_needed -= energy_from_pv
                            pv_energy_available -= energy_from_pv
                            # CORRETTO: Degrado
                            degradation_cost_pv = (energy_from_pv * DEGRADATION_COST_PER_MWH / (2 * self.battery.eol_cycles))
                            cumulative_profit -= degradation_cost_pv

                            if pv_energy_available > 0:
                                profit_pv = pv_energy_available * price_sell
                                cumulative_profit += profit_pv
                                pv_to_grid_this_hour = pv_energy_available
                                pv_energy_available = 0

                        if energy_needed > 0.01:
                            energy_from_grid = self.battery.charge(energy_needed / 1.0, dt=1.0, source='grid')
                            grid_to_battery_this_hour = energy_from_grid
                            profit = -energy_from_grid * price_buy
                            cumulative_profit += profit
                            # CORRETTO: Degrado
                            degradation_cost_grid = (energy_from_grid * DEGRADATION_COST_PER_MWH / (2 * self.battery.eol_cycles))
                            cumulative_profit -= degradation_cost_grid
                    else:
                        if pv_energy_available > 0:
                            profit_pv = pv_energy_available * price_sell
                            cumulative_profit += profit_pv
                            pv_to_grid_this_hour = pv_energy_available
                else:
                    actual_action = 0.0
                    if pv_energy_available > 0:
                        profit_pv = pv_energy_available * price_sell
                        cumulative_profit += profit_pv
                        pv_to_grid_this_hour = pv_energy_available

            elif action < -0.01:  # SCARICA
                if self.battery.soc > self.battery.soc_min:
                    max_energy_available = (self.battery.soc - self.battery.soc_min) * self.battery.trading_capacity
                    max_power_by_soc = max_energy_available * self.battery.discharge_efficiency / 1.0
                    actual_power = min(-action, max_power_by_soc, max_power_available_trading)
                    actual_action = -actual_power

                    if actual_power > 0.01:
                        energy_to_grid = self.battery.discharge(actual_power, dt=1.0)
                        total_energy_sold = energy_to_grid + pv_energy_available
                        profit = total_energy_sold * price_sell
                        cumulative_profit += profit
                        # CORRETTO: Degrado
                        degradation_cost = (energy_to_grid * DEGRADATION_COST_PER_MWH / (2 * self.battery.eol_cycles))
                        cumulative_profit -= degradation_cost

                        if pv_energy_available > 0:
                            pv_to_grid_this_hour = pv_energy_available
                    else:
                        if pv_energy_available > 0:
                            profit_pv = pv_energy_available * price_sell
                            cumulative_profit += profit_pv
                            pv_to_grid_this_hour = pv_energy_available
                else:
                    actual_action = 0.0
                    if pv_energy_available > 0:
                        profit_pv = pv_energy_available * price_sell
                        cumulative_profit += profit_pv
                        pv_to_grid_this_hour = pv_energy_available

            else:  # IDLE
                actual_action = 0.0
                if pv_energy_available > 0:
                    profit_pv = pv_energy_available * price_sell
                    cumulative_profit += profit_pv
                    pv_to_grid_this_hour = pv_energy_available

            if self.pv_system and (pv_to_battery_this_hour > 0 or pv_to_grid_this_hour > 0 or pv_to_load_this_hour > 0):
                self.pv_system.allocate_energy(pv_to_battery_this_hour, pv_to_grid_this_hour, pv_to_load_this_hour)

            if self.load_profile:
                self.load_profile.register_supply(load_from_pv_this_hour, load_from_battery_this_hour, load_from_grid_this_hour)

            macse_available = self.battery.check_macse_availability() if MACSE_ENABLED else False
            if MACSE_ENABLED:
                self.battery.update_macse_availability(macse_available)

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
        print("Simulazione completata!")

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
# SEZIONE 8: MACSE E JSON EXPORT
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
    actions = results_df['Azione_MW'].values
    prices_sell = results_df['€/MWh'].values
    prices_buy = results_df['Prezzo_Acquisto_€/MWh'].values

    pv_stats = pv_system.get_statistics() if pv_system else {
        'total_production_mwh': 0, 'energy_to_battery_mwh': 0, 'energy_to_grid_mwh': 0,
        'energy_to_load_mwh': 0, 'curtailed_energy_mwh': 0, 'battery_utilization_percent': 0,
        'grid_sale_percent': 0, 'load_service_percent': 0, 'curtailment_percent': 0
    }

    load_stats = load_profile.get_statistics() if load_profile else {
        'total_energy_required_mwh': 0, 'energy_from_pv_mwh': 0, 'energy_from_battery_mwh': 0,
        'energy_from_grid_mwh': 0, 'pv_coverage_percent': 0, 'battery_coverage_percent': 0,
        'grid_dependency_percent': 0
    }

    charge_hours = np.sum(actions > 0.01)
    discharge_hours = np.sum(actions < -0.01)
    idle_hours = len(actions) - charge_hours - discharge_hours
    total_energy_charged = np.sum(actions[actions > 0] * 1.0)
    total_energy_discharged = np.sum(np.abs(actions[actions < 0]) * 1.0)

    total_revenue = trading_profit + macse_revenue
    annual_profit = total_revenue
    roi_percent = (annual_profit / battery_investment) * 100 if battery_investment > 0 else 0
    payback_years = battery_investment / annual_profit if annual_profit > 0 else float('inf')

    results_json = {
        "simulation_info": {
            "version": "2.5.0-CORRECTED",
            "accounting_method": "REAL_ONLY (no virtual revenues)",
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

    json_file = os.path.join('results', f'simulation_results_{battery.technology.lower().replace("-", "_")}_v250_corrected.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON salvato: {json_file}")


# ========================================================================================================
# SEZIONE 9: GRAFICI (placeholder - implementa come vuoi)
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
        fig.suptitle(f'Analisi Dettagliata PV - {mesi_nomi[mese - 1]} {data_str}\n{battery.technology}',
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
                ax4.axvspan(row['Ora'] - 0.4, row['Ora'] + 0.4,
                            color='#06A77D', alpha=0.15, zorder=0)
        ax4.axhline(y=battery.soc_min * 100, color='red', linewidth=1.5,
                    linestyle='--', alpha=0.7, label=f'SOC min ({battery.soc_min * 100:.0f}%)')
        ax4.axhline(y=battery.soc_max * 100, color='green', linewidth=1.5,
                    linestyle='--', alpha=0.7, label=f'SOC max ({battery.soc_max * 100:.0f}%)')
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
STATISTICHE GIORNALIERE {mesi_nomi[mese - 1].upper()} - {data_str}

PRODUZIONE FOTOVOLTAICA:
  • Produzione totale PV:           {total_pv_prod:.3f} MWh
  • PV utilizzato per carico:       {total_pv_to_load:.3f} MWh ({total_pv_to_load / total_pv_prod * 100 if total_pv_prod > 0 else 0:.1f}%)
  • PV utilizzato per batteria:     {total_pv_to_batt:.3f} MWh ({total_pv_to_batt / total_pv_prod * 100 if total_pv_prod > 0 else 0:.1f}%)
  • PV venduto direttamente:        {total_pv_to_grid:.3f} MWh ({total_pv_to_grid / total_pv_prod * 100 if total_pv_prod > 0 else 0:.1f}%)

CARICA BATTERIA:
  • Energia da PV (GRATIS):         {total_pv_to_batt:.3f} MWh ({pv_charge_percent:.1f}%)
  • Energia da RETE (ACQUISTO):     {total_grid_to_batt:.3f} MWh ({(100 - pv_charge_percent):.1f}%)
  • Totale caricato:                {total_charge:.3f} MWh

CARICO UTENTE:
  • Carico totale:                  {total_load:.3f} MWh
  • Servito da PV:                  {load_from_pv:.3f} MWh ({load_from_pv / total_load * 100 if total_load > 0 else 0:.1f}%)
  • Servito da Batteria:            {load_from_batt:.3f} MWh ({load_from_batt / total_load * 100 if total_load > 0 else 0:.1f}%)
  • Servito da Rete:                {load_from_grid:.3f} MWh ({load_from_grid / total_load * 100 if total_load > 0 else 0:.1f}%)

ECONOMIA:
  • Profitto giornaliero:           {profit_day:.2f} €
  • Prezzo medio energia:           {avg_price:.2f} €/MWh
  • SOC iniziale → finale:          {df_giorno['SOC'].iloc[0] * 100:.1f}% → {df_giorno['SOC'].iloc[-1] * 100:.1f}%
        """
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        filename = f'{mese:02d}_{mesi_nomi[mese - 1]}_dettaglio_pv_{battery.technology}.png'
        plt.savefig(os.path.join(monthly_pv_folder, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Salvato: {filename}")

    print("\n" + "=" * 80)
    print(f"Grafici mensili PV dettagliati salvati in '{monthly_pv_folder}/'")
    print("=" * 80)


def create_pv_impact_comparison(results_df, battery, pv_system):
    """
    Lorenzo Giannuzzo: Confronto impatto PV su profitto e utilizzo batteria
    """
    if not SAVE_PLOTS or not PV_ENABLED or pv_system is None:
        return

    viz_folder = 'visualization'
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)

    print("\n" + "=" * 80)
    print("GENERAZIONE GRAFICO CONFRONTO IMPATTO PV")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Impatto Sistema Fotovoltaico - {battery.technology}', fontsize=16, fontweight='bold')

    pv_stats = pv_system.get_statistics()

    # Subplot 1: Pie chart allocazione energia PV
    ax1 = axes[0, 0]
    labels = ['Batteria', 'Rete', 'Carico', 'Curtailment']
    sizes = [
        pv_stats['energy_to_battery_mwh'],
        pv_stats['energy_to_grid_mwh'],
        pv_stats['energy_to_load_mwh'],
        pv_stats['curtailed_energy_mwh']
    ]
    colors = ['#06A77D', '#457B9D', '#F77F00', '#E63946']
    explode = (0.05, 0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title('Allocazione Energia PV', fontsize=12, fontweight='bold')

    # Subplot 2: Grafico a barre energia PV per destinazione
    ax2 = axes[0, 1]
    destinations = ['Batteria', 'Rete\nDiretta', 'Carico\nDiretto', 'Curtailment']
    values = [
        pv_stats['energy_to_battery_mwh'],
        pv_stats['energy_to_grid_mwh'],
        pv_stats['energy_to_load_mwh'],
        pv_stats['curtailed_energy_mwh']
    ]
    bars = ax2.bar(destinations, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Energia (MWh)', fontsize=11, fontweight='bold')
    ax2.set_title('Energia PV per Destinazione', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.1f} MWh\n({val / pv_stats["total_production_mwh"] * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 3: Timeline profitto cumulativo (stima contributo PV)
    ax3 = axes[1, 0]
    if 'Profitto_Euro' in results_df.columns and len(results_df) > 0:
        hours = range(len(results_df))
        ax3.plot(hours, results_df['Profitto_Euro'], color='#06A77D', linewidth=2, label='Profitto Cumulativo')
        ax3.fill_between(hours, 0, results_df['Profitto_Euro'], color='#06A77D', alpha=0.2)
        ax3.set_xlabel('Ora', fontsize=11)
        ax3.set_ylabel('Profitto Cumulativo (€)', fontsize=11, fontweight='bold')
        ax3.set_title('Evoluzione Profitto nel Tempo', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)

    # Subplot 4: Statistiche testuali PV
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
STATISTICHE SISTEMA FOTOVOLTAICO

Produzione Totale:
  • Energia prodotta: {pv_stats['total_production_mwh']:.2f} MWh

Utilizzo Energia PV:
  • Per batteria: {pv_stats['energy_to_battery_mwh']:.2f} MWh ({pv_stats['battery_utilization_percent']:.1f}%)
  • Vendita diretta rete: {pv_stats['energy_to_grid_mwh']:.2f} MWh ({pv_stats['grid_sale_percent']:.1f}%)
  • Per carico utente: {pv_stats['energy_to_load_mwh']:.2f} MWh ({pv_stats['load_service_percent']:.1f}%)
  • Energia curtailed: {pv_stats['curtailed_energy_mwh']:.2f} MWh ({pv_stats['curtailment_percent']:.1f}%)

Efficienza Sistema:
  • Inverter: {pv_system.inverter_efficiency * 100:.1f}%
  • Perdite sistema: {pv_system.system_losses * 100:.1f}%
  • Efficienza totale: {pv_system.total_efficiency * 100:.1f}%

Impatto su Batteria:
  • Energia da PV: {battery.energy_from_pv_mwh:.2f} MWh
  • Energia da rete: {battery.energy_from_grid_mwh:.2f} MWh
  • % carica da PV: {battery.energy_from_pv_mwh / (battery.energy_from_pv_mwh + battery.energy_from_grid_mwh) * 100 if (battery.energy_from_pv_mwh + battery.energy_from_grid_mwh) > 0 else 0:.1f}%
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    filename = os.path.join(viz_folder, f'pv_impact_comparison_{battery.technology}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvato: {filename}")
    print("=" * 80)


def create_pv_impact_summary(results_df, battery, pv_system):
    """
    Lorenzo Giannuzzo: Sommario impatto PV su operazioni sistema
    """
    if not SAVE_PLOTS or not PV_ENABLED or pv_system is None:
        return

    viz_folder = 'visualization'
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)

    print("\n" + "=" * 80)
    print("GENERAZIONE GRAFICO SOMMARIO PV")
    print("=" * 80)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Sommario Impatto Fotovoltaico - {battery.technology}', fontsize=16, fontweight='bold')

    pv_stats = pv_system.get_statistics()

    # Grafico 1: Produzione PV oraria (sample 7 giorni)
    ax1 = fig.add_subplot(gs[0, :])
    if 'PV_Production_MWh' in results_df.columns:
        sample_hours = min(168, len(results_df))  # 7 giorni max
        hours = range(sample_hours)
        ax1.plot(hours, results_df['PV_Production_MWh'].iloc[:sample_hours],
                 color='#F4A300', linewidth=2, label='Produzione PV')
        ax1.fill_between(hours, 0, results_df['PV_Production_MWh'].iloc[:sample_hours],
                         color='#F4A300', alpha=0.2)
        ax1.set_xlabel('Ora', fontsize=11)
        ax1.set_ylabel('Produzione (MWh)', fontsize=11, fontweight='bold')
        ax1.set_title('Profilo Produzione PV (Sample 7 giorni)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

    # Grafico 2: Confronto fonti carica batteria
    ax2 = fig.add_subplot(gs[1, 0])
    sources = ['PV', 'Rete']
    energy_values = [battery.energy_from_pv_mwh, battery.energy_from_grid_mwh]
    colors_sources = ['#06A77D', '#E63946']
    bars = ax2.bar(sources, energy_values, color=colors_sources, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Energia (MWh)', fontsize=11, fontweight='bold')
    ax2.set_title('Fonti di Carica Batteria', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, energy_values):
        height = bar.get_height()
        total = sum(energy_values)
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.1f} MWh\n({val / total * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Grafico 3: Utilizzo PV nel tempo (aggregato giornaliero)
    ax3 = fig.add_subplot(gs[1, 1:])
    if 'PV_to_Battery_MWh' in results_df.columns and 'PV_to_Grid_MWh' in results_df.columns:
        # Aggrega per giorno
        results_df_copy = results_df.copy()
        results_df_copy['Giorno'] = results_df_copy.index // 24
        daily_data = results_df_copy.groupby('Giorno').agg({
            'PV_to_Battery_MWh': 'sum',
            'PV_to_Grid_MWh': 'sum',
            'PV_to_Load_MWh': 'sum' if 'PV_to_Load_MWh' in results_df.columns else lambda x: 0
        })

        days = range(len(daily_data))
        width = 0.8

        ax3.bar(days, daily_data['PV_to_Battery_MWh'], width=width, label='PV→Batteria',
                color='#06A77D', alpha=0.8, edgecolor='black')
        ax3.bar(days, daily_data['PV_to_Grid_MWh'], width=width,
                bottom=daily_data['PV_to_Battery_MWh'], label='PV→Rete',
                color='#457B9D', alpha=0.8, edgecolor='black')
        if 'PV_to_Load_MWh' in results_df.columns:
            ax3.bar(days, daily_data['PV_to_Load_MWh'], width=width,
                    bottom=daily_data['PV_to_Battery_MWh'] + daily_data['PV_to_Grid_MWh'],
                    label='PV→Carico', color='#F77F00', alpha=0.8, edgecolor='black')

        ax3.set_xlabel('Giorno', fontsize=11)
        ax3.set_ylabel('Energia (MWh)', fontsize=11, fontweight='bold')
        ax3.set_title('Allocazione PV Giornaliera', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

    # Grafico 4: Impatto economico PV (stima ricavi)
    ax4 = fig.add_subplot(gs[2, 0])
    if '€/MWh' in results_df.columns:
        avg_price = results_df['€/MWh'].mean()
        revenue_components = {
            'Vendita\nDiretta': pv_stats['energy_to_grid_mwh'] * avg_price,
            'Risparmio\nBatteria': pv_stats['energy_to_battery_mwh'] * avg_price * 0.5,  # Stima
            'Risparmio\nCarico': pv_stats['energy_to_load_mwh'] * avg_price  # Risparmio acquisto
        }

        labels = list(revenue_components.keys())
        values = list(revenue_components.values())
        colors_rev = ['#457B9D', '#06A77D', '#F77F00']

        bars = ax4.bar(labels, values, color=colors_rev, alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Valore Stimato (€)', fontsize=11, fontweight='bold')
        ax4.set_title('Contributo Economico PV (Stima)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.0f} €', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Grafico 5: Percentuali utilizzo PV
    ax5 = fig.add_subplot(gs[2, 1])
    categories = ['Batteria', 'Rete', 'Carico']
    percentages = [
        pv_stats['battery_utilization_percent'],
        pv_stats['grid_sale_percent'],
        pv_stats['load_service_percent']
    ]
    colors_perc = ['#06A77D', '#457B9D', '#F77F00']

    bars = ax5.barh(categories, percentages, color=colors_perc, alpha=0.8, edgecolor='black', linewidth=2)
    ax5.set_xlabel('Percentuale (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Distribuzione Utilizzo PV', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, percentages):
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height() / 2.,
                 f' {val:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

    # Grafico 6: Indici prestazione PV
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    total_pv = pv_stats['total_production_mwh']
    total_used = pv_stats['energy_to_battery_mwh'] + pv_stats['energy_to_grid_mwh'] + pv_stats['energy_to_load_mwh']
    utilization_rate = (total_used / total_pv * 100) if total_pv > 0 else 0

    indices_text = f"""
INDICI PRESTAZIONE PV

Produzione:
  • Totale: {total_pv:.2f} MWh
  • Utilizzata: {total_used:.2f} MWh
  • Tasso utilizzo: {utilization_rate:.1f}%

Autoconsumo:
  • Diretto carico: {pv_stats['energy_to_load_mwh']:.2f} MWh
  • Via batteria: {pv_stats['energy_to_battery_mwh']:.2f} MWh
  • Totale: {pv_stats['energy_to_load_mwh'] + pv_stats['energy_to_battery_mwh']:.2f} MWh
  • % autoconsumo: {(pv_stats['energy_to_load_mwh'] + pv_stats['energy_to_battery_mwh']) / total_pv * 100 if total_pv > 0 else 0:.1f}%

Export:
  • Energia venduta: {pv_stats['energy_to_grid_mwh']:.2f} MWh
  • % export: {pv_stats['grid_sale_percent']:.1f}%
    """

    ax6.text(0.05, 0.95, indices_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.tight_layout()
    filename = os.path.join(viz_folder, f'pv_impact_summary_{battery.technology}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvato: {filename}")
    print("=" * 80)


def create_pv_visualizations(results_df, battery, pv_system):
    """
    Lorenzo Giannuzzo: Visualizzazioni analisi PV complete
    """
    if not SAVE_PLOTS or not PV_ENABLED or pv_system is None:
        return

    viz_folder = 'visualization'
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)

    print("\n" + "=" * 80)
    print("GENERAZIONE VISUALIZZAZIONI ANALISI PV")
    print("=" * 80)

    # Grafico correlazione PV-SOC-Prezzo
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Analisi Correlazioni PV-Batteria-Prezzi - {battery.technology}',
                 fontsize=16, fontweight='bold')

    sample_hours = min(168, len(results_df))  # 7 giorni
    hours = range(sample_hours)

    # Subplot 1: Produzione PV e allocazione
    ax1 = axes[0]
    if 'PV_Production_MWh' in results_df.columns:
        ax1.plot(hours, results_df['PV_Production_MWh'].iloc[:sample_hours],
                 color='#F4A300', linewidth=2, label='Produzione PV', zorder=3)
        ax1.fill_between(hours, 0, results_df['PV_Production_MWh'].iloc[:sample_hours],
                         color='#F4A300', alpha=0.2)

        if 'PV_to_Battery_MWh' in results_df.columns:
            ax1.plot(hours, results_df['PV_to_Battery_MWh'].iloc[:sample_hours],
                     color='#06A77D', linewidth=1.5, linestyle='--', label='PV→Batteria', zorder=2)

        if 'PV_to_Load_MWh' in results_df.columns:
            ax1.plot(hours, results_df['PV_to_Load_MWh'].iloc[:sample_hours],
                     color='#F77F00', linewidth=1.5, linestyle='--', label='PV→Carico', zorder=2)

    ax1.set_ylabel('Energia (MWh)', fontsize=11, fontweight='bold')
    ax1.set_title('Produzione e Allocazione PV', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: SOC batteria con evidenza carica da PV
    ax2 = axes[1]
    if 'SOC' in results_df.columns:
        ax2.plot(hours, results_df['SOC'].iloc[:sample_hours] * 100,
                 color='#457B9D', linewidth=2.5, label='SOC', zorder=3)
        ax2.fill_between(hours, battery.soc_min * 100,
                         results_df['SOC'].iloc[:sample_hours] * 100,
                         color='#457B9D', alpha=0.2)

        # Evidenzia ore con carica da PV
        if 'Energy_from_PV_MWh' in results_df.columns:
            for h in hours:
                if h < len(results_df) and results_df['Energy_from_PV_MWh'].iloc[h] > 0.01:
                    ax2.axvspan(h - 0.5, h + 0.5, color='#06A77D', alpha=0.15, zorder=1)

        ax2.axhline(y=battery.soc_min * 100, color='red', linewidth=1,
                    linestyle='--', alpha=0.7, label=f'SOC min ({battery.soc_min * 100:.0f}%)')
        ax2.axhline(y=battery.soc_max * 100, color='green', linewidth=1,
                    linestyle='--', alpha=0.7, label=f'SOC max ({battery.soc_max * 100:.0f}%)')

    ax2.set_ylabel('SOC (%)', fontsize=11, fontweight='bold')
    ax2.set_title('State of Charge (sfondo verde = carica da PV)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Subplot 3: Prezzo energia
    ax3 = axes[2]
    if '€/MWh' in results_df.columns:
        ax3.plot(hours, results_df['€/MWh'].iloc[:sample_hours],
                 color='#E63946', linewidth=2, label='Prezzo Vendita', zorder=3)
        ax3.fill_between(hours, 0, results_df['€/MWh'].iloc[:sample_hours],
                         color='#E63946', alpha=0.2)

        if 'Prezzo_Acquisto_€/MWh' in results_df.columns:
            ax3.plot(hours, results_df['Prezzo_Acquisto_€/MWh'].iloc[:sample_hours],
                     color='#8B0000', linewidth=1.5, linestyle='--',
                     label='Prezzo Acquisto', zorder=2)

    ax3.set_xlabel('Ora', fontsize=11)
    ax3.set_ylabel('Prezzo (€/MWh)', fontsize=11, fontweight='bold')
    ax3.set_title('Andamento Prezzi Energia', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = os.path.join(viz_folder, f'pv_analysis_{battery.technology}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvato: {filename}")
    print("=" * 80)


def create_load_analysis_plots(results_df, battery, load_profile):
    """
    Lorenzo Giannuzzo: Analisi completa gestione carico utente
    """
    if not SAVE_PLOTS or not LOAD_ENABLED or load_profile is None:
        return

    viz_folder = 'visualization'
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)

    print("\n" + "=" * 80)
    print("GENERAZIONE GRAFICI ANALISI CARICO")
    print("=" * 80)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Analisi Gestione Carico Utente - {battery.technology}',
                 fontsize=16, fontweight='bold')

    load_stats = load_profile.get_statistics()

    # Grafico 1: Pie chart fonti fornitura carico
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['PV Diretto', 'Batteria', 'Rete']
    sizes = [
        load_stats['energy_from_pv_mwh'],
        load_stats['energy_from_battery_mwh'],
        load_stats['energy_from_grid_mwh']
    ]
    colors = ['#F4A300', '#06A77D', '#E63946']
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title('Fonti di Fornitura Carico', fontsize=12, fontweight='bold')

    # Grafico 2: Barre energia per fonte
    ax2 = fig.add_subplot(gs[0, 1])
    sources = ['PV\nDiretto', 'Batteria', 'Rete']
    values = [
        load_stats['energy_from_pv_mwh'],
        load_stats['energy_from_battery_mwh'],
        load_stats['energy_from_grid_mwh']
    ]
    bars = ax2.bar(sources, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Energia (MWh)', fontsize=11, fontweight='bold')
    ax2.set_title('Energia Carico per Fonte', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.2f} MWh\n({val / load_stats["total_energy_required_mwh"] * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Grafico 3: Timeline fornitura carico (sample 7 giorni)
    ax3 = fig.add_subplot(gs[1, :])
    if all(col in results_df.columns for col in ['Load_Demand_MWh', 'Load_from_PV_MWh',
                                                 'Load_from_Battery_MWh', 'Load_from_Grid_MWh']):
        sample_hours = min(168, len(results_df))
        hours = range(sample_hours)

        ax3.plot(hours, results_df['Load_Demand_MWh'].iloc[:sample_hours],
                 color='black', linewidth=2.5, label='Carico Totale', zorder=4, linestyle='-')

        ax3.bar(hours, results_df['Load_from_PV_MWh'].iloc[:sample_hours],
                width=0.8, color='#F4A300', alpha=0.8, label='Da PV', edgecolor='black', linewidth=0.5)
        ax3.bar(hours, results_df['Load_from_Battery_MWh'].iloc[:sample_hours],
                width=0.8, bottom=results_df['Load_from_PV_MWh'].iloc[:sample_hours],
                color='#06A77D', alpha=0.8, label='Da Batteria', edgecolor='black', linewidth=0.5)
        ax3.bar(hours, results_df['Load_from_Grid_MWh'].iloc[:sample_hours],
                width=0.8,
                bottom=(results_df['Load_from_PV_MWh'] + results_df['Load_from_Battery_MWh']).iloc[:sample_hours],
                color='#E63946', alpha=0.8, label='Da Rete', edgecolor='black', linewidth=0.5)

        ax3.set_xlabel('Ora', fontsize=11)
        ax3.set_ylabel('Energia (MWh)', fontsize=11, fontweight='bold')
        ax3.set_title('Soddisfacimento Carico nel Tempo (Sample 7 giorni)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10, loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')

    # Grafico 4: Percentuali coverage
    ax4 = fig.add_subplot(gs[2, 0])
    coverage_types = ['PV\nDiretto', 'Batteria', 'Rete\n(dipendenza)']
    coverage_values = [
        load_stats['pv_coverage_percent'],
        load_stats['battery_coverage_percent'],
        load_stats['grid_dependency_percent']
    ]
    colors_coverage = ['#F4A300', '#06A77D', '#E63946']

    bars = ax4.barh(coverage_types, coverage_values, color=colors_coverage,
                    alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_xlabel('Coverage (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Copertura Carico per Fonte', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, 100)

    for bar, val in zip(bars, coverage_values):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height() / 2.,
                 f' {val:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

    # Grafico 5: Statistiche testuali
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    autosufficienza = 100 - load_stats['grid_dependency_percent']
    risparmio_stima = load_stats['energy_from_pv_mwh'] + load_stats['energy_from_battery_mwh']

    stats_text = f"""
STATISTICHE CARICO UTENTE

Fabbisogno Totale:
  • Energia richiesta: {load_stats['total_energy_required_mwh']:.2f} MWh

Fonti di Fornitura:
  • PV diretto: {load_stats['energy_from_pv_mwh']:.2f} MWh ({load_stats['pv_coverage_percent']:.1f}%)
  • Batteria: {load_stats['energy_from_battery_mwh']:.2f} MWh ({load_stats['battery_coverage_percent']:.1f}%)
  • Rete: {load_stats['energy_from_grid_mwh']:.2f} MWh ({load_stats['grid_dependency_percent']:.1f}%)

Indici Prestazione:
  • Autosufficienza: {autosufficienza:.1f}%
  • Dipendenza rete: {load_stats['grid_dependency_percent']:.1f}%
  • Energia risparmiata: {risparmio_stima:.2f} MWh

Strategia Ottimizzazione:
  • Priorità 1: PV diretto (GRATIS)
  • Priorità 2: Batteria (risparmio)
  • Priorità 3: Rete (COSTO)
    """

    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    filename = os.path.join(viz_folder, f'load_analysis_{battery.technology}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvato: {filename}")
    print("=" * 80)

# ========================================================================================================
# FUNZIONE MAIN
# ========================================================================================================
def main(file_name, pv_file_name=None, load_file_name=None):
    file_path = os.path.join('data', file_name)

    print("=" * 80)
    print("BESS OPTIMIZATION v2.5.0 CORRECTED")
    print("=" * 80)

    try:
        df = pd.read_excel(file_path)
        if df['€/MWh'].dtype == 'object':
            df['€/MWh'] = df['€/MWh'].astype(str).str.replace(',', '.').astype(float)
        print(f"✓ Prezzi vendita: {len(df)} righe, media {df['€/MWh'].mean():.2f} €/MWh")
    except Exception as e:
        print(f"❌ Errore caricamento prezzi: {e}")
        return

    print(f"✓ Prezzi acquisto calcolati con mark-up {PRICE_MARKUP_PERCENT}%")

    pv_df = None
    pv_system = None
    if PV_ENABLED and pv_file_name:
        try:
            pv_file_path = os.path.join('data', pv_file_name)
            pv_df = pd.read_csv(pv_file_path, sep=';')
            pv_system = PhotovoltaicSystem()
            print(f"✓ PV caricato: {PV_NOMINAL_POWER_KWP:.0f} kWp, media {pv_df['P'].mean():.2f} kW")
        except Exception as e:
            print(f"❌ Errore PV: {e}")
            pv_df = None
            pv_system = None

    load_df = None
    load_profile = None
    if LOAD_ENABLED and load_file_name:
        try:
            load_file_path = os.path.join('data', load_file_name)
            if LOAD_SHEET_NAME:
                load_df_raw = pd.read_excel(load_file_path, sheet_name=LOAD_SHEET_NAME)
            else:
                xls = pd.ExcelFile(load_file_path)
                first_sheet = xls.sheet_names[0]
                print(f"  Sheet: {first_sheet}")
                load_df_raw = pd.read_excel(load_file_path, sheet_name=first_sheet)

            load_column = None
            for col_name in ['value', 'Value', 'VALUE', 'load', 'Load', 'LOAD', 'Power', 'power', 'POWER']:
                if col_name in load_df_raw.columns:
                    load_column = col_name
                    break

            if load_column is None:
                numeric_cols = load_df_raw.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    load_column = numeric_cols[0]
                else:
                    raise ValueError("Nessuna colonna numerica trovata")

            load_df = pd.DataFrame()
            load_df['value'] = load_df_raw[load_column].copy()
            load_profile = LoadProfile()
            print(f"✓ Carico: {len(load_df)} righe, media {load_df['value'].mean():.2f} kW, totale {load_df['value'].sum()/1000:.2f} MWh")
        except Exception as e:
            print(f"❌ Errore carico: {e}")
            load_df = None
            load_profile = None

    battery = Battery()
    optimizer = PSOOptimizer(n_particles=50, n_iterations=100)
    simulator = RollingHorizonSimulator(battery, optimizer, pv_system=pv_system, load_profile=load_profile)

    start_time = datetime.now()
    results_df, trading_profit = simulator.simulate(df, pv_df, load_df)
    end_time = datetime.now()

    macse_revenue, macse_base, macse_penalty, macse_bonus = calculate_macse_revenue(battery)
    total_system_profit = trading_profit + macse_revenue

    print("\n" + "=" * 80)
    print("RISULTATI v2.5.0")
    print("=" * 80)

    if LOAD_ENABLED and load_profile:
        load_stats = load_profile.get_statistics()
        print(f"\nCARICO:")
        print(f"  Totale: {load_stats['total_energy_required_mwh']:.2f} MWh")
        print(f"  Da PV: {load_stats['energy_from_pv_mwh']:.2f} MWh ({load_stats['pv_coverage_percent']:.1f}%)")
        print(f"  Da Batteria: {load_stats['energy_from_battery_mwh']:.2f} MWh ({load_stats['battery_coverage_percent']:.1f}%)")
        print(f"  Da Rete: {load_stats['energy_from_grid_mwh']:.2f} MWh ({load_stats['grid_dependency_percent']:.1f}%)")

    if PV_ENABLED and pv_system:
        pv_stats = pv_system.get_statistics()
        print(f"\nPV:")
        print(f"  Produzione: {pv_stats['total_production_mwh']:.2f} MWh")
        print(f"  → Carico: {pv_stats['energy_to_load_mwh']:.2f} MWh ({pv_stats['load_service_percent']:.1f}%)")
        print(f"  → Batteria: {pv_stats['energy_to_battery_mwh']:.2f} MWh ({pv_stats['battery_utilization_percent']:.1f}%)")
        print(f"  → Rete: {pv_stats['energy_to_grid_mwh']:.2f} MWh ({pv_stats['grid_sale_percent']:.1f}%)")

    print(f"\nBATTERIA:")
    print(f"  SOH: {battery.get_soh():.2f}%")
    print(f"  Cicli: {battery.equivalent_cycles:.2f}")
    print(f"  Throughput: {battery.throughput_kwh:.2f} kWh")

    print(f"\nECONOMIA (REALE):")
    print(f"  Profitto trading: {trading_profit:,.2f} €")
    if MACSE_ENABLED:
        print(f"  Ricavi MACSE: {macse_revenue:,.2f} €")
    print(f"  Profitto totale: {total_system_profit:,.2f} €")
    print(f"  Tempo: {(end_time - start_time).total_seconds():.1f} s")
    print("=" * 80)

    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    output_file = os.path.join(results_folder, f'risultati_{battery.technology.lower().replace("-", "_")}_v250.xlsx')
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Risultati', index=False)
    print(f"✓ Excel: {output_file}")

    export_results_to_json(results_df, battery, pv_system, load_profile, trading_profit, macse_revenue,
                          macse_base, macse_penalty, macse_bonus, 600000, (end_time - start_time).total_seconds())

    if PV_ENABLED and pv_system:
        create_pv_impact_comparison(results_df, battery, pv_system)
        create_pv_impact_summary(results_df, battery, pv_system)
        create_pv_visualizations(results_df, battery, pv_system)
        create_detailed_monthly_pv_plots(results_df, battery, pv_system)

    if LOAD_ENABLED and load_profile:
        create_load_analysis_plots(results_df, battery, load_profile)

    print("\n" + "=" * 80)
    print("✓ SIMULAZIONE COMPLETATA v2.5.0 CORRECTED")
    print("=" * 80)


if __name__ == "__main__":
    pv_file = pv_production_file if PV_ENABLED else None
    load_file_input = load_file if LOAD_ENABLED else None
    main(energy_selling_price_name, pv_file, load_file_input)