"""
------------------------------------------------------------------------------------------------------------------------
BATTERY ENERGY STORAGE SYSTEM (BESS) OPTIMIZATION - AUTONOMOUS LOAD DECISIONS
Particle Swarm Optimization with Rolling Horizon, MACSE, Autonomous PV and AUTONOMOUS Load Management
Support Litio-ione e Grafene + Sistema Fotovoltaico + Carico Utente con DECISIONE OTTIMALE AUTONOMA
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
    - **AUTONOMOUS LOAD: Batteria DECIDE quando servire carico vs trading**

Version: 3.7.0
Date: 25 November 2025
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
energy_selling_price_name = 'Prezzo_Vendita.xlsx'
energy_buying_price_name = 'Prezzo_Acquisto.xlsx'
pv_production_file = 'PV_formattato.csv'
load_file = 'Consumo.xlsx'

# ---------------------------------- PARAMETRI POINT OF DELIVERY (POD) --------------------------------
POD_POWER_MW = 0.8  # Potenza massima scambio con rete [MW]

# ---------------------------------- SCELTA TECNOLOGIA BATTERIA -------------------------------------------
BATTERY_TECHNOLOGY = "GRAFENE"

# ---------------------------------- PARAMETRI BATTERIA ---------------------------------------------------
BATTERY_CAPACITY_MWH = 1.0
BATTERY_MAX_POWER_MW = 1.0
BATTERY_MAX_C_RATE = 1.0
BATTERY_BASE_EFFICIENCY = 0.95 #todo dare come input

# ---------------------------------- PARAMETRI FOTOVOLTAICO -----------------------------------------------
PV_ENABLED = True
PV_NOMINAL_POWER_KWP = 1.0
PV_INVERTER_EFFICIENCY = 0.98
PV_SYSTEM_LOSSES = 0.0

# ---------------------------------- PARAMETRI CARICO UTENTE -----------------------------------------------
LOAD_ENABLED = True
LOAD_SHEET_NAME = None  # None = primo sheet disponibile


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
DEGRADATION_COST_PER_MWH = 0.0

# ---------------------------------- PARAMETRI MACSE ------------------------------------------------------
MACSE_ENABLED = False
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
        power_mw = (irradiance_w_per_kwp * self.nominal_power_kwp * self.total_efficiency) * 1000
        return power_mw

    def get_energy(self, irradiance_w_per_kwp, dt=1.0):
        """Calcola energia prodotta in un timestep [MWh]"""
        power_mw = self.get_production(irradiance_w_per_kwp)
        energy_mwh = power_mw * dt
        self.total_production_mwh += energy_mwh
        return energy_mwh

    def load_pv_production(self, pv_value_kw, dt=1.0):
        """
        Carica produzione PV già calcolata in kW
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
    Lorenzo Giannuzzo: Modello carico elettrico utente con tracking decisioni autonome
    NUOVO v2.6.0: Traccia quando batteria SCEGLIE di servire carico vs lasciare a rete
    """
    def __init__(self):
        self.total_energy_required_mwh = 0.0
        self.energy_from_pv_mwh = 0.0
        self.energy_from_battery_mwh = 0.0
        self.energy_from_grid_mwh = 0.0

        self.battery_served_load_count = 0  # Ore in cui batteria ha SCELTO di servire carico
        self.grid_served_load_count = 0      # Ore in cui batteria ha SCELTO di lasciare carico a rete
        self.total_decision_hours = 0

    def register_supply(self, from_pv, from_battery, from_grid):
        """Registra fonti di fornitura carico"""
        self.energy_from_pv_mwh += from_pv
        self.energy_from_battery_mwh += from_battery
        self.energy_from_grid_mwh += from_grid
        self.total_energy_required_mwh += (from_pv + from_battery + from_grid)

    def register_battery_decision(self, battery_served: bool):
        """
        battery_served=True → Batteria ha SCELTO di servire carico
        battery_served=False → Batteria ha SCELTO di lasciare carico a rete (per trading futuro)
        """
        self.total_decision_hours += 1
        if battery_served:
            self.battery_served_load_count += 1
        else:
            self.grid_served_load_count += 1

    def get_statistics(self):
        """Ritorna statistiche fornitura carico con decisioni autonome"""
        total = self.total_energy_required_mwh
        stats = {
            'total_energy_required_mwh': total,
            'energy_from_pv_mwh': self.energy_from_pv_mwh,
            'energy_from_battery_mwh': self.energy_from_battery_mwh,
            'energy_from_grid_mwh': self.energy_from_grid_mwh,
            'pv_coverage_percent': (self.energy_from_pv_mwh / total * 100) if total > 0 else 0,
            'battery_coverage_percent': (self.energy_from_battery_mwh / total * 100) if total > 0 else 0,
            'grid_dependency_percent': (self.energy_from_grid_mwh / total * 100) if total > 0 else 0
        }

        # NUOVO: Statistiche decisioni
        if self.total_decision_hours > 0:
            stats['battery_served_decisions_percent'] = (self.battery_served_load_count / self.total_decision_hours * 100)
            stats['grid_served_decisions_percent'] = (self.grid_served_load_count / self.total_decision_hours * 100)
            stats['battery_served_count'] = self.battery_served_load_count
            stats['grid_served_count'] = self.grid_served_load_count
            stats['total_decisions'] = self.total_decision_hours
        else:
            stats['battery_served_decisions_percent'] = 0
            stats['grid_served_decisions_percent'] = 0
            stats['battery_served_count'] = 0
            stats['grid_served_count'] = 0
            stats['total_decisions'] = 0

        return stats

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
            self.equivalent_cycles = self.throughput_kwh / (2 * 10 * self.nominal_capacity * 1000)
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
        """C-rate sempre su nominal_capacity"""
        return self.nominal_capacity * self.max_c_rate


# ========================================================================================================
# SEZIONE 6: OTTIMIZZATORE PSO
# ========================================================================================================
class PSOOptimizer:
    """
    PSO con decisioni autonome multi-dimensionali
    Ogni ora: 3 decisioni continue [p_batt_trading, alpha_pv_load, p_batt_load]

    OBIETTIVO ECONOMICO REALE:
    Minimizzare: Costo Netto = Costi - Ricavi + Degrado

    Dove:
    - Ricavi = SOLO vendite energia alla rete (PV + batteria)
    - Costi = SOLO acquisti energia dalla rete (per batteria + per carico)
    - Degrado = Costo cicli batteria

    L'AUTOCONSUMO NON GENERA RICAVI - riduce solo il carico che deve essere servito dalla rete
    """

    def __init__(self, n_particles=120, n_iterations=300, w_start=0.95, w_end=0.2, c1=2.0, c2=2.0):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.stagnation_limit = 15

    def optimize(self, battery, prices_sell, prices_buy, pv_production, load_demand, horizon_hours=24):
        """Ottimizzazione PSO con vincolo POD"""
        n_hours = min(horizon_hours, len(prices_sell))
        max_power_limit = min(battery.trading_power, battery.get_max_power_by_crate())

        # Inizializzazione smart
        positions = self._smart_initialization(battery, prices_sell, prices_buy, pv_production, load_demand,
                                               max_power_limit)
        velocities = np.random.uniform(-0.5, 0.5, (self.n_particles, n_hours, 3))

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
                r1, r2 = np.random.random((n_hours, 3)), np.random.random((n_hours, 3))
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social

                # Limiti velocità
                max_vel = np.array([max_power_limit * 0.5, 0.3, max_power_limit * 0.5])
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)

                positions[i] += velocities[i]

                # Clipping azioni con vincolo POD
                positions[i, :, 0] = np.clip(positions[i, :, 0], -min(max_power_limit, POD_POWER_MW),
                                             min(max_power_limit, POD_POWER_MW))
                positions[i, :, 1] = np.clip(positions[i, :, 1], 0, 1)
                positions[i, :, 2] = np.clip(positions[i, :, 2], 0, min(max_power_limit, POD_POWER_MW))

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
                    noise = np.random.uniform(-0.3, 0.3, (n_hours, 3))
                    noise[:, 0] *= min(max_power_limit, POD_POWER_MW)
                    noise[:, 2] *= min(max_power_limit, POD_POWER_MW)
                    positions[idx] = global_best_position + noise
                    positions[idx, :, 0] = np.clip(positions[idx, :, 0], -min(max_power_limit, POD_POWER_MW),
                                                   min(max_power_limit, POD_POWER_MW))
                    positions[idx, :, 1] = np.clip(positions[idx, :, 1], 0, 1)
                    positions[idx, :, 2] = np.clip(positions[idx, :, 2], 0, min(max_power_limit, POD_POWER_MW))
                    velocities[idx] = np.random.uniform(-0.5, 0.5, (n_hours, 3))
                stagnation_counter = 0

        return global_best_position

    def _smart_initialization(self, battery, prices_sell, prices_buy, pv_production, load_demand, max_power):
        """
        Inizializzazione smart con euristiche economiche E vincolo POD
        """
        n_hours = len(prices_sell)
        positions = np.zeros((self.n_particles, n_hours, 3))

        price_low = np.percentile(prices_sell, 25)
        price_high = np.percentile(prices_sell, 75)

        # VINCOLO POD: Limita max_power al POD
        max_power_with_pod = min(max_power, POD_POWER_MW)

        for i in range(self.n_particles):
            if i < self.n_particles // 3:  # STRATEGIA 1: Price-driven
                for h in range(n_hours):
                    # === BATTERIA TRADING ===
                    if prices_sell[h] < price_low:
                        positions[i, h, 0] = np.random.uniform(0.4 * max_power_with_pod, max_power_with_pod)
                    elif prices_sell[h] > price_high:
                        positions[i, h, 0] = np.random.uniform(-max_power_with_pod, -0.4 * max_power_with_pod)
                    else:
                        positions[i, h, 0] = np.random.uniform(-0.3 * max_power_with_pod, 0.3 * max_power_with_pod)

                    # === ALLOCAZIONE PV ===
                    if prices_buy[h] > prices_sell[h] * 1.2:
                        positions[i, h, 1] = np.random.uniform(0.7, 1.0)
                    else:
                        positions[i, h, 1] = np.random.uniform(0.3, 0.7)

                    # === BATTERIA PER CARICO ===
                    if prices_buy[h] > np.mean(prices_buy) and load_demand[h] > np.mean(load_demand):
                        positions[i, h, 2] = np.random.uniform(0.3 * max_power_with_pod, 0.8 * max_power_with_pod)
                    else:
                        positions[i, h, 2] = np.random.uniform(0, 0.3 * max_power_with_pod)

            elif i < 2 * self.n_particles // 3:  # STRATEGIA 2: Load-priority
                for h in range(n_hours):
                    positions[i, h, 0] = np.random.uniform(-0.4 * max_power_with_pod, 0.4 * max_power_with_pod)
                    positions[i, h, 1] = np.random.uniform(0.8, 1.0)
                    if load_demand[h] > 0.001:
                        positions[i, h, 2] = np.random.uniform(0.2 * max_power_with_pod, max_power_with_pod)
                    else:
                        positions[i, h, 2] = 0.0

            else:  # STRATEGIA 3: Random exploration
                positions[i, :, 0] = np.random.uniform(-max_power_with_pod, max_power_with_pod, n_hours)
                positions[i, :, 1] = np.random.uniform(0, 1, n_hours)
                positions[i, :, 2] = np.random.uniform(0, max_power_with_pod, n_hours)

        return positions

    def _evaluate(self, battery, actions, prices_sell, prices_buy, pv_production, load_demand):
        """
        ===============================================================================
        FUNZIONE OBIETTIVO - CON VINCOLO POD (POINT OF DELIVERY)
        ===============================================================================

        VINCOLO FISICO:
        - POD_POWER_MW: Potenza massima scambio con rete elettrica
        - Limita PRELIEVO dalla rete: ≤ POD_POWER_MW
        - Limita IMMISSIONE in rete: ≤ POD_POWER_MW

        APPLICAZIONE:
        - Prelievo = Energia per batteria + Energia per carico
        - Immissione = Scarica batteria trading + PV venduto

        PENALITÀ:
        - Se violazione POD → penalità pesante nel profitto
        ===============================================================================
        """
        bat_sim = battery.copy()
        profit = 0.0
        pod_violation_penalty = 0.0  # Penalità per violazioni POD

        for hour, (action, price_sell, price_buy, pv_available, load_required) in enumerate(
                zip(actions, prices_sell, prices_buy, pv_production, load_demand)
        ):
            p_batt_trading = action[0]
            alpha_pv_load = action[1]
            p_batt_load = action[2]

            pv_available = pv_available if PV_ENABLED else 0.0
            load_required = load_required if LOAD_ENABLED else 0.0

            # ========================================================================
            # FASE 1: ALLOCAZIONE PV
            # ========================================================================
            pv_to_load = min(alpha_pv_load * pv_available, load_required)
            pv_remaining = pv_available - pv_to_load
            load_remaining = load_required - pv_to_load

            # ========================================================================
            # FASE 2: VINCOLO XOR BATTERIA
            # ========================================================================
            if p_batt_trading > 0.001:
                charge_request = p_batt_trading
                discharge_request_trading = 0.0
                discharge_request_load = 0.0
                action_mode = "CHARGE"
            elif p_batt_trading < -0.001:
                charge_request = 0.0
                discharge_request_trading = -p_batt_trading
                discharge_request_load = p_batt_load
                action_mode = "DISCHARGE"
            elif p_batt_load > 0.001:
                charge_request = 0.0
                discharge_request_trading = 0.0
                discharge_request_load = p_batt_load
                action_mode = "DISCHARGE"
            else:
                charge_request = 0.0
                discharge_request_trading = 0.0
                discharge_request_load = 0.0
                action_mode = "IDLE"

            total_discharge_request = discharge_request_trading + discharge_request_load

            # Tracciamento flussi per vincolo POD
            grid_withdrawal_this_hour = 0.0  # Prelievo totale dalla rete
            grid_injection_this_hour = 0.0  # Immissione totale in rete

            # ========================================================================
            # FASE 3: ESECUZIONE AZIONI BATTERIA CON VINCOLO POD
            # ========================================================================
            if action_mode == "DISCHARGE" and total_discharge_request > 0.001:
                # === SCARICA ===
                max_discharge = (
                                            bat_sim.soc - bat_sim.soc_min) * bat_sim.trading_capacity * bat_sim.discharge_efficiency
                actual_discharge_total = min(total_discharge_request, max_discharge)

                if actual_discharge_total > 0.001:
                    ratio_trading = discharge_request_trading / total_discharge_request if total_discharge_request > 0 else 0
                    ratio_load = discharge_request_load / total_discharge_request if total_discharge_request > 0 else 0

                    discharge_for_trading = actual_discharge_total * ratio_trading
                    discharge_for_load_raw = actual_discharge_total * ratio_load
                    discharge_for_load = min(discharge_for_load_raw, load_remaining)

                    # VINCOLO POD: Limita scarica trading se eccede POD
                    discharge_for_trading = min(discharge_for_trading, POD_POWER_MW)

                    actual_discharge_used = discharge_for_trading + discharge_for_load

                    # Controlla violazione POD per immissione
                    if discharge_for_trading > POD_POWER_MW:
                        pod_violation_penalty += (discharge_for_trading - POD_POWER_MW) * price_sell * 10.0

                    # Esegui scarica fisica
                    energy_consumed = actual_discharge_used / bat_sim.discharge_efficiency
                    new_soc = bat_sim.soc - (energy_consumed / bat_sim.capacity)
                    bat_sim.soc = max(new_soc, bat_sim.soc_min)
                    bat_sim.throughput_kwh += energy_consumed * 1000

                    # RICAVO: Vendita trading
                    if discharge_for_trading > 0.001:
                        profit += discharge_for_trading * price_sell
                        profit -= discharge_for_trading * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                        grid_injection_this_hour += discharge_for_trading

                    # AUTOCONSUMO: Scarica per carico
                    if discharge_for_load > 0.001:
                        load_remaining -= discharge_for_load
                        profit -= discharge_for_load * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)

            elif action_mode == "CHARGE" and charge_request > 0.001:
                # === CARICA ===
                if bat_sim.soc < bat_sim.soc_max:
                    max_storable = (bat_sim.soc_max - bat_sim.soc) * bat_sim.trading_capacity
                    max_power = max_storable / (1.0 * bat_sim.charge_efficiency)
                    actual_power = min(charge_request, max_power)

                    if actual_power > 0.01:
                        energy_needed = actual_power * 1.0

                        # Priorità 1: PV (se disponibile)
                        if pv_remaining > 0:
                            energy_from_pv = min(energy_needed, pv_remaining)
                            bat_sim.charge(energy_from_pv / 1.0, dt=1.0, source='pv')
                            pv_remaining -= energy_from_pv
                            energy_needed -= energy_from_pv
                            profit -= energy_from_pv * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)

                        # Priorità 2: Rete (con VINCOLO POD)
                        if energy_needed > 0.01:
                            # VINCOLO POD: Limita prelievo da rete
                            energy_from_grid_requested = energy_needed
                            energy_from_grid = min(energy_from_grid_requested, POD_POWER_MW)

                            # Controlla violazione POD per prelievo
                            if energy_from_grid_requested > POD_POWER_MW:
                                pod_violation_penalty += (energy_from_grid_requested - POD_POWER_MW) * price_buy * 10.0

                            if energy_from_grid > 0.001:
                                actual_charged = bat_sim.charge(energy_from_grid / 1.0, dt=1.0, source='grid')
                                profit -= actual_charged * price_buy
                                profit -= actual_charged * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                                grid_withdrawal_this_hour += actual_charged

            # ========================================================================
            # FASE 4: GESTIONE PV RESIDUO E CARICO RESIDUO CON VINCOLO POD
            # ========================================================================

            # PV residuo → Vendi a rete (CON VINCOLO POD)
            if pv_remaining > 0.001:
                # VINCOLO POD: Limita immissione PV
                pv_to_grid = min(pv_remaining, POD_POWER_MW - grid_injection_this_hour)

                if pv_to_grid > 0.001:
                    profit += pv_to_grid * price_sell
                    grid_injection_this_hour += pv_to_grid

                # Penalità per PV curtailed causa POD
                pv_curtailed = pv_remaining - pv_to_grid
                if pv_curtailed > 0.001:
                    # Penalità minore: è energia persa ma non costa
                    pod_violation_penalty += pv_curtailed * price_sell * 0.5

            # Carico residuo → Acquista da rete (CON VINCOLO POD)
            if load_remaining > 0.001:
                # VINCOLO POD: Limita prelievo per carico
                load_from_grid = min(load_remaining, POD_POWER_MW - grid_withdrawal_this_hour)

                if load_from_grid > 0.001:
                    profit -= load_from_grid * price_buy
                    grid_withdrawal_this_hour += load_from_grid

                # Penalità PESANTE per carico non servito
                load_unserved = load_remaining - load_from_grid
                if load_unserved > 0.001:
                    # Penalità molto alta: il carico DEVE essere servito
                    pod_violation_penalty += load_unserved * price_buy * 100.0

            # ========================================================================
            # CONTROLLO FINALE VIOLAZIONI POD
            # ========================================================================
            if grid_withdrawal_this_hour > POD_POWER_MW + 0.001:
                pod_violation_penalty += (grid_withdrawal_this_hour - POD_POWER_MW) * price_buy * 50.0

            if grid_injection_this_hour > POD_POWER_MW + 0.001:
                pod_violation_penalty += (grid_injection_this_hour - POD_POWER_MW) * price_sell * 50.0

        # Sottrai penalità POD dal profitto finale
        final_profit = profit - pod_violation_penalty

        return final_profit


# ========================================================================================================
# SEZIONE 7: ROLLING HORIZON SIMULATOR - v3.2 CORRECTED
# ========================================================================================================
class RollingHorizonSimulator:
    """Scarica SOLO energia effettivamente utilizzata"""

    def __init__(self, battery, optimizer, pv_system=None, load_profile=None, horizon_hours=24, step_hours=1):
        self.battery = battery
        self.optimizer = optimizer
        self.pv_system = pv_system
        self.load_profile = load_profile
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours

    def simulate(self, prices_df, price_df2, pv_df=None, load_df=None):
        """
        CON VINCOLO POD (POINT OF DELIVERY)

        NUOVO: Rispetta limite potenza scambio con rete (POD_POWER_MW)
        - Limita prelievo dalla rete
        - Limita immissione in rete
        - Traccia curtailment PV e carico non servito causa POD
        """
        prices_sell = prices_df['€/MWh'].values
        prices_buy = price_df2['€/MWh'].values
        n_hours = len(prices_sell)

        # ========================================================================
        # PREPARAZIONE DATI
        # ========================================================================
        if PV_ENABLED and pv_df is not None and self.pv_system is not None:
            pv_production = pv_df['P'].values / 1000.0  # kW → MW
            self.pv_system.total_production_mwh = np.sum(pv_production)
            if len(pv_production) < n_hours:
                pv_production = np.pad(pv_production, (0, n_hours - len(pv_production)), 'constant')
            elif len(pv_production) > n_hours:
                pv_production = pv_production[:n_hours]
        else:
            pv_production = np.zeros(n_hours)

        if LOAD_ENABLED and load_df is not None:
            load_demand = load_df['value'].values / 1000.0  # kW → MW
            if len(load_demand) < n_hours:
                load_demand = np.pad(load_demand, (0, n_hours - len(load_demand)), 'constant')
            elif len(load_demand) > n_hours:
                load_demand = load_demand[:n_hours]
        else:
            load_demand = np.zeros(n_hours)

        # ========================================================================
        # ARRAYS RISULTATI + TRACKING POD
        # ========================================================================
        actions_trading_effective = []
        actions_alpha_pv = []
        actions_p_load_effective = []

        soc_history = []
        capacity_history = []
        soh_history = []
        profits_history = []
        macse_availability_history = []

        pv_production_history = []
        pv_to_battery_history = []
        pv_to_grid_history = []
        pv_to_load_history = []
        pv_curtailed_history = []

        load_demand_history = []
        load_from_pv_history = []
        load_from_battery_history = []
        load_from_grid_history = []
        load_unserved_history = []

        energy_from_grid_to_battery_history = []
        energy_from_pv_to_battery_history = []
        price_buy_history = []

        battery_decision_history = []
        load_discharge_history = []
        trading_discharge_history = []

        # Tracking POD
        grid_withdrawal_history = []  # Prelievo orario totale
        grid_injection_history = []  # Immissione oraria totale
        pod_violation_history = []  # Flag violazione POD

        cumulative_profit = 0.0

        print("=" * 80)
        print("SIMULAZIONE BESS")
        print("=" * 80)
        print(f"Tecnologia: {self.battery.technology}")
        print(f"Capacità: {self.battery.nominal_capacity} MWh")
        print(f"POD Power: {POD_POWER_MW} MW ⚡")
        if PV_ENABLED and self.pv_system:
            print(f"PV: {self.pv_system.nominal_power_kwp:.0f} kWp")
        if LOAD_ENABLED:
            print(f"Carico: {np.sum(load_demand):.2f} MWh totali")
        print(f"Obiettivo: Minimizzare costo netto energia")
        print(f"Vincolo: Scambio rete ≤ {POD_POWER_MW} MW")
        print("=" * 80)

        current_hour = 0
        last_progress = 0
        degradation_update_interval = 24

        # ========================================================================
        # ROLLING HORIZON LOOP
        # ========================================================================
        while current_hour < n_hours:
            progress = int((current_hour / n_hours) * 100)
            if progress >= last_progress + 20:
                print(
                    f"Progresso: {progress}% - SOH: {self.battery.get_soh():.2f}% - SOC: {self.battery.get_soc() * 100:.1f}%")
                last_progress = progress

            if current_hour % degradation_update_interval == 0 and current_hour > 0:
                self.battery.update_degradation()

            # Ottimizzazione rolling horizon
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
            p_batt_trading_requested = action[0]
            alpha_pv_load = action[1]
            p_batt_load_requested = action[2]

            price_sell = prices_sell[current_hour]
            price_buy = prices_buy[current_hour]
            pv_available = pv_production[current_hour]
            load_required = load_demand[current_hour]

            # ====================================================================
            # VARIABILI TRACKING POD - RESET OGNI ORA
            # ====================================================================
            grid_withdrawal_this_hour = 0.0  # Prelievo totale dalla rete
            grid_injection_this_hour = 0.0  # Immissione totale in rete

            pv_to_battery_this_hour = 0.0
            pv_to_grid_this_hour = 0.0
            pv_to_load_this_hour = 0.0
            pv_curtailed_this_hour = 0.0

            load_from_pv_this_hour = 0.0
            load_from_battery_this_hour = 0.0
            load_from_grid_this_hour = 0.0
            load_unserved_this_hour = 0.0

            grid_to_battery_this_hour = 0.0
            pv_for_battery_charging = 0.0
            battery_served_load = False
            load_discharge_this_hour = 0.0
            trading_discharge_this_hour = 0.0

            actual_charge_this_hour = 0.0
            actual_trading_discharge_this_hour = 0.0
            actual_load_discharge_this_hour = 0.0

            # ====================================================================
            # FASE 1: ALLOCAZIONE PV AL CARICO
            # ====================================================================
            pv_to_load_this_hour = min(alpha_pv_load * pv_available, load_required)
            load_from_pv_this_hour = pv_to_load_this_hour
            pv_remaining = pv_available - pv_to_load_this_hour
            load_remaining = load_required - pv_to_load_this_hour

            # ====================================================================
            # FASE 2: VINCOLO XOR ESPLICITO
            # ====================================================================
            if p_batt_trading_requested > 0.001:
                charge_request = p_batt_trading_requested
                discharge_request_trading = 0.0
                discharge_request_load = 0.0
                action_mode = "CHARGE"
            elif p_batt_trading_requested < -0.001:
                charge_request = 0.0
                discharge_request_trading = -p_batt_trading_requested
                discharge_request_load = p_batt_load_requested
                action_mode = "DISCHARGE"
            elif p_batt_load_requested > 0.001:
                charge_request = 0.0
                discharge_request_trading = 0.0
                discharge_request_load = p_batt_load_requested
                action_mode = "DISCHARGE"
            else:
                charge_request = 0.0
                discharge_request_trading = 0.0
                discharge_request_load = 0.0
                action_mode = "IDLE"

            total_discharge_request = discharge_request_trading + discharge_request_load

            max_power_c_rate = self.battery.nominal_capacity * self.battery.max_c_rate
            max_power_physical = min(self.battery.max_power, max_power_c_rate)
            max_discharge_soc = (
                                            self.battery.soc - self.battery.soc_min) * self.battery.trading_capacity * self.battery.discharge_efficiency
            max_energy_storable = (self.battery.soc_max - self.battery.soc) * self.battery.trading_capacity

            # ====================================================================
            # FASE 3: ESECUZIONE AZIONI BATTERIA CON VINCOLO POD
            # ====================================================================

            if action_mode == "DISCHARGE" and total_discharge_request > 0.001:
                # ===== SCARICA =====
                actual_discharge_total = min(total_discharge_request, max_power_physical, max_discharge_soc)

                if actual_discharge_total > 0.001:
                    ratio_trading = discharge_request_trading / total_discharge_request
                    ratio_load = discharge_request_load / total_discharge_request

                    discharge_for_trading = actual_discharge_total * ratio_trading
                    discharge_for_load_raw = actual_discharge_total * ratio_load
                    discharge_for_load = min(discharge_for_load_raw, load_remaining)

                    # VINCOLO POD: Limita scarica trading per immissione rete
                    pod_available_for_injection = POD_POWER_MW - grid_injection_this_hour
                    discharge_for_trading = min(discharge_for_trading, pod_available_for_injection)

                    actual_discharge_used = discharge_for_trading + discharge_for_load

                    actual_trading_discharge_this_hour = discharge_for_trading
                    actual_load_discharge_this_hour = discharge_for_load

                    # Esegui scarica fisica
                    total_energy_discharge = actual_discharge_used / self.battery.discharge_efficiency
                    new_soc = self.battery.soc - (total_energy_discharge / self.battery.capacity)
                    self.battery.soc = max(new_soc, self.battery.soc_min)
                    self.battery.throughput_kwh += total_energy_discharge * 1000

                    # RICAVO: Vendita batteria alla rete
                    if discharge_for_trading > 0.001:
                        revenue_discharge = discharge_for_trading * price_sell
                        cumulative_profit += revenue_discharge
                        trading_discharge_this_hour = discharge_for_trading
                        grid_injection_this_hour += discharge_for_trading

                    # AUTOCONSUMO: Batteria al carico
                    if discharge_for_load > 0.001:
                        load_from_battery_this_hour = discharge_for_load
                        load_remaining -= discharge_for_load
                        battery_served_load = True
                        load_discharge_this_hour = discharge_for_load

            elif action_mode == "CHARGE" and charge_request > 0.001:
                # ===== CARICA =====
                if self.battery.soc < self.battery.soc_max:
                    max_power_charge = max_energy_storable / (1.0 * self.battery.charge_efficiency)
                    actual_power = min(charge_request, max_power_charge, max_power_physical)

                    if actual_power > 0.01:
                        energy_needed = actual_power * 1.0
                        actual_charge_this_hour = actual_power

                        # Priorità 1: PV (TIME-SHIFTING)
                        if pv_remaining > 0:
                            energy_from_pv = min(energy_needed, pv_remaining)
                            self.battery.charge(energy_from_pv / 1.0, dt=1.0, source='pv')
                            pv_to_battery_this_hour += energy_from_pv
                            pv_for_battery_charging += energy_from_pv
                            energy_needed -= energy_from_pv
                            pv_remaining -= energy_from_pv

                        # Priorità 2: Rete CON VINCOLO POD
                        if energy_needed > 0.01:
                            # VINCOLO POD: Limita prelievo da rete
                            pod_available_for_withdrawal = POD_POWER_MW - grid_withdrawal_this_hour
                            energy_from_grid_allowed = min(energy_needed, pod_available_for_withdrawal)

                            if energy_from_grid_allowed > 0.01:
                                energy_from_grid = self.battery.charge(energy_from_grid_allowed / 1.0, dt=1.0,
                                                                       source='grid')
                                grid_to_battery_this_hour += energy_from_grid
                                grid_withdrawal_this_hour += energy_from_grid

                                # COSTO: Acquisto energia dalla rete
                                cost_grid_charge = energy_from_grid * price_buy
                                cumulative_profit -= cost_grid_charge

            # ====================================================================
            # FASE 4: CARICO RESIDUO DALLA RETE CON VINCOLO POD
            # ====================================================================
            if load_remaining > 0.001:
                # VINCOLO POD: Limita prelievo per carico
                pod_available_for_withdrawal = POD_POWER_MW - grid_withdrawal_this_hour
                load_from_grid_allowed = min(load_remaining, pod_available_for_withdrawal)

                if load_from_grid_allowed > 0.001:
                    load_from_grid_this_hour = load_from_grid_allowed
                    grid_withdrawal_this_hour += load_from_grid_allowed

                    # COSTO: Acquisto energia dalla rete per carico
                    cost_grid_load = load_from_grid_this_hour * price_buy
                    cumulative_profit -= cost_grid_load

                # NUOVO: Carico non servito causa POD
                load_unserved_this_hour = load_remaining - load_from_grid_allowed
                if load_unserved_this_hour > 0.001:
                    # PENALITÀ PESANTE: Carico non servito
                    penalty = load_unserved_this_hour * price_buy * 000.0
                    cumulative_profit -= penalty
                    print(f"⚠️  Ora {current_hour}: Carico non servito {load_unserved_this_hour:.3f} MWh (POD limit)")

            # Registra decisione batteria per carico
            if self.load_profile and (load_from_battery_this_hour > 0.001 or load_from_grid_this_hour > 0.001):
                self.load_profile.register_battery_decision(battery_served_load)

            # ====================================================================
            # FASE 5: VENDITA PV RESIDUO CON VINCOLO POD
            # ====================================================================
            if pv_remaining > 0.001:
                # VINCOLO POD: Limita immissione PV
                pod_available_for_injection = POD_POWER_MW - grid_injection_this_hour
                pv_to_grid_allowed = min(pv_remaining, pod_available_for_injection)

                if pv_to_grid_allowed > 0.001:
                    pv_to_grid_this_hour += pv_to_grid_allowed
                    grid_injection_this_hour += pv_to_grid_allowed

                    # RICAVO: Vendita PV alla rete
                    revenue_pv = pv_to_grid_allowed * price_sell
                    cumulative_profit += revenue_pv

                # NUOVO: PV curtailed causa POD
                pv_curtailed_this_hour = pv_remaining - pv_to_grid_allowed
                if pv_curtailed_this_hour > 0.001:
                    # Penalità leggera: energia persa ma non costo diretto
                    penalty = pv_curtailed_this_hour * price_sell * 0.0
                    cumulative_profit -= penalty
                    if pv_curtailed_this_hour > 0.1:  # Log solo se significativo
                        print(f"⚠️  Ora {current_hour}: PV curtailed {pv_curtailed_this_hour:.3f} MWh (POD limit)")

            # ====================================================================
            # TRACKING E REGISTRAZIONE
            # ====================================================================
            if self.pv_system and (pv_to_battery_this_hour > 0 or pv_to_grid_this_hour > 0 or pv_to_load_this_hour > 0):
                self.pv_system.allocate_energy(pv_to_battery_this_hour, pv_to_grid_this_hour, pv_to_load_this_hour)
                if pv_curtailed_this_hour > 0:
                    self.pv_system.curtailed_energy_mwh += pv_curtailed_this_hour

            if self.load_profile:
                self.load_profile.register_supply(load_from_pv_this_hour, load_from_battery_this_hour,
                                                  load_from_grid_this_hour)

            macse_available = self.battery.check_macse_availability() if MACSE_ENABLED else False
            if MACSE_ENABLED:
                self.battery.update_macse_availability(macse_available)

            # Converti scarica in negativo per compatibilità grafici
            net_trading_action = actual_charge_this_hour if actual_charge_this_hour > 0 else -actual_trading_discharge_this_hour

            # ====================================================================
            # APPEND RISULTATI + POD TRACKING
            # ====================================================================
            actions_trading_effective.append(net_trading_action)
            actions_alpha_pv.append(alpha_pv_load)
            actions_p_load_effective.append(actual_load_discharge_this_hour)

            soc_history.append(self.battery.get_soc())
            capacity_history.append(self.battery.capacity)
            soh_history.append(self.battery.get_soh())
            profits_history.append(cumulative_profit)
            macse_availability_history.append(self.battery.get_macse_availability_factor() if MACSE_ENABLED else 0)

            pv_production_history.append(pv_production[current_hour])
            pv_to_battery_history.append(pv_to_battery_this_hour)
            pv_to_grid_history.append(pv_to_grid_this_hour)
            pv_to_load_history.append(pv_to_load_this_hour)
            pv_curtailed_history.append(pv_curtailed_this_hour)  # NUOVO

            load_demand_history.append(load_demand[current_hour])
            load_from_pv_history.append(load_from_pv_this_hour)
            load_from_battery_history.append(load_from_battery_this_hour)
            load_from_grid_history.append(load_from_grid_this_hour)
            load_unserved_history.append(load_unserved_this_hour)  # NUOVO

            energy_from_grid_to_battery_history.append(grid_to_battery_this_hour)
            energy_from_pv_to_battery_history.append(pv_for_battery_charging)
            price_buy_history.append(price_buy)

            battery_decision_history.append(1 if battery_served_load else 0)
            load_discharge_history.append(load_discharge_this_hour)
            trading_discharge_history.append(trading_discharge_this_hour)

            # NUOVO: POD tracking
            grid_withdrawal_history.append(grid_withdrawal_this_hour)
            grid_injection_history.append(grid_injection_this_hour)
            pod_violated = (grid_withdrawal_this_hour > POD_POWER_MW + 0.001) or (
                        grid_injection_this_hour > POD_POWER_MW + 0.001)
            pod_violation_history.append(1 if pod_violated else 0)

            current_hour += self.step_hours

        # ========================================================================
        # AGGIORNAMENTO FINALE DEGRADO
        # ========================================================================
        self.battery.update_degradation()

        # Statistiche POD
        total_pod_violations = sum(pod_violation_history)
        total_pv_curtailed = sum(pv_curtailed_history)
        total_load_unserved = sum(load_unserved_history)

        print("\nSimulazione completata!")
        print(f"Profitto finale: {cumulative_profit:.2f} €")
        print(f"\n STATISTICHE POD:")
        print(f"  • Violazioni POD: {total_pod_violations} ore su {len(pod_violation_history)}")
        print(f"  • PV curtailed: {total_pv_curtailed:.2f} MWh")
        print(f"  • Carico non servito: {total_load_unserved:.2f} MWh")
        if total_pod_violations > 0:
            print(f"  ⚠️  ATTENZIONE: Rilevate {total_pod_violations} violazioni POD")

        # ========================================================================
        # CREAZIONE DATAFRAME RISULTATI CON COLONNE POD
        # ========================================================================
        results_df = prices_df.copy()
        pad_length = len(results_df) - len(actions_trading_effective)

        results_df['Prezzo_Acquisto_€/MWh'] = price_buy_history + [price_buy_history[-1]] * pad_length

        # Azioni PSO effettive
        results_df['Azione_Trading_MW'] = actions_trading_effective + [0] * pad_length
        results_df['Azione_Alpha_PV_Load'] = actions_alpha_pv + [0] * pad_length
        results_df['Azione_P_Batt_Load_MW'] = actions_p_load_effective + [0] * pad_length

        # Stati batteria
        results_df['SOC'] = soc_history + [soc_history[-1]] * pad_length
        results_df['Capacita_MWh'] = capacity_history + [capacity_history[-1]] * pad_length
        results_df['SOH_%'] = soh_history + [soh_history[-1]] * pad_length

        # Economia
        results_df['Profitto_Euro'] = profits_history + [profits_history[-1]] * pad_length
        results_df['MACSE_Availability'] = macse_availability_history + [macse_availability_history[-1]] * pad_length

        # PV + NUOVO: curtailment
        results_df['PV_Production_MWh'] = pv_production_history + [0] * pad_length
        results_df['PV_to_Battery_MWh'] = pv_to_battery_history + [0] * pad_length
        results_df['PV_to_Grid_MWh'] = pv_to_grid_history + [0] * pad_length
        results_df['PV_to_Load_MWh'] = pv_to_load_history + [0] * pad_length
        results_df['PV_Curtailed_MWh'] = pv_curtailed_history + [0] * pad_length  # NUOVO

        # Load + NUOVO: unserved
        results_df['Load_Demand_MWh'] = load_demand_history + [0] * pad_length
        results_df['Load_from_PV_MWh'] = load_from_pv_history + [0] * pad_length
        results_df['Load_from_Battery_MWh'] = load_from_battery_history + [0] * pad_length
        results_df['Load_from_Grid_MWh'] = load_from_grid_history + [0] * pad_length
        results_df['Load_Unserved_MWh'] = load_unserved_history + [0] * pad_length  # NUOVO

        # Energy sources
        results_df['Energy_from_Grid_MWh'] = energy_from_grid_to_battery_history + [0] * pad_length
        results_df['Energy_from_PV_MWh'] = energy_from_pv_to_battery_history + [0] * pad_length

        # Decisioni
        results_df['Battery_Decision'] = battery_decision_history + [0] * pad_length
        results_df['Load_Discharge_MW'] = load_discharge_history + [0] * pad_length
        results_df['Trading_Discharge_MW'] = trading_discharge_history + [0] * pad_length

        # NUOVO: POD tracking
        results_df['Grid_Withdrawal_MW'] = grid_withdrawal_history + [0] * pad_length
        results_df['Grid_Injection_MW'] = grid_injection_history + [0] * pad_length
        results_df['POD_Violation'] = pod_violation_history + [0] * pad_length

        # Retrocompatibilità
        results_df['Azione_MW'] = actions_trading_effective + [0] * pad_length

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
                       macse_penalty, macse_bonus, battery_investment, simulation_time, baseline_scenario=None):
    """
    Esporta risultati simulazione in formato JSON con confronto baseline
    """
    actions = results_df['Azione_Trading_MW'].values
    prices_sell = results_df['€/MWh'].values
    prices_buy = results_df['Prezzo_Acquisto_€/MWh'].values

    # ========================================================================
    # STATISTICHE PV
    # ========================================================================
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

    # ========================================================================
    # STATISTICHE CARICO
    # ========================================================================
    load_stats = load_profile.get_statistics() if load_profile else {
        'total_energy_required_mwh': 0,
        'energy_from_pv_mwh': 0,
        'energy_from_battery_mwh': 0,
        'energy_from_grid_mwh': 0,
        'pv_coverage_percent': 0,
        'battery_coverage_percent': 0,
        'grid_dependency_percent': 0,
        'battery_served_decisions_percent': 0,
        'grid_served_decisions_percent': 0,
        'battery_served_count': 0,
        'grid_served_count': 0,
        'total_decisions': 0
    }

    # ========================================================================
    # STATISTICHE TRADING
    # ========================================================================
    charge_hours = np.sum(actions > 0.01)
    discharge_hours = np.sum(actions < -0.01)
    idle_hours = len(actions) - charge_hours - discharge_hours
    total_energy_charged = np.sum(actions[actions > 0] * 1.0)
    total_energy_discharged = np.sum(np.abs(actions[actions < 0]) * 1.0)

    # ========================================================================
    # CALCOLI ECONOMICI
    # ========================================================================
    total_revenue = trading_profit + macse_revenue
    annual_profit = total_revenue
    roi_percent = (annual_profit / battery_investment) * 100 if battery_investment > 0 else 0
    payback_years = battery_investment / annual_profit if annual_profit > 0 else float('inf')

    # ========================================================================
    # CONFRONTO CON BASELINE
    # ========================================================================
    if baseline_scenario:
        battery_benefit = total_revenue - baseline_scenario['net_balance']
        benefit_percent = (battery_benefit / abs(baseline_scenario['net_balance']) * 100) if baseline_scenario[
                                                                                                 'net_balance'] != 0 else 0

        autosufficienza_with_bess = 100 - load_stats['grid_dependency_percent']
        delta_autosufficienza = autosufficienza_with_bess - baseline_scenario['autosufficienza_percent']
        delta_grid_dependency = baseline_scenario['total_load_from_grid'] - load_stats['energy_from_grid_mwh']
    else:
        battery_benefit = 0
        benefit_percent = 0
        delta_autosufficienza = 0
        delta_grid_dependency = 0

    # ========================================================================
    # COSTRUZIONE JSON
    # ========================================================================
    results_json = {
        "simulation_info": {
            "version": "2.7.0-AUTONOMOUS-MULTIDIM",
            "feature": "Fully Autonomous Multi-Dimensional PSO - Zero Hardcoded Logic",
            "description": "PSO decides all allocations: [p_batt_trading, alpha_pv_load, p_batt_load]",
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
            "grid_dependency_percent": load_stats['grid_dependency_percent'],
            "autonomous_decisions": {
                "battery_served_load_percent": load_stats['battery_served_decisions_percent'],
                "grid_served_load_percent": load_stats['grid_served_decisions_percent'],
                "battery_served_count": load_stats['battery_served_count'],
                "grid_served_count": load_stats['grid_served_count'],
                "total_decision_hours": load_stats['total_decisions']
            },
        "point_of_delivery": {
                "pod_power_mw": POD_POWER_MW,
                "total_violations": int(results_df['POD_Violation'].sum()),
                "violation_hours": int(results_df['POD_Violation'].sum()),
                "total_hours": len(results_df),
                "violation_rate_percent": float(results_df['POD_Violation'].sum() / len(results_df) * 100),
                "pv_curtailed_mwh": float(results_df['PV_Curtailed_MWh'].sum()),
                "load_unserved_mwh": float(results_df['Load_Unserved_MWh'].sum()),
                "max_grid_withdrawal_mw": float(results_df['Grid_Withdrawal_MW'].max()),
                "max_grid_injection_mw": float(results_df['Grid_Injection_MW'].max()),
                "avg_grid_withdrawal_mw": float(results_df['Grid_Withdrawal_MW'].mean()),
                "avg_grid_injection_mw": float(results_df['Grid_Injection_MW'].mean()),
                "energy_lost_to_pod_mwh": float(
                    results_df['PV_Curtailed_MWh'].sum() + results_df['Load_Unserved_MWh'].sum())
            },

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

        },

        # ====================================================================
        # NUOVO: SEZIONE CONFRONTO CON BASELINE
        # ====================================================================
        "baseline_comparison": {
            "scenario_without_battery": {
                "description": "Scenario base senza sistema di accumulo",
                "net_balance_euro": float(baseline_scenario['net_balance']) if baseline_scenario else 0,
                "total_costs_euro": float(baseline_scenario['total_cost_buy']) if baseline_scenario else 0,
                "total_revenues_euro": float(baseline_scenario['total_revenue_sell']) if baseline_scenario else 0,
                "energy_allocation": {
                    "pv_to_load_mwh": float(baseline_scenario['total_pv_to_load']) if baseline_scenario else 0,
                    "pv_to_grid_mwh": float(baseline_scenario['total_pv_to_grid']) if baseline_scenario else 0,
                    "load_from_grid_mwh": float(baseline_scenario['total_load_from_grid']) if baseline_scenario else 0,
                    "total_load_mwh": float(baseline_scenario['total_load_required']) if baseline_scenario else 0
                },
                "autosufficienza_percent": float(baseline_scenario['autosufficienza_percent']) if baseline_scenario else 0
            },

            "scenario_with_battery": {
                "description": "Scenario ottimizzato con sistema di accumulo",
                "net_balance_euro": float(total_revenue),
                "autosufficienza_percent": float(100 - load_stats['grid_dependency_percent']) if load_stats[
                                                                                                     'total_energy_required_mwh'] > 0 else 0
            },

            "battery_benefits": {
                "economic_benefit_euro": float(battery_benefit),
                "benefit_percent": float(benefit_percent),
                "is_profitable": bool(battery_benefit > 0),
                "autosufficienza_improvement_points": float(delta_autosufficienza),
                "grid_dependency_reduction_mwh": float(delta_grid_dependency),
                "interpretation": {
                    "economic": "Profitable" if battery_benefit > 0 else "Not profitable",
                    "energy": "Increased self-sufficiency" if delta_autosufficienza > 0 else "No change in self-sufficiency"
                }
            },

            "key_metrics": {
                "break_even_point_years": float(payback_years) if payback_years != float('inf') else None,
                "annual_savings_euro": float(battery_benefit),
                "energy_independence_gain_percent": float(delta_autosufficienza),
                "load_served_by_battery_percent": float(load_stats['battery_coverage_percent']) if load_stats[
                                                                                                       'total_energy_required_mwh'] > 0 else 0
            }
        },

        # ====================================================================
        # METADATI DECISIONI PSO
        # ====================================================================
        "pso_decision_space": {
            "dimensions": 3,
            "decision_variables": [
                {
                    "name": "p_batt_trading",
                    "description": "Battery power for trading (MW)",
                    "range": [-battery.max_power, battery.max_power],
                    "unit": "MW"
                },
                {
                    "name": "alpha_pv_load",
                    "description": "Fraction of PV allocated to load",
                    "range": [0, 1],
                    "unit": "fraction"
                },
                {
                    "name": "p_batt_load",
                    "description": "Battery power for load service (MW)",
                    "range": [0, battery.max_power],
                    "unit": "MW"
                }
            ],
            "optimization_objective": "Maximize: (revenues - costs - degradation - load_penalty)",
            "hardcoded_logic": "None - fully autonomous decision making"
        }
    }

    # ========================================================================
    # SALVATAGGIO JSON
    # ========================================================================
    json_file = os.path.join('results',
                             f'simulation_results_{battery.technology.lower().replace("-", "_")}_v270_autonomous.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"✓ JSON salvato: {json_file}")

    return results_json


# ========================================================================================================
# SEZIONE 9: GRAFICI (placeholder - implementa come vuoi)
# ========================================================================================================
def create_detailed_monthly_pv_plots(results_df, battery, pv_system):
    """
    Lorenzo Giannuzzo: GRAFICI DETTAGLIATI MENSILI con IMPATTO PV
    VERSIONE CORRETTA - Usa colonne DataFrame corrette
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
        charge_from_grid = []
        charge_from_pv = []
        for idx, row in df_giorno.iterrows():
            if row['Azione_Trading_MW'] > 0.01:  # Usa colonna corretta
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

        # Subplot 3: AZIONI BATTERIA (CORRETTO v3.2)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3_twin = ax3.twinx()

        # USA DIRETTAMENTE LE COLONNE CORRETTE DAL DATAFRAME
        charge_pv_bars = []
        charge_grid_bars = []
        discharge_load_bars = []
        discharge_trading_bars = []

        for idx, row in df_giorno.iterrows():
            # Usa Azione_Trading_MW che contiene già l'azione trading corretta
            trading_action = row['Azione_Trading_MW']  # Positivo=carica, Negativo=scarica trading

            # Usa le colonne dedicate già presenti nel DataFrame
            load_discharge = row.get('Load_Discharge_MW', 0.0)
            trading_discharge = row.get('Trading_Discharge_MW', 0.0)

            # Carica (trading_action positivo)
            if trading_action > 0.001:
                # Recupera fonti di carica
                pv_to_batt = row.get('PV_to_Battery_MWh', 0.0)
                grid_to_batt = row.get('Energy_from_Grid_MWh', 0.0)

                charge_pv_bars.append(pv_to_batt)
                charge_grid_bars.append(grid_to_batt)
                discharge_load_bars.append(0.0)
                discharge_trading_bars.append(0.0)

            # Scarica trading (trading_action negativo)
            elif trading_action < -0.001:
                charge_pv_bars.append(0.0)
                charge_grid_bars.append(0.0)

                # Scarica trading (già registrata correttamente)
                discharge_trading_bars.append(-trading_discharge)  # Negativo per grafico

                # Scarica carico (può coesistere con trading se erano entrambi richiesti)
                if load_discharge > 0.001:
                    discharge_load_bars.append(-load_discharge)
                else:
                    discharge_load_bars.append(0.0)

            # Solo scarica per carico (nessun trading)
            elif load_discharge > 0.001:
                charge_pv_bars.append(0.0)
                charge_grid_bars.append(0.0)
                discharge_trading_bars.append(0.0)
                discharge_load_bars.append(-load_discharge)

            # IDLE
            else:
                charge_pv_bars.append(0.0)
                charge_grid_bars.append(0.0)
                discharge_load_bars.append(0.0)
                discharge_trading_bars.append(0.0)

        width = 0.8

        # Plotta SCARICA per TRADING (viola/rosso)
        ax3.bar(df_giorno['Ora'], discharge_trading_bars, width=width,
                color='#E63946', alpha=0.8, edgecolor='black', linewidth=1,
                label='Scarica Trading')

        # Plotta SCARICA per CARICO (magenta)
        ax3.bar(df_giorno['Ora'], discharge_load_bars, width=width,
                bottom=discharge_trading_bars,
                color='#9D4EDD', alpha=0.8, edgecolor='black', linewidth=1,
                label='Scarica Carico')

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
        ax3.set_title('Azioni Batteria (Carico vs Trading) vs Prezzo', fontsize=13, fontweight='bold')

        # Legends
        ax3.legend(fontsize=9, loc='upper left', framealpha=0.95)
        ax3_twin.legend(fontsize=9, loc='upper right', framealpha=0.95)

        # Grid e limiti
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_xticks(range(0, 24, 2))

        # Forza limiti corretti asse Y
        all_actions = (discharge_trading_bars + discharge_load_bars +
                       [sum(x) for x in zip(charge_pv_bars, charge_grid_bars)])
        if any(x != 0 for x in all_actions):
            y_max = max(abs(min(all_actions)), max(all_actions)) * 1.1
            ax3.set_ylim(-y_max, y_max)

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
            # Costi: acquisto da rete
            if row['Energy_from_Grid_MWh'] > 0:
                profit_hour -= row['Energy_from_Grid_MWh'] * row.get('Prezzo_Acquisto_€/MWh', row['€/MWh'])
            if 'Load_from_Grid_MWh' in row and row['Load_from_Grid_MWh'] > 0:
                profit_hour -= row['Load_from_Grid_MWh'] * row.get('Prezzo_Acquisto_€/MWh', row['€/MWh'])

            # Ricavi: vendite a rete
            if row.get('Trading_Discharge_MW', 0) > 0:
                profit_hour += row['Trading_Discharge_MW'] * row['€/MWh']
            if row['PV_to_Grid_MWh'] > 0:
                profit_hour += row['PV_to_Grid_MWh'] * row['€/MWh']

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

        #plt.tight_layout()
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


def calculate_baseline_scenario(prices_sell, prices_buy, pv_production, load_demand):
    """
    Calcola scenario BASELINE senza batteria CON VINCOLO POD
    """
    total_cost_buy = 0.0
    total_revenue_sell = 0.0

    total_pv_to_load = 0.0
    total_pv_to_grid = 0.0
    total_load_from_grid = 0.0
    total_load_required = 0.0

    # NUOVO: tracking POD baseline
    total_pv_curtailed_baseline = 0.0
    total_load_unserved_baseline = 0.0

    n_hours = len(prices_sell)

    for h in range(n_hours):
        pv_available = pv_production[h] if PV_ENABLED else 0.0
        load_required = load_demand[h] if LOAD_ENABLED else 0.0
        price_sell = prices_sell[h]
        price_buy = prices_buy[h]

        total_load_required += load_required

        # LOGICA BASELINE: PV al carico prima
        pv_to_load = min(pv_available, load_required)
        pv_remaining = pv_available - pv_to_load
        load_remaining = load_required - pv_to_load

        total_pv_to_load += pv_to_load

        # PV eccesso venduto CON VINCOLO POD
        if pv_remaining > 0.001:
            pv_to_grid = min(pv_remaining, POD_POWER_MW)  # ✅ VINCOLO POD
            pv_curtailed = pv_remaining - pv_to_grid

            total_pv_to_grid += pv_to_grid
            total_revenue_sell += pv_to_grid * price_sell

            if pv_curtailed > 0.001:
                total_pv_curtailed_baseline += pv_curtailed
                # Penalità leggera per PV perso
                total_cost_buy += pv_curtailed * price_sell * 0.0

        # Carico residuo comprato da rete CON VINCOLO POD
        if load_remaining > 0.001:
            load_from_grid = min(load_remaining, POD_POWER_MW)  # ✅ VINCOLO POD
            load_unserved = load_remaining - load_from_grid

            total_load_from_grid += load_from_grid
            total_cost_buy += load_from_grid * price_buy

            if load_unserved > 0.001:
                total_load_unserved_baseline += load_unserved
                # Penalità pesante per carico non servito
                total_cost_buy += load_unserved * price_buy * 000.0

    net_balance = total_revenue_sell - total_cost_buy
    autosufficienza = (total_pv_to_load / total_load_required * 100) if total_load_required > 0 else 0

    return {
        'net_balance': net_balance,
        'total_cost_buy': total_cost_buy,
        'total_revenue_sell': total_revenue_sell,
        'total_pv_to_load': total_pv_to_load,
        'total_pv_to_grid': total_pv_to_grid,
        'total_load_from_grid': total_load_from_grid,
        'total_load_required': total_load_required,
        'autosufficienza_percent': autosufficienza,
        'pv_curtailed_baseline': total_pv_curtailed_baseline,  # NUOVO
        'load_unserved_baseline': total_load_unserved_baseline  # NUOVO
    }

# ========================================================================================================
# FUNZIONE MAIN
# ========================================================================================================
def main(file_name, file_name2, pv_file_name=None, load_file_name=None):
    file_path = os.path.join('data', file_name)
    file_path2 = os.path.join('data', file_name2)

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

    try:
        df2 = pd.read_excel(file_path2)
        if df2['€/MWh'].dtype == 'object':
            df2['€/MWh'] = df2['€/MWh'].astype(str).str.replace(',', '.').astype(float)
        print(f"✓ Prezzi vendita: {len(df2)} righe, media {df2['€/MWh'].mean():.2f} €/MWh")
    except Exception as e:
        print(f"❌ Errore caricamento prezzi: {e}")
        return


    pv_df = None
    pv_system = None
    if PV_ENABLED and pv_file_name:
        try:
            pv_file_path = os.path.join('data', pv_file_name)
            pv_df = pd.read_csv(pv_file_path, sep=';')
            pv_system = PhotovoltaicSystem()
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
    optimizer = PSOOptimizer(n_particles=50, n_iterations=200)
    simulator = RollingHorizonSimulator(battery, optimizer, pv_system=pv_system, load_profile=load_profile)

    start_time = datetime.now()
    results_df, trading_profit = simulator.simulate(df, df2, pv_df, load_df)
    end_time = datetime.now()

    macse_revenue, macse_base, macse_penalty, macse_bonus = calculate_macse_revenue(battery)
    total_system_profit = trading_profit + macse_revenue

    # ========================================================================
    # NUOVO: CALCOLO SCENARIO BASELINE (SENZA BATTERIA)
    # ========================================================================
    prices_sell = df['€/MWh'].values
    prices_buy = df2['€/MWh'].values

    if PV_ENABLED and pv_df is not None:
        pv_production = pv_df['P'].values / 1000.0  # kW → MWh
        if len(pv_production) < len(prices_sell):
            pv_production = np.pad(pv_production, (0, len(prices_sell) - len(pv_production)), 'constant')
        elif len(pv_production) > len(prices_sell):
            pv_production = pv_production[:len(prices_sell)]
    else:
        pv_production = np.zeros(len(prices_sell))

    if LOAD_ENABLED and load_df is not None:
        load_demand = load_df['value'].values / 1000.0  # kW → MWh
        if len(load_demand) < len(prices_sell):
            load_demand = np.pad(load_demand, (0, len(prices_sell) - len(load_demand)), 'constant')
        elif len(load_demand) > len(prices_sell):
            load_demand = load_demand[:len(prices_sell)]
    else:
        load_demand = np.zeros(len(prices_sell))

    baseline_scenario = calculate_baseline_scenario(prices_sell, prices_buy, pv_production, load_demand)

    print("\n" + "=" * 80)
    print("RISULTATI FINALI - CONFRONTO ECONOMICO")
    print("=" * 80)

    # ========================================================================
    # PRINT 1: SCENARIO SENZA BATTERIA (BASELINE)
    # ========================================================================
    print("\n" + "🔵 " * 40)
    print("SCENARIO 1: SENZA BATTERIA (BASELINE)")
    print("🔵 " * 40)
    print(f"\n📊 BILANCIO ENERGETICO:")
    print(f"  • Carico totale richiesto:        {baseline_scenario['total_load_required']:>10.2f} MWh")
    print(
        f"  • PV copre carico direttamente:   {baseline_scenario['total_pv_to_load']:>10.2f} MWh ({baseline_scenario['autosufficienza_percent']:.1f}%)")
    print(
        f"  • Carico coperto da rete:         {baseline_scenario['total_load_from_grid']:>10.2f} MWh ({baseline_scenario['total_load_from_grid'] / baseline_scenario['total_load_required'] * 100 if baseline_scenario['total_load_required'] > 0 else 0:.1f}%)")
    print(f"  • PV venduto a rete:              {baseline_scenario['total_pv_to_grid']:>10.2f} MWh")

    print(f"\n💰 BILANCIO ECONOMICO:")
    print(f"  • Costi acquisto energia:         {baseline_scenario['total_cost_buy']:>10,.2f} €  ❌")
    print(f"  • Ricavi vendita PV:              {baseline_scenario['total_revenue_sell']:>10,.2f} €  ✅")
    print(f"  • {'─' * 60}")

    baseline_sign = "✅" if baseline_scenario['net_balance'] >= 0 else "❌"
    print(f"  • BILANCIO NETTO (senza batteria): {baseline_scenario['net_balance']:>10,.2f} €  {baseline_sign}")
    print(f"  • Autosufficienza energetica:     {baseline_scenario['autosufficienza_percent']:>10.1f} %")

    # ========================================================================
    # PRINT 2: SCENARIO CON BATTERIA (SISTEMA OTTIMIZZATO)
    # ========================================================================
    print("\n" + "🟢 " * 40)
    print("SCENARIO 2: CON BATTERIA (SISTEMA OTTIMIZZATO)")
    print("🟢 " * 40)

    if LOAD_ENABLED and load_profile:
        load_stats = load_profile.get_statistics()
        print(f"\n📊 BILANCIO ENERGETICO:")
        print(f"  • Carico totale richiesto:        {load_stats['total_energy_required_mwh']:>10.2f} MWh")
        print(
            f"  • Carico da PV diretto:           {load_stats['energy_from_pv_mwh']:>10.2f} MWh ({load_stats['pv_coverage_percent']:.1f}%)")
        print(
            f"  • Carico da BATTERIA:             {load_stats['energy_from_battery_mwh']:>10.2f} MWh ({load_stats['battery_coverage_percent']:.1f}%)")
        print(
            f"  • Carico da rete:                 {load_stats['energy_from_grid_mwh']:>10.2f} MWh ({load_stats['grid_dependency_percent']:.1f}%)")
        autosufficienza_with_bess = 100 - load_stats['grid_dependency_percent']
        print(f"  • Autosufficienza energetica:     {autosufficienza_with_bess:>10.1f} %")

    if PV_ENABLED and pv_system:
        pv_stats = pv_system.get_statistics()
        print(f"\n☀️ UTILIZZO PV:")
        print(f"  • Produzione totale:              {pv_stats['total_production_mwh']:>10.2f} MWh")
        print(
            f"  • PV → Carico diretto:            {pv_stats['energy_to_load_mwh']:>10.2f} MWh ({pv_stats['load_service_percent']:.1f}%)")
        print(
            f"  • PV → Batteria (storage):        {pv_stats['energy_to_battery_mwh']:>10.2f} MWh ({pv_stats['battery_utilization_percent']:.1f}%)")
        print(
            f"  • PV → Vendita diretta:           {pv_stats['energy_to_grid_mwh']:>10.2f} MWh ({pv_stats['grid_sale_percent']:.1f}%)")

    print(f"\n🔋 STATO BATTERIA:")
    print(f"  • SOH finale:                     {battery.get_soh():>10.2f} %")
    print(f"  • Cicli equivalenti:              {battery.equivalent_cycles:>10.2f}")
    print(f"  • Throughput totale:              {battery.throughput_kwh:>10,.0f} kWh")
    print(f"  • Energia da rete → batteria:     {battery.energy_from_grid_mwh:>10.2f} MWh")
    print(f"  • Energia da PV → batteria:       {battery.energy_from_pv_mwh:>10.2f} MWh")

    print(f"\n💰 BILANCIO ECONOMICO:")
    print(f"  • Profitto trading batteria:      {trading_profit:>10,.2f} €")
    if MACSE_ENABLED:
        print(f"  • Ricavi servizi MACSE:           {macse_revenue:>10,.2f} €")
    print(f"  • {'─' * 60}")

    system_sign = "✅" if total_system_profit >= 0 else "❌"
    print(f"  • BILANCIO NETTO (con batteria):   {total_system_profit:>10,.2f} €  {system_sign}")

    if 'POD_Violation' in results_df.columns:
        print("\n" + "⚡ " * 40)
        print("ANALISI VINCOLO POD (POINT OF DELIVERY)")
        print("⚡ " * 40)

        total_violations = results_df['POD_Violation'].sum()
        violation_rate = (total_violations / len(results_df) * 100)
        total_pv_curtailed = results_df['PV_Curtailed_MWh'].sum()
        total_load_unserved = results_df['Load_Unserved_MWh'].sum()
        max_withdrawal = results_df['Grid_Withdrawal_MW'].max()
        max_injection = results_df['Grid_Injection_MW'].max()

        print(f"\n📊 LIMITE POD: {POD_POWER_MW} MW")
        print(f"\n🔴 VIOLAZIONI:")
        print(
            f"  • Ore con violazione:             {int(total_violations)} / {len(results_df)} ({violation_rate:.1f}%)")
        print(f"  • PV curtailed (perso):           {total_pv_curtailed:.2f} MWh")
        print(f"  • Carico non servito:             {total_load_unserved:.2f} MWh")

        print(f"\n📈 SCAMBI MASSIMI:")
        print(
            f"  • Max prelievo rete:              {max_withdrawal:.2f} MW {'⚠️ VIOLA POD' if max_withdrawal > POD_POWER_MW else '✅'}")
        print(
            f"  • Max immissione rete:            {max_injection:.2f} MW {'⚠️ VIOLA POD' if max_injection > POD_POWER_MW else '✅'}")

        if baseline_scenario and 'pv_curtailed_baseline' in baseline_scenario:
            print(f"\n🔵 CONFRONTO CON BASELINE:")
            print(f"  • PV curtailed SENZA batteria:    {baseline_scenario['pv_curtailed_baseline']:.2f} MWh")
            print(f"  • PV curtailed CON batteria:      {total_pv_curtailed:.2f} MWh")
            delta_curtailment = baseline_scenario['pv_curtailed_baseline'] - total_pv_curtailed
            print(
                f"  • Riduzione curtailment:          {delta_curtailment:.2f} MWh {'✅' if delta_curtailment > 0 else '❌'}")

    # ========================================================================
    # CONFRONTO E DELTA
    # ========================================================================
    print("\n" + "⚡ " * 40)
    print("CONFRONTO E BENEFICI BATTERIA")
    print("⚡ " * 40)

    delta_economic = total_system_profit - baseline_scenario['net_balance']
    delta_percent = (delta_economic / abs(baseline_scenario['net_balance']) * 100) if baseline_scenario[
                                                                                          'net_balance'] != 0 else float(
        'inf')

    if LOAD_ENABLED and load_profile:
        delta_autosufficienza = autosufficienza_with_bess - baseline_scenario['autosufficienza_percent']
        delta_grid_dependency = baseline_scenario['total_load_from_grid'] - load_stats['energy_from_grid_mwh']
    else:
        delta_autosufficienza = 0
        delta_grid_dependency = 0

    print(f"\n💵 IMPATTO ECONOMICO:")
    print(f"  • Bilancio SENZA batteria:        {baseline_scenario['net_balance']:>10,.2f} €")
    print(f"  • Bilancio CON batteria:          {total_system_profit:>10,.2f} €")
    print(f"  • {'─' * 60}")

    delta_sign = "✅ MIGLIORAMENTO" if delta_economic > 0 else "❌ PEGGIORAMENTO"
    delta_arrow = "📈" if delta_economic > 0 else "📉"
    print(f"  • DELTA (beneficio batteria):     {delta_economic:>10,.2f} €  {delta_arrow} {delta_sign}")

    if baseline_scenario['net_balance'] != 0:
        print(f"  • Variazione percentuale:         {delta_percent:>10.1f} %")

    print(f"\n⚡ IMPATTO ENERGETICO:")
    print(f"  • Autosufficienza SENZA batteria: {baseline_scenario['autosufficienza_percent']:>10.1f} %")
    if LOAD_ENABLED and load_profile:
        print(f"  • Autosufficienza CON batteria:   {autosufficienza_with_bess:>10.1f} %")
        print(
            f"  • Incremento autosufficienza:     {delta_autosufficienza:>10.1f} punti %  {'✅' if delta_autosufficienza > 0 else '➖'}")
        print(
            f"  • Riduzione dipendenza rete:      {delta_grid_dependency:>10.2f} MWh  {'✅' if delta_grid_dependency > 0 else '❌'}")

    print(f"\n⏱️ PRESTAZIONI:")
    print(f"  • Tempo simulazione:              {(end_time - start_time).total_seconds():>10.1f} s")
    print(f"  • Ore simulate:                   {len(prices_sell):>10.0f} h")
    print(
        f"  • Velocità:                       {len(prices_sell) / (end_time - start_time).total_seconds():>10.1f} ore/s")

    print("\n" + "=" * 80)

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
    print("✓ SIMULAZIONE COMPLETATA")
    print("=" * 80)


if __name__ == "__main__":
    pv_file = pv_production_file if PV_ENABLED else None
    load_file_input = load_file if LOAD_ENABLED else None
    main(energy_selling_price_name,energy_buying_price_name, pv_file, load_file_input)