"""
------------------------------------------------------------------------------------------------------------------------
BATTERY ENERGY STORAGE SYSTEM (BESS) OPTIMIZATION - MULTI TECHNOLOGY + AUTONOMOUS PV INTEGRATION
Particle Swarm Optimization with Rolling Horizon, MACSE Integration and Autonomous Photovoltaic Strategy
Supporto per Litio-ione e Grafene + Sistema Fotovoltaico con Decisione Autonoma
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
    - **AUTONOMOUS PV: L'algoritmo decide autonomamente se usare PV per batteria o vendere direttamente**
Version: 2.3.0-PV-AUTONOMOUS
Date: November 2025
------------------------------------------------------------------------------------------------------------------------
CARATTERISTICHE VERSIONE 2.3.0-PV-AUTONOMOUS:
- Algoritmo PSO decide autonomamente allocazione energia PV
- Nessuna priorità forzata: sistema valuta economicamente ogni opzione
- PV può essere: venduto direttamente, usato per caricare batteria, o combinazione
- Decisione basata su: prezzo corrente, prezzo futuro atteso, SOC batteria, degrado
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
file_name = '20240101_20241231_PUN.xlsx'
pv_production_file = 'year_PV.csv'

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
    Modello sistema fotovoltaico con tracking completo allocazione energia
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

    def allocate_energy(self, to_battery_mwh, to_grid_mwh):
        """Registra allocazione energia PV"""
        self.energy_to_battery_mwh += to_battery_mwh
        self.energy_to_grid_mwh += to_grid_mwh

    def get_statistics(self):
        """Ritorna statistiche produzione PV"""
        total = self.total_production_mwh
        return {
            'total_production_mwh': total,
            'energy_to_battery_mwh': self.energy_to_battery_mwh,
            'energy_to_grid_mwh': self.energy_to_grid_mwh,
            'curtailed_energy_mwh': self.curtailed_energy_mwh,
            'battery_utilization_percent': (self.energy_to_battery_mwh / total * 100) if total > 0 else 0,
            'grid_sale_percent': (self.energy_to_grid_mwh / total * 100) if total > 0 else 0,
            'curtailment_percent': (self.curtailed_energy_mwh / total * 100) if total > 0 else 0
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
    Modello batteria con tracking separato carica da rete vs PV
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
        self.equivalent_cycles = self.throughput_kwh / (2 * self.nominal_capacity * 1000)
        capacity_percentage = degradation(self.equivalent_cycles)
        self.capacity = self.nominal_capacity * (capacity_percentage / 100.0)
        if MACSE_ENABLED:
            macse_percentage_original = MACSE_CAPACITY_MWH / self.nominal_capacity
            self.macse_capacity = self.capacity * macse_percentage_original
            self.trading_capacity = self.capacity - self.macse_capacity

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
# SEZIONE 6: OTTIMIZZATORE PSO CON DECISIONE AUTONOMA PV
# ========================================================================================================
class PSOOptimizer:
    """
    PSO con decisione COMPLETAMENTE AUTONOMA per allocazione PV:
    - L'algoritmo valuta economicamente ogni opzione
    - Nessuna priorità forzata: PV può essere venduto o usato per batteria
    - Decisione basata su: prezzo corrente, prezzo futuro, SOC, degrado

    LOGICA DECISIONALE:
    Azione > 0 (CARICA): algoritmo decide se usare PV o rete per caricare
    Azione < 0 (SCARICA): algoritmo decide se scaricare batteria, con PV venduto direttamente
    Azione ≈ 0 (IDLE): PV venduto direttamente al prezzo di mercato
    """

    def __init__(self, n_particles=50, n_iterations=150, w_start=0.95, w_end=0.2, c1=2.0, c2=2.0):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.stagnation_limit = 15

    def optimize(self, battery, prices, pv_production, horizon_hours=24):
        n_hours = min(horizon_hours, len(prices))
        max_power_limit = min(battery.trading_power, battery.get_max_power_by_crate())

        positions = self._smart_initialization(battery, prices[:n_hours], pv_production[:n_hours], max_power_limit)
        velocities = np.random.uniform(-1.0, 1.0, (self.n_particles, n_hours))
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self._evaluate(battery, p, prices[:n_hours], pv_production[:n_hours])
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
                score = self._evaluate(battery, positions[i], prices[:n_hours], pv_production[:n_hours])
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

    def _smart_initialization(self, battery, prices, pv_production, max_power):
        """Inizializzazione considerando sia prezzi che disponibilità PV"""
        n_hours = len(prices)
        positions = np.zeros((self.n_particles, n_hours))
        price_low = np.percentile(prices, 25)
        price_high = np.percentile(prices, 75)

        for i in range(self.n_particles):
            if i < self.n_particles // 3:
                # Strategia aggressiva
                for h in range(n_hours):
                    if prices[h] < price_low:
                        # Prezzo basso: carica (da rete o PV, deciderà l'evaluate)
                        positions[i, h] = np.random.uniform(0.5 * max_power, max_power)
                    elif prices[h] > price_high:
                        # Prezzo alto: scarica
                        positions[i, h] = np.random.uniform(-max_power, -0.5 * max_power)
                    else:
                        positions[i, h] = np.random.uniform(-0.3 * max_power, 0.3 * max_power)
            elif i < 2 * self.n_particles // 3:
                # Strategia moderata
                for h in range(n_hours):
                    if prices[h] < price_low:
                        positions[i, h] = np.random.uniform(0, 0.7 * max_power)
                    elif prices[h] > price_high:
                        positions[i, h] = np.random.uniform(-0.7 * max_power, 0)
                    else:
                        positions[i, h] = np.random.uniform(-0.2 * max_power, 0.2 * max_power)
            else:
                # Strategia random per esplorazione
                positions[i] = np.random.uniform(-max_power, max_power, n_hours)
        return positions

    def _evaluate(self, battery, actions, prices, pv_production):
        """
        LOGICA DECISIONALE AUTONOMA PER PV:

        L'algoritmo valuta TUTTE le opzioni disponibili e sceglie quella più profittevole:

        1. Quando PV è disponibile:
           - Opzione A: Vendere PV direttamente → profitto = pv_energy * price
           - Opzione B: Usare PV per caricare batteria → profitto futuro - costo degrado
           - Scelta: l'algoritmo confronta implicitamente attraverso la fitness

        2. Quando batteria deve caricare (action > 0):
           - Se PV disponibile: usa PV (gratis) invece di rete (a pagamento)
           - Se PV insufficiente: completa con rete
           - Valuta costo degrado in entrambi i casi

        3. Quando batteria deve scaricare (action < 0):
           - Scarica batteria al prezzo corrente
           - Vende anche PV disponibile al prezzo corrente

        4. Quando batteria idle (action ≈ 0):
           - Vende PV direttamente se disponibile

        NESSUNA PRIORITÀ FORZATA: ogni decisione è valutata economicamente
        """
        bat_sim = battery.copy()
        profit = 0.0

        for hour, (power, price, pv_energy) in enumerate(zip(actions, prices, pv_production)):
            pv_available = pv_energy if PV_ENABLED else 0.0

            if power > 0.01:  # RICHIESTA CARICA
                if bat_sim.soc >= bat_sim.soc_max:
                    # Batteria piena: vendi PV se disponibile
                    if pv_available > 0:
                        profit += pv_available * price
                    continue

                # Calcola energia caricabile
                max_energy_storable = (bat_sim.soc_max - bat_sim.soc) * bat_sim.trading_capacity
                max_power_available = max_energy_storable / (1.0 * bat_sim.charge_efficiency)
                actual_power = min(power, max_power_available)

                if actual_power > 0.01:
                    energy_needed = actual_power * 1.0

                    # VALUTAZIONE ECONOMICA: Usare PV per batteria o venderlo?
                    # L'algoritmo implicitamente decide attraverso la fitness:
                    # - Se usa PV per batteria: risparmia costo acquisto da rete
                    # - Se vende PV: guadagna immediatamente
                    # La scelta ottimale emerge dalle iterazioni PSO

                    if pv_available > 0:
                        # Opzione: usa PV per caricare (GRATIS, no costo acquisto)
                        energy_from_pv = min(energy_needed, pv_available)
                        bat_sim.charge(energy_from_pv / 1.0, dt=1.0, source='pv')
                        # Costo degrado per energia PV usata
                        degradation_cost_pv = energy_from_pv * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                        profit -= degradation_cost_pv

                        energy_needed -= energy_from_pv
                        pv_available -= energy_from_pv

                        # Surplus PV venduto direttamente
                        if pv_available > 0:
                            profit += pv_available * price

                    # Se serve ancora energia, compra da rete
                    if energy_needed > 0.01:
                        energy_from_grid = bat_sim.charge(energy_needed / 1.0, dt=1.0, source='grid')
                        profit -= energy_from_grid * price  # COSTO acquisto
                        degradation_cost_grid = energy_from_grid * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                        profit -= degradation_cost_grid
                else:
                    # Non può caricare: vendi PV
                    if pv_available > 0:
                        profit += pv_available * price

            elif power < -0.01:  # RICHIESTA SCARICA
                if bat_sim.soc <= bat_sim.soc_min:
                    # Batteria scarica: vendi solo PV se disponibile
                    if pv_available > 0:
                        profit += pv_available * price
                    continue

                max_energy_available = (bat_sim.soc - bat_sim.soc_min) * bat_sim.trading_capacity
                max_power_available = max_energy_available * bat_sim.discharge_efficiency / 1.0
                actual_power = min(-power, max_power_available)

                if actual_power > 0.01:
                    # Scarica batteria
                    energy_to_grid_battery = bat_sim.discharge(actual_power, dt=1.0)

                    # Vendi anche PV disponibile
                    total_energy_sold = energy_to_grid_battery + pv_available
                    profit += total_energy_sold * price

                    # Costo degrado solo per scarica batteria
                    degradation_cost = energy_to_grid_battery * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                    profit -= degradation_cost
                else:
                    # Non può scaricare: vendi solo PV
                    if pv_available > 0:
                        profit += pv_available * price

            else:  # IDLE (power ≈ 0)
                # Batteria ferma: vendi PV se disponibile
                if pv_available > 0:
                    profit += pv_available * price

        return profit


# ========================================================================================================
# SEZIONE 7: ROLLING HORIZON SIMULATOR CON PV
# ========================================================================================================
class RollingHorizonSimulator:
    def __init__(self, battery, optimizer, pv_system=None, horizon_hours=24, step_hours=1):
        self.battery = battery
        self.optimizer = optimizer
        self.pv_system = pv_system
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours

    def simulate(self, prices_df, pv_df=None):
        prices = prices_df['€/MWh'].values
        n_hours = len(prices)

        # Prepara produzione PV
        if PV_ENABLED and pv_df is not None and self.pv_system is not None:
            pv_production = np.array([self.pv_system.get_energy(irr, dt=1.0) for irr in pv_df['P'].values])
            if len(pv_production) < n_hours:
                pv_production = np.pad(pv_production, (0, n_hours - len(pv_production)), 'constant')
            elif len(pv_production) > n_hours:
                pv_production = pv_production[:n_hours]
        else:
            pv_production = np.zeros(n_hours)

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
        energy_from_grid_history = []
        energy_from_pv_history = []

        cumulative_profit = 0.0

        print("=" * 80)
        print("SIMULAZIONE BESS CON PSO, ROLLING HORIZON, MACSE E PV AUTONOMO")
        print("=" * 80)
        print(f"TECNOLOGIA BATTERIA: {self.battery.technology}")
        print(f"Capacita nominale: {self.battery.nominal_capacity} MWh")
        print(f"Potenza massima: {self.battery.max_power} MW")
        print(f"C-rate massimo: {self.battery.max_c_rate}C")
        print(f"DoD utilizzabile: {self.battery.dod * 100}%")
        print(f"SOC range: {self.battery.soc_min * 100}% - {self.battery.soc_max * 100}%")
        print(f"Efficienza round-trip: {self.battery.efficiency * 100:.1f}%")
        print(f"EOL cycles: {self.battery.eol_cycles} cicli")

        if PV_ENABLED and self.pv_system is not None:
            print(f"\nFOTOVOLTAICO ABILITATO (DECISIONE AUTONOMA):")
            print(f"  - Potenza nominale: {self.pv_system.nominal_power_kwp:.0f} kWp")
            print(f"  - Efficienza totale: {self.pv_system.total_efficiency * 100:.1f}%")
            print(f"  - Strategia: COMPLETAMENTE AUTONOMA")
            print(f"  - Algoritmo decide: vendita diretta vs carica batteria")
        else:
            print(f"\nFOTOVOLTAICO DISABILITATO")

        if MACSE_ENABLED:
            print(f"\nMACSE ABILITATO:")
            print(f"  - Capacita MACSE: {self.battery.macse_capacity:.2f} MWh")
            print(f"  - Capacita trading: {self.battery.trading_capacity:.2f} MWh")

        print(f"\nOrizzonte: {self.horizon_hours} ore")
        print(f"Ore totali: {n_hours}")
        print("=" * 80)
        print()
        print(f"Progresso: 0% - Inizio ottimizzazione...")

        current_hour = 0
        last_progress = 0
        degradation_update_interval = 24

        while current_hour < n_hours:
            progress = int((current_hour / n_hours) * 100)
            if progress >= last_progress + 20:
                avail_factor = self.battery.get_macse_availability_factor() if MACSE_ENABLED else 0
                print(f"Progresso: {progress}% ({current_hour}/{n_hours} h) - SOH: {self.battery.get_soh():.2f}%")
                last_progress = progress

            if current_hour % degradation_update_interval == 0 and current_hour > 0:
                self.battery.update_degradation()

            end_hour = min(current_hour + self.horizon_hours, n_hours)
            window_prices = prices[current_hour:end_hour]
            window_pv = pv_production[current_hour:end_hour]

            # OTTIMIZZAZIONE: PSO decide autonomamente allocazione PV
            optimal_actions = self.optimizer.optimize(
                self.battery, window_prices, window_pv, self.horizon_hours
            )

            action = optimal_actions[0]
            price = prices[current_hour]
            pv_energy_available = pv_production[current_hour]

            # Tracking per questa ora
            pv_to_battery_this_hour = 0.0
            pv_to_grid_this_hour = 0.0
            grid_to_battery_this_hour = 0.0

            # ESECUZIONE DECISIONE PSO
            if action > 0.01:  # CARICA
                if self.battery.soc < self.battery.soc_max:
                    max_energy_storable = (self.battery.soc_max - self.battery.soc) * self.battery.trading_capacity
                    max_power_available = max_energy_storable / (1.0 * self.battery.charge_efficiency)
                    actual_power = min(action, max_power_available)

                    if actual_power > 0.01:
                        energy_needed = actual_power * 1.0

                        # Usa PV disponibile (già ottimizzato da PSO)
                        if pv_energy_available > 0:
                            energy_from_pv = min(energy_needed, pv_energy_available)
                            self.battery.charge(energy_from_pv / 1.0, dt=1.0, source='pv')
                            pv_to_battery_this_hour = energy_from_pv
                            energy_needed -= energy_from_pv
                            pv_energy_available -= energy_from_pv

                            # Surplus PV venduto
                            if pv_energy_available > 0:
                                profit_pv = pv_energy_available * price
                                cumulative_profit += profit_pv
                                pv_to_grid_this_hour = pv_energy_available
                                if self.pv_system:
                                    self.pv_system.allocate_energy(0, pv_energy_available)
                                pv_energy_available = 0

                        # Completa con rete
                        if energy_needed > 0.01:
                            energy_from_grid = self.battery.charge(energy_needed / 1.0, dt=1.0, source='grid')
                            grid_to_battery_this_hour = energy_from_grid
                            profit = -energy_from_grid * price
                            cumulative_profit += profit
                    else:
                        # Vendi PV se disponibile
                        if pv_energy_available > 0:
                            profit_pv = pv_energy_available * price
                            cumulative_profit += profit_pv
                            pv_to_grid_this_hour = pv_energy_available
                            if self.pv_system:
                                self.pv_system.allocate_energy(0, pv_energy_available)
                else:
                    action = 0.0
                    # Batteria piena: vendi PV
                    if pv_energy_available > 0:
                        profit_pv = pv_energy_available * price
                        cumulative_profit += profit_pv
                        pv_to_grid_this_hour = pv_energy_available
                        if self.pv_system:
                            self.pv_system.allocate_energy(0, pv_energy_available)

            elif action < -0.01:  # SCARICA
                if self.battery.soc > self.battery.soc_min:
                    max_energy_available = (self.battery.soc - self.battery.soc_min) * self.battery.trading_capacity
                    max_power_available = max_energy_available * self.battery.discharge_efficiency / 1.0
                    actual_power = min(-action, max_power_available)

                    if actual_power > 0.01:
                        energy_to_grid = self.battery.discharge(actual_power, dt=1.0)

                        # Vendi anche PV
                        total_energy_sold = energy_to_grid + pv_energy_available
                        profit = total_energy_sold * price
                        cumulative_profit += profit

                        if pv_energy_available > 0:
                            pv_to_grid_this_hour = pv_energy_available
                            if self.pv_system:
                                self.pv_system.allocate_energy(0, pv_energy_available)
                    else:
                        # Vendi solo PV
                        if pv_energy_available > 0:
                            profit_pv = pv_energy_available * price
                            cumulative_profit += profit_pv
                            pv_to_grid_this_hour = pv_energy_available
                            if self.pv_system:
                                self.pv_system.allocate_energy(0, pv_energy_available)
                else:
                    action = 0.0
                    # Batteria scarica: vendi solo PV
                    if pv_energy_available > 0:
                        profit_pv = pv_energy_available * price
                        cumulative_profit += profit_pv
                        pv_to_grid_this_hour = pv_energy_available
                        if self.pv_system:
                            self.pv_system.allocate_energy(0, pv_energy_available)

            else:  # IDLE
                # Vendi PV disponibile
                if pv_energy_available > 0:
                    profit_pv = pv_energy_available * price
                    cumulative_profit += profit_pv
                    pv_to_grid_this_hour = pv_energy_available
                    if self.pv_system:
                        self.pv_system.allocate_energy(0, pv_energy_available)

            # Tracking PV allocato a batteria
            if self.pv_system and pv_to_battery_this_hour > 0:
                self.pv_system.allocate_energy(pv_to_battery_this_hour, 0)

            macse_available = self.battery.check_macse_availability() if MACSE_ENABLED else False
            if MACSE_ENABLED:
                self.battery.update_macse_availability(macse_available)

            actions_taken.append(action)
            soc_history.append(self.battery.get_soc())
            capacity_history.append(self.battery.capacity)
            soh_history.append(self.battery.get_soh())
            profits_history.append(cumulative_profit)
            macse_availability_history.append(self.battery.get_macse_availability_factor() if MACSE_ENABLED else 0)
            pv_production_history.append(pv_production[current_hour])
            pv_to_battery_history.append(pv_to_battery_this_hour)
            pv_to_grid_history.append(pv_to_grid_this_hour)
            energy_from_grid_history.append(grid_to_battery_this_hour)
            energy_from_pv_history.append(pv_to_battery_this_hour)

            current_hour += self.step_hours

        self.battery.update_degradation()

        print(f"Progresso: 100% completato ({n_hours}/{n_hours} ore)")
        print()

        results_df = prices_df.copy()
        pad_length = len(results_df) - len(actions_taken)
        results_df['Azione_MW'] = actions_taken + [0] * pad_length
        results_df['SOC'] = soc_history + [soc_history[-1]] * pad_length
        results_df['Capacita_MWh'] = capacity_history + [capacity_history[-1]] * pad_length
        results_df['SOH_%'] = soh_history + [soh_history[-1]] * pad_length
        results_df['Profitto_Euro'] = profits_history + [profits_history[-1]] * pad_length
        results_df['MACSE_Availability'] = macse_availability_history + [macse_availability_history[-1]] * pad_length
        results_df['PV_Production_MWh'] = pv_production_history + [0] * pad_length
        results_df['PV_to_Battery_MWh'] = pv_to_battery_history + [0] * pad_length
        results_df['PV_to_Grid_MWh'] = pv_to_grid_history + [0] * pad_length
        results_df['Energy_from_Grid_MWh'] = energy_from_grid_history + [0] * pad_length
        results_df['Energy_from_PV_MWh'] = energy_from_pv_history + [0] * pad_length

        return results_df, cumulative_profit


# ========================================================================================================
# SEZIONE 8: CALCOLO RICAVI MACSE
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


# ========================================================================================================
# SEZIONE 9: EXPORT JSON
# ========================================================================================================
def export_results_to_json(results_df, battery, pv_system, trading_profit, macse_revenue, macse_base,
                           macse_penalty, macse_bonus, battery_investment, simulation_time):
    actions = results_df['Azione_MW'].values
    prices = results_df['€/MWh'].values
    energy_charged = sum([abs(a) for a in actions if a > 0])
    energy_discharged = sum([abs(a) for a in actions if a < 0])
    energy_traded = energy_charged + energy_discharged
    buy_prices = [prices[i] for i, a in enumerate(actions) if a > 0]
    sell_prices = [prices[i] for i, a in enumerate(actions) if a < 0]
    avg_buy_price = np.mean(buy_prices) if buy_prices else 0
    avg_sell_price = np.mean(sell_prices) if sell_prices else 0

    pv_stats = pv_system.get_statistics() if pv_system else {}

    total_revenue = trading_profit + macse_revenue
    residual_value = battery_investment * (battery.get_soh() / 100.0)
    net_return = total_revenue + residual_value - battery_investment
    roi = (net_return / battery_investment) * 100 if battery_investment > 0 else 0
    payback = battery_investment / total_revenue if total_revenue > 0 else float('inf')

    sampling_rate = 24
    sampled_data = []
    for i in range(0, len(results_df), sampling_rate):
        row = results_df.iloc[i]
        timestamp_str = None
        if pd.notnull(row['Data']):
            if isinstance(row['Data'], str):
                try:
                    timestamp_obj = pd.to_datetime(row['Data'], format='%d/%m/%Y')
                    timestamp_str = timestamp_obj.strftime('%Y-%m-%dT%H:%M:%S')
                except:
                    timestamp_str = row['Data']
            else:
                timestamp_str = row['Data'].strftime('%Y-%m-%dT%H:%M:%S')
        sampled_data.append({
            "hour": int(i),
            "timestamp": timestamp_str,
            "price_eur_per_mwh": float(row['€/MWh']),
            "action_mw": float(row['Azione_MW']),
            "soc": float(row['SOC']),
            "soh_percent": float(row['SOH_%']),
            "profit_eur": float(row['Profitto_Euro']),
            "pv_production_mwh": float(row['PV_Production_MWh']) if 'PV_Production_MWh' in row else 0,
            "pv_to_battery_mwh": float(row['PV_to_Battery_MWh']) if 'PV_to_Battery_MWh' in row else 0,
            "pv_to_grid_mwh": float(row['PV_to_Grid_MWh']) if 'PV_to_Grid_MWh' in row else 0
        })

    output_data = {
        "simulation_info": {
            "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "version": "2.3.0-PV-AUTONOMOUS",
            "pv_strategy": "AUTONOMOUS_DECISION",
            "simulation_duration_seconds": simulation_time
        },
        "battery_configuration": {
            "technology": battery.technology,
            "nominal_capacity_mwh": battery.nominal_capacity,
            "max_power_mw": battery.max_power,
            "max_c_rate": battery.max_c_rate,
            "dod": battery.dod,
            "soc_min": battery.soc_min,
            "soc_max": battery.soc_max,
            "eol_cycles": battery.eol_cycles,
            "efficiency_roundtrip": battery.efficiency,
            "efficiency_charge": battery.charge_efficiency,
            "efficiency_discharge": battery.discharge_efficiency
        },
        "pv_configuration": {
            "enabled": PV_ENABLED,
            "nominal_power_kwp": PV_NOMINAL_POWER_KWP if PV_ENABLED else 0,
            "inverter_efficiency": PV_INVERTER_EFFICIENCY if PV_ENABLED else 0,
            "system_losses": PV_SYSTEM_LOSSES if PV_ENABLED else 0,
            "total_efficiency": pv_system.total_efficiency if pv_system else 0,
            "decision_strategy": "AUTONOMOUS_PSO_OPTIMIZATION"
        },
        "macse_configuration": {
            "enabled": MACSE_ENABLED,
            "capacity_mwh": battery.macse_capacity if MACSE_ENABLED else 0,
            "power_mw": battery.macse_power if MACSE_ENABLED else 0,
            "contract_price_eur_per_mw_year": MACSE_PRICE_PER_MW_YEAR if MACSE_ENABLED else 0,
            "availability_target": MACSE_AVAILABILITY_TARGET if MACSE_ENABLED else 0,
            "penalty_threshold": MACSE_PENALTY_THRESHOLD if MACSE_ENABLED else 0,
            "bonus_threshold": MACSE_BONUS_THRESHOLD if MACSE_ENABLED else 0
        },
        "trading_results": {
            "gross_profit_eur": trading_profit,
            "energy_traded_mwh": energy_traded,
            "energy_charged_mwh": energy_charged,
            "energy_discharged_mwh": energy_discharged,
            "energy_from_grid_mwh": battery.energy_from_grid_mwh,
            "energy_from_pv_mwh": battery.energy_from_pv_mwh,
            "average_buy_price_eur_per_mwh": avg_buy_price,
            "average_sell_price_eur_per_mwh": avg_sell_price
        },
        "pv_results": pv_stats if pv_stats else {
            "total_production_mwh": 0,
            "energy_to_battery_mwh": 0,
            "energy_to_grid_mwh": 0,
            "curtailed_energy_mwh": 0,
            "battery_utilization_percent": 0,
            "grid_sale_percent": 0,
            "curtailment_percent": 0
        },
        "macse_results": {
            "annual_revenue_eur": macse_revenue,
            "base_revenue_eur": macse_base,
            "penalty_eur": macse_penalty,
            "bonus_eur": macse_bonus,
            "availability_factor": battery.get_macse_availability_factor() if MACSE_ENABLED else 0,
            "available_hours": battery.macse_available_hours if MACSE_ENABLED else 0,
            "total_hours": battery.macse_total_hours if MACSE_ENABLED else 0
        },
        "battery_health": {
            "final_soc": battery.get_soc(),
            "final_soh_percent": battery.get_soh(),
            "final_capacity_mwh": battery.capacity,
            "equivalent_cycles": battery.equivalent_cycles,
            "throughput_kwh": battery.throughput_kwh,
            "capacity_fade_percent": 100 - battery.get_soh()
        },
        "economic_summary": {
            "total_revenue_eur": total_revenue,
            "trading_profit_eur": trading_profit,
            "macse_revenue_eur": macse_revenue,
            "battery_investment_eur": battery_investment,
            "residual_value_eur": residual_value,
            "net_return_eur": net_return,
            "roi_percent": roi,
            "payback_years": payback if payback != float('inf') else None
        },
        "time_series_data": {
            "sample_size": len(sampled_data),
            "sampling_rate_hours": sampling_rate,
            "data": sampled_data
        }
    }

    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    json_filename = os.path.join(results_folder,
                                 f'simulation_results_{battery.technology.lower().replace("-", "_")}_pv_autonomous.json')

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Risultati JSON salvati in: {json_filename}")
    return json_filename


# ========================================================================================================
# SEZIONE 10: VISUALIZZAZIONI
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

    # CREA LE CARTELLE SE NON ESISTONO
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    if not os.path.exists(monthly_pv_folder):
        os.makedirs(monthly_pv_folder)

    # CREA LE CARTELLE SE NON ESISTONO
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

        # =============================
        # Subplot 1: PRODUZIONE PV E ALLOCAZIONE
        # =============================
        ax1 = fig.add_subplot(gs[0, :])

        # Produzione PV totale
        ax1.plot(df_giorno['Ora'], df_giorno['PV_Production_MWh'],
                color='#F4A300', linewidth=3, marker='o', markersize=6,
                label='Produzione PV Totale', zorder=5)
        ax1.fill_between(df_giorno['Ora'], 0, df_giorno['PV_Production_MWh'],
                         color='#F4A300', alpha=0.2)

        # PV verso batteria
        ax1.bar(df_giorno['Ora'], df_giorno['PV_to_Battery_MWh'],
               width=0.7, color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1,
               label='PV → Batteria (PSO Decision)')

        # PV verso rete
        ax1.bar(df_giorno['Ora'], df_giorno['PV_to_Grid_MWh'],
               width=0.7, bottom=df_giorno['PV_to_Battery_MWh'],
               color='#457B9D', alpha=0.8, edgecolor='black', linewidth=1,
               label='PV → Rete Diretta (PSO Decision)')

        ax1.set_ylabel('Energia (MWh)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Ora', fontsize=11)
        ax1.set_title('Produzione e Allocazione Fotovoltaica', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper left', framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(0, 24, 2))

        # =============================
        # Subplot 2: FONTI DI CARICA BATTERIA (CHIAVE!)
        # =============================
        ax2 = fig.add_subplot(gs[1, 0])

        # Crea array per visualizzazione carica
        charge_from_grid = []
        charge_from_pv = []
        for idx, row in df_giorno.iterrows():
            if row['Azione_MW'] > 0.01:  # Carica
                charge_from_grid.append(row['Energy_from_Grid_MWh'])
                charge_from_pv.append(row['Energy_from_PV_MWh'])
            else:
                charge_from_grid.append(0)
                charge_from_pv.append(0)

        # Barre impilate per mostrare fonte energia
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

        # =============================
        # Subplot 3: AZIONI BATTERIA E PREZZO
        # =============================
        ax3 = fig.add_subplot(gs[1, 1])
        ax3_twin = ax3.twinx()

        # Colora azioni: verde=carica da PV, arancione=carica da rete, rosso=scarica
        colors_actions = []
        for idx, row in df_giorno.iterrows():
            if row['Azione_MW'] > 0.01:
                if row['Energy_from_PV_MWh'] > row['Energy_from_Grid_MWh']:
                    colors_actions.append('#06A77D')  # Verde: principalmente PV
                else:
                    colors_actions.append('#F77F00')  # Arancione: principalmente rete
            elif row['Azione_MW'] < -0.01:
                colors_actions.append('#E63946')  # Rosso: scarica
            else:
                colors_actions.append('#CCCCCC')  # Grigio: idle

        ax3.bar(df_giorno['Ora'], df_giorno['Azione_MW'],
               color=colors_actions, alpha=0.8, width=0.8, edgecolor='black', linewidth=1)

        # Prezzo
        ax3_twin.plot(df_giorno['Ora'], df_giorno['€/MWh'],
                     color='#457B9D', linewidth=2.5, marker='s', markersize=5,
                     label='Prezzo Energia', zorder=10)

        ax3.axhline(y=0, color='black', linewidth=1, linestyle='-')
        ax3.set_ylabel('Potenza Batteria (MW)', fontsize=12, fontweight='bold')
        ax3_twin.set_ylabel('Prezzo (€/MWh)', fontsize=12, fontweight='bold', color='#457B9D')
        ax3_twin.tick_params(axis='y', labelcolor='#457B9D')
        ax3.set_xlabel('Ora', fontsize=11)
        ax3.set_title('Azioni Batteria vs Prezzo Energia', fontsize=13, fontweight='bold')

        # Legenda personalizzata
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#06A77D', alpha=0.8, edgecolor='black', label='Carica da PV'),
            Patch(facecolor='#F77F00', alpha=0.8, edgecolor='black', label='Carica da Rete'),
            Patch(facecolor='#E63946', alpha=0.8, edgecolor='black', label='Scarica'),
        ]
        ax3.legend(handles=legend_elements, fontsize=9, loc='upper left', framealpha=0.95)
        ax3_twin.legend(fontsize=9, loc='upper right', framealpha=0.95)

        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_xticks(range(0, 24, 2))

        # =============================
        # Subplot 4: SOC E IMPATTO PV
        # =============================
        ax4 = fig.add_subplot(gs[2, 0])

        ax4.plot(df_giorno['Ora'], df_giorno['SOC'] * 100,
                color='#457B9D', linewidth=3, marker='o', markersize=6,
                label='State of Charge')
        ax4.fill_between(df_giorno['Ora'], battery.soc_min * 100, df_giorno['SOC'] * 100,
                         color='#457B9D', alpha=0.2)

        # Evidenzia ore con carica da PV
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

        # =============================
        # Subplot 5: BILANCIO ECONOMICO ORARIO
        # =============================
        ax5 = fig.add_subplot(gs[2, 1])

        # Calcola profitto/costo orario
        profitto_orario = []
        for idx, row in df_giorno.iterrows():
            profit_hour = 0
            # Costo carica da rete
            if row['Energy_from_Grid_MWh'] > 0:
                profit_hour -= row['Energy_from_Grid_MWh'] * row['€/MWh']
            # Ricavo scarica
            if row['Azione_MW'] < -0.01:
                energy_discharged = abs(row['Azione_MW']) * 1.0 * battery.discharge_efficiency
                profit_hour += energy_discharged * row['€/MWh']
            # Ricavo vendita PV diretta
            if row['PV_to_Grid_MWh'] > 0:
                profit_hour += row['PV_to_Grid_MWh'] * row['€/MWh']
            # Risparmio PV per batteria (costo evitato)
            if row['Energy_from_PV_MWh'] > 0:
                profit_hour += row['Energy_from_PV_MWh'] * row['€/MWh'] * 0.5  # 50% del valore

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

        # =============================
        # Subplot 6: STATISTICHE GIORNALIERE
        # =============================
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')

        # Calcola statistiche giorno
        total_pv_prod = df_giorno['PV_Production_MWh'].sum()
        total_pv_to_batt = df_giorno['PV_to_Battery_MWh'].sum()
        total_pv_to_grid = df_giorno['PV_to_Grid_MWh'].sum()
        total_grid_to_batt = df_giorno['Energy_from_Grid_MWh'].sum()
        total_charge = total_pv_to_batt + total_grid_to_batt
        pv_charge_percent = (total_pv_to_batt / total_charge * 100) if total_charge > 0 else 0

        profit_day = sum(profitto_orario)
        avg_price = df_giorno['€/MWh'].mean()

        stats_text = f"""
STATISTICHE GIORNALIERE {mesi_nomi[mese-1].upper()} - {data_str}

PRODUZIONE FOTOVOLTAICA:
  • Produzione totale PV:           {total_pv_prod:.3f} MWh
  • PV utilizzato per batteria:     {total_pv_to_batt:.3f} MWh ({total_pv_to_batt/total_pv_prod*100 if total_pv_prod > 0 else 0:.1f}% della produzione)
  • PV venduto direttamente:        {total_pv_to_grid:.3f} MWh ({total_pv_to_grid/total_pv_prod*100 if total_pv_prod > 0 else 0:.1f}% della produzione)
  
CARICA BATTERIA:
  • Energia da PV (GRATIS):         {total_pv_to_batt:.3f} MWh ({pv_charge_percent:.1f}% del totale caricato)
  • Energia da RETE (ACQUISTO):     {total_grid_to_batt:.3f} MWh ({(100-pv_charge_percent):.1f}% del totale caricato)
  • Totale caricato:                {total_charge:.3f} MWh
  
BENEFICIO PV:
  • Risparmio acquisto rete:        {total_pv_to_batt * avg_price:.2f} €
  • Ricavo vendita diretta:         {total_pv_to_grid * avg_price:.2f} €
  • Valore totale PV giornaliero:   {(total_pv_to_batt + total_pv_to_grid) * avg_price:.2f} €

ECONOMIA:
  • Profitto giornaliero:           {profit_day:.2f} €
  • Prezzo medio energia:           {avg_price:.2f} €/MWh
  • SOC iniziale → finale:          {df_giorno['SOC'].iloc[0]*100:.1f}% → {df_giorno['SOC'].iloc[-1]*100:.1f}%
        """

        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Salva figura
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
    Mostra CHIARAMENTE la differenza tra avere o non avere il PV
    """
    if not SAVE_PLOTS or not PV_ENABLED or pv_system is None:
        return

    viz_folder = 'visualization'
    pv_folder = os.path.join(viz_folder, 'analisi_pv_autonomous')

    # CREA LE CARTELLE SE NON ESISTONO
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    if not os.path.exists(pv_folder):
        os.makedirs(pv_folder)

    # CREA LE CARTELLE SE NON ESISTONO
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    if not os.path.exists(pv_folder):
        os.makedirs(pv_folder)

    # CREA LE CARTELLE SE NON ESISTONO
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    if not os.path.exists(pv_folder):
        os.makedirs(pv_folder)

    print("\n" + "=" * 80)
    print("GENERAZIONE GRAFICO IMPATTO PV - CONFRONTO CON/SENZA FOTOVOLTAICO")
    print("=" * 80)

    # Calcola metriche totali
    pv_stats = pv_system.get_statistics()

    # Energia totale caricata
    energia_da_pv = battery.energy_from_pv_mwh
    energia_da_rete = battery.energy_from_grid_mwh
    energia_totale = energia_da_pv + energia_da_rete

    # PV prodotto
    pv_prodotto = pv_stats['total_production_mwh']
    pv_a_batteria = pv_stats['energy_to_battery_mwh']
    pv_venduto = pv_stats['energy_to_grid_mwh']

    # Calcola costi/benefici
    prezzo_medio = results_df['€/MWh'].mean()

    # SCENARIO SENZA PV: tutta l'energia verrebbe dalla rete
    costo_senza_pv = energia_totale * prezzo_medio

    # SCENARIO CON PV: parte gratis da PV, parte da rete
    costo_con_pv = energia_da_rete * prezzo_medio
    ricavo_vendita_pv = pv_venduto * prezzo_medio

    # RISPARMIO NETTO
    risparmio_acquisto = energia_da_pv * prezzo_medio
    beneficio_totale_pv = risparmio_acquisto + ricavo_vendita_pv

    # Percentuali
    percentuale_pv = (energia_da_pv / energia_totale * 100) if energia_totale > 0 else 0
    percentuale_rete = 100 - percentuale_pv

    # ========================================
    # CREA FIGURA IMPATTO
    # ========================================
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    fig.suptitle('IMPATTO FOTOVOLTAICO SUL SISTEMA BESS\nConfronto Diretto: CON vs SENZA Fotovoltaico',
                 fontsize=18, fontweight='bold', y=0.98)

    # ========================================
    # 1. PRIMA DOMANDA: Da dove viene l'energia?
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])

    labels1 = ['DA PV\n(GRATIS)', 'DA RETE\n(ACQUISTO)']
    sizes1 = [energia_da_pv, energia_da_rete]
    colors1 = ['#06A77D', '#E63946']
    explode1 = (0.1, 0)

    wedges1, texts1, autotexts1 = ax1.pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%',
                                           startangle=90, explode=explode1, textprops={'fontsize': 13, 'fontweight': 'bold'},
                                           shadow=True, wedgeprops={'edgecolor': 'black', 'linewidth': 2})

    ax1.set_title('FONTI DI CARICA BATTERIA\n(Da dove viene l\'energia?)',
                  fontsize=14, fontweight='bold', pad=15)

    # Aggiungi valori assoluti
    ax1.text(0, -1.4, f'PV: {energia_da_pv:.1f} MWh | Rete: {energia_da_rete:.1f} MWh',
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========================================
    # 2. SECONDA DOMANDA: Cosa fa il PV prodotto?
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])

    labels2 = ['→ Batteria\n(Stoccato)', '→ Rete\n(Venduto)']
    sizes2 = [pv_a_batteria, pv_venduto]
    colors2 = ['#06A77D', '#457B9D']
    explode2 = (0.1, 0)

    wedges2, texts2, autotexts2 = ax2.pie(sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%',
                                           startangle=90, explode=explode2, textprops={'fontsize': 13, 'fontweight': 'bold'},
                                           shadow=True, wedgeprops={'edgecolor': 'black', 'linewidth': 2})

    ax2.set_title('ALLOCAZIONE ENERGIA PV\n(Dove va il PV prodotto?)',
                  fontsize=14, fontweight='bold', pad=15)

    ax2.text(0, -1.4, f'Produzione Totale PV: {pv_prodotto:.1f} MWh',
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # ========================================
    # 3. TERZA DOMANDA: Quanto si risparmia?
    # ========================================
    ax3 = fig.add_subplot(gs[0, 2])

    categories3 = ['SENZA PV\n(tutto da rete)', 'CON PV\n(parziale rete)']
    values3 = [costo_senza_pv, costo_con_pv]
    colors3 = ['#E63946', '#06A77D']

    bars3 = ax3.bar(categories3, values3, color=colors3, alpha=0.8,
                    edgecolor='black', linewidth=2, width=0.6)

    # Freccia risparmio
    ax3.annotate('', xy=(1, costo_con_pv), xytext=(1, costo_senza_pv),
                arrowprops=dict(arrowstyle='<->', color='gold', lw=4))
    ax3.text(1.15, (costo_senza_pv + costo_con_pv)/2,
             f'RISPARMIO\n{risparmio_acquisto:,.0f} €',
             fontsize=12, fontweight='bold', color='gold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax3.set_ylabel('Costo Acquisto Energia (€)', fontsize=12, fontweight='bold')
    ax3.set_title('RISPARMIO ECONOMICO\n(Quanto costa caricare?)',
                  fontsize=14, fontweight='bold', pad=15)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f} €',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ========================================
    # 4. CONFRONTO DIRETTO: Con PV vs Senza PV
    # ========================================
    ax4 = fig.add_subplot(gs[1, :])

    categories4 = ['Energia\nCaricata', 'Costo\nAcquisto', 'Ricavo\nVendita PV', 'BILANCIO\nNETTO']

    senza_pv = [energia_totale, -costo_senza_pv, 0, -costo_senza_pv]
    con_pv = [energia_totale, -costo_con_pv, ricavo_vendita_pv, -costo_con_pv + ricavo_vendita_pv]

    x = np.arange(len(categories4))
    width = 0.35

    bars_senza = ax4.bar(x - width/2, senza_pv, width, label='SENZA PV',
                         color='#E63946', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars_con = ax4.bar(x + width/2, con_pv, width, label='CON PV',
                       color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax4.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
    ax4.set_ylabel('Valore (MWh o €)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories4, fontsize=12, fontweight='bold')
    ax4.set_title('CONFRONTO DIRETTO: Sistema CON vs SENZA Fotovoltaico',
                  fontsize=15, fontweight='bold', pad=15)
    ax4.legend(fontsize=12, loc='upper left', framealpha=0.95)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    # Aggiungi valori sopra barre
    for bars in [bars_senza, bars_con]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.1:
                label = f'{abs(height):.0f}'
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center',
                        va='bottom' if height > 0 else 'top',
                        fontsize=10, fontweight='bold')

    # ========================================
    # 5. BENEFICIO ECONOMICO DETTAGLIATO
    # ========================================
    ax5 = fig.add_subplot(gs[2, 0])

    benefit_categories = ['Risparmio\nAcquisto', 'Ricavo\nVendita', 'TOTALE\nBeneficio']
    benefit_values = [risparmio_acquisto, ricavo_vendita_pv, beneficio_totale_pv]
    benefit_colors = ['#06A77D', '#457B9D', '#F4A300']

    bars5 = ax5.bar(benefit_categories, benefit_values, color=benefit_colors,
                   alpha=0.8, edgecolor='black', linewidth=2, width=0.6)

    ax5.set_ylabel('Beneficio (€)', fontsize=12, fontweight='bold')
    ax5.set_title('BENEFICIO ECONOMICO PV\n(Quanto vale il fotovoltaico?)',
                  fontsize=14, fontweight='bold', pad=15)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f} €',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ========================================
    # 6. IMPATTO PERCENTUALE
    # ========================================
    ax6 = fig.add_subplot(gs[2, 1])

    # Gauge chart per impatto PV
    sizes_gauge = [percentuale_pv, percentuale_rete]
    colors_gauge = ['#06A77D', '#E63946']

    wedges_g, texts_g = ax6.pie(sizes_gauge, colors=colors_gauge, startangle=90,
                                 counterclock=False, wedgeprops={'edgecolor': 'black', 'linewidth': 2})

    # Aggiungi testo centrale
    ax6.text(0, 0, f'{percentuale_pv:.1f}%\nPV', ha='center', va='center',
            fontsize=28, fontweight='bold', color='#06A77D',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))

    ax6.set_title('IMPATTO PV SULLA CARICA\n(% energia dalla fotovoltaico)',
                  fontsize=14, fontweight='bold', pad=15)

    # ========================================
    # 7. RIEPILOGO NUMERICO
    # ========================================
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    summary_text = f"""
╔══════════════════════════════════════╗
║   RIEPILOGO IMPATTO PV        ║
╚══════════════════════════════════════╝

ENERGIA BATTERIA:
   • Totale caricata:  {energia_totale:.1f} MWh
   • Da PV (gratis):   {energia_da_pv:.1f} MWh ({percentuale_pv:.1f}%)
   • Da Rete (acquisto): {energia_da_rete:.1f} MWh ({percentuale_rete:.1f}%)

PRODUZIONE PV:
   • Totale prodotto:  {pv_prodotto:.1f} MWh
   • → Batteria:       {pv_a_batteria:.1f} MWh ({pv_a_batteria/pv_prodotto*100:.1f}%)
   • → Rete:           {pv_venduto:.1f} MWh ({pv_venduto/pv_prodotto*100:.1f}%)

IMPATTO ECONOMICO:
   • Senza PV:         {costo_senza_pv:,.0f} € (costo)
   • Con PV:           {costo_con_pv:,.0f} € (costo)
   • Risparmio:        {risparmio_acquisto:,.0f} € ✓
   • Ricavo vendita:   {ricavo_vendita_pv:,.0f} € ✓
   • BENEFICIO TOTALE: {beneficio_totale_pv:,.0f} € ✓✓

ROI FOTOVOLTAICO:
   • Valore/MWh:       {beneficio_totale_pv/pv_prodotto:.2f} €/MWh
   • Riduzione costo:  {(risparmio_acquisto/costo_senza_pv*100):.1f}%
    """

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3,
                     edgecolor='black', linewidth=2))

    plt.tight_layout()
    plt.savefig(os.path.join(pv_folder, f'IMPATTO_PV_CONFRONTO_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Salvato: IMPATTO_PV_CONFRONTO_{battery.technology}.png")
    print()
    print("=" * 80)
    print("🌞 SINTESI IMPATTO FOTOVOLTAICO:")
    print("=" * 80)
    print(f"  ⚡ Energia caricata DA PV:     {energia_da_pv:.1f} MWh ({percentuale_pv:.1f}%)")
    print(f"  ⚡ Energia caricata DA RETE:   {energia_da_rete:.1f} MWh ({percentuale_rete:.1f}%)")
    print(f"  💰 Risparmio acquisto rete:    {risparmio_acquisto:,.0f} €")
    print(f"  💰 Ricavo vendita PV:          {ricavo_vendita_pv:,.0f} €")
    print(f"  💰 BENEFICIO TOTALE PV:        {beneficio_totale_pv:,.0f} €")
    print(f"  📊 Riduzione costo carica:     {(risparmio_acquisto/costo_senza_pv*100):.1f}%")
    print("=" * 80)


def create_pv_impact_summary(results_df, battery, pv_system):
    """
    Lorenzo Giannuzzo: Grafico sintesi annuale IMPATTO PV sul sistema
    """
    if not SAVE_PLOTS or not PV_ENABLED or pv_system is None:
        return

    viz_folder = 'visualization'
    pv_folder = os.path.join(viz_folder, 'analisi_pv_autonomous')

    # CREA LE CARTELLE SE NON ESISTONO
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    if not os.path.exists(pv_folder):
        os.makedirs(pv_folder)

    # CREA LE CARTELLE SE NON ESISTONO
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    if not os.path.exists(pv_folder):
        os.makedirs(pv_folder)

    # CREA LE CARTELLE SE NON ESISTONO
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    if not os.path.exists(pv_folder):
        os.makedirs(pv_folder)

    print("\n" + "=" * 80)
    print("GENERAZIONE GRAFICO SINTESI IMPATTO PV ANNUALE")
    print("=" * 80)

    if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
        results_df['Data'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')

    # Aggrega per mese
    results_df['Mese'] = results_df['Data'].dt.month
    df_monthly = results_df.groupby('Mese').agg({
        '€/MWh': 'mean',
        'PV_Production_MWh': 'sum',
        'PV_to_Battery_MWh': 'sum',
        'PV_to_Grid_MWh': 'sum',
        'Energy_from_Grid_MWh': 'sum',
        'Energy_from_PV_MWh': 'sum',
        'Profitto_Euro': lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 0 else 0
    }).reset_index()

    mesi_nomi = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu',
                 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
    df_monthly['Mese_Nome'] = df_monthly['Mese'].apply(lambda x: mesi_nomi[x-1])

    # Crea figura grande
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    fig.suptitle(f'Sintesi Annuale Impatto Fotovoltaico - {battery.technology}\nDecisione Autonoma Algoritmo PSO',
                 fontsize=17, fontweight='bold', y=0.995)

    # =============================
    # 1. PRODUZIONE E ALLOCAZIONE PV MENSILE
    # =============================
    ax1 = fig.add_subplot(gs[0, :])

    x = np.arange(len(df_monthly))
    width = 0.7

    ax1.bar(x, df_monthly['PV_to_Battery_MWh'], width=width,
           color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.5,
           label='PV → Batteria')
    ax1.bar(x, df_monthly['PV_to_Grid_MWh'], width=width,
           bottom=df_monthly['PV_to_Battery_MWh'],
           color='#457B9D', alpha=0.8, edgecolor='black', linewidth=1.5,
           label='PV → Rete Diretta')
    ax1.plot(x, df_monthly['PV_Production_MWh'], color='#F4A300', linewidth=3,
            marker='o', markersize=8, label='Produzione Totale PV', zorder=10)

    ax1.set_ylabel('Energia (MWh)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Mese', fontsize=12)
    ax1.set_title('Produzione e Allocazione Fotovoltaica Mensile', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_monthly['Mese_Nome'])
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # =============================
    # 2. FONTI CARICA BATTERIA MENSILE
    # =============================
    ax2 = fig.add_subplot(gs[1, 0])

    ax2.bar(x, df_monthly['Energy_from_PV_MWh'], width=width,
           color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.5,
           label='Carica da PV (GRATIS)')
    ax2.bar(x, df_monthly['Energy_from_Grid_MWh'], width=width,
           bottom=df_monthly['Energy_from_PV_MWh'],
           color='#E63946', alpha=0.8, edgecolor='black', linewidth=1.5,
           label='Carica da Rete (ACQUISTO)')

    ax2.set_ylabel('Energia Caricata (MWh)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Mese', fontsize=11)
    ax2.set_title('Fonti di Carica Batteria', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_monthly['Mese_Nome'], rotation=45)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # =============================
    # 3. PERCENTUALE CARICA DA PV
    # =============================
    ax3 = fig.add_subplot(gs[1, 1])

    df_monthly['PV_Charge_Percent'] = (df_monthly['Energy_from_PV_MWh'] /
                                        (df_monthly['Energy_from_PV_MWh'] + df_monthly['Energy_from_Grid_MWh']) * 100)
    df_monthly['PV_Charge_Percent'].fillna(0, inplace=True)

    colors_percent = ['#06A77D' if p >= 50 else '#F77F00' for p in df_monthly['PV_Charge_Percent']]
    bars_percent = ax3.bar(x, df_monthly['PV_Charge_Percent'], width=width,
                          color=colors_percent, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='50% Soglia')

    ax3.set_ylabel('% Carica da PV', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Mese', fontsize=11)
    ax3.set_title('Percentuale Carica da Fotovoltaico', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_monthly['Mese_Nome'], rotation=45)
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Aggiungi valori sopra barre
    for i, (bar, val) in enumerate(zip(bars_percent, df_monthly['PV_Charge_Percent'])):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # =============================
    # 4. DECISIONI PSO: BATTERIA VS VENDITA DIRETTA
    # =============================
    ax4 = fig.add_subplot(gs[1, 2])

    df_monthly['PV_Battery_Percent'] = (df_monthly['PV_to_Battery_MWh'] /
                                         df_monthly['PV_Production_MWh'] * 100)
    df_monthly['PV_Grid_Percent'] = (df_monthly['PV_to_Grid_MWh'] /
                                      df_monthly['PV_Production_MWh'] * 100)
    df_monthly['PV_Battery_Percent'].fillna(0, inplace=True)
    df_monthly['PV_Grid_Percent'].fillna(0, inplace=True)

    ax4.bar(x, df_monthly['PV_Battery_Percent'], width=width,
           color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.5,
           label='% PV → Batteria')
    ax4.bar(x, df_monthly['PV_Grid_Percent'], width=width,
           bottom=df_monthly['PV_Battery_Percent'],
           color='#457B9D', alpha=0.8, edgecolor='black', linewidth=1.5,
           label='% PV → Rete')

    ax4.set_ylabel('% Produzione PV', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Mese', fontsize=11)
    ax4.set_title('Decisioni PSO: Allocazione PV', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df_monthly['Mese_Nome'], rotation=45)
    ax4.set_ylim(0, 105)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    # =============================
    # 5. BENEFICIO ECONOMICO PV MENSILE
    # =============================
    ax5 = fig.add_subplot(gs[2, :2])

    # Calcola beneficio economico
    df_monthly['Risparmio_Acquisto'] = df_monthly['Energy_from_PV_MWh'] * df_monthly['€/MWh']
    df_monthly['Ricavo_Vendita'] = df_monthly['PV_to_Grid_MWh'] * df_monthly['€/MWh']
    df_monthly['Beneficio_Totale'] = df_monthly['Risparmio_Acquisto'] + df_monthly['Ricavo_Vendita']

    width2 = 0.35
    ax5.bar(x - width2/2, df_monthly['Risparmio_Acquisto'], width=width2,
           color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.5,
           label='Risparmio Acquisto Rete')
    ax5.bar(x + width2/2, df_monthly['Ricavo_Vendita'], width=width2,
           color='#457B9D', alpha=0.8, edgecolor='black', linewidth=1.5,
           label='Ricavo Vendita Diretta')
    ax5.plot(x, df_monthly['Beneficio_Totale'], color='#F4A300', linewidth=3,
            marker='D', markersize=8, label='Beneficio Totale PV', zorder=10)

    ax5.set_ylabel('Valore Economico (€)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Mese', fontsize=11)
    ax5.set_title('Beneficio Economico Mensile Sistema Fotovoltaico', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(df_monthly['Mese_Nome'])
    ax5.legend(fontsize=10, loc='upper left')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')

    # =============================
    # 6. STATISTICHE ANNUALI
    # =============================
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    # Calcola totali
    pv_stats = pv_system.get_statistics()
    total_pv_prod = pv_stats['total_production_mwh']
    total_pv_to_batt = pv_stats['energy_to_battery_mwh']
    total_pv_to_grid = pv_stats['energy_to_grid_mwh']
    total_grid_to_batt = battery.energy_from_grid_mwh
    total_charge = total_pv_to_batt + total_grid_to_batt
    pv_charge_percent_year = (total_pv_to_batt / total_charge * 100) if total_charge > 0 else 0

    avg_price_year = df_monthly['€/MWh'].mean()
    total_risparmio = total_pv_to_batt * avg_price_year
    total_ricavo = total_pv_to_grid * avg_price_year
    total_beneficio = total_risparmio + total_ricavo

    stats_text = f"""
STATISTICHE ANNUALI
FOTOVOLTAICO

Produzione:
 • Totale: {total_pv_prod:.1f} MWh

Allocazione:
 • → Batteria: {total_pv_to_batt:.1f} MWh
   ({pv_stats['battery_utilization_percent']:.1f}%)
 • → Rete: {total_pv_to_grid:.1f} MWh
   ({pv_stats['grid_sale_percent']:.1f}%)

Carica Batteria:
 • Da PV: {total_pv_to_batt:.1f} MWh
   ({pv_charge_percent_year:.1f}%)
 • Da Rete: {total_grid_to_batt:.1f} MWh
   ({100-pv_charge_percent_year:.1f}%)

Beneficio Economico:
 • Risparmio: {total_risparmio:,.0f} €
 • Ricavo: {total_ricavo:,.0f} €
 • TOTALE: {total_beneficio:,.0f} €

ROI PV:
 • Valore/MWh: {total_beneficio/total_pv_prod:.2f} €/MWh
    """

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4, edgecolor='black', linewidth=2))

    plt.tight_layout()
    plt.savefig(os.path.join(pv_folder, f'00_sintesi_impatto_pv_annuale_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: 00_sintesi_impatto_pv_annuale_{battery.technology}.png")

    print("=" * 80)


def create_pv_visualizations(results_df, battery, pv_system):
    """Grafici analisi fotovoltaico con decisione autonoma"""
    if not SAVE_PLOTS or not PV_ENABLED or pv_system is None:
        return

    viz_folder = 'visualization'
    pv_folder = os.path.join(viz_folder, 'analisi_pv_autonomous')

    # CREA LE CARTELLE SE NON ESISTONO
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    if not os.path.exists(pv_folder):
        os.makedirs(pv_folder)

    # CREA LE CARTELLE SE NON ESISTONO
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    if not os.path.exists(pv_folder):
        os.makedirs(pv_folder)

    print("\n" + "=" * 80)
    print("GENERAZIONE GRAFICI ANALISI FOTOVOLTAICO (STRATEGIA AUTONOMA)")
    print("=" * 80)

    if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
        results_df['Data'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')

    df_daily = results_df.groupby(results_df['Data'].dt.date, as_index=False).agg({
        'Data': 'first',
        '€/MWh': 'mean',
        'PV_Production_MWh': 'sum',
        'PV_to_Battery_MWh': 'sum',
        'PV_to_Grid_MWh': 'sum',
        'Energy_from_Grid_MWh': 'sum',
        'Profitto_Euro': 'last'
    })
    df_daily['Data'] = pd.to_datetime(df_daily['Data'])

    # Grafico 1: Decisioni Allocazione PV
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(df_daily['Data'], df_daily['PV_Production_MWh'],
            color='#F4A300', linewidth=2.5, label='Produzione Totale PV', marker='o', markersize=4)
    ax1.fill_between(df_daily['Data'], 0, df_daily['PV_to_Battery_MWh'],
                     color='#06A77D', alpha=0.6, label='PV → Batteria (Decisione PSO)')
    ax1.fill_between(df_daily['Data'], df_daily['PV_to_Battery_MWh'],
                     df_daily['PV_to_Battery_MWh'] + df_daily['PV_to_Grid_MWh'],
                     color='#457B9D', alpha=0.6, label='PV → Rete Diretta (Decisione PSO)')
    ax1.set_ylabel('Energia (MWh)', fontsize=12, fontweight='bold')
    ax1.set_title('Decisioni Autonome Allocazione PV\n(Algoritmo PSO ottimizza tra carica batteria e vendita diretta)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Subplot 2: Prezzo vs decisioni PV
    ax2_twin = ax2.twinx()
    ax2.bar(df_daily['Data'], df_daily['PV_to_Battery_MWh'],
           width=0.8, color='#06A77D', alpha=0.7, label='PV → Batteria')
    ax2.bar(df_daily['Data'], df_daily['PV_to_Grid_MWh'],
           width=0.8, bottom=df_daily['PV_to_Battery_MWh'],
           color='#457B9D', alpha=0.7, label='PV → Rete')
    ax2_twin.plot(df_daily['Data'], df_daily['€/MWh'],
                  color='#E63946', linewidth=2, marker='s', markersize=4, label='Prezzo Medio')
    ax2.set_ylabel('Energia PV (MWh)', fontsize=12, fontweight='bold')
    ax2_twin.set_ylabel('Prezzo Energia (€/MWh)', fontsize=12, fontweight='bold', color='#E63946')
    ax2_twin.tick_params(axis='y', labelcolor='#E63946')
    ax2.set_xlabel('Data', fontsize=11)
    ax2.set_title('Correlazione Prezzi vs Decisioni PV', fontsize=13, fontweight='bold')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(pv_folder, f'01_decisioni_pv_autonome_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: 01_decisioni_pv_autonome_{battery.technology}.png")

    # Grafico 2: Statistiche PV
    pv_stats = pv_system.get_statistics()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Statistiche Sistema Fotovoltaico - Strategia Autonoma - {battery.technology}',
                 fontsize=15, fontweight='bold')

    # Pie chart decisioni
    labels = ['Verso Batteria\n(PSO Decision)', 'Vendita Diretta\n(PSO Decision)', 'Curtailment']
    sizes = [pv_stats['battery_utilization_percent'],
             pv_stats['grid_sale_percent'],
             pv_stats['curtailment_percent']]
    colors = ['#06A77D', '#457B9D', '#E63946']
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax1.set_title('Allocazione Produzione PV\n(Decisione Algoritmo)', fontsize=12, fontweight='bold')

    # Bar chart energie
    categories = ['Produzione\nTotale', 'Verso\nBatteria', 'Vendita\nDiretta', 'Curtailment']
    values = [pv_stats['total_production_mwh'],
              pv_stats['energy_to_battery_mwh'],
              pv_stats['energy_to_grid_mwh'],
              pv_stats['curtailed_energy_mwh']]
    bars = ax2.bar(categories, values, color=['#F4A300', '#06A77D', '#457B9D', '#E63946'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Energia (MWh)', fontsize=11, fontweight='bold')
    ax2.set_title('Bilancio Energetico PV', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Confronto fonti carica
    total_charge_grid = battery.energy_from_grid_mwh
    total_charge_pv = battery.energy_from_pv_mwh
    total_charge = total_charge_grid + total_charge_pv

    sources = ['Da Rete\n(Acquisto)', 'Da PV\n(Gratuito)']
    charge_values = [total_charge_grid, total_charge_pv]
    charge_colors = ['#E63946', '#06A77D']
    bars3 = ax3.bar(sources, charge_values, color=charge_colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Energia Caricata (MWh)', fontsize=11, fontweight='bold')
    ax3.set_title('Fonti di Carica Batteria', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars3:
        height = bar.get_height()
        percentage = (height / total_charge * 100) if total_charge > 0 else 0
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f} MWh\n({percentage:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Risparmio energetico PV
    cost_avoided = total_charge_pv * df_daily['€/MWh'].mean()
    revenue_from_sales = pv_stats['energy_to_grid_mwh'] * df_daily['€/MWh'].mean()

    benefit_categories = ['Risparmio\nAcquisto Rete', 'Ricavo\nVendita Diretta', 'Valore\nTotale PV']
    benefit_values = [cost_avoided, revenue_from_sales, cost_avoided + revenue_from_sales]
    benefit_colors = ['#06A77D', '#457B9D', '#F4A300']
    bars4 = ax4.bar(benefit_categories, benefit_values, color=benefit_colors,
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Valore Economico (€)', fontsize=11, fontweight='bold')
    ax4.set_title('Beneficio Economico Sistema PV', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:,.0f} €',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(pv_folder, f'02_statistiche_pv_autonomous_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: 02_statistiche_pv_autonomous_{battery.technology}.png")

    print("\n" + "=" * 80)
    print(f"Grafici PV salvati in '{pv_folder}/'")
    print("  - Decisioni autonome allocazione PV")
    print("  - Statistiche complete sistema PV")
    print("  - Analisi economica benefici PV")
    print("=" * 80)


# ========================================================================================================
# FUNZIONE MAIN
# ========================================================================================================
def main(file_name, pv_file_name=None):
    file_path = os.path.join('data', file_name)

    print("=" * 80)
    print("BESS OPTIMIZATION WITH AUTONOMOUS PV STRATEGY v2.3.0")
    print("=" * 80)
    print("Caricamento dati prezzi energia...")

    try:
        df = pd.read_excel(file_path)
        if df['€/MWh'].dtype == 'object':
            df['€/MWh'] = df['€/MWh'].astype(str).str.replace(',', '.').astype(float)

        print(f"Dati caricati: {len(df)} righe")
        print(f"Range prezzi: {df['€/MWh'].min():.2f} - {df['€/MWh'].max():.2f} Euro/MWh")

        if 'Data' in df.columns:
            date_range = pd.to_datetime(df['Data'], format='%d/%m/%Y')
            print(f"Periodo: {date_range.min().date()} -> {date_range.max().date()}")
            print(f"Durata: {(date_range.max() - date_range.min()).days} giorni")
        print()
    except Exception as e:
        print(f"Errore caricamento prezzi: {e}")
        return

    # Caricamento dati PV
    pv_df = None
    pv_system = None
    if PV_ENABLED and pv_file_name:
        print("Caricamento dati produzione fotovoltaica...")
        try:
            pv_file_path = os.path.join('data', pv_file_name)
            pv_df = pd.read_csv(pv_file_path, sep=';')
            print(f"Dati PV caricati: {len(pv_df)} righe")
            print(f"Range produzione: {pv_df['P'].min():.2f} - {pv_df['P'].max():.2f} W/kWp")

            pv_system = PhotovoltaicSystem(
                nominal_power_kwp=PV_NOMINAL_POWER_KWP,
                inverter_efficiency=PV_INVERTER_EFFICIENCY,
                system_losses=PV_SYSTEM_LOSSES
            )
            print(f"Sistema PV inizializzato: {PV_NOMINAL_POWER_KWP:.0f} kWp")
            print(f"Efficienza totale PV: {pv_system.total_efficiency * 100:.1f}%")
            print(f"STRATEGIA: DECISIONE COMPLETAMENTE AUTONOMA")
            print()
        except Exception as e:
            print(f"Errore caricamento PV: {e}")
            print("Procedo senza sistema fotovoltaico")
            pv_df = None
            pv_system = None
            print()

    battery = Battery(
        technology=BATTERY_TECHNOLOGY,
        capacity_mwh=BATTERY_CAPACITY_MWH,
        max_power_mw=BATTERY_MAX_POWER_MW,
        max_c_rate=BATTERY_MAX_C_RATE
    )

    optimizer = PSOOptimizer(n_particles=50, n_iterations=100,
                             w_start=0.9, w_end=0.4, c1=2.0, c2=2.0)

    simulator = RollingHorizonSimulator(battery, optimizer, pv_system=pv_system,
                                        horizon_hours=24, step_hours=1)

    start_time = datetime.now()
    results_df, trading_profit = simulator.simulate(df, pv_df)
    end_time = datetime.now()

    macse_revenue, macse_base, macse_penalty, macse_bonus = calculate_macse_revenue(battery)
    total_system_profit = trading_profit + macse_revenue

    print("=" * 80)
    print("RISULTATI SIMULAZIONE")
    print("=" * 80)
    print(f"\nTECNOLOGIA: {battery.technology}")
    print(f"  - DoD: {battery.dod * 100}%")
    print(f"  - SOC range: {battery.soc_min * 100}% - {battery.soc_max * 100}%")
    print(f"  - Efficienza round-trip: {battery.efficiency * 100:.1f}%")
    print(f"  - EOL cycles: {battery.eol_cycles} cicli")

    print("\nTRADING:")
    print(f"  Capacita trading: {battery.trading_capacity:.2f} MWh")
    print(f"  Profitto operativo: {trading_profit:,.2f} Euro")
    print(f"  Energia da rete: {battery.energy_from_grid_mwh:.2f} MWh")
    if PV_ENABLED and pv_system:
        print(f"  Energia da PV: {battery.energy_from_pv_mwh:.2f} MWh")
        pv_percentage = (battery.energy_from_pv_mwh / (battery.energy_from_grid_mwh + battery.energy_from_pv_mwh) * 100) if (battery.energy_from_grid_mwh + battery.energy_from_pv_mwh) > 0 else 0
        print(f"  % Carica da PV: {pv_percentage:.1f}%")

    if PV_ENABLED and pv_system:
        pv_stats = pv_system.get_statistics()
        print(f"\nFOTOVOLTAICO (STRATEGIA AUTONOMA PSO):")
        print(f"  Produzione totale: {pv_stats['total_production_mwh']:.2f} MWh")
        print(f"  → Batteria (decisione PSO): {pv_stats['energy_to_battery_mwh']:.2f} MWh ({pv_stats['battery_utilization_percent']:.1f}%)")
        print(f"  → Rete diretta (decisione PSO): {pv_stats['energy_to_grid_mwh']:.2f} MWh ({pv_stats['grid_sale_percent']:.1f}%)")
        print(f"  → Curtailment: {pv_stats['curtailed_energy_mwh']:.2f} MWh ({pv_stats['curtailment_percent']:.1f}%)")

    if MACSE_ENABLED:
        print(f"\nMACSE:")
        print(f"  Capacita MACSE: {MACSE_CAPACITY_MWH:.2f} MWh")
        print(f"  Ricavo netto: {macse_revenue:,.2f} Euro")
        print(f"  Disponibilita: {battery.get_macse_availability_factor() * 100:.2f}%")

    print(f"\nSISTEMA TOTALE:")
    print(f"  Profitto totale: {total_system_profit:,.2f} Euro")

    print(f"\nSTATO BATTERIA:")
    print(f"  SOC finale: {battery.get_soc() * 100:.1f}%")
    print(f"  SOH finale: {battery.get_soh():.2f}%")
    print(f"  Cicli equivalenti: {battery.equivalent_cycles:.1f}")
    print(f"  Tempo esecuzione: {(end_time - start_time).total_seconds():.1f} secondi")
    print("=" * 80)

    # Salva risultati
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    output_file = os.path.join(results_folder,
                               f'risultati_{battery.technology.lower().replace("-", "_")}_pv_autonomous.xlsx')

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Risultati Orari', index=False)

        params_data = {
            'Parametro': [
                'Versione',
                'Tecnologia',
                'Strategia PV',
                'Capacita nominale (MWh)',
                'PV potenza (kWp)',
                'PV produzione (MWh)',
                'PV verso batteria (MWh)',
                'PV verso rete (MWh)',
                'Profitto Trading (Euro)',
                'Ricavi MACSE (Euro)',
                'Profitto Totale (Euro)',
                'SOH finale (%)',
            ],
            'Valore': [
                '2.3.0-PV-AUTONOMOUS',
                battery.technology,
                'AUTONOMOUS_PSO_DECISION',
                battery.nominal_capacity,
                PV_NOMINAL_POWER_KWP if PV_ENABLED else 0,
                pv_stats['total_production_mwh'] if PV_ENABLED and pv_system else 0,
                pv_stats['energy_to_battery_mwh'] if PV_ENABLED and pv_system else 0,
                pv_stats['energy_to_grid_mwh'] if PV_ENABLED and pv_system else 0,
                trading_profit,
                macse_revenue,
                total_system_profit,
                battery.get_soh(),
            ]
        }
        params_df = pd.DataFrame(params_data)
        params_df.to_excel(writer, sheet_name='Parametri', index=False)

    print(f"\nRisultati Excel salvati in: {output_file}")

    # Export JSON
    export_results_to_json(
        results_df, battery, pv_system, trading_profit, macse_revenue, macse_base,
        macse_penalty, macse_bonus, 600000, (end_time - start_time).total_seconds()
    )

    # Visualizzazioni PV
    print("\n" + "=" * 80)
    print("INIZIO GENERAZIONE GRAFICI PV")
    print("=" * 80)
    print(f"PV_ENABLED = {PV_ENABLED}")
    print(f"SAVE_PLOTS = {SAVE_PLOTS}")
    print(f"pv_system = {pv_system}")
    print(f"Condizione check: {PV_ENABLED and pv_system}")
    print("=" * 80)

    if PV_ENABLED and pv_system:
        print("→ Chiamata create_pv_impact_comparison...")
        create_pv_impact_comparison(results_df, battery, pv_system)  # QUESTO È IL GRAFICO CHIAVE!
        print("→ Chiamata create_pv_impact_summary...")
        create_pv_impact_summary(results_df, battery, pv_system)
        print("→ Chiamata create_pv_visualizations...")
        create_pv_visualizations(results_df, battery, pv_system)
        print("→ Chiamata create_detailed_monthly_pv_plots...")
        create_detailed_monthly_pv_plots(results_df, battery, pv_system)
        print("✓ TUTTE LE FUNZIONI CHIAMATE!")
    else:
        print("✗ CONDIZIONE FALSA - GRAFICI NON GENERATI!")
        print(f"   PV_ENABLED = {PV_ENABLED}")
        print(f"   pv_system = {pv_system}")

    print("\n" + "=" * 80)
    print("SIMULAZIONE COMPLETATA CON SUCCESSO")
    print("=" * 80)


if __name__ == "__main__":
    pv_file = pv_production_file if PV_ENABLED else None
    main(file_name, pv_file)