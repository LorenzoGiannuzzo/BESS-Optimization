"""
------------------------------------------------------------------------------------------------------------------------
BATTERY ENERGY STORAGE SYSTEM (BESS) OPTIMIZATION - MULTI TECHNOLOGY
Particle Swarm Optimization with Rolling Horizon and MACSE Integration
Supporto per Litio-ione e Grafene
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
    - Supporto per tecnologie Litio-ione e Grafene
Version: 2.2.0 - CORRECTED
Date: November 2025
------------------------------------------------------------------------------------------------------------------------
MIT License
Copyright (c) 2024 Lorenzo Giannuzzo
------------------------------------------------------------------------------------------------------------------------
MODIFICHE VERSIONE 2.2.0:
- DEGRADATION_COST_PER_MWH ridotto da 25 a 10 Euro/MWh
- MACSE_CAPACITY fisso a 1.0 MWh (non più percentuale)
- Penalizzazione degrado applicata sia in carica che scarica
- Cicli equivalenti mantenuti con formula originale
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
file_name = '20240101_20241231_PUN.xlsx'        # Lorenzo Giannuzzo: Da mettere dentro la cartella data
# ---------------------------------- SCELTA TECNOLOGIA BATTERIA -------------------------------------------
BATTERY_TECHNOLOGY = "LITIO-IONE"               # Lorenzo Giannuzzo: Tecnologia batteria ["LITIO-IONE" | "GRAFENE"]

# ---------------------------------- PARAMETRI BATTERIA ---------------------------------------------------
BATTERY_CAPACITY_MWH = 4.0                      # Lorenzo Giannuzzo: Capacità nominale batteria [MWh]
BATTERY_MAX_POWER_MW = 2.0                      # Lorenzo Giannuzzo: Potenza massima sistema [MW]
BATTERY_MAX_C_RATE = 0.5                        # Lorenzo Giannuzzo: C-rate massimo operativo [C]
BATTERY_BASE_EFFICIENCY = 0.95                  # Lorenzo Giannuzzo: Efficienza round-trip nominale [adimensionale]

# ---------------------------------- PARAMETRI SPECIFICI PER TECNOLOGIA -----------------------------------
# Lorenzo Giannuzzo: Configurazione LITIO-IONE
LITHIUM_ION_SOC_MIN = 0.1                       # Lorenzo Giannuzzo: State of Charge minimo [adimensionale]
LITHIUM_ION_SOC_MAX = 0.9                       # Lorenzo Giannuzzo: State of Charge massimo [adimensionale]
LITHIUM_ION_DOD = 0.8                           # Lorenzo Giannuzzo: Depth of Discharge operativo [adimensionale]
LITHIUM_ION_EOL_CYCLES = 6000                   # Lorenzo Giannuzzo: Cicli nominali End-of-Life [cycles]

# Lorenzo Giannuzzo: Configurazione GRAFENE
GRAPHENE_SOC_MIN = 0.0                          # Lorenzo Giannuzzo: State of Charge minimo [adimensionale]
GRAPHENE_SOC_MAX = 1.0                          # Lorenzo Giannuzzo: State of Charge massimo [adimensionale]
GRAPHENE_DOD = 1.0                              # Lorenzo Giannuzzo: Depth of Discharge operativo [adimensionale]
GRAPHENE_EOL_CYCLES = 500000                    # Lorenzo Giannuzzo: Cicli nominali End-of-Life [cycles]

# ---------------------------------- PARAMETRI GENERALI ---------------------------------------------------
SAVE_PLOTS = True                               # Lorenzo Giannuzzo: Flag salvataggio grafici [bool]
DEGRADATION_COST_PER_MWH = 10.0                 # Lorenzo Giannuzzo: Costo degrado per ottimizzatore [Euro/MWh]

# ---------------------------------- PARAMETRI MACSE ------------------------------------------------------
MACSE_ENABLED = True                            # Lorenzo Giannuzzo: Abilitazione servizio MACSE [bool]
MACSE_CAPACITY_MWH = 1.0                        # Lorenzo Giannuzzo: Capacità fissa allocata MACSE [MWh]
MACSE_CONTRACT_YEARS = 1                        # Lorenzo Giannuzzo: Durata contratto MACSE [anni]
MACSE_PRICE_PER_MW_YEAR = 50000                 # Lorenzo Giannuzzo: Remunerazione contratto MACSE [Euro/MW/anno]
MACSE_AVAILABILITY_TARGET = 0.95                # Lorenzo Giannuzzo: Target disponibilità contrattuale [adimensionale]
MACSE_PENALTY_THRESHOLD = 0.92                  # Lorenzo Giannuzzo: Soglia applicazione penali [adimensionale]
MACSE_BONUS_THRESHOLD = 0.98                    # Lorenzo Giannuzzo: Soglia erogazione bonus [adimensionale]
MACSE_PENALTY_MULTIPLIER = 1.5                  # Lorenzo Giannuzzo: Fattore moltiplicativo penali [adimensionale]
MACSE_BONUS_PERCENTAGE = 0.10                   # Lorenzo Giannuzzo: Percentuale bonus disponibilità [adimensionale]
MACSE_RESPONSE_TIME_MINUTES = 15                # Lorenzo Giannuzzo: Tempo risposta servizio [minuti]
MACSE_MIN_SOC_FOR_DISCHARGE = 0.20              # Lorenzo Giannuzzo: SOC minimo per erogazione MACSE [adimensionale]
MACSE_MAX_SOC_FOR_CHARGE = 0.80                 # Lorenzo Giannuzzo: SOC massimo per assorbimento MACSE [adimensionale]

np.random.seed(42)                              # Lorenzo Giannuzzo: Seed riproducibilità algoritmo PSO

# ========================================================================================================
# SEZIONE 2: DATI SPERIMENTALI CURVE DI EFFICIENZA
# ========================================================================================================
# Lorenzo Giannuzzo: Dati sperimentali LITIO-IONE @ 0.5C
# Lorenzo Giannuzzo: Test effettuati su 3 cicli completi per validazione statistica
LITHIUM_ION_05C_DATA = {
    'charge_energy_kwh': [11.45, 11.42, 11.48],
    'discharge_energy_kwh': [10.85, 10.88, 10.82],
    'energy_efficiency': [0.948, 0.952, 0.942],
    'coulombic_efficiency': [0.995, 0.997, 0.994]
}
# Lorenzo Giannuzzo: Dati sperimentali LITIO-IONE @ 1C
# Lorenzo Giannuzzo: Incremento perdite per effetto Joule e polarizzazione
LITHIUM_ION_1C_DATA = {
    'charge_energy_kwh': [11.38, 11.42, 11.35],
    'discharge_energy_kwh': [10.45, 10.52, 10.48],
    'energy_efficiency': [0.918, 0.921, 0.923],
    'coulombic_efficiency': [0.985, 0.988, 0.986]
}

"""
DIFFERENZE RISPETTO AL GRAFENE:
1. EFFICIENZA ENERGETICA:
   - Litio-ione @ 0.5C: ~94-95% (vs Grafene ~96%)
   - Litio-ione @ 1C: ~91-92% (vs Grafene ~94-95%)
2. EFFICIENZA COULOMBICA:
   - Litio-ione @ 0.5C: ~99.5% (vs Grafene ~98%)
   - Litio-ione @ 1C: ~98.5-99% (vs Grafene ~96-97%)
3. CARATTERISTICHE:
   - Litio-ione (LFP): Più maturo, costi inferiori, buona durata (3000-5000 cicli @ 80% DoD)
   - Grafene: Maggiore efficienza energetica, DoD 100%, cicli vita superiori (10000+)
4. DEGRADAZIONE:
   - Litio-ione: Più sensibile a DoD elevati, limitazione a 80% DoD
   - Grafene: Migliore stabilità elettrochimica, 100% DoD utilizzabile
5. PERDITE:
   - Litio-ione: Perdite interne ~5-8% per ciclo completo
   - Grafene: Perdite interne ~4-6% per ciclo completo
"""
# Lorenzo Giannuzzo: Dati sperimentali GRAFENE @ 0.5C
# Lorenzo Giannuzzo: Prestazioni superiori grazie a bassa resistenza interna
GRAPHENE_05C_DATA = {
    'charge_energy_kwh': [11.5, 11.308, 11.3],
    'discharge_energy_kwh': [11.0, 10.7, 10.9],
    'energy_efficiency': [0.956, 0.964, 0.964],
    'coulombic_efficiency': [0.96, 0.983, 0.984]
}
# Lorenzo Giannuzzo: Dati sperimentali GRAFENE @ 1C
# Lorenzo Giannuzzo: Mantenimento efficienza elevata anche a C-rate superiori
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
    """
    Lorenzo Giannuzzo: Modello parametrico efficienza basato su caratterizzazione sperimentale
    Lorenzo Giannuzzo: Implementa curve di efficienza funzione del C-rate operativo
    """
    def __init__(self, technology, c_rate):
        """
        Inizializza il modello di efficienza
        Args:
            technology: Tecnologia batteria ["LITIO-IONE" | "GRAFENE"]
            c_rate: C-rate operativo [C]
        """
        self.technology = technology
        self.c_rate = c_rate
        # Lorenzo Giannuzzo: Selezione dataset sperimentale in base a tecnologia e C-rate
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
        # Lorenzo Giannuzzo: Calcolo statistiche medie da dataset sperimentale
        self.avg_energy_efficiency = np.mean(self.data['energy_efficiency'])
        self.avg_coulombic_efficiency = np.mean(self.data['coulombic_efficiency'])
        self.avg_charge_energy = np.mean(self.data['charge_energy_kwh'])
        self.avg_discharge_energy = np.mean(self.data['discharge_energy_kwh'])
    def get_charge_efficiency(self):
        """
        Lorenzo Giannuzzo: Calcolo efficienza fase di carica
        Lorenzo Giannuzzo: Approssimazione η_charge ≈ √(η_roundtrip)
        Returns:
            Efficienza carica [adimensionale]
        """
        return np.sqrt(self.avg_energy_efficiency)
    def get_discharge_efficiency(self):
        """
        Lorenzo Giannuzzo: Calcolo efficienza fase di scarica
        Lorenzo Giannuzzo: Approssimazione η_discharge ≈ √(η_roundtrip)

        Returns:
            Efficienza scarica [adimensionale]
        """
        return np.sqrt(self.avg_energy_efficiency)
    def get_roundtrip_efficiency(self):
        """
        Lorenzo Giannuzzo: Efficienza ciclo completo carica-scarica
        Returns:
            Efficienza round-trip [adimensionale]
        """
        return self.avg_energy_efficiency

# ========================================================================================================
# SEZIONE 4: DEGRADO
# ========================================================================================================
def degradation(cycle_num):
    """
    Lorenzo Giannuzzo: Modello polinomiale degrado capacità batteria
    Lorenzo Giannuzzo: Funzione di ordine 9 calibrata su dati sperimentali LFP
    Lorenzo Giannuzzo: Rappresenta capacità residua in funzione dei cicli equivalenti
    Args:
        cycle_num: Numero cicli equivalenti accumulati [cycles]
    Returns:
        Capacità residua rispetto a nominale [%]
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

# ========================================================================================================
# SEZIONE 5: MODELLO BATTERIA
# ========================================================================================================
class Battery:
    """
    Lorenzo Giannuzzo: Modello fisico-elettrico batteria multi-tecnologia
    Lorenzo Giannuzzo: Implementa dinamiche SOC, degrado e vincoli operativi
    Lorenzo Giannuzzo: Integrazione servizi ancillari MACSE con allocazione fissa
    """
    def __init__(self, technology=BATTERY_TECHNOLOGY,
                 capacity_mwh=BATTERY_CAPACITY_MWH,
                 max_power_mw=BATTERY_MAX_POWER_MW,
                 max_c_rate=BATTERY_MAX_C_RATE):
        """
        Inizializza modello batteria
        Args:
            technology: Tecnologia elettrochimica ["LITIO-IONE" | "GRAFENE"]
            capacity_mwh: Capacità energetica nominale [MWh]
            max_power_mw: Potenza nominale inverter [MW]
            max_c_rate: Tasso scarica/carica massimo [C]
        """
        self.technology = technology
        self.nominal_capacity = capacity_mwh
        self.capacity = capacity_mwh
        self.max_power = max_power_mw
        self.max_c_rate = max_c_rate
        # Lorenzo Giannuzzo: Inizializzazione parametri tecnologia-dipendenti
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
        # Lorenzo Giannuzzo: SOC iniziale al punto medio del range operativo
        self.soc = (self.soc_min + self.soc_max) / 2
        # Lorenzo Giannuzzo: Inizializzazione modello efficienza da dati sperimentali
        self.efficiency_model = BatteryEfficiencyModel(technology, max_c_rate)
        self.efficiency = self.efficiency_model.get_roundtrip_efficiency()
        self.charge_efficiency = self.efficiency_model.get_charge_efficiency()
        self.discharge_efficiency = self.efficiency_model.get_discharge_efficiency()
        # Lorenzo Giannuzzo: Contatori degrado batteria
        self.equivalent_cycles = 0.0
        self.throughput_kwh = 0.0
        # Lorenzo Giannuzzo: Allocazione capacità per servizi MACSE - CAPACITÀ FISSA
        if MACSE_ENABLED:
            self.macse_capacity = MACSE_CAPACITY_MWH
            self.trading_capacity = capacity_mwh - MACSE_CAPACITY_MWH
            # Potenza proporzionale alla capacità allocata
            macse_percentage = MACSE_CAPACITY_MWH / capacity_mwh
            self.macse_power = max_power_mw * macse_percentage
            self.trading_power = max_power_mw * (1 - macse_percentage)
        else:
            self.macse_capacity = 0
            self.trading_capacity = capacity_mwh
            self.macse_power = 0
            self.trading_power = max_power_mw
        # Lorenzo Giannuzzo: Metriche disponibilità servizio MACSE
        self.macse_available_hours = 0
        self.macse_total_hours = 0
    def charge(self, power_mw, dt=1.0):
        """
        Lorenzo Giannuzzo: Gestione fase di carica batteria
        Lorenzo Giannuzzo: Implementa vincoli C-rate e saturazione SOC
        Lorenzo Giannuzzo: Opera solo su capacità allocata al trading
        Args:
            power_mw: Potenza richiesta assorbimento [MW]
            dt: Timestep simulazione [h]
        Returns:
            Energia effettivamente prelevata da rete [MWh]
        """
        # Lorenzo Giannuzzo: Calcolo potenza massima ammissibile
        max_power_c_rate = self.trading_capacity * self.max_c_rate
        max_power_trading = min(self.trading_power, max_power_c_rate)
        power_mw = min(power_mw, max_power_trading)
        # Lorenzo Giannuzzo: Calcolo energia stoccabile con perdite efficienza
        energy = power_mw * dt * self.charge_efficiency
        new_soc = self.soc + (energy / self.capacity)
        new_soc = min(new_soc, self.soc_max)
        # Lorenzo Giannuzzo: Aggiornamento stato batteria
        actual_energy = (new_soc - self.soc) * self.capacity
        self.soc = new_soc
        self.throughput_kwh += actual_energy * 1000
        return actual_energy / self.charge_efficiency

    def discharge(self, power_mw, dt=1.0):
        """
        Lorenzo Giannuzzo: Gestione fase di scarica batteria
        Lorenzo Giannuzzo: Implementa vincoli C-rate e protezione deep discharge
        Lorenzo Giannuzzo: Opera solo su capacità allocata al trading
        Args:
            power_mw: Potenza richiesta erogazione [MW]
            dt: Timestep simulazione [h]
        Returns:
            Energia effettivamente immessa in rete [MWh]
        """
        # Lorenzo Giannuzzo: Calcolo potenza massima erogabile
        max_power_c_rate = self.trading_capacity * self.max_c_rate
        max_power_trading = min(self.trading_power, max_power_c_rate)
        power_mw = min(power_mw, max_power_trading)
        # Lorenzo Giannuzzo: Calcolo energia prelevabile con perdite efficienza
        energy = power_mw * dt / self.discharge_efficiency
        new_soc = self.soc - (energy / self.capacity)
        new_soc = max(new_soc, self.soc_min)
        # Lorenzo Giannuzzo: Aggiornamento stato batteria
        actual_energy = (self.soc - new_soc) * self.capacity
        self.soc = new_soc
        self.throughput_kwh += actual_energy * 1000
        return actual_energy * self.discharge_efficiency

    def check_macse_availability(self):
        """
        Lorenzo Giannuzzo: Verifica requisiti tecnici disponibilità MACSE
        Lorenzo Giannuzzo: Controlla vincoli SOC e potenza disponibile
        Returns:
            Flag disponibilità servizio [bool]
        """
        if not MACSE_ENABLED:
            return False
        soc_ok_for_discharge = self.soc >= MACSE_MIN_SOC_FOR_DISCHARGE
        soc_ok_for_charge = self.soc <= MACSE_MAX_SOC_FOR_CHARGE
        power_ok = self.macse_power >= 0.9 * (self.max_power * (MACSE_CAPACITY_MWH / self.nominal_capacity))
        return (soc_ok_for_discharge or soc_ok_for_charge) and power_ok

    def update_macse_availability(self, is_available):
        """
        Lorenzo Giannuzzo: Aggiornamento contatori disponibilità MACSE
        Lorenzo Giannuzzo: Utilizzato per calcolo KPI contrattuali
        Args:
            is_available: Flag disponibilità timestep corrente [bool]
        """
        self.macse_total_hours += 1
        if is_available:
            self.macse_available_hours += 1

    def get_macse_availability_factor(self):
        """
        Lorenzo Giannuzzo: Calcolo fattore disponibilità MACSE
        Lorenzo Giannuzzo: Metrica per valutazione penali/bonus contrattuali
        Returns:
            Fattore disponibilità cumulativo [adimensionale 0-1]
        """
        if self.macse_total_hours == 0:
            return 1.0
        return self.macse_available_hours / self.macse_total_hours

    def update_degradation(self):
        """
        Lorenzo Giannuzzo: Aggiornamento capacità per effetto degrado
        Lorenzo Giannuzzo: Applica modello polinomiale in funzione throughput
        """
        self.equivalent_cycles = self.throughput_kwh / (2 * self.nominal_capacity * 1000)
        capacity_percentage = degradation(self.equivalent_cycles)
        self.capacity = self.nominal_capacity * (capacity_percentage / 100.0)
        if MACSE_ENABLED:
            # La capacità MACSE si degrada proporzionalmente
            macse_percentage_original = MACSE_CAPACITY_MWH / self.nominal_capacity
            self.macse_capacity = self.capacity * macse_percentage_original
            self.trading_capacity = self.capacity - self.macse_capacity

    def get_soh(self):
        """
        Lorenzo Giannuzzo: State of Health batteria
        Returns:
            Percentuale capacità residua [%]
        """
        return (self.capacity / self.nominal_capacity) * 100.0

    def get_soc(self):
        """
        Lorenzo Giannuzzo: State of Charge batteria
        Returns:
            Livello carica normalizzato [adimensionale 0-1]
        """
        return self.soc

    def get_dod(self):
        """
        Lorenzo Giannuzzo: Depth of Discharge operativo
        Returns:
            Profondità scarica utilizzabile [adimensionale 0-1]
        """
        return self.dod

    def get_usable_capacity(self):
        """
        Lorenzo Giannuzzo: Capacità energetica effettivamente utilizzabile
        Returns:
            Capacità netta considerando DoD [MWh]
        """
        return self.capacity * self.dod

    def copy(self):
        """
        Lorenzo Giannuzzo: Deep copy istanza batteria per simulazioni PSO
        Returns:
            Nuova istanza Battery con stato identico
        """
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
        return b

    def get_max_power_by_crate(self):
        """
        Lorenzo Giannuzzo: Calcolo potenza massima limitata da C-rate
        Returns:
            Potenza massima trading ammissibile [MW]
        """
        return self.trading_capacity * self.max_c_rate


# ========================================================================================================
# SEZIONE 6: OTTIMIZZATORE PSO
# ========================================================================================================
class PSOOptimizer:
    """PSO ottimizzato con inizializzazione smart e penalizzazione degrado corretta"""

    def __init__(self, n_particles=50, n_iterations=150, w_start=0.95, w_end=0.2, c1=2.0, c2=2.0):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.stagnation_limit = 15

    def optimize(self, battery, prices, horizon_hours=24):
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

    def _evaluate(self, battery, actions, prices):
        """
        Valuta la fitness di una soluzione PSO
        Args:
            battery: Istanza di Battery
            actions: Array di azioni [MW] per ogni ora
            prices: Array di prezzi [Euro/MWh] per ogni ora
        Returns:
            Profitto penalizzato per degrado [Euro] - solo per PSO
        """
        bat_sim = battery.copy()
        profit = 0.0
        for power, price in zip(actions, prices):
            if power > 0:  # CARICA
                if bat_sim.soc >= bat_sim.soc_max:
                    continue
                max_energy_storable = (bat_sim.soc_max - bat_sim.soc) * bat_sim.trading_capacity
                max_power_available = max_energy_storable / (1.0 * bat_sim.charge_efficiency)
                actual_power = min(power, max_power_available)
                if actual_power > 0.01:
                    energy_from_grid = bat_sim.charge(actual_power, dt=1.0)
                    profit -= energy_from_grid * price
                    degradation_penalty = abs(energy_from_grid) * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                    profit -= degradation_penalty
            elif power < 0:  # SCARICA
                if bat_sim.soc <= bat_sim.soc_min:
                    continue
                max_energy_available = (bat_sim.soc - bat_sim.soc_min) * bat_sim.trading_capacity
                max_power_available = max_energy_available * bat_sim.discharge_efficiency / 1.0
                actual_power = min(-power, max_power_available)
                if actual_power > 0.01:
                    energy_to_grid = bat_sim.discharge(actual_power, dt=1.0)
                    profit += energy_to_grid * price
                    degradation_penalty = abs(energy_to_grid) * DEGRADATION_COST_PER_MWH / (2 * bat_sim.eol_cycles)
                    profit -= degradation_penalty
        return profit


# ========================================================================================================
# SEZIONE 7: ROLLING HORIZON SIMULATOR
# ========================================================================================================
class RollingHorizonSimulator:

    def __init__(self, battery, optimizer, horizon_hours=24, step_hours=1):
        self.battery = battery
        self.optimizer = optimizer
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours

    def simulate(self, prices_df):
        prices = prices_df['€/MWh'].values
        n_hours = len(prices)
        actions_taken = []
        soc_history = []
        capacity_history = []
        soh_history = []
        profits_history = []
        macse_availability_history = []
        cumulative_profit = 0.0
        print("=" * 80)
        print("SIMULAZIONE BESS CON PSO, ROLLING HORIZON E MACSE")
        print("=" * 80)
        print(f"TECNOLOGIA BATTERIA: {self.battery.technology}")
        print(f"Capacita nominale: {self.battery.nominal_capacity} MWh")
        print(f"Potenza massima: {self.battery.max_power} MW")
        print(f"C-rate massimo: {self.battery.max_c_rate}C")
        print(f"DoD utilizzabile: {self.battery.dod * 100}%")
        print(f"SOC range: {self.battery.soc_min * 100}% - {self.battery.soc_max * 100}%")
        print(f"Efficienza round-trip: {self.battery.efficiency * 100:.1f}%")
        print(f"  - Efficienza carica: {self.battery.charge_efficiency * 100:.1f}%")
        print(f"  - Efficienza scarica: {self.battery.discharge_efficiency * 100:.1f}%")
        print(f"EOL cycles: {self.battery.eol_cycles} cicli")
        if MACSE_ENABLED:
            print(f"\nMACSE ABILITATO:")
            print(f"  - Capacita allocata MACSE: {self.battery.macse_capacity:.2f} MWh (FISSO)")
            print(f"  - Capacita per trading: {self.battery.trading_capacity:.2f} MWh")
            print(f"  - Potenza MACSE: {self.battery.macse_power:.2f} MW")
            print(f"  - Potenza trading: {self.battery.trading_power:.2f} MW")
            print(f"  - Prezzo contratto: {MACSE_PRICE_PER_MW_YEAR:,.0f} Euro/MW/anno")
        else:
            print(f"\nMACSE DISABILITATO")
            print(f"  - Intera capacita batteria ({self.battery.capacity:.2f} MWh) per trading")
            print(f"  - Potenza totale per trading: {self.battery.max_power:.2f} MW")
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
                print(
                    f"Progresso: {progress}% ({current_hour}/{n_hours} h) - SOH: {self.battery.get_soh():.2f}% - MACSE Avail: {avail_factor * 100:.1f}%")
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
            if action > 0.01:
                if self.battery.soc < self.battery.soc_max:
                    max_energy_storable = (self.battery.soc_max - self.battery.soc) * self.battery.trading_capacity
                    max_power_available = max_energy_storable / (1.0 * self.battery.charge_efficiency)
                    actual_power = min(action, max_power_available)
                    action = actual_power
                    if actual_power > 0.01:
                        energy_from_grid = self.battery.charge(actual_power, dt=1.0)
                        profit = -energy_from_grid * price
                    else:
                        profit = 0.0
                else:
                    profit = 0.0
                    action = 0.0
            elif action < -0.01:
                if self.battery.soc > self.battery.soc_min:
                    max_energy_available = (self.battery.soc - self.battery.soc_min) * self.battery.trading_capacity
                    max_power_available = max_energy_available * self.battery.discharge_efficiency / 1.0
                    actual_power = min(-action, max_power_available)
                    action = -actual_power
                    if actual_power > 0.01:
                        energy_to_grid = self.battery.discharge(actual_power, dt=1.0)
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
            cumulative_profit += profit
            actions_taken.append(action)
            soc_history.append(self.battery.get_soc())
            capacity_history.append(self.battery.capacity)
            soh_history.append(self.battery.get_soh())
            profits_history.append(cumulative_profit)
            macse_availability_history.append(self.battery.get_macse_availability_factor() if MACSE_ENABLED else 0)
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
        return results_df, cumulative_profit


# ========================================================================================================
# SEZIONE 8: CALCOLO RICAVI MACSE
# ========================================================================================================
def calculate_macse_revenue(battery):
    """
    Calcola ricavi e penali MACSE basati sulla capacità fissa allocata
    Args:
        battery: Istanza di Battery
    Returns:
        Tuple (ricavo_annuale, base_revenue, penalty, bonus) tutti in [Euro]
    """
    if not MACSE_ENABLED:
        return 0, 0, 0, 0
    # Usa la potenza corrispondente alla capacità MACSE allocata
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
# SEZIONE 9: EXPORT JSON RISULTATI
# ========================================================================================================

"""
STRUTTURA FILE JSON OUTPUT
--------------------------

Il file JSON contiene tutti i risultati della simulazione organizzati in sezioni logiche.
Ogni valore è accompagnato dalla sua unità di misura come chiave separata.

SEZIONE 1: simulation_info
    - Informazioni generali sulla simulazione
    - timestamp: Data e ora di esecuzione [ISO 8601 format]
    - version: Versione del codice [string]
    - simulation_duration_seconds: Tempo di esecuzione [s]

SEZIONE 2: battery_configuration
    - Parametri configurazione batteria
    - technology: Tecnologia batteria [string: "LITIO-IONE" o "GRAFENE"]
    - nominal_capacity_mwh: Capacità nominale [MWh]
    - max_power_mw: Potenza massima [MW]
    - max_c_rate: C-rate massimo [C]
    - dod: Depth of Discharge [adimensionale 0-1]
    - soc_min: SOC minimo [adimensionale 0-1]
    - soc_max: SOC massimo [adimensionale 0-1]
    - eol_cycles: Cicli a fine vita [cycles]
    - efficiency_roundtrip: Efficienza round-trip [adimensionale 0-1]
    - efficiency_charge: Efficienza carica [adimensionale 0-1]
    - efficiency_discharge: Efficienza scarica [adimensionale 0-1]

SEZIONE 3: macse_configuration
    - Parametri configurazione MACSE
    - enabled: Flag attivazione [boolean]
    - capacity_mwh: Capacità allocata FISSA [MWh]
    - power_mw: Potenza allocata [MW]
    - contract_price_eur_per_mw_year: Prezzo contratto [Euro/MW/anno]
    - availability_target: Target disponibilità [adimensionale 0-1]
    - penalty_threshold: Soglia penale [adimensionale 0-1]
    - bonus_threshold: Soglia bonus [adimensionale 0-1]

SEZIONE 4: trading_results
    - Risultati economici del trading energetico
    - gross_profit_eur: Profitto lordo da arbitraggio [Euro]
    - energy_traded_mwh: Energia totale scambiata [MWh]
    - energy_charged_mwh: Energia totale caricata [MWh]
    - energy_discharged_mwh: Energia totale scaricata [MWh]
    - average_buy_price_eur_per_mwh: Prezzo medio acquisto [Euro/MWh]
    - average_sell_price_eur_per_mwh: Prezzo medio vendita [Euro/MWh]

SEZIONE 5: macse_results
    - Risultati economici MACSE
    - annual_revenue_eur: Ricavo annuale netto [Euro]
    - base_revenue_eur: Remunerazione base [Euro]
    - penalty_eur: Penali applicate [Euro]
    - bonus_eur: Bonus ricevuti [Euro]
    - availability_factor: Fattore disponibilità [adimensionale 0-1]
    - available_hours: Ore disponibili [h]
    - total_hours: Ore totali [h]

SEZIONE 6: battery_health
    - Stato di salute batteria
    - final_soc: SOC finale [adimensionale 0-1]
    - final_soh_percent: SOH finale [%]
    - final_capacity_mwh: Capacità finale [MWh]
    - equivalent_cycles: Cicli equivalenti [cycles]
    - throughput_kwh: Throughput totale [kWh]
    - capacity_fade_percent: Perdita capacità [%]

SEZIONE 7: economic_summary
    - Riepilogo economico totale
    - total_revenue_eur: Ricavi totali (trading + MACSE) [Euro]
    - trading_profit_eur: Profitto da trading [Euro]
    - macse_revenue_eur: Ricavi da MACSE [Euro]
    - battery_investment_eur: Investimento batteria [Euro]
    - residual_value_eur: Valore residuo batteria [Euro]
    - net_return_eur: Ritorno netto [Euro]
    - roi_percent: Return on Investment [%]
    - payback_years: Tempo di ritorno [anni]

SEZIONE 8: time_series_data
    - Dati orari della simulazione (solo campione per evitare file troppo grandi)
    - sample_size: Numero di campioni salvati [adimensionale]
    - sampling_rate: Ogni quante ore è salvato un campione [h]
    - data: Array di oggetti con:
        - hour: Ora progressiva [h]
        - timestamp: Timestamp [ISO 8601]
        - price_eur_per_mwh: Prezzo energia [Euro/MWh]
        - action_mw: Azione (+ carica, - scarica) [MW]
        - soc: State of Charge [adimensionale 0-1]
        - soh_percent: State of Health [%]
        - profit_eur: Profitto cumulativo [Euro]

NOTE IMPORTANTI:
- Tutti i valori monetari sono in Euro
- Le energie sono in MWh o kWh come specificato
- Le potenze sono in MW
- I valori adimensionali (SOC, efficienza, disponibilità) sono nel range 0-1
- Le percentuali sono esplicitate con _percent nel nome
- I timestamp seguono il formato ISO 8601 (YYYY-MM-DDTHH:MM:SS)
"""
def export_results_to_json(results_df, battery, trading_profit, macse_revenue, macse_base,
                           macse_penalty, macse_bonus, battery_investment, simulation_time):
    """
    Esporta i risultati della simulazione in formato JSON
    Args:
        results_df: DataFrame risultati orari
        battery: Istanza di Battery
        trading_profit: Profitto da trading [Euro]
        macse_revenue: Ricavo MACSE [Euro]
        macse_base: Base revenue MACSE [Euro]
        macse_penalty: Penali MACSE [Euro]
        macse_bonus: Bonus MACSE [Euro]
        battery_investment: Investimento batteria [Euro]
        simulation_time: Tempo esecuzione [s]
    """
    # Calcola statistiche trading
    actions = results_df['Azione_MW'].values
    prices = results_df['€/MWh'].values
    energy_charged = sum([abs(a) for a in actions if a > 0])
    energy_discharged = sum([abs(a) for a in actions if a < 0])
    energy_traded = energy_charged + energy_discharged
    buy_prices = [prices[i] for i, a in enumerate(actions) if a > 0]
    sell_prices = [prices[i] for i, a in enumerate(actions) if a < 0]
    avg_buy_price = np.mean(buy_prices) if buy_prices else 0
    avg_sell_price = np.mean(sell_prices) if sell_prices else 0
    # Calcola metriche economiche
    total_revenue = trading_profit + macse_revenue
    residual_value = battery_investment * (battery.get_soh() / 100.0)
    net_return = total_revenue + residual_value - battery_investment
    roi = (net_return / battery_investment) * 100 if battery_investment > 0 else 0
    payback = battery_investment / total_revenue if total_revenue > 0 else float('inf')
    # CORREZIONE: Gestione timestamp - converti string a datetime se necessario
    sampling_rate = 24
    sampled_data = []
    for i in range(0, len(results_df), sampling_rate):
        row = results_df.iloc[i]
        # Gestione sicura del timestamp
        timestamp_str = None
        if pd.notnull(row['Data']):
            if isinstance(row['Data'], str):
                # Se è una stringa, prova a convertirla
                try:
                    timestamp_obj = pd.to_datetime(row['Data'], format='%d/%m/%Y')
                    timestamp_str = timestamp_obj.strftime('%Y-%m-%dT%H:%M:%S')
                except:
                    timestamp_str = row['Data']  # Usa la stringa così com'è
            else:
                # Se è già un datetime, usa strftime
                timestamp_str = row['Data'].strftime('%Y-%m-%dT%H:%M:%S')
        sampled_data.append({
            "hour": int(i),
            "timestamp": timestamp_str,
            "price_eur_per_mwh": float(row['€/MWh']),
            "action_mw": float(row['Azione_MW']),
            "soc": float(row['SOC']),
            "soh_percent": float(row['SOH_%']),
            "profit_eur": float(row['Profitto_Euro'])
        })
    # Costruisci struttura JSON
    output_data = {
        "simulation_info": {
            "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "version": "2.2.0",
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
            "average_buy_price_eur_per_mwh": avg_buy_price,
            "average_sell_price_eur_per_mwh": avg_sell_price
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
                                 f'simulation_results_{battery.technology.lower().replace("-", "_")}.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Risultati JSON salvati in: {json_filename}")
    return json_filename

# ========================================================================================================
# SEZIONE 10: VISUALIZZAZIONI
# ========================================================================================================
def create_visualizations(results_df, battery):
    """Crea visualizzazioni incluso MACSE e dettagli giornalieri mensili"""
    if not SAVE_PLOTS:
        print("\nSalvataggio grafici disabilitato")
        return
    viz_folder = 'visualization'
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    print("\n" + "=" * 80)
    print("GENERAZIONE VISUALIZZAZIONI")
    print("=" * 80)
    if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
        results_df['Data'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')
    results_df['Mese'] = results_df['Data'].dt.month
    results_df['Giorno'] = results_df['Data'].dt.day
    results_df['Anno'] = results_df['Data'].dt.year
    if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
        results_df['Data_dt'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')
    else:
        results_df['Data_dt'] = results_df['Data']
    df_daily = results_df.groupby(results_df['Data_dt'].dt.date, as_index=False).last()
    df_daily['Data_dt'] = pd.to_datetime(df_daily['Data_dt'])
    # ===========================================================================================
    # Grafico 1: Profitto e MACSE
    # ===========================================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(df_daily['Data_dt'], df_daily['Profitto_Euro'],
             color='#06A77D', linewidth=2, label=f'{battery.technology}')
    ax1.fill_between(df_daily['Data_dt'], 0, df_daily['Profitto_Euro'],
                     alpha=0.3, color='#06A77D')
    ax1.set_ylabel('Profitto Trading (Euro)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Profitto Trading nel Tempo - {battery.technology}', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if MACSE_ENABLED:
        ax2.plot(df_daily['Data_dt'], df_daily['MACSE_Availability'] * 100,
                 color='#457B9D', linewidth=2)
        ax2.axhline(y=MACSE_AVAILABILITY_TARGET * 100, color='green', linestyle='--',
                    alpha=0.5, label=f'Target {MACSE_AVAILABILITY_TARGET * 100}%')
        ax2.axhline(y=MACSE_PENALTY_THRESHOLD * 100, color='red', linestyle='--',
                    alpha=0.5, label=f'Soglia penale {MACSE_PENALTY_THRESHOLD * 100}%')
        ax2.set_ylabel('Disponibilita MACSE (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Disponibilita MACSE nel Tempo', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, f'01_profitto_macse_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvato: 01_profitto_macse_{battery.technology}.png")

    # ===========================================================================================
    # Grafico 2: SOH
    # ===========================================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_daily['Data_dt'], df_daily['SOH_%'], color='#E63946', linewidth=2,
            label=f'{battery.technology} (DoD {battery.dod * 100}%)')
    ax.set_ylabel('SOH (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'State of Health - {battery.technology}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, f'02_soh_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvato: 02_soh_{battery.technology}.png")

    # ===========================================================================================
    # Grafico 3: Griglia 12 mesi (un giorno rappresentativo per mese)
    # ===========================================================================================
    print("\nGenerazione grafico griglia 12 mesi...")

    mesi_nomi = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno',
                 'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']

    # Figura con 12 subplot (4 righe x 3 colonne)
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'Operazioni Giornaliere BESS - {battery.technology}\n(Un giorno rappresentativo per ogni mese)',
                 fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    for mese in range(1, 13):
        ax = axes[mese - 1]
        # Filtra dati per il mese corrente
        df_mese = results_df[results_df['Mese'] == mese].copy()
        if len(df_mese) == 0:
            ax.text(0.5, 0.5, 'Dati non disponibili',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{mesi_nomi[mese - 1]}', fontweight='bold')
            continue
        # Seleziona il 15° giorno del mese (o il giorno più vicino disponibile)
        giorni_disponibili = df_mese['Giorno'].unique()
        giorno_target = 15 if 15 in giorni_disponibili else giorni_disponibili[len(giorni_disponibili) // 2]
        df_giorno = df_mese[df_mese['Giorno'] == giorno_target].copy()
        if len(df_giorno) < 24:  # Se non abbiamo 24 ore, prendo il primo giorno completo
            for g in giorni_disponibili:
                df_test = df_mese[df_mese['Giorno'] == g]
                if len(df_test) >= 24:
                    df_giorno = df_test.copy()
                    giorno_target = g
                    break
        # Crea array ore (0-23)
        df_giorno = df_giorno.head(24).copy()
        df_giorno['Ora'] = range(len(df_giorno))
        # Crea il grafico a doppio asse Y
        ax2 = ax.twinx()
        # Plot potenza (asse sinistro) - barre colorate per carica/scarica
        colori_azioni = ['#E63946' if a < 0 else '#06A77D' if a > 0 else '#CCCCCC'
                         for a in df_giorno['Azione_MW'].values]
        bars = ax.bar(df_giorno['Ora'], df_giorno['Azione_MW'],
                      color=colori_azioni, alpha=0.7, width=0.8)
        # Plot prezzo (asse destro)
        line_prezzo = ax2.plot(df_giorno['Ora'], df_giorno['€/MWh'],
                               color='#F77F00', linewidth=2, marker='o', markersize=3,
                               label='Prezzo', zorder=10)
        # Plot SOC (asse destro)
        line_soc = ax2.plot(df_giorno['Ora'], df_giorno['SOC'] * 100,
                            color='#457B9D', linewidth=2, linestyle='--',
                            marker='s', markersize=3, label='SOC', zorder=11)
        # Configurazione assi
        ax.set_xlabel('Ora', fontsize=9)
        ax.set_ylabel('Potenza [MW]', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Prezzo [€/MWh] / SOC [%]', fontsize=9, fontweight='bold')
        # Titolo con data
        data_str = df_giorno['Data'].iloc[0].strftime('%d/%m/%Y') if len(df_giorno) > 0 else ''
        ax.set_title(f'{mesi_nomi[mese - 1]} - {data_str}', fontsize=10, fontweight='bold')
        # Griglia
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        # Limiti assi
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 4))
        # Linea zero per potenza
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
        # Legenda (solo nel primo subplot)
        if mese == 1:
            from matplotlib.patches import Patch
            carica_patch = Patch(facecolor='#06A77D', alpha=0.7, label='Carica (+)')
            scarica_patch = Patch(facecolor='#E63946', alpha=0.7, label='Scarica (-)')

            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend([carica_patch, scarica_patch] + lines2,
                      ['Carica (+)', 'Scarica (-)'] + labels2,
                      loc='upper left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, f'03_griglia_12_mesi_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvato: 03_griglia_12_mesi_{battery.technology}.png")
    # ===========================================================================================
    # Grafico 4: Grafici individuali ad alta risoluzione per ogni mese
    # ===========================================================================================
    print("\nGenerazione grafici individuali mensili ad alta risoluzione...")
    monthly_folder = os.path.join(viz_folder, 'dettaglio_mensile')
    if not os.path.exists(monthly_folder):
        os.makedirs(monthly_folder)
    for mese in range(1, 13):
        # Filtra dati per il mese corrente
        df_mese = results_df[results_df['Mese'] == mese].copy()
        if len(df_mese) == 0:
            continue
        # Seleziona il 15° giorno del mese (o il giorno più vicino disponibile)
        giorni_disponibili = df_mese['Giorno'].unique()
        giorno_target = 15 if 15 in giorni_disponibili else giorni_disponibili[len(giorni_disponibili) // 2]
        df_giorno = df_mese[df_mese['Giorno'] == giorno_target].copy()
        if len(df_giorno) < 24:  # Se non abbiamo 24 ore, prendi il primo giorno completo
            for g in giorni_disponibili:
                df_test = df_mese[df_mese['Giorno'] == g]
                if len(df_test) >= 24:
                    df_giorno = df_test.copy()
                    giorno_target = g
                    break
        # Crea array ore (0-23)
        df_giorno = df_giorno.head(24).copy()
        df_giorno['Ora'] = range(len(df_giorno))
        # Crea figura per questo mese con 3 subplot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        data_str = df_giorno['Data'].iloc[0].strftime('%d/%m/%Y') if len(df_giorno) > 0 else ''
        fig.suptitle(f'Analisi Dettagliata BESS - {mesi_nomi[mese - 1]} {data_str}\n{battery.technology}',
                     fontsize=14, fontweight='bold')
        # =============================
        # Subplot 1: Potenza e Prezzo
        # =============================
        ax1_twin = ax1.twinx()
        colori_azioni = ['#E63946' if a < 0 else '#06A77D' if a > 0 else '#CCCCCC'
                         for a in df_giorno['Azione_MW'].values]
        ax1.bar(df_giorno['Ora'], df_giorno['Azione_MW'],
                color=colori_azioni, alpha=0.7, width=0.8)
        ax1_twin.plot(df_giorno['Ora'], df_giorno['€/MWh'],
                      color='#F77F00', linewidth=2.5, marker='o', markersize=4,
                      label='Prezzo Energia')
        ax1.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        ax1.set_ylabel('Potenza [MW]', fontsize=11, fontweight='bold')
        ax1_twin.set_ylabel('Prezzo [€/MWh]', fontsize=11, fontweight='bold', color='#F77F00')
        ax1_twin.tick_params(axis='y', labelcolor='#F77F00')
        ax1.set_xlabel('Ora', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_title('Potenza BESS e Prezzo Energia', fontsize=12, fontweight='bold')
        # Legenda
        from matplotlib.patches import Patch
        carica_patch = Patch(facecolor='#06A77D', alpha=0.7, label='Carica (+)')
        scarica_patch = Patch(facecolor='#E63946', alpha=0.7, label='Scarica (-)')
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend([carica_patch, scarica_patch] + lines2,
                   ['Carica (+)', 'Scarica (-)'] + labels2,
                   loc='upper left', fontsize=9)
        # =============================
        # Subplot 2: SOC
        # =============================
        ax2.plot(df_giorno['Ora'], df_giorno['SOC'] * 100,
                 color='#457B9D', linewidth=2.5, marker='o', markersize=5)
        ax2.axhline(y=battery.soc_min * 100, color='red', linewidth=1.5,
                    linestyle='--', alpha=0.7, label=f'SOC min ({battery.soc_min * 100:.0f}%)')
        ax2.axhline(y=battery.soc_max * 100, color='green', linewidth=1.5,
                    linestyle='--', alpha=0.7, label=f'SOC max ({battery.soc_max * 100:.0f}%)')
        ax2.fill_between(df_giorno['Ora'], battery.soc_min * 100, df_giorno['SOC'] * 100,
                         color='#457B9D', alpha=0.2)
        ax2.set_ylabel('SOC [%]', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Ora', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(-0.5, 23.5)
        ax2.set_xticks(range(0, 24, 2))
        ax2.set_ylim(0, 100)
        ax2.set_title('State of Charge', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        # =============================
        # Subplot 3: Profitto giornaliero
        # =============================
        profitto_giornaliero = df_giorno['Profitto_Euro'].values - df_giorno['Profitto_Euro'].values[0]
        ax3.plot(df_giorno['Ora'], profitto_giornaliero,
                 color='#06A77D', linewidth=2.5, marker='o', markersize=5)
        ax3.fill_between(df_giorno['Ora'], 0, profitto_giornaliero,
                         color='#06A77D', alpha=0.3)
        ax3.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        ax3.set_ylabel('Profitto [€]', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Ora', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_xticks(range(0, 24, 2))
        ax3.set_title(f'Profitto Cumulativo Giornaliero (Totale: {profitto_giornaliero[-1]:.2f} €)',
                      fontsize=12, fontweight='bold')
        plt.tight_layout()
        filename = f'{mese:02d}_{mesi_nomi[mese - 1]}_dettaglio_{battery.technology}.png'
        plt.savefig(os.path.join(monthly_folder, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Salvato: dettaglio_mensile/{filename}")
    print("\n" + "=" * 80)
    print(f"Visualizzazioni salvate in '{viz_folder}/'")
    print(f"  - 2 grafici principali (profitto/MACSE, SOH)")
    print(f"  - 1 grafico griglia 12 mesi")
    print(f"  - 12 grafici individuali ad alta risoluzione in 'dettaglio_mensile/'")
    print("=" * 80)


# ========================================================================================================
# SEZIONE 10B: VISUALIZZAZIONI ECONOMICHE AVANZATE
# ========================================================================================================
def create_economic_analysis_charts(results_df, battery, trading_profit, macse_revenue,
                                    macse_base, macse_penalty, macse_bonus):
    """
    Lorenzo Giannuzzo: Crea visualizzazioni avanzate per analisi economica MACSE vs Trading
    Args:
        results_df: DataFrame risultati simulazione
        battery: Istanza Battery
        trading_profit: Profitto trading [Euro]
        macse_revenue: Ricavo netto MACSE [Euro]
        macse_base: Remunerazione base MACSE [Euro]
        macse_penalty: Penali MACSE [Euro]
        macse_bonus: Bonus MACSE [Euro]
    """
    if not SAVE_PLOTS:
        return
    viz_folder = 'visualization'
    economic_folder = os.path.join(viz_folder, 'analisi_economica')
    if not os.path.exists(economic_folder):
        os.makedirs(economic_folder)
    print("\n" + "=" * 80)
    print("GENERAZIONE GRAFICI ANALISI ECONOMICA AVANZATA")
    print("=" * 80)
    # Prepara dati temporali
    if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
        results_df['Data'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')
    results_df['Mese'] = results_df['Data'].dt.month
    results_df['Anno'] = results_df['Data'].dt.year
    # ===========================================================================================
    # Grafico 1: DASHBOARD ECONOMICA COMPARATIVA (4 subplot)
    # ===========================================================================================
    print("Generando dashboard economica comparativa...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    # Subplot 1: Confronto Ricavi Totali
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Trading\nArbitraggio', 'MACSE\nServizi Rete', 'TOTALE\nSistema']
    values = [trading_profit, macse_revenue if MACSE_ENABLED else 0,
              trading_profit + (macse_revenue if MACSE_ENABLED else 0)]
    colors = ['#06A77D', '#457B9D', '#F77F00']
    bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Ricavi (Euro)', fontsize=12, fontweight='bold')
    ax1.set_title('Confronto Ricavi: Trading vs MACSE vs Totale', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    # Aggiungi valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:,.0f} €\n({height / (values[2] if values[2] > 0 else 1) * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Subplot 2: Breakdown Ricavi MACSE
    ax2 = fig.add_subplot(gs[0, 1])
    if MACSE_ENABLED:
        macse_components = ['Base\nRemunerazione', 'Bonus\nDisponibilità', 'Penali\nContrattuali', 'NETTO\nMERCE']
        macse_values = [macse_base, macse_bonus, -macse_penalty, macse_revenue]
        macse_colors = ['#06A77D', '#2A9D8F', '#E63946', '#457B9D']
        bars2 = ax2.bar(macse_components, macse_values, color=macse_colors, alpha=0.8,
                        edgecolor='black', linewidth=1.5)
        ax2.axhline(y=0, color='black', linewidth=1, linestyle='-')
        ax2.set_ylabel('Importo (Euro)', fontsize=12, fontweight='bold')
        ax2.set_title('Breakdown Ricavi MACSE', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        for bar in bars2:
            height = bar.get_height()
            label_y = height if height > 0 else height
            va_pos = 'bottom' if height > 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width() / 2., label_y,
                     f'{abs(height):,.0f} €',
                     ha='center', va=va_pos, fontsize=10, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'MACSE Non Abilitato', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=14, fontweight='bold', color='gray')
        ax2.set_title('Breakdown Ricavi MACSE', fontsize=13, fontweight='bold')
    # Subplot 3: Evoluzione Profitti Cumulativi Separati
    ax3 = fig.add_subplot(gs[1, 0])
    # Calcola profitti giornalieri
    df_daily = results_df.groupby(results_df['Data'].dt.date, as_index=False).agg({
        'Data': 'first',
        'Profitto_Euro': 'last'
    })
    df_daily['Data'] = pd.to_datetime(df_daily['Data'])
    # Simula split profitti (proporzionale alle capacità)
    if MACSE_ENABLED:
        trading_fraction = battery.trading_capacity / battery.nominal_capacity
        macse_fraction = battery.macse_capacity / battery.nominal_capacity
        df_daily['Profitto_Trading'] = df_daily['Profitto_Euro'] * trading_fraction
        df_daily['Profitto_MACSE'] = (macse_revenue / len(df_daily)) * np.arange(1, len(df_daily) + 1)
    else:
        df_daily['Profitto_Trading'] = df_daily['Profitto_Euro']
        df_daily['Profitto_MACSE'] = 0
    ax3.plot(df_daily['Data'], df_daily['Profitto_Trading'],
             color='#06A77D', linewidth=2.5, label='Trading', marker='o', markersize=3, alpha=0.8)
    ax3.fill_between(df_daily['Data'], 0, df_daily['Profitto_Trading'],
                     color='#06A77D', alpha=0.2)
    if MACSE_ENABLED:
        ax3.plot(df_daily['Data'], df_daily['Profitto_MACSE'],
                 color='#457B9D', linewidth=2.5, label='MACSE', marker='s', markersize=3, alpha=0.8)
        ax3.fill_between(df_daily['Data'], 0, df_daily['Profitto_MACSE'],
                         color='#457B9D', alpha=0.2)
    ax3.set_ylabel('Profitto Cumulativo (Euro)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Data', fontsize=11)
    ax3.set_title('Evoluzione Profitti: Trading vs MACSE', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=11, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    # Subplot 4: Pie Chart Contribuzione Percentuale
    ax4 = fig.add_subplot(gs[1, 1])
    total_revenue = trading_profit + (macse_revenue if MACSE_ENABLED else 0)
    if total_revenue > 0:
        sizes = [trading_profit, macse_revenue if MACSE_ENABLED else 0]
        labels = [f'Trading\n{trading_profit:,.0f} €\n({trading_profit / total_revenue * 100:.1f}%)',
                  f'MACSE\n{macse_revenue if MACSE_ENABLED else 0:,.0f} €\n({(macse_revenue if MACSE_ENABLED else 0) / total_revenue * 100:.1f}%)']
        colors_pie = ['#06A77D', '#457B9D']
        explode = (0.05, 0.05)
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='',
                                           startangle=90, explode=explode, shadow=True,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax4.set_title('Contribuzione Ricavi al Sistema Totale', fontsize=13, fontweight='bold', pad=20)

    fig.suptitle(f'Dashboard Economica Comparativa - {battery.technology}',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(economic_folder, f'01_dashboard_economica_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: 01_dashboard_economica_{battery.technology}.png")

    # ===========================================================================================
    # Grafico 2: ANALISI MENSILE COMPARATIVA
    # ===========================================================================================
    print("Generando analisi mensile comparativa...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'Analisi Mensile Comparativa - {battery.technology}',
                 fontsize=15, fontweight='bold')
    # Calcola metriche mensili
    df_monthly = results_df.groupby('Mese').agg({
        'Profitto_Euro': lambda x: x.iloc[-1] - (x.iloc[0] if len(x) > 0 else 0),
        'MACSE_Availability': 'mean' if MACSE_ENABLED else lambda x: 0
    }).reset_index()
    if MACSE_ENABLED:
        trading_fraction = battery.trading_capacity / battery.nominal_capacity
        df_monthly['Profitto_Trading'] = df_monthly['Profitto_Euro'] * trading_fraction
        df_monthly['Profitto_MACSE'] = (macse_revenue / 12)
    else:
        df_monthly['Profitto_Trading'] = df_monthly['Profitto_Euro']
        df_monthly['Profitto_MACSE'] = 0
    mesi_nomi = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu',
                 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
    df_monthly['Mese_Nome'] = df_monthly['Mese'].apply(lambda x: mesi_nomi[x - 1])

    # Subplot 1: Profitti mensili affiancati
    x = np.arange(len(df_monthly))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, df_monthly['Profitto_Trading'], width,
                    label='Trading', color='#06A77D', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width / 2, df_monthly['Profitto_MACSE'], width,
                    label='MACSE', color='#457B9D', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Profitto Mensile (Euro)', fontsize=11, fontweight='bold')
    ax1.set_title('Confronto Profitti Mensili: Trading vs MACSE', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_monthly['Mese_Nome'])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    # Subplot 2: Disponibilità MACSE mensile
    if MACSE_ENABLED:
        ax2.plot(df_monthly['Mese_Nome'], df_monthly['MACSE_Availability'] * 100,
                 color='#457B9D', linewidth=2.5, marker='o', markersize=8)
        ax2.axhline(y=MACSE_AVAILABILITY_TARGET * 100, color='green', linestyle='--',
                    linewidth=2, alpha=0.7, label=f'Target {MACSE_AVAILABILITY_TARGET * 100}%')
        ax2.axhline(y=MACSE_PENALTY_THRESHOLD * 100, color='red', linestyle='--',
                    linewidth=2, alpha=0.7, label=f'Soglia Penale {MACSE_PENALTY_THRESHOLD * 100}%')
        ax2.fill_between(range(len(df_monthly)), MACSE_PENALTY_THRESHOLD * 100,
                         df_monthly['MACSE_Availability'] * 100,
                         where=df_monthly['MACSE_Availability'] * 100 >= MACSE_PENALTY_THRESHOLD * 100,
                         color='green', alpha=0.2, label='Zona Sicura')
        ax2.fill_between(range(len(df_monthly)), 0, df_monthly['MACSE_Availability'] * 100,
                         where=df_monthly['MACSE_Availability'] * 100 < MACSE_PENALTY_THRESHOLD * 100,
                         color='red', alpha=0.2, label='Zona Penale')
    else:
        ax2.text(0.5, 0.5, 'MACSE Non Abilitato', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, color='gray')
    ax2.set_ylabel('Disponibilità MACSE (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Trend Disponibilità MACSE Mensile', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    # Subplot 3: Rapporto profitto/capacità allocata
    df_monthly['Trading_per_MWh'] = df_monthly['Profitto_Trading'] / battery.trading_capacity
    df_monthly['MACSE_per_MWh'] = df_monthly['Profitto_MACSE'] / (battery.macse_capacity if MACSE_ENABLED else 1)
    bars3 = ax3.bar(x - width / 2, df_monthly['Trading_per_MWh'], width,
                    label='Trading (€/MWh)', color='#06A77D', alpha=0.8, edgecolor='black')
    bars4 = ax3.bar(x + width / 2, df_monthly['MACSE_per_MWh'], width,
                    label='MACSE (€/MWh)', color='#457B9D', alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Profitto per MWh Allocato (€/MWh)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Mese', fontsize=11)
    ax3.set_title('Efficienza Economica: Profitto Normalizzato per Capacità', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_monthly['Mese_Nome'])
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(economic_folder, f'02_analisi_mensile_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: 02_analisi_mensile_{battery.technology}.png")

    # ===========================================================================================
    # Grafico 3: EFFICIENZA ECONOMICA E ROI
    # ===========================================================================================
    print("Generando analisi efficienza economica...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Analisi Efficienza Economica - {battery.technology}',
                 fontsize=15, fontweight='bold')
    # Subplot 1: ROI per MWh allocato
    categories_roi = ['Trading', 'MACSE']
    capacities = [battery.trading_capacity, battery.macse_capacity if MACSE_ENABLED else 0]
    profits = [trading_profit, macse_revenue if MACSE_ENABLED else 0]
    roi_per_mwh = [p / c if c > 0 else 0 for p, c in zip(profits, capacities)]
    bars_roi = ax1.barh(categories_roi, roi_per_mwh, color=['#06A77D', '#457B9D'],
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('ROI per MWh Allocato (€/MWh)', fontsize=11, fontweight='bold')
    ax1.set_title('Return on Investment Normalizzato', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    for i, bar in enumerate(bars_roi):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{width:,.0f} €/MWh\n(Cap: {capacities[i]:.2f} MWh)',
                 ha='left', va='center', fontsize=10, fontweight='bold')
    # Subplot 2: Revenue per ciclo equivalente
    revenue_per_cycle_trading = trading_profit / battery.equivalent_cycles if battery.equivalent_cycles > 0 else 0
    revenue_per_cycle_macse = macse_revenue / battery.equivalent_cycles if (
                battery.equivalent_cycles > 0 and MACSE_ENABLED) else 0
    categories_cycle = ['Trading', 'MACSE', 'Sistema\nTotale']
    values_cycle = [revenue_per_cycle_trading, revenue_per_cycle_macse,
                    (trading_profit + (
                        macse_revenue if MACSE_ENABLED else 0)) / battery.equivalent_cycles if battery.equivalent_cycles > 0 else 0]
    colors_cycle = ['#06A77D', '#457B9D', '#F77F00']
    bars_cycle = ax2.bar(categories_cycle, values_cycle, color=colors_cycle,
                         alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Revenue per Ciclo Equivalente (€/ciclo)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Efficienza per Ciclo\n(Tot: {battery.equivalent_cycles:.1f} cicli)',
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars_cycle:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:,.2f} €',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Subplot 3: Costo degrado vs ricavi
    # Stima costo degrado totale
    capacity_lost = battery.nominal_capacity - battery.capacity
    estimated_degradation_cost = capacity_lost * 150000  # Stima €/MWh capacità persa
    categories_cost = ['Ricavi\nTrading', 'Ricavi\nMACSE', 'Costo\nDegrado\n(stimato)', 'NETTO']
    values_cost = [trading_profit, macse_revenue if MACSE_ENABLED else 0,
                   -estimated_degradation_cost,
                   trading_profit + (macse_revenue if MACSE_ENABLED else 0) - estimated_degradation_cost]
    colors_cost = ['#06A77D', '#457B9D', '#E63946', '#F77F00']
    bars_cost = ax3.bar(categories_cost, values_cost, color=colors_cost,
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax3.set_ylabel('Importo (Euro)', fontsize=11, fontweight='bold')
    ax3.set_title('Analisi Costi: Degrado vs Ricavi', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars_cost:
        height = bar.get_height()
        label_y = height if height > 0 else height
        va_pos = 'bottom' if height > 0 else 'top'
        ax3.text(bar.get_x() + bar.get_width() / 2., label_y,
                 f'{abs(height):,.0f} €',
                 ha='center', va=va_pos, fontsize=9, fontweight='bold')
    # Subplot 4: Comparazione efficienza per tecnologia (informativo)
    tech_comparison = {
        'Efficienza\nRound-Trip': [battery.efficiency * 100],
        'DoD\nUtilizzabile': [battery.dod * 100],
        'EOL Cycles': [battery.eol_cycles / 1000],  # In migliaia
        'SOH\nFinale': [battery.get_soh()]
    }
    x_pos = np.arange(len(tech_comparison))
    values_tech = [list(v)[0] for v in tech_comparison.values()]
    bars_tech = ax4.bar(x_pos, values_tech, color='#F77F00', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(tech_comparison.keys(), fontsize=9)
    ax4.set_ylabel('Valore', fontsize=11, fontweight='bold')
    ax4.set_title(f'Parametri Prestazionali\n{battery.technology}', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for i, bar in enumerate(bars_tech):
        height = bar.get_height()
        unit = '%' if i < 3 else ('k cicli' if i == 2 else '%')
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}{unit}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(economic_folder, f'03_efficienza_economica_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: 03_efficienza_economica_{battery.technology}.png")

    # ===========================================================================================
    # Grafico 4: WATERFALL CHART RICAVI TOTALI
    # ===========================================================================================
    print("Generando waterfall chart ricavi...")
    fig, ax = plt.subplots(figsize=(12, 7))
    # Dati per waterfall
    categories_wf = ['Trading\nBase', 'MACSE\nBase', 'Bonus\nMERCE', 'Penali\nMERCE',
                     'TOTALE\nSistema']
    values_wf = [trading_profit,
                 macse_base if MACSE_ENABLED else 0,
                 macse_bonus if MACSE_ENABLED else 0,
                 -macse_penalty if MACSE_ENABLED else 0,
                 trading_profit + macse_revenue if MACSE_ENABLED else trading_profit]
    # Calcola posizioni cumulative
    cumulative = [0]
    for i, val in enumerate(values_wf[:-1]):
        cumulative.append(cumulative[-1] + val)
    # Colori
    colors_wf = ['#06A77D', '#457B9D', '#2A9D8F', '#E63946', '#F77F00']
    # Plot barre waterfall
    for i in range(len(values_wf) - 1):
        if values_wf[i] >= 0:
            ax.bar(i, values_wf[i], bottom=cumulative[i], color=colors_wf[i],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
        else:
            ax.bar(i, -values_wf[i], bottom=cumulative[i] + values_wf[i],
                   color=colors_wf[i], alpha=0.8, edgecolor='black', linewidth=1.5)
        # Linea di connessione
        if i < len(values_wf) - 2:
            ax.plot([i + 0.4, i + 0.6], [cumulative[i + 1], cumulative[i + 1]],
                    'k--', linewidth=1, alpha=0.5)
    # Barra finale totale
    ax.bar(len(values_wf) - 1, values_wf[-1], color=colors_wf[-1],
           alpha=0.8, edgecolor='black', linewidth=2)
    # Annotazioni valori
    for i, val in enumerate(values_wf):
        if i < len(values_wf) - 1:
            y_pos = cumulative[i] + val / 2 if val >= 0 else cumulative[i] + val / 2
        else:
            y_pos = val / 2
        ax.text(i, y_pos, f'{abs(val):,.0f} €',
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    ax.set_xticks(range(len(categories_wf)))
    ax.set_xticklabels(categories_wf, fontsize=10)
    ax.set_ylabel('Ricavi (Euro)', fontsize=12, fontweight='bold')
    ax.set_title(f'Waterfall Chart Ricavi Totali - {battery.technology}\n' +
                 'Composizione Ricavi da Trading a Totale Sistema',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(economic_folder, f'04_waterfall_ricavi_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: 04_waterfall_ricavi_{battery.technology}.png")

    # ===========================================================================================
    # Grafico 5: HEATMAP OPERAZIONI vs DISPONIBILITÀ MACSE
    # ===========================================================================================
    print("Generando heatmap operazioni vs MACSE...")

    # Seleziona 30 giorni campione distribuiti nell'anno
    results_df['Data_dt'] = pd.to_datetime(results_df['Data'])
    total_days = (results_df['Data_dt'].max() - results_df['Data_dt'].min()).days
    sample_days = min(30, total_days)

    day_indices = np.linspace(0, len(results_df) - 24, sample_days, dtype=int)

    # Crea matrice operazioni (giorni x ore)
    operations_matrix = np.zeros((sample_days, 24))
    macse_availability_matrix = np.zeros((sample_days, 24))
    dates_labels = []

    for idx, day_start in enumerate(day_indices):
        day_data = results_df.iloc[day_start:day_start + 24]
        if len(day_data) == 24:
            operations_matrix[idx, :] = day_data['Azione_MW'].values
            if MACSE_ENABLED:
                macse_availability_matrix[idx, :] = day_data['MACSE_Availability'].values
            dates_labels.append(day_data['Data_dt'].iloc[0].strftime('%d/%m'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Heatmap operazioni trading
    im1 = ax1.imshow(operations_matrix, cmap='RdYlGn', aspect='auto',
                     vmin=-battery.trading_power, vmax=battery.trading_power)
    ax1.set_xlabel('Ora del Giorno', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Giorno Campione', fontsize=11, fontweight='bold')
    ax1.set_title('Operazioni Trading\n(Verde=Carica, Rosso=Scarica)',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_yticks(range(0, len(dates_labels), max(1, len(dates_labels) // 10)))
    ax1.set_yticklabels([dates_labels[i] for i in range(0, len(dates_labels), max(1, len(dates_labels) // 10))])
    cbar1 = plt.colorbar(im1, ax=ax1, label='Potenza [MW]')

    # Heatmap disponibilità MACSE
    if MACSE_ENABLED:
        im2 = ax2.imshow(macse_availability_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax2.set_xlabel('Ora del Giorno', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Giorno Campione', fontsize=11, fontweight='bold')
        ax2.set_title('Disponibilità MACSE\n(Più scuro = Maggiore Disponibilità)',
                      fontsize=12, fontweight='bold')
        ax2.set_xticks(range(0, 24, 2))
        ax2.set_yticks(range(0, len(dates_labels), max(1, len(dates_labels) // 10)))
        ax2.set_yticklabels([dates_labels[i] for i in range(0, len(dates_labels), max(1, len(dates_labels) // 10))])
        cbar2 = plt.colorbar(im2, ax=ax2, label='Disponibilità [0-1]')
    else:
        ax2.text(0.5, 0.5, 'MACSE Non Abilitato', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=14, color='gray', fontweight='bold')
        ax2.set_title('Disponibilità MACSE', fontsize=12, fontweight='bold')

    fig.suptitle(f'Heatmap Operazioni vs Disponibilità MACSE - {battery.technology}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(economic_folder, f'05_heatmap_operazioni_{battery.technology}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: 05_heatmap_operazioni_{battery.technology}.png")

    print("\n" + "=" * 80)
    print(f"Grafici analisi economica salvati in '{economic_folder}/'")
    print("  - Dashboard economica comparativa (4 subplot)")
    print("  - Analisi mensile comparativa (3 subplot)")
    print("  - Efficienza economica e ROI (4 subplot)")
    print("  - Waterfall chart ricavi totali")
    print("  - Heatmap operazioni vs disponibilità MACSE")
    print("=" * 80)

# ========================================================================================================
# FUNZIONE MAIN
# ========================================================================================================

def main(file_name):
    file_path = os.path.join('data', file_name)

    print("=" * 80)
    print("BESS OPTIMIZATION WITH MACSE - MULTI TECHNOLOGY v2.2.0")
    print("=" * 80)
    print("Caricamento dati...")

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
        print(f"Errore: {e}")
        return

    # Inizializza batteria con tecnologia selezionata
    battery = Battery(
        technology=BATTERY_TECHNOLOGY,
        capacity_mwh=BATTERY_CAPACITY_MWH,
        max_power_mw=BATTERY_MAX_POWER_MW,
        max_c_rate=BATTERY_MAX_C_RATE
    )

    optimizer = PSOOptimizer(n_particles=50, n_iterations=100,
                             w_start=0.9, w_end=0.4, c1=2.0, c2=2.0)

    simulator = RollingHorizonSimulator(battery, optimizer,
                                        horizon_hours=24, step_hours=1)

    start_time = datetime.now()
    results_df, trading_profit = simulator.simulate(df)
    end_time = datetime.now()

    # Ricavi MACSE
    macse_revenue, macse_base, macse_penalty, macse_bonus = calculate_macse_revenue(battery)

    # Profitto totale sistema
    total_system_profit = trading_profit + macse_revenue

    # Report finale
    print("=" * 80)
    print("RISULTATI SIMULAZIONE")
    print("=" * 80)
    print(f"\nTECNOLOGIA: {battery.technology}")
    print(f"  - DoD: {battery.dod * 100}%")
    print(f"  - SOC range: {battery.soc_min * 100}% - {battery.soc_max * 100}%")
    print(f"  - Efficienza round-trip: {battery.efficiency * 100:.1f}%")
    print(f"  - EOL cycles: {battery.eol_cycles} cicli")
    print(f"  - Costo degrado ottimizzatore: {DEGRADATION_COST_PER_MWH} Euro/MWh")

    print("\nTRADING (Capacita disponibile per arbitraggio):")
    print(f"  Capacita trading: {battery.trading_capacity:.2f} MWh")
    print(f"  Profitto operativo trading: {trading_profit:,.2f} Euro")

    if MACSE_ENABLED:
        print(f"\nMACSE (Capacita riservata alla rete):")
        print(f"  Capacita MACSE (fissa): {MACSE_CAPACITY_MWH:.2f} MWh")
        print(f"  Remunerazione base annuale: {macse_base:,.2f} Euro")
        print(f"  Penali disponibilita: {macse_penalty:,.2f} Euro")
        print(f"  Bonus disponibilita: {macse_bonus:,.2f} Euro")
        print(f"  Ricavo netto MACSE: {macse_revenue:,.2f} Euro")
        print(f"  Disponibilita finale: {battery.get_macse_availability_factor() * 100:.2f}%")

    print(f"\nSISTEMA TOTALE:")
    print(f"  Profitto totale (Trading + MACSE): {total_system_profit:,.2f} Euro")

    print(f"\nSTATO BATTERIA:")
    print(f"  SOC finale: {battery.get_soc() * 100:.1f}%")
    print(f"  Capacita finale: {battery.capacity:.3f} MWh (nominale: {battery.nominal_capacity:.3f} MWh)")
    print(f"  SOH finale: {battery.get_soh():.2f}%")
    print(f"  Cicli equivalenti: {battery.equivalent_cycles:.1f}")
    print(f"  Tempo esecuzione: {(end_time - start_time).total_seconds():.1f} secondi")
    print("=" * 80)

    # Salva risultati Excel
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    output_file = os.path.join(results_folder,
                               f'risultati_simulazione_{battery.technology.lower().replace("-", "_")}_v2.2.0.xlsx')

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Risultati Orari', index=False)

        params_data = {
            'Parametro': [
                'Versione',
                'Tecnologia',
                'Capacita nominale (MWh)',
                'Potenza massima (MW)',
                'C-rate massimo',
                'DoD (%)',
                'SOC min (%)',
                'SOC max (%)',
                'EOL cycles',
                'Efficienza round-trip (%)',
                'Costo degrado (Euro/MWh)',
                'MACSE abilitato',
                'Capacita MACSE (MWh)',
                'Capacita Trading (MWh)',
                'Profitto Trading (Euro)',
                'Ricavi MACSE (Euro)',
                'Profitto Totale (Euro)',
                'SOH finale (%)',
            ],
            'Valore': [
                '2.2.0',
                battery.technology,
                battery.nominal_capacity,
                battery.max_power,
                battery.max_c_rate,
                battery.dod * 100,
                battery.soc_min * 100,
                battery.soc_max * 100,
                battery.eol_cycles,
                battery.efficiency * 100,
                DEGRADATION_COST_PER_MWH,
                'Si' if MACSE_ENABLED else 'No',
                battery.macse_capacity if MACSE_ENABLED else 0,
                battery.trading_capacity,
                trading_profit,
                macse_revenue,
                total_system_profit,
                battery.get_soh(),
            ]
        }
        params_df = pd.DataFrame(params_data)
        params_df.to_excel(writer, sheet_name='Parametri', index=False)

    print(f"Risultati Excel salvati in: {output_file}")

    # Export JSON
    json_file = export_results_to_json(
        results_df, battery, trading_profit, macse_revenue, macse_base,
        macse_penalty, macse_bonus, 600000, (end_time - start_time).total_seconds()
    )

    # Genera visualizzazioni
    create_visualizations(results_df, battery)

    create_economic_analysis_charts(results_df, battery, trading_profit,
                                    macse_revenue, macse_base, macse_penalty, macse_bonus)

    print("\n" + "=" * 80)
    print("SIMULAZIONE COMPLETATA CON SUCCESSO")
    print("=" * 80)


if __name__ == "__main__":
    main(file_name)