""""
------------------------------------------------------------------------------------------------------------------------
BATTERY ENERGY STORAGE SYSTEM (BESS) OPTIMIZATION - AUTONOMOUS LOAD DECISIONS
Particle Swarm Optimization with Rolling Horizon, MACSE, Autonomous PV and AUTONOMOUS Load Management
Support Litio-ione e Grafene + Sistema Fotovoltaico + Carico Utente con DECISIONE OTTIMALE AUTONOMA
------------------------------------------------------------------------------------------------------------------------
Author: Lorenzo Giannuzzo
Affiliation: Politecnico di Torino
             Dipartimento Energia (DENERG)
             Energy Center Lab

Description:
    Sistema di ottimizzazione per Battery Energy Storage System (BESS) che
    implementa un Particle Swarm Optimization (PSO) con Rolling
    Horizon per la massimizzazione dei profitti attraverso:
    - Arbitraggio sul mercato elettrico
    - Partecipazione al mercato MACSE (servizi ancillari)
    - Gestione ottimale del degrado della batteria
    - Supporto per tecnologie Litio-ione e Grafene
    - Autonomous PV: L'algoritmo decide autonomamente allocazione energia PV
    - **AUTONOMOUS LOAD: Batteria DECIDE quando servire carico vs trading**

Version: 3.8.0
Date: 12 December 2025
------------------------------------------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from joblib import Parallel, delayed
import multiprocessing
import time
import argparse
import sys

def parse_arguments():
    """
    Lorenzo Giannuzzo: Parser argomenti da command line per BESS Optimization
    """
    parser = argparse.ArgumentParser(
        description='BESS Optimization with PSO - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Simulazione base con batteria litio-ione
  python script.py --price-sell data/vendita.xlsx --price-buy data/acquisto.xlsx

  # Con PV e carico
  python script.py --price-sell data/vendita.xlsx --price-buy data/acquisto.xlsx \
                   --pv-file data/pv.csv --pv-enabled \
                   --load-file data/load.xlsx --load-enabled

  # Batteria grafene custom
  python script.py --price-sell data/vendita.xlsx --price-buy data/acquisto.xlsx \
                   --battery-tech GRAFENE --battery-capacity 2.0 --battery-power 2.0 \
                   --graphene-soc-min 0.0 --graphene-soc-max 1.0

  # Con MACSE
  python script.py --price-sell data/vendita.xlsx --price-buy data/acquisto.xlsx \
                   --macse-enabled --macse-capacity 0.5 --macse-contract-years 2
        """
    )

    # ========================================================================
    # Lorenzo Giannuzzo: FILE PATHS (REQUIRED)
    # ========================================================================
    required = parser.add_argument_group('File Paths (Obbligatori)')
    required.add_argument('--price-sell', type=str, required=True,
                          help='Path assoluto file prezzi vendita (.xlsx)')
    required.add_argument('--price-buy', type=str, required=True,
                          help='Path assoluto file prezzi acquisto (.xlsx)')

    # ========================================================================
    # Lorenzo Giannuzzo: FILE PATHS (OPTIONAL)
    # ========================================================================
    optional_files = parser.add_argument_group('File Paths (Opzionali)')
    optional_files.add_argument('--pv-file', type=str, default=None,
                                help='Path assoluto file produzione PV (.csv)')
    optional_files.add_argument('--load-file', type=str, default=None,
                                help='Path assoluto file carico (.xlsx)')

    # ========================================================================
    # Lorenzo Giannuzzo: POD (POINT OF DELIVERY)
    # ========================================================================
    pod_group = parser.add_argument_group('Point of Delivery')
    pod_group.add_argument('--pod-limit', type=float, default=1.5,
                           help='Potenza massima scambio rete [MW] (default: 1.5)')

    # ========================================================================
    # Lorenzo Giannuzzo: BATTERY PARAMETERS
    # ========================================================================
    battery_group = parser.add_argument_group('Parametri Batteria')
    battery_group.add_argument('--battery-tech', type=str,
                               choices=['LITIO-IONE', 'GRAFENE'],
                               default='LITIO-IONE',
                               help='Tecnologia batteria (default: LITIO-IONE)')
    battery_group.add_argument('--battery-capacity', type=float, default=1.0,
                               help='Capacità batteria [MWh] (default: 1.0)')
    battery_group.add_argument('--battery-power', type=float, default=1.0,
                               help='Potenza massima batteria [MW] (default: 1.0)')
    battery_group.add_argument('--battery-c-rate', type=float, default=1.0,
                               help='C-rate massimo batteria (default: 1.0)')

    battery_group.add_argument('--custom-efficiency', action='store_true',
                               help='Usa efficienza custom invece dati sperimentali')
    battery_group.add_argument('--charge-efficiency', type=float, default=None,
                               help='Efficienza carica custom [0-1] (es: 0.95)')
    battery_group.add_argument('--discharge-efficiency', type=float, default=None,
                               help='Efficienza scarica custom [0-1] (es: 0.95)')
    battery_group.add_argument('--roundtrip-efficiency', type=float, default=None,
                               help='Efficienza roundtrip custom [0-1] (es: 0.90)')

    # ========================================================================
    # Lorenzo Giannuzzo: SOC LIMITS
    # ========================================================================
    soc_group = parser.add_argument_group('SOC Limits')
    soc_group.add_argument('--lithium-soc-min', type=float, default=0.1,
                           help='SOC minimo litio-ione (default: 0.1)')
    soc_group.add_argument('--lithium-soc-max', type=float, default=0.9,
                           help='SOC massimo litio-ione (default: 0.9)')
    soc_group.add_argument('--graphene-soc-min', type=float, default=0.0,
                           help='SOC minimo grafene (default: 0.0)')
    soc_group.add_argument('--graphene-soc-max', type=float, default=1.0,
                           help='SOC massimo grafene (default: 1.0)')

    # ========================================================================
    # Lorenzo Giannuzzo: PV & LOAD
    # ========================================================================
    pv_load_group = parser.add_argument_group('PV e Carico')
    pv_load_group.add_argument('--pv-enabled', action='store_true',
                               help='Abilita sistema fotovoltaico')
    pv_load_group.add_argument('--load-enabled', action='store_true',
                               help='Abilita carico utente')
    pv_load_group.add_argument('--pv-nominal-power', type=float, default=1.0,
                               help='Potenza nominale PV [kWp] (default: 1.0)') # non caricare questo parametro, sballerebbe i risultati

    # ========================================================================
    # Lorenzo Giannuzzo: MACSE
    # ========================================================================
    macse_group = parser.add_argument_group('MACSE (Servizi Ancillari)')
    macse_group.add_argument('--macse-enabled', action='store_true',
                             help='Abilita servizi MACSE')
    macse_group.add_argument('--macse-capacity', type=float, default=1.0,
                             help='Capacità riservata MACSE [MWh] (default: 1.0)')
    macse_group.add_argument('--macse-contract-years', type=int, default=1,
                             help='Anni contratto MACSE (default: 1)')
    macse_group.add_argument('--macse-price-per-mw-year', type=float, default=50000,
                             help='Prezzo MACSE [€/MW/anno] (default: 50000)')

    # ========================================================================
    # Lorenzo Giannuzzo: PARALLELIZATION
    # ========================================================================
    parallel_group = parser.add_argument_group('Parallelizzazione')
    parallel_group.add_argument('--no-parallel', action='store_true',
                                help='Disabilita parallelizzazione PSO')
    parallel_group.add_argument('--n-cores', type=int, default=-2,
                                help='Numero cores (-1=tutti-1, -2=tutti, N=specifico)')

    # ========================================================================
    # Lorenzo Giannuzzo: PSO PARAMETERS
    # ========================================================================
    pso_group = parser.add_argument_group('PSO Parameters')
    pso_group.add_argument('--n-particles', type=int, default=50,
                           help='Numero particelle PSO')
    pso_group.add_argument('--n-iterations', type=int, default=100,
                           help='Numero iterazioni PSO (default: 100)')

    # ========================================================================
    # Lorenzo Giannuzzo: OUTPUT
    # ========================================================================
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--save-plots', action='store_true',
                              help='Salva grafici visualizzazione')
    output_group.add_argument('--output-dir', type=str, default='results',
                              help='Directory output risultati (default: results)')
    output_group.add_argument('--output-filename', type=str, default=None,
                              help='Nome base file output (senza estensione). Se non specificato, usa nome automatico')
    
    # ========================================================================
    # Lorenzo Giannuzzo: TESTING E DEBUG
    # ========================================================================
    test_group = parser.add_argument_group('Testing e Debug')
    test_group.add_argument('--compare-pso', action='store_true',
                           help='Confronta PSO originale vs ottimizzato (solo test, non simulazione completa)')
    test_group.add_argument('--disable-optimizations', action='store_true',
                           help='Disabilita tutte le ottimizzazioni PSO (usa versione originale)')
    
    # ========================================================================
    # Lorenzo Giannuzzo: BIAS OTTIMIZZAZIONE
    # ========================================================================
    bias_group = parser.add_argument_group('Bias Ottimizzazione')
    bias_group.add_argument('--bias-arbitrage', type=float, default=1.0,
                           help='Moltiplicatore ricavi arbitraggio (>1.0 favorisce trading, default: 1.0)')
    bias_group.add_argument('--bias-autoconsumo', type=float, default=1.0,
                           help='Moltiplicatore ricavi autoconsumo (>1.0 favorisce autoconsumo, default: 1.0)')
    
    args = parser.parse_args()

    # ========================================================================
    # Lorenzo Giannuzzo: VALIDAZIONE
    # ========================================================================
    # Lorenzo Giannuzzo: Verifica esistenza file
    if not os.path.exists(args.price_sell):
        parser.error(f"File prezzi vendita non trovato: {args.price_sell}")
    if not os.path.exists(args.price_buy):
        parser.error(f"File prezzi acquisto non trovato: {args.price_buy}")

    if args.pv_enabled and args.pv_file and not os.path.exists(args.pv_file):
        parser.error(f"File PV non trovato: {args.pv_file}")

    if args.load_enabled and args.load_file and not os.path.exists(args.load_file):
        parser.error(f"File carico non trovato: {args.load_file}")

    # Lorenzo Giannuzzo: Verifica SOC
    if not (0 <= args.lithium_soc_min < args.lithium_soc_max <= 1):
        parser.error(f"SOC litio invalido: min={args.lithium_soc_min}, max={args.lithium_soc_max}")

    if not (0 <= args.graphene_soc_min < args.graphene_soc_max <= 1):
        parser.error(f"SOC grafene invalido: min={args.graphene_soc_min}, max={args.graphene_soc_max}")

    # Lorenzo Giannuzzo: Verifica capacità
    if args.battery_capacity <= 0:
        parser.error(f"Capacità batteria deve essere > 0: {args.battery_capacity}")

    if args.battery_power <= 0:
        parser.error(f"Potenza batteria deve essere > 0: {args.battery_power}")

    if args.pod_limit <= 0:
        parser.error(f"POD limit deve essere > 0: {args.pod_limit}")

    # Lorenzo Giannuzzo: Verifica MACSE
    if args.macse_enabled and args.macse_capacity > args.battery_capacity:
        parser.error(f"Capacità MACSE ({args.macse_capacity}) > capacità batteria ({args.battery_capacity})")

    # Lorenzo Giannuzzo: Validazione efficienza custom
    if args.custom_efficiency:
        if args.charge_efficiency is None and args.discharge_efficiency is None and args.roundtrip_efficiency is None:
            parser.error("--custom-efficiency richiede almeno un parametro efficienza")

        if args.charge_efficiency is not None:
            if not (0 < args.charge_efficiency <= 1):
                parser.error(f"Charge efficiency deve essere tra 0 e 1: {args.charge_efficiency}")

        if args.discharge_efficiency is not None:
            if not (0 < args.discharge_efficiency <= 1):
                parser.error(f"Discharge efficiency deve essere tra 0 e 1: {args.discharge_efficiency}")

        if args.roundtrip_efficiency is not None:
            if not (0 < args.roundtrip_efficiency <= 1):
                parser.error(f"Roundtrip efficiency deve essere tra 0 e 1: {args.roundtrip_efficiency}")

    # Lorenzo Giannuzzo: Auto-calcola particelle se non specificato
    if args.n_particles is None:
        n_cores = multiprocessing.cpu_count()
        if args.no_parallel:
            args.n_particles = 120
        else:
            cores_used = n_cores if args.n_cores == -2 else (n_cores - 1 if args.n_cores == -1 else args.n_cores)
            args.n_particles = max(120, cores_used * 40)

    return args

# ========================================================================================================
# Lorenzo Giannuzzo: SEZIONE 1: PARAMETRI CONFIGURABILI PRINCIPALI
# ========================================================================================================
energy_selling_price_name = 'Prezzo_Vendita.xlsx'
energy_buying_price_name = 'Prezzo_Acquisto.xlsx'
pv_production_file = 'PV_formattato.csv'
load_file = 'Consumo.xlsx'

# ---------------------------------- PARAMETRI POINT OF DELIVERY (POD) --------------------------------
POD_POWER_MW = 1.5  # Lorenzo Giannuzzo: Potenza massima scambio con rete [MW]

# ---------------------------------- PARALLELIZZAZIONE ------------------------------------------------
ENABLE_MULTIPROCESSING = True   # Lorenzo Giannuzzo: True = PSO parallelo, False = PSO sequenziale
MULTIPROCESSING_CORES = -2     # Lorenzo Giannuzzo: -1 = auto (tutti-1), -2 = tutti, N = N cores specifici


# ---------------------------------- SCELTA TECNOLOGIA BATTERIA -------------------------------------------
BATTERY_TECHNOLOGY = "LITIO-IONE"

# ---------------------------------- PARAMETRI BATTERIA ---------------------------------------------------
BATTERY_CAPACITY_MWH = 1.0
BATTERY_MAX_POWER_MW = 1.0
BATTERY_MAX_C_RATE = 1.0
BATTERY_BASE_EFFICIENCY = 0.95

# ---------------------------------- PARAMETRI FOTOVOLTAICO -----------------------------------------------
PV_ENABLED = True
PV_NOMINAL_POWER_KWP = 1.0
PV_INVERTER_EFFICIENCY = 0.98
PV_SYSTEM_LOSSES = 0.0

# ---------------------------------- PARAMETRI CARICO UTENTE -----------------------------------------------
LOAD_ENABLED = True
LOAD_SHEET_NAME = None  # Lorenzo Giannuzzo: None = primo sheet disponibile


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
DEGRADATION_COST_PER_MWH = 500000  # Lorenzo Giannuzzo: €/MWh, da dividere poi per il numero di cicli EOL (x2)

# ---------------------------------- BIAS OTTIMIZZAZIONE -----------------------------------------------
ARBITRAGE_BIAS_MULTIPLIER = 1.0    # Lorenzo Giannuzzo: Moltiplicatore ricavi arbitraggio
AUTOCONSUMO_BIAS_MULTIPLIER = 1.0  # Lorenzo Giannuzzo: Moltiplicatore ricavi autoconsumo

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
# Lorenzo Giannuzzo: SEZIONE 2: DATI SPERIMENTALI CURVE DI EFFICIENZA
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
# Lorenzo Giannuzzo: SEZIONE 3: CLASSI
# ========================================================================================================
class BatteryEfficiencyModel:
    def __init__(self, technology, c_rate, custom_efficiency=None):
        self.technology = technology
        self.c_rate = c_rate
        self.custom_mode = custom_efficiency is not None

        if self.custom_mode:
            print(f" Utilizzo di parametri di efficienza Custom")

            charge_eff = custom_efficiency.get('charge_efficiency')
            discharge_eff = custom_efficiency.get('discharge_efficiency')
            roundtrip_eff = custom_efficiency.get('roundtrip_efficiency')

            if roundtrip_eff is not None:
                if charge_eff is None and discharge_eff is None:
                    charge_eff = discharge_eff = np.sqrt(roundtrip_eff)
                elif charge_eff is not None and discharge_eff is None:
                    discharge_eff = roundtrip_eff / charge_eff
                elif discharge_eff is not None and charge_eff is None:
                    charge_eff = roundtrip_eff / discharge_eff
            else:
                if charge_eff is not None and discharge_eff is not None:
                    roundtrip_eff = charge_eff * discharge_eff
                else:
                    raise ValueError("Devi fornire almeno roundtrip_efficiency o entrambi charge/discharge")

            self.avg_charge_efficiency = charge_eff
            self.avg_discharge_efficiency = discharge_eff
            self.avg_energy_efficiency = roundtrip_eff
            self.avg_coulombic_efficiency = 0.99

            self.data = {
                'charge_energy_kwh': [10.0],
                'discharge_energy_kwh': [10.0 * roundtrip_eff],
                'energy_efficiency': [roundtrip_eff],
                'coulombic_efficiency': [0.99]
            }

            print(
                f"      • Efficienza carica:     {self.avg_charge_efficiency:.4f} ({self.avg_charge_efficiency * 100:.2f}%)")
            print(
                f"      • Efficienza scarica:    {self.avg_discharge_efficiency:.4f} ({self.avg_discharge_efficiency * 100:.2f}%)")
            print(
                f"      • Efficienza roundtrip:  {self.avg_energy_efficiency:.4f} ({self.avg_energy_efficiency * 100:.2f}%)")

        else:
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

            self.avg_charge_efficiency = np.sqrt(self.avg_energy_efficiency)
            self.avg_discharge_efficiency = np.sqrt(self.avg_energy_efficiency)

    def get_charge_efficiency(self):
        return self.avg_charge_efficiency

    def get_discharge_efficiency(self):
        return self.avg_discharge_efficiency

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
        """Lorenzo Giannuzzo: Calcola produzione effettiva dato irraggiamento [MW]"""
        power_mw = (irradiance_w_per_kwp * self.nominal_power_kwp * self.total_efficiency) * 1000
        return power_mw

    def get_energy(self, irradiance_w_per_kwp, dt=1.0):
        """Lorenzo Giannuzzo: Calcola energia prodotta in un timestep [MWh]"""
        power_mw = self.get_production(irradiance_w_per_kwp)
        energy_mwh = power_mw * dt
        self.total_production_mwh += energy_mwh
        return energy_mwh

    def load_pv_production(self, pv_value_kw, dt=1.0):
        """
        Lorenzo Giannuzzo: Carica produzione PV già calcolata in kW
        """
        energy_mwh = (pv_value_kw / 1000.0) * dt
        self.total_production_mwh += energy_mwh
        return energy_mwh

    def allocate_energy(self, to_battery_mwh, to_grid_mwh, to_load_mwh):
        """Lorenzo Giannuzzo: Registra allocazione energia PV"""
        self.energy_to_battery_mwh += to_battery_mwh
        self.energy_to_grid_mwh += to_grid_mwh
        self.energy_to_load_mwh += to_load_mwh

    def get_statistics(self):
        """Lorenzo Giannuzzo: Ritorna statistiche produzione PV"""
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
    """
    def __init__(self):
        self.total_energy_required_mwh = 0.0
        self.energy_from_pv_mwh = 0.0
        self.energy_from_battery_mwh = 0.0
        self.energy_from_grid_mwh = 0.0

        self.battery_served_load_count = 0   # Lorenzo Giannuzzo: Ore in cui batteria ha SCELTO di servire carico
        self.grid_served_load_count = 0      # Lorenzo Giannuzzo: Ore in cui batteria ha SCELTO di lasciare carico a rete
        self.total_decision_hours = 0

    def register_supply(self, from_pv, from_battery, from_grid):
        """Registra fonti di fornitura carico"""
        self.energy_from_pv_mwh += from_pv
        self.energy_from_battery_mwh += from_battery
        self.energy_from_grid_mwh += from_grid
        self.total_energy_required_mwh += (from_pv + from_battery + from_grid)

    def register_battery_decision(self, battery_served: bool):
        """
        Lorenzo Giannuzzo: battery_served=True → Batteria ha SCELTO di servire carico
        Lorenzo Giannuzzo: battery_served=False → Batteria ha SCELTO di lasciare carico a rete (per trading futuro)
        """
        self.total_decision_hours += 1
        if battery_served:
            self.battery_served_load_count += 1
        else:
            self.grid_served_load_count += 1

    def get_statistics(self):
        """Lorenzo Giannuzzo: Ritorna statistiche fornitura carico con decisioni autonome"""
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

        # Statistiche decisioni
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
# Lorenzo Giannuzzo: SEZIONE 4: DEGRADO
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
# Lorenzo Giannuzzo: SEZIONE 5: MODELLO BATTERIA
# ========================================================================================================
class Battery:
    """
    Lorenzo Giannuzzo: Modello batteria con tracking separato carica da rete vs PV
    """

    def __init__(self, technology=BATTERY_TECHNOLOGY,
                 capacity_mwh=BATTERY_CAPACITY_MWH,
                 max_power_mw=BATTERY_MAX_POWER_MW,
                 max_c_rate=BATTERY_MAX_C_RATE,
                 custom_efficiency=None):
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
        self.efficiency_model = BatteryEfficiencyModel(technology, max_c_rate, custom_efficiency=custom_efficiency)
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
        """Lorenzo Giannuzzo: Scarica batteria"""
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
# Lorenzo Giannuzzo: SEZIONE 6: OTTIMIZZATORE PSO
# ========================================================================================================
# Lorenzo Giannuzzo: SEZIONE NUMBA: VALUTAZIONE ULTRA-VELOCE PARTICELLE PSO
# ========================================================================================================

from numba import njit, prange
import numba

@njit(fastmath=True, cache=True)
def evaluate_particle_numba(
        # Lorenzo Giannuzzo: Stati batteria
        soc_init, capacity, trading_capacity, soc_min, soc_max,
        charge_eff, discharge_eff, eol_cycles,
        # Lorenzo Giannuzzo: Azioni particella
        actions,  # (n_hours, 3): [p_batt_trading, alpha_pv_load, p_batt_load]
        # Lorenzo Giannuzzo: Dati mercato
        prices_sell, prices_buy, pv_production, load_demand,
        # Lorenzo Giannuzzo: Flags
        pv_enabled, load_enabled,
        # Lorenzo Giannuzzo: Parametri globali
        pod_power_mw, degradation_cost_per_mwh,
        # Lorenzo Giannuzzo: Bias ottimizzazione
        arbitrage_bias, autoconsumo_bias
):
    """
    Lorenzo Giannuzzo: Valutazione SINGOLA particella con Numba JIT
    
    VERSIONE CORRETTA v3.8.1:
    - Fix bilancio energetico (load_remaining aggiornato)
    - Fix logica economica autoconsumo (risparmio vs ricavo)
    - Fix gestione POD coordinata (batteria + PV)
    - Penalità POD realistiche
    """
    n_hours = len(prices_sell)
    soc = soc_init
    profit = 0.0
    penalty = 0.0

    for h in range(n_hours):
        # Estrai azioni
        p_batt_trading = actions[h, 0]
        alpha_pv_load = actions[h, 1]
        p_batt_load = actions[h, 2]

        # Lorenzo Giannuzzo: Dati ora corrente
        price_sell = prices_sell[h]
        price_buy = prices_buy[h]
        pv_available = pv_production[h] if pv_enabled else 0.0
        load_required = load_demand[h] if load_enabled else 0.0

        # Lorenzo Giannuzzo: FASE 1: Allocazione PV al carico
        pv_to_load = min(alpha_pv_load * pv_available, load_required)
        pv_remaining = pv_available - pv_to_load
        load_remaining = load_required - pv_to_load

        # Lorenzo Giannuzzo: FASE 2: Vincolo XOR batteria
        if p_batt_trading > 0.001:  # Lorenzo Giannuzzo: CARICA
            charge_request = p_batt_trading
            action_mode = 1  # Lorenzo Giannuzzo: CHARGE
        elif p_batt_trading < -0.001:  # Lorenzo Giannuzzo: SCARICA TRADING
            discharge_request_trading = -p_batt_trading
            discharge_request_load = p_batt_load
            action_mode = 2  # Lorenzo Giannuzzo: DISCHARGE
        elif p_batt_load > 0.001:  # Lorenzo Giannuzzo: SCARICA SOLO LOAD
            discharge_request_trading = 0.0
            discharge_request_load = p_batt_load
            action_mode = 2  # Lorenzo Giannuzzo: DISCHARGE
        else:  # Lorenzo Giannuzzo: IDLE
            action_mode = 0

        grid_withdrawal = 0.0
        grid_injection = 0.0

        # ====================================================================
        # Lorenzo Giannuzzo: FASE 3: SCARICA BATTERIA
        # ====================================================================
        if action_mode == 2:
            total_discharge_request = discharge_request_trading + discharge_request_load
            max_discharge = (soc - soc_min) * trading_capacity * discharge_eff
            actual_discharge_total = min(total_discharge_request, max_discharge)

            if actual_discharge_total > 0.001:
                ratio_trading = discharge_request_trading / total_discharge_request if total_discharge_request > 0 else 0.0
                ratio_load = discharge_request_load / total_discharge_request if total_discharge_request > 0 else 0.0

                discharge_for_trading = actual_discharge_total * ratio_trading
                discharge_for_load_raw = actual_discharge_total * ratio_load
                discharge_for_load = min(discharge_for_load_raw, load_remaining)

                discharge_for_trading = min(discharge_for_trading, pod_power_mw)
                actual_discharge_used = discharge_for_trading + discharge_for_load
                
                # Lorenzo Giannuzzo: CORREZIONE CRITICA - Aggiorna carico rimanente
                load_remaining -= discharge_for_load

                # Lorenzo Giannuzzo: Penalità POD disabilitata per confronto
                # if discharge_for_trading > pod_power_mw:
                #     penalty += (discharge_for_trading - pod_power_mw) * price_sell * 10.0

                energy_consumed = actual_discharge_used / discharge_eff
                new_soc = soc - (energy_consumed / capacity)
                soc = max(new_soc, soc_min)

                # ============================================================
                # TRADING: Ricavo ESPLICITO vendita rete (con bias)
                # ============================================================
                if discharge_for_trading > 0.001:
                    profit += discharge_for_trading * price_sell * arbitrage_bias
                    profit -= discharge_for_trading * degradation_cost_per_mwh / (2 * eol_cycles)
                    grid_injection += discharge_for_trading

                # ============================================================
                # AUTOCONSUMO (con bias)
                # ============================================================
                if discharge_for_load > 0.001:
                    # Lorenzo Giannuzzo: Autoconsumo = "ricavo equivalente" (risparmio acquisto rete)
                    profit += discharge_for_load * price_buy * autoconsumo_bias
                    profit -= discharge_for_load * degradation_cost_per_mwh / (2 * eol_cycles)
                    # Nota: load_remaining già aggiornato sopra

        # ====================================================================
        # Lorenzo Giannuzzo: FASE 4: CARICO DALLA RETE
        # ====================================================================
        if load_remaining > 0.001:
            pod_available = pod_power_mw - grid_withdrawal
            load_from_grid = min(load_remaining, pod_available)

            if load_from_grid > 0.001:
                # COSTO acquisto energia per carico
                # Questo bilancia il "risparmio" della FASE 3
                profit -= load_from_grid * price_buy
                grid_withdrawal += load_from_grid
                load_remaining -= load_from_grid

            # Carico non servito (penalità pesante)
            if load_remaining > 0.001:
                penalty += load_remaining * price_buy * 1000.0

        # ====================================================================
        # Lorenzo Giannuzzo: FASE 5: CARICA BATTERIA
        # ====================================================================
        if action_mode == 1 and soc < soc_max:
            max_storable = (soc_max - soc) * trading_capacity
            max_power = max_storable / (1.0 * charge_eff)
            actual_power = min(charge_request, max_power)

            if actual_power > 0.01:
                energy_needed = actual_power * 1.0

                if pv_remaining > 0.0:
                    energy_from_pv = min(energy_needed, pv_remaining)
                    energy_stored = energy_from_pv * charge_eff
                    soc += energy_stored / capacity
                    pv_remaining -= energy_from_pv
                    energy_needed -= energy_from_pv
                    profit -= energy_from_pv * degradation_cost_per_mwh / (2 * eol_cycles)

                # Lorenzo Giannuzzo:  Carica da rete
                if energy_needed > 0.01:
                    pod_available = pod_power_mw - grid_withdrawal
                    energy_from_grid = min(energy_needed, pod_available)

                    if energy_from_grid > 0.001:
                        energy_stored = energy_from_grid * charge_eff
                        soc += energy_stored / capacity
                        profit -= energy_from_grid * price_buy
                        profit -= energy_from_grid * degradation_cost_per_mwh / (2 * eol_cycles)
                        grid_withdrawal += energy_from_grid

        # ====================================================================
        # Lorenzo Giannuzzo: FASE 6: PV residuo alla rete
        # ====================================================================
        if pv_remaining > 0.001:
            pv_to_grid = min(pv_remaining, pod_power_mw - grid_injection)
            if pv_to_grid > 0.001:
                profit += pv_to_grid * price_sell
                grid_injection += pv_to_grid

            pv_curtailed = pv_remaining - pv_to_grid
            if pv_curtailed > 0.001:
                penalty += pv_curtailed * price_sell * 50.0

        # Lorenzo Giannuzzo: Penalità POD disabilitate per confronto
        # if grid_withdrawal > pod_power_mw + 0.001:
        #     penalty += (grid_withdrawal - pod_power_mw) * price_buy * 100.0
        # if grid_injection > pod_power_mw + 0.001:
        #     penalty += (grid_injection - pod_power_mw) * price_sell * 100.0

    return profit - penalty


@njit(parallel=True, fastmath=True, cache=True)
def evaluate_all_particles_numba(
        # Lorenzo Giannuzzo: Stati batteria (scalari)
        soc_init, capacity, trading_capacity, soc_min, soc_max,
        charge_eff, discharge_eff, eol_cycles,
        # Lorenzo Giannuzzo: Azioni TUTTE le particelle
        positions,  # (n_particles, n_hours, 3)
        # Lorenzo Giannuzzo: Dati mercato
        prices_sell, prices_buy, pv_production, load_demand,
        # Lorenzo Giannuzzo: Flags
        pv_enabled, load_enabled,
        # Lorenzo Giannuzzo: Parametri globali
        pod_power_mw, degradation_cost_per_mwh,
        # Lorenzo Giannuzzo: Bias ottimizzazione
        arbitrage_bias, autoconsumo_bias
):
    """
    Lorenzo Giannuzzo: Valutazione PARALLELA di TUTTE le particelle con Numba

    Lorenzo Giannuzzo: ULTRA-VELOCE: parallel=True usa threads nativi C
    Speedup: 50-200x rispetto a Python puro
    """
    n_particles = positions.shape[0]
    scores = np.empty(n_particles, dtype=np.float64)

    # Lorenzo Giannuzzo: PARALLELIZZAZIONE AUTOMATICA NUMBA
    for i in prange(n_particles):
        scores[i] = evaluate_particle_numba(
            soc_init, capacity, trading_capacity, soc_min, soc_max,
            charge_eff, discharge_eff, eol_cycles,
            positions[i],  # Azioni particella i
            prices_sell, prices_buy, pv_production, load_demand,
            pv_enabled, load_enabled,
            pod_power_mw, degradation_cost_per_mwh,
            arbitrage_bias, autoconsumo_bias
        )

    return scores


# ========================================================================================================
# Lorenzo Giannuzzo: WRAPPER PYTHON
# ========================================================================================================
def evaluate_particles_fast(battery, positions, prices_sell, prices_buy, pv_production, load_demand):
    """
    Lorenzo Giannuzzo: Wrapper Python-friendly per chiamare Numba
    """
    # Lorenzo Giannuzzo: Converti bool Python in int per Numba (0=False, 1=True)
    pv_enabled_int = 1 if PV_ENABLED else 0
    load_enabled_int = 1 if LOAD_ENABLED else 0

    scores = evaluate_all_particles_numba(
        # Lorenzo Giannuzzo:  Stati batteria
        battery.soc,
        battery.capacity,
        battery.trading_capacity,
        battery.soc_min,
        battery.soc_max,
        battery.charge_efficiency,
        battery.discharge_efficiency,
        battery.eol_cycles,
        # Lorenzo Giannuzzo: Azioni
        positions,
        # Lorenzo Giannuzzo: Dati mercato
        prices_sell,
        prices_buy,
        pv_production,
        load_demand,
        # Lorenzo Giannuzzo: Flags
        pv_enabled_int,
        load_enabled_int,
        # Parametri
        POD_POWER_MW,
        DEGRADATION_COST_PER_MWH,
        # Lorenzo Giannuzzo: Bias ottimizzazione
        ARBITRAGE_BIAS_MULTIPLIER,
        AUTOCONSUMO_BIAS_MULTIPLIER
    )

    return scores
# ========================================================================================================
# Lorenzo Giannuzzo: CLASSE PSO OTTIMIZZATA - VERSIONE MIGLIORATA
# ========================================================================================================
class PSOOptimizer:
    """
    Lorenzo Giannuzzo: PSO OTTIMIZZATO con tutte le migliorie per risultati superiori
    
    Migliorie implementate:
    1. Inizializzazione intelligente multi-strategia
    2. Riduzione dimensionalità con pattern temporali
    3. Constraint handling con repair
    4. Parametri PSO adattivi
    5. Euristica greedy come seed
    6. Diversificazione dinamica
    """

    def __init__(self, n_particles=120, n_iterations=300, w_start=0.9, w_end=0.1, c1=1.5, c2=2.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1_start = c1
        self.c2_start = c2
        self.c1 = c1
        self.c2 = c2
        self.stagnation_limit = 20
        self.use_temporal_patterns = False
        self.use_intelligent_init = True
        self.use_constraint_repair = False
        self.use_adaptive_parameters = False
        
        # Tracking per parametri adattivi
        self.best_score_history = []
        self.diversity_history = []
        
        # Pattern temporali per riduzione dimensionalità
        self.time_blocks = [
            (0, 6),    # Notte (00:00-06:00)
            (6, 9),    # Mattina (06:00-09:00)
            (9, 12),   # Tarda mattina (09:00-12:00)
            (12, 15),  # Primo pomeriggio (12:00-15:00)
            (15, 18),  # Tardo pomeriggio (15:00-18:00)
            (18, 21),  # Sera (18:00-21:00)
            (21, 24)   # Notte (21:00-24:00)
        ]

        # print(f"PSO OTTIMIZZATO v2.0 - TUTTE LE MIGLIORIE ATTIVE")
        # print(f"   • Particelle: {n_particles}")
        # print(f"   • Iterazioni: {n_iterations}")
        # print(f"   • Inizializzazione intelligente: {self.use_intelligent_init}")
        # print(f"   • Pattern temporali: {self.use_temporal_patterns}")
        # print(f"   • Constraint repair: {self.use_constraint_repair}")
        # print(f"   • Parametri adattivi: {self.use_adaptive_parameters}")
        # print(f"   • Blocchi temporali: {len(self.time_blocks)}")
        # print(f"   • Parametri PSO: w=[{w_start:.2f}-{w_end:.2f}], c1={c1:.1f}, c2={c2:.1f}")
        # print(f"   • Numba parallel: TRUE")
        # print(f"   • Threads Numba: {numba.get_num_threads()}")
        # print(f"   • NOTA: Prima iterazione lenta (compilazione JIT), poi 50-200x più veloce")

    def configure_optimizations(self, intelligent_init=True, temporal_patterns=True, 
                              constraint_repair=True, adaptive_parameters=True):
        """
        Lorenzo Giannuzzo: Configura quali ottimizzazioni usare
        """
        self.use_intelligent_init = intelligent_init
        self.use_temporal_patterns = temporal_patterns
        self.use_constraint_repair = constraint_repair
        self.use_adaptive_parameters = adaptive_parameters
        
        #print(f"Configurazione ottimizzazioni aggiornata:")
        #print(f"   • Inizializzazione intelligente: {self.use_intelligent_init}")
        #print(f"   • Pattern temporali: {self.use_temporal_patterns}")
        #print(f"   • Constraint repair: {self.use_constraint_repair}")
        #print(f"   • Parametri adattivi: {self.use_adaptive_parameters}")

    def get_optimization_stats(self):
        """
        Lorenzo Giannuzzo: Ritorna statistiche ottimizzazione
        """
        if not self.best_score_history:
            return {}
        
        return {
            'initial_score': self.best_score_history[0],
            'final_score': self.best_score_history[-1],
            'improvement': self.best_score_history[-1] - self.best_score_history[0],
            'improvement_percent': ((self.best_score_history[-1] - self.best_score_history[0]) / 
                                  abs(self.best_score_history[0]) * 100) if self.best_score_history[0] != 0 else 0,
            'iterations_to_best': len(self.best_score_history) - 1,
            'convergence_rate': len([i for i in range(1, len(self.best_score_history)) 
                                   if self.best_score_history[i] > self.best_score_history[i-1]]),
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'avg_diversity': np.mean(self.diversity_history) if self.diversity_history else 0
        }

    def optimize(self, battery, prices_sell, prices_buy, pv_production, load_demand, horizon_hours=24):
        """
        Lorenzo Giannuzzo: Ottimizzazione PSO MIGLIORATA con tutte le ottimizzazioni
        """
        n_hours = min(horizon_hours, len(prices_sell))
        max_power_limit = min(battery.trading_power, battery.get_max_power_by_crate())

        # ====================================================================
        # 1. INIZIALIZZAZIONE INTELLIGENTE MULTI-STRATEGIA
        # ====================================================================
        if self.use_intelligent_init:
            positions = self._intelligent_initialization(
                battery, prices_sell, prices_buy, pv_production, load_demand, max_power_limit, n_hours
            )
        else:
            positions = self._smart_initialization(
                battery, prices_sell, prices_buy, pv_production, load_demand, max_power_limit
            )

        # ====================================================================
        # 2. CONSTRAINT REPAIR INIZIALE
        # ====================================================================
        if self.use_constraint_repair:
            for i in range(self.n_particles):
                positions[i] = self._repair_solution(positions[i], battery, pv_production, load_demand, max_power_limit)

        velocities = np.random.uniform(-0.3, 0.3, (self.n_particles, n_hours, 3))

        # ====================================================================
        # 3. VALUTAZIONE INIZIALE
        # ====================================================================
        personal_best_scores = evaluate_particles_fast(
            battery, positions, prices_sell, prices_buy, pv_production, load_demand
        )

        personal_best_positions = positions.copy()
        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        # Reset tracking
        self.best_score_history = [global_best_score]
        self.diversity_history = []
        stagnation_counter = 0

        # print(f"   • Score iniziale: {global_best_score:.2f} €")

        # ====================================================================
        # 4. LOOP OTTIMIZZAZIONE CON PARAMETRI ADATTIVI
        # ====================================================================
        for iteration in range(self.n_iterations):
            
            # 4a. PARAMETRI ADATTIVI
            if self.use_adaptive_parameters and iteration > 10:
                self._update_adaptive_parameters(iteration)
            else:
                w = self.w_start - (self.w_start - self.w_end) * (iteration / self.n_iterations)

            # 4b. CALCOLA DIVERSITÀ POPOLAZIONE
            diversity = self._calculate_diversity(positions)
            self.diversity_history.append(diversity)

            # 4c. AGGIORNA VELOCITÀ E POSIZIONI
            for i in range(self.n_particles):
                r1, r2 = np.random.random((n_hours, 3)), np.random.random((n_hours, 3))
                
                # Componenti PSO standard
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (global_best_position - positions[i])
                
                # Aggiorna velocità
                if self.use_adaptive_parameters:
                    velocities[i] = w * velocities[i] + cognitive + social
                else:
                    w = self.w_start - (self.w_start - self.w_end) * (iteration / self.n_iterations)
                    velocities[i] = w * velocities[i] + cognitive + social

                # Limita velocità
                max_vel = np.array([max_power_limit * 0.4, 0.25, max_power_limit * 0.4])
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
                
                # Aggiorna posizione
                positions[i] += velocities[i]

                # 4d. CONSTRAINT REPAIR
                if self.use_constraint_repair:
                    positions[i] = self._repair_solution(positions[i], battery, pv_production, load_demand, max_power_limit)
                else:
                    # Clip bounds standard
                    positions[i, :, 0] = np.clip(positions[i, :, 0],
                                                 -min(max_power_limit, POD_POWER_MW),
                                                 min(max_power_limit, POD_POWER_MW))
                    positions[i, :, 1] = np.clip(positions[i, :, 1], 0, 1)
                    positions[i, :, 2] = np.clip(positions[i, :, 2], 0, min(max_power_limit, POD_POWER_MW))

            # ====================================================================
            # 5. VALUTAZIONE NUMBA PARALLELA
            # ====================================================================
            scores = evaluate_particles_fast(
                battery, positions, prices_sell, prices_buy, pv_production, load_demand
            )

            # ====================================================================
            # 6. AGGIORNA BEST SOLUTIONS
            # ====================================================================
            improved_mask = scores > personal_best_scores
            personal_best_scores[improved_mask] = scores[improved_mask]
            personal_best_positions[improved_mask] = positions[improved_mask].copy()

            current_best_idx = np.argmax(personal_best_scores)
            if personal_best_scores[current_best_idx] > global_best_score:
                improvement = personal_best_scores[current_best_idx] - global_best_score
                global_best_score = personal_best_scores[current_best_idx]
                global_best_position = personal_best_positions[current_best_idx].copy()
                stagnation_counter = 0
                
                if iteration % 50 == 0 or improvement > 10:
                    #print(f"   • Iter {iteration}: Nuovo best {global_best_score:.2f} € (+{improvement:.2f})")
                    continue
            else:
                stagnation_counter += 1

            self.best_score_history.append(global_best_score)

            # ====================================================================
            # 7. DIVERSIFICAZIONE DINAMICA
            # ====================================================================
            if stagnation_counter > self.stagnation_limit:
                self._dynamic_diversification(positions, velocities, global_best_position, 
                                            battery, pv_production, load_demand, max_power_limit, n_hours)
                stagnation_counter = 0

            # ====================================================================
            # 8. REINIZIALIZZAZIONE PARTICELLE PEGGIORI (ogni 100 iter)
            # ====================================================================
            if iteration > 0 and iteration % 100 == 0 and diversity < 0.1:
                self._reinitialize_worst_particles(positions, personal_best_scores, global_best_position,
                                                 battery, prices_sell, prices_buy, pv_production, load_demand, 
                                                 max_power_limit, n_hours)

        # print(f"   • Score finale: {global_best_score:.2f} € (miglioramento: {global_best_score - self.best_score_history[0]:.2f} €)")
        return global_best_position

    def _repair_solution(self, position, battery, pv_production, load_demand, max_power):
        """
        Lorenzo Giannuzzo: Ripara soluzioni violando vincoli invece di penalizzarle
        
        Vincoli gestiti:
        1. SOC limits (min/max)
        2. Power limits (C-rate, max_power)
        3. POD limits (grid exchange)
        4. Physical constraints (energy balance)
        """
        n_hours = len(position)
        repaired_position = position.copy()
        
        # Simula SOC per verificare vincoli
        current_soc = battery.soc
        
        for h in range(n_hours):
            p_batt_trading = repaired_position[h, 0]
            alpha_pv_load = repaired_position[h, 1]
            p_batt_load = repaired_position[h, 2]
            
            # Clip alpha_pv_load
            repaired_position[h, 1] = np.clip(alpha_pv_load, 0.0, 1.0)
            
            # Calcola energia disponibile/richiesta
            pv_avail = pv_production[h]
            load_req = load_demand[h]
            
            # ================================================================
            # VINCOLO 1: SOC LIMITS
            # ================================================================
            if p_batt_trading > 0:  # Carica
                max_charge_energy = (battery.soc_max - current_soc) * battery.capacity
                max_charge_power = max_charge_energy / (1.0 * battery.charge_efficiency)
                max_charge_power = min(max_charge_power, max_power, POD_POWER_MW)
                
                repaired_position[h, 0] = min(p_batt_trading, max_charge_power)
                
                # Aggiorna SOC simulato
                energy_stored = repaired_position[h, 0] * 1.0 * battery.charge_efficiency
                current_soc += energy_stored / battery.capacity
                
            elif p_batt_trading < 0:  # Scarica trading
                max_discharge_energy = (current_soc - battery.soc_min) * battery.capacity
                max_discharge_power = max_discharge_energy * battery.discharge_efficiency / 1.0
                max_discharge_power = min(max_discharge_power, max_power, POD_POWER_MW)
                
                repaired_position[h, 0] = max(-max_discharge_power, p_batt_trading)
                
                # Aggiorna SOC simulato (scarica trading)
                energy_discharged = abs(repaired_position[h, 0]) * 1.0 / battery.discharge_efficiency
                current_soc -= energy_discharged / battery.capacity
            
            # ================================================================
            # VINCOLO 2: SCARICA PER CARICO (separata da trading)
            # ================================================================
            if p_batt_load > 0:
                # Calcola energia batteria ancora disponibile dopo trading
                remaining_energy = (current_soc - battery.soc_min) * battery.capacity
                max_load_power = remaining_energy * battery.discharge_efficiency / 1.0
                max_load_power = min(max_load_power, max_power)
                
                # Limita anche in base al carico effettivo
                pv_to_load = repaired_position[h, 1] * pv_avail
                load_remaining = max(0, load_req - pv_to_load)
                max_load_power = min(max_load_power, load_remaining)
                
                repaired_position[h, 2] = min(p_batt_load, max_load_power)
                
                # Aggiorna SOC simulato (scarica carico)
                if repaired_position[h, 2] > 0:
                    energy_discharged_load = repaired_position[h, 2] * 1.0 / battery.discharge_efficiency
                    current_soc -= energy_discharged_load / battery.capacity
            else:
                repaired_position[h, 2] = 0.0
            
            # ================================================================
            # VINCOLO 3: POD LIMITS (verifica totale scambio rete)
            # ================================================================
            # Calcola flussi rete approssimativi
            pv_to_load = repaired_position[h, 1] * pv_avail
            pv_remaining = pv_avail - pv_to_load
            load_remaining = max(0, load_req - pv_to_load - repaired_position[h, 2])
            
            # Prelievo da rete (carica batteria + carico residuo)
            grid_withdrawal = 0.0
            if repaired_position[h, 0] > 0:  # Carica da rete
                grid_withdrawal += max(0, repaired_position[h, 0] - pv_remaining)
            grid_withdrawal += load_remaining
            
            # Immissione in rete (scarica batteria + PV eccesso)
            grid_injection = 0.0
            if repaired_position[h, 0] < 0:  # Scarica trading
                grid_injection += abs(repaired_position[h, 0])
            grid_injection += max(0, pv_remaining - max(0, repaired_position[h, 0]))
            
            # Correggi se viola POD
            if grid_withdrawal > POD_POWER_MW:
                scale_factor = POD_POWER_MW / grid_withdrawal
                if repaired_position[h, 0] > 0:
                    repaired_position[h, 0] *= scale_factor
                repaired_position[h, 2] *= scale_factor
            
            if grid_injection > POD_POWER_MW:
                scale_factor = POD_POWER_MW / grid_injection
                if repaired_position[h, 0] < 0:
                    repaired_position[h, 0] *= scale_factor
            
            # ================================================================
            # VINCOLO 4: LIMITI FISICI FINALI
            # ================================================================
            current_soc = np.clip(current_soc, battery.soc_min, battery.soc_max)
            
            # Clip finali
            repaired_position[h, 0] = np.clip(repaired_position[h, 0], -max_power, max_power)
            repaired_position[h, 1] = np.clip(repaired_position[h, 1], 0.0, 1.0)
            repaired_position[h, 2] = np.clip(repaired_position[h, 2], 0.0, max_power)
        
        return repaired_position

    def _update_adaptive_parameters(self, iteration):
        """
        Lorenzo Giannuzzo: Aggiorna parametri PSO in base alla convergenza
        """
        if len(self.best_score_history) < 10:
            return
        
        # Calcola miglioramento recente
        recent_improvement = self.best_score_history[-1] - self.best_score_history[-10]
        
        # Calcola diversità media recente
        if len(self.diversity_history) >= 5:
            avg_diversity = np.mean(self.diversity_history[-5:])
        else:
            avg_diversity = 0.5
        
        # Adatta parametri
        if recent_improvement < 0.1:  # Stagnazione
            # Aumenta esplorazione
            self.w_start = min(0.95, self.w_start * 1.02)
            self.w_end = min(0.2, self.w_end * 1.05)
            self.c1 = min(2.5, self.c1 * 1.01)
            
        elif recent_improvement > 5.0:  # Buona convergenza
            # Aumenta sfruttamento
            self.w_start = max(0.7, self.w_start * 0.98)
            self.w_end = max(0.05, self.w_end * 0.95)
            self.c2 = min(3.0, self.c2 * 1.01)
        
        # Adatta in base alla diversità
        if avg_diversity < 0.1:  # Popolazione troppo concentrata
            self.c1 = min(2.5, self.c1 * 1.02)  # Più esplorazione individuale
        elif avg_diversity > 0.8:  # Popolazione troppo dispersa
            self.c2 = min(3.0, self.c2 * 1.02)  # Più attrazione verso best
        
        # Calcola w corrente
        w = self.w_start - (self.w_start - self.w_end) * (iteration / self.n_iterations)
        
        return w

    def _calculate_diversity(self, positions):
        """
        Lorenzo Giannuzzo: Calcola diversità popolazione (distanza media dal centroide)
        """
        if len(positions) < 2:
            return 1.0
        
        # Calcola centroide
        centroid = np.mean(positions, axis=0)
        
        # Calcola distanze dal centroide
        distances = []
        for i in range(len(positions)):
            dist = np.linalg.norm(positions[i] - centroid)
            distances.append(dist)
        
        # Normalizza per dimensione spazio
        max_possible_dist = np.linalg.norm(np.ones_like(centroid))
        avg_distance = np.mean(distances)
        
        return min(1.0, avg_distance / (max_possible_dist + 1e-6))

    def _dynamic_diversification(self, positions, velocities, global_best_position, 
                               battery, pv_production, load_demand, max_power, n_hours):
        """
        Lorenzo Giannuzzo: Diversificazione dinamica quando stagnazione
        """
        n_diversify = self.n_particles // 3
        worst_indices = np.argsort([np.linalg.norm(pos - global_best_position) 
                                  for pos in positions])[:n_diversify]
        
        # print(f"   • Diversificazione: reinizializzando {n_diversify} particelle")
        
        for idx in worst_indices:
            # Strategia 1: Perturbazione del best (50%)
            if np.random.random() < 0.5:
                noise_scale = np.random.uniform(0.3, 0.7)
                noise = np.random.uniform(-noise_scale, noise_scale, (n_hours, 3))
                noise[:, 0] *= max_power
                noise[:, 2] *= max_power
                positions[idx] = global_best_position + noise
            
            # Strategia 2: Nuova inizializzazione intelligente (50%)
            else:
                new_positions = self._intelligent_initialization(
                    battery, np.ones(n_hours), np.ones(n_hours), 
                    pv_production, load_demand, max_power, n_hours
                )
                positions[idx] = new_positions[0]
            
            # Repair della nuova posizione
            positions[idx] = self._repair_solution(positions[idx], battery, pv_production, load_demand, max_power)
            
            # Reset velocità
            velocities[idx] = np.random.uniform(-0.3, 0.3, (n_hours, 3))

    def _reinitialize_worst_particles(self, positions, personal_best_scores, global_best_position,
                                    battery, prices_sell, prices_buy, pv_production, load_demand, 
                                    max_power, n_hours):
        """
        Lorenzo Giannuzzo: Reinizializza particelle peggiori periodicamente
        """
        n_reinit = self.n_particles // 5
        worst_indices = np.argsort(personal_best_scores)[:n_reinit]
        
        #print(f"   • Reinizializzazione: {n_reinit} particelle peggiori")
        
        for idx in worst_indices:
            # Crea nuova soluzione intelligente
            new_positions = self._intelligent_initialization(
                battery, prices_sell, prices_buy, pv_production, load_demand, max_power, n_hours
            )
            positions[idx] = new_positions[np.random.randint(len(new_positions))]
            
            # Repair
            positions[idx] = self._repair_solution(positions[idx], battery, pv_production, load_demand, max_power)

    def compare_pso_versions(battery, prices_sell, prices_buy, pv_production, load_demand, n_particles=50, n_iterations=100):
        """
        Lorenzo Giannuzzo: Confronta PSO originale vs ottimizzato
        """
        print("\n" + "="*80)
        print("CONFRONTO PSO: ORIGINALE vs OTTIMIZZATO")
        print("="*80)

        results = {}

        # Test PSO originale (parametri vecchi)
        print("\n1. TEST PSO ORIGINALE:")
        pso_original = PSOOptimizer(n_particles=n_particles, n_iterations=n_iterations,
                                   w_start=0.95, w_end=0.005, c1=2.0, c2=2.0)
        pso_original.configure_optimizations(intelligent_init=False, temporal_patterns=False,
                                           constraint_repair=False, adaptive_parameters=False)

        start_time = time.time()
        solution_original = pso_original.optimize(battery, prices_sell, prices_buy, pv_production, load_demand)
        time_original = time.time() - start_time

        score_original = evaluate_particles_fast(battery, solution_original.reshape(1, -1, 3),
                                               prices_sell, prices_buy, pv_production, load_demand)[0]

        results['original'] = {
            'score': score_original,
            'time': time_original,
            'stats': pso_original.get_optimization_stats()
        }

        # Test PSO ottimizzato
        print("\n2. TEST PSO OTTIMIZZATO:")
        pso_optimized = PSOOptimizer(n_particles=n_particles, n_iterations=n_iterations,
                                    w_start=0.9, w_end=0.1, c1=1.5, c2=2.5)
        # Tutte le ottimizzazioni attive per default

        start_time = time.time()
        solution_optimized = pso_optimized.optimize(battery, prices_sell, prices_buy, pv_production, load_demand)
        time_optimized = time.time() - start_time

        score_optimized = evaluate_particles_fast(battery, solution_optimized.reshape(1, -1, 3),
                                                prices_sell, prices_buy, pv_production, load_demand)[0]

        results['optimized'] = {
            'score': score_optimized,
            'time': time_optimized,
            'stats': pso_optimized.get_optimization_stats()
        }

        # Confronto risultati
        print("\n" + "="*80)
        print("RISULTATI CONFRONTO:")
        print("="*80)

        improvement = score_optimized - score_original
        improvement_percent = (improvement / abs(score_original) * 100) if score_original != 0 else 0
        time_ratio = time_optimized / time_original if time_original > 0 else 1

        print(f"\nPSO ORIGINALE:")
        print(f"   • Score finale:        {score_original:.2f} €")
        print(f"   • Tempo esecuzione:    {time_original:.1f} secondi")
        print(f"   • Miglioramento:       {results['original']['stats'].get('improvement', 0):.2f} €")

        print(f"\nPSO OTTIMIZZATO:")
        print(f"   • Score finale:        {score_optimized:.2f} €")
        print(f"   • Tempo esecuzione:    {time_optimized:.1f} secondi")
        print(f"   • Miglioramento:       {results['optimized']['stats'].get('improvement', 0):.2f} €")

        print(f"\nCONFRONTO:")
        print(f"   • Miglioramento score: {improvement:+.2f} € ({improvement_percent:+.1f}%)")
        print(f"   • Rapporto tempo:      {time_ratio:.2f}x")

        if improvement > 0:
            print(f"   ✅ PSO OTTIMIZZATO È MIGLIORE!")
        else:
            print(f"   ❌ PSO originale ancora migliore")

        print(f"\nSTATISTICHE DETTAGLIATE:")
        print(f"   • Convergenza originale:   {results['original']['stats'].get('convergence_rate', 0)} miglioramenti")
        print(f"   • Convergenza ottimizzato: {results['optimized']['stats'].get('convergence_rate', 0)} miglioramenti")
        print(f"   • Diversità finale orig:   {results['original']['stats'].get('final_diversity', 0):.3f}")
        print(f"   • Diversità finale ott:    {results['optimized']['stats'].get('final_diversity', 0):.3f}")

        print("="*80)

        return results

    def _intelligent_initialization(self, battery, prices_sell, prices_buy, pv_production, load_demand, max_power, n_hours):
        """
        Lorenzo Giannuzzo: Inizializzazione intelligente multi-strategia

        Strategie implementate:
        1. Price-based (25%): Carica prezzi bassi, scarica prezzi alti
        2. PV-aware (25%): Ottimizza uso PV per carico e storage
        3. Load-first (25%): Priorità assoluta al carico
        4. Greedy baseline (15%): Soluzione euristica ottima
        5. Random migliorato (10%): Diversificazione controllata
        """
        positions = np.zeros((self.n_particles, n_hours, 3))
        max_power_with_pod = min(max_power, POD_POWER_MW)

        # Calcola percentili prezzi per strategie
        price_low = np.percentile(prices_sell, 25)
        price_high = np.percentile(prices_sell, 75)
        price_very_low = np.percentile(prices_sell, 10)
        price_very_high = np.percentile(prices_sell, 90)

        particle_idx = 0

        # ====================================================================
        # STRATEGIA 1: PRICE-BASED (25% particelle)
        # ====================================================================
        n_price_based = int(0.25 * self.n_particles)
        for i in range(n_price_based):
            for h in range(n_hours):
                price = prices_sell[h]

                # Logica carica/scarica basata su prezzi
                if price <= price_very_low:
                    # Prezzo molto basso → carica aggressiva
                    positions[particle_idx, h, 0] = np.random.uniform(0.7, 1.0) * max_power_with_pod
                elif price <= price_low:
                    # Prezzo basso → carica moderata
                    positions[particle_idx, h, 0] = np.random.uniform(0.3, 0.7) * max_power_with_pod
                elif price >= price_very_high:
                    # Prezzo molto alto → scarica aggressiva
                    positions[particle_idx, h, 0] = -np.random.uniform(0.7, 1.0) * max_power_with_pod
                elif price >= price_high:
                    # Prezzo alto → scarica moderata
                    positions[particle_idx, h, 0] = -np.random.uniform(0.3, 0.7) * max_power_with_pod
                else:
                    # Prezzo medio → azione casuale leggera
                    positions[particle_idx, h, 0] = np.random.uniform(-0.2, 0.2) * max_power_with_pod

                # PV allocation intelligente
                if pv_production[h] > 0.01 and load_demand[h] > 0.01:
                    positions[particle_idx, h, 1] = min(1.0, load_demand[h] / pv_production[h])
                else:
                    positions[particle_idx, h, 1] = np.random.uniform(0.5, 1.0)

                # Load discharge conservativo
                positions[particle_idx, h, 2] = np.random.uniform(0, 0.3) * max_power_with_pod

            particle_idx += 1

        # ====================================================================
        # STRATEGIA 2: PV-AWARE (25% particelle)
        # ====================================================================
        n_pv_aware = int(0.25 * self.n_particles)
        for i in range(n_pv_aware):
            for h in range(n_hours):
                pv_avail = pv_production[h]
                load_req = load_demand[h]

                if pv_avail > 0.01:
                    # Priorità PV al carico
                    if load_req > 0.01:
                        positions[particle_idx, h, 1] = min(1.0, load_req / pv_avail)
                        pv_excess = max(0, pv_avail - load_req)
                    else:
                        positions[particle_idx, h, 1] = 0.0
                        pv_excess = pv_avail

                    # Se c'è eccesso PV, carica batteria
                    if pv_excess > 0.1:
                        positions[particle_idx, h, 0] = min(max_power_with_pod, pv_excess * np.random.uniform(0.6, 1.0))
                    else:
                        positions[particle_idx, h, 0] = np.random.uniform(-0.1, 0.1) * max_power_with_pod
                else:
                    # Nessun PV → strategia prezzo
                    positions[particle_idx, h, 1] = 0.0
                    if prices_sell[h] >= price_high:
                        positions[particle_idx, h, 0] = -np.random.uniform(0.2, 0.6) * max_power_with_pod
                    elif prices_sell[h] <= price_low:
                        positions[particle_idx, h, 0] = np.random.uniform(0.2, 0.6) * max_power_with_pod
                    else:
                        positions[particle_idx, h, 0] = np.random.uniform(-0.1, 0.1) * max_power_with_pod

                # Load discharge moderato
                positions[particle_idx, h, 2] = np.random.uniform(0, 0.4) * max_power_with_pod

            particle_idx += 1

        # ====================================================================
        # STRATEGIA 3: LOAD-FIRST (25% particelle)
        # ====================================================================
        n_load_first = int(0.25 * self.n_particles)
        for i in range(n_load_first):
            for h in range(n_hours):
                load_req = load_demand[h]
                pv_avail = pv_production[h]

                # Massima priorità al carico
                if pv_avail > 0.01 and load_req > 0.01:
                    positions[particle_idx, h, 1] = 1.0  # Tutto il PV possibile al carico
                else:
                    positions[particle_idx, h, 1] = np.random.uniform(0.7, 1.0)

                # Batteria serve carico quando conveniente
                if load_req > pv_avail and prices_sell[h] > np.mean(prices_buy):
                    # Conviene usare batteria invece di comprare da rete
                    positions[particle_idx, h, 2] = min(max_power_with_pod,
                                                       (load_req - pv_avail) * np.random.uniform(0.5, 1.0))
                    positions[particle_idx, h, 0] = np.random.uniform(-0.1, 0.1) * max_power_with_pod
                else:
                    positions[particle_idx, h, 2] = np.random.uniform(0, 0.2) * max_power_with_pod
                    # Trading normale
                    if prices_sell[h] >= price_high:
                        positions[particle_idx, h, 0] = -np.random.uniform(0.2, 0.5) * max_power_with_pod
                    elif prices_sell[h] <= price_low:
                        positions[particle_idx, h, 0] = np.random.uniform(0.2, 0.5) * max_power_with_pod
                    else:
                        positions[particle_idx, h, 0] = np.random.uniform(-0.1, 0.1) * max_power_with_pod

            particle_idx += 1

        # ====================================================================
        # STRATEGIA 4: GREEDY BASELINE (15% particelle)
        # ====================================================================
        n_greedy = int(0.15 * self.n_particles)
        greedy_solution = self._create_greedy_solution(prices_sell, prices_buy, pv_production, load_demand, max_power_with_pod)

        for i in range(n_greedy):
            # Usa soluzione greedy con piccole variazioni
            noise_factor = 0.1 + (i / n_greedy) * 0.2  # Variazione crescente
            for h in range(n_hours):
                for dim in range(3):
                    noise = np.random.uniform(-noise_factor, noise_factor)
                    positions[particle_idx, h, dim] = greedy_solution[h, dim] * (1 + noise)

            particle_idx += 1

        # ====================================================================
        # STRATEGIA 5: RANDOM MIGLIORATO (10% particelle rimanenti)
        # ====================================================================
        while particle_idx < self.n_particles:
            for h in range(n_hours):
                # Random ma con bias verso azioni sensate
                positions[particle_idx, h, 0] = np.random.uniform(-0.6, 0.6) * max_power_with_pod
                positions[particle_idx, h, 1] = np.random.uniform(0.3, 1.0)
                positions[particle_idx, h, 2] = np.random.uniform(0, 0.5) * max_power_with_pod

            particle_idx += 1

        # print(f"   • Inizializzazione intelligente completata:")
        # print(f"     - Price-based: {n_price_based} particelle")
        # print(f"     - PV-aware: {n_pv_aware} particelle")
        # print(f"     - Load-first: {n_load_first} particelle")
        # print(f"     - Greedy: {n_greedy} particelle")
        # print(f"     - Random: {self.n_particles - particle_idx + (self.n_particles - particle_idx)} particelle")

        return positions

    def _create_greedy_solution(self, prices_sell, prices_buy, pv_production, load_demand, max_power):
        """
        Lorenzo Giannuzzo: Crea soluzione greedy euristica come baseline
        """
        n_hours = len(prices_sell)
        solution = np.zeros((n_hours, 3))
        
        # Calcola soglie prezzo
        price_low_threshold = np.percentile(prices_sell, 30)
        price_high_threshold = np.percentile(prices_sell, 70)
        
        for h in range(n_hours):
            price_sell = prices_sell[h]
            price_buy = prices_buy[h]
            pv_avail = pv_production[h]
            load_req = load_demand[h]
            
            # Regola 1: PV al carico sempre quando possibile
            if pv_avail > 0.01 and load_req > 0.01:
                solution[h, 1] = min(1.0, load_req / pv_avail)
            else:
                solution[h, 1] = 0.5
            
            # Regola 2: Trading basato su prezzi
            if price_sell <= price_low_threshold:
                # Prezzo basso → carica
                solution[h, 0] = 0.6 * max_power
            elif price_sell >= price_high_threshold:
                # Prezzo alto → scarica
                solution[h, 0] = -0.6 * max_power
            else:
                # Prezzo medio → idle
                solution[h, 0] = 0.0
            
            # Regola 3: Batteria per carico solo se molto conveniente
            spread = price_sell - price_buy
            if spread > np.percentile(prices_sell - prices_buy, 80):
                # Spread alto → usa batteria per carico
                solution[h, 2] = min(0.4 * max_power, load_req - pv_avail * solution[h, 1])
            else:
                solution[h, 2] = 0.0
        
        return solution

    def _smart_initialization(self, battery, prices_sell, prices_buy, pv_production, load_demand, max_power):
        """Lorenzo Giannuzzo: Fallback per inizializzazione semplice"""
        n_hours = len(prices_sell)
        positions = np.zeros((self.n_particles, n_hours, 3))
        max_power_with_pod = min(max_power, POD_POWER_MW)

        # Inizializzazione RANDOM UNIFORME migliorata
        positions[:, :, 0] = np.random.uniform(-max_power_with_pod * 0.8, max_power_with_pod * 0.8,
                                               (self.n_particles, n_hours))
        positions[:, :, 1] = np.random.uniform(0.3, 1.0, (self.n_particles, n_hours))
        positions[:, :, 2] = np.random.uniform(0, max_power_with_pod * 0.6, (self.n_particles, n_hours))

        return positions


# ============================================================================
# Lorenzo Giannuzzo: ROLLING HORIZON PARALLELIZZATO
# ============================================================================

class RollingHorizonSimulator:
    """
    Lorenzo Giannuzzo: Rolling Horizon Simulator - VERSIONE PULITA

    Lorenzo Giannuzzo: Funziona con PSO parallelizzato senza nested parallelism.
    Lorenzo Giannuzzo: IDENTICA alla tua originale - zero modifiche inutili.

    Author: Lorenzo Giannuzzo
    """

    def __init__(self, battery, optimizer, pv_system=None, load_profile=None, horizon_hours=24, step_hours=1):
        self.battery = battery
        self.optimizer = optimizer
        self.pv_system = pv_system
        self.load_profile = load_profile
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours

    def simulate(self, prices_df, price_df2, pv_df=None, load_df=None):
        """
        Lorenzo Giannuzzo: Simulazione Rolling Horizon CON VINCOLO POD
        """
        prices_sell = prices_df['€/MWh'].values
        prices_buy = price_df2['€/MWh'].values
        n_hours = len(prices_sell)

        # ========================================================================
        # Lorenzo Giannuzzo: PREPARAZIONE DATI PV
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

        # ========================================================================
        # Lorenzo Giannuzzo: PREPARAZIONE DATI CARICO
        # ========================================================================
        if LOAD_ENABLED and load_df is not None:
            load_demand = load_df['value'].values / 1000.0  # kW → MW
            if len(load_demand) < n_hours:
                load_demand = np.pad(load_demand, (0, n_hours - len(load_demand)), 'constant')
            elif len(load_demand) > n_hours:
                load_demand = load_demand[:n_hours]
        else:
            load_demand = np.zeros(n_hours)

        # ========================================================================
        # Lorenzo Giannuzzo: ARRAYS RISULTATI
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

        grid_withdrawal_history = []
        grid_injection_history = []
        pod_violation_history = []

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

        # Lorenzo Giannuzzo: STAMPA PROGRESSO INIZIALE
        print(f"Progresso: 0% - SOH: {self.battery.get_soh():.2f}% - SOC: {self.battery.get_soc() * 100:.1f}%")

        while current_hour < n_hours:
            progress = int((current_hour / n_hours) * 100)
            if progress >= last_progress + 20:
                print(
                    f"Progresso: {progress}% - SOH: {self.battery.get_soh():.2f}% - SOC: {self.battery.get_soc() * 100:.1f}%")
                last_progress = progress

            if current_hour % degradation_update_interval == 0 and current_hour > 0:
                self.battery.update_degradation()

            # ====================================================================
            # Lorenzo Giannuzzo: OTTIMIZZAZIONE PSO
            # ====================================================================
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
            # Lorenzo Giannuzzo: VARIABILI TRACKING POD - RESET OGNI ORA
            # ====================================================================
            grid_withdrawal_this_hour = 0.0
            grid_injection_this_hour = 0.0

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
            # Lorenzo Giannuzzo: FASE 1: ALLOCAZIONE PV AL CARICO
            # ====================================================================
            pv_to_load_this_hour = min(alpha_pv_load * pv_available, load_required)
            load_from_pv_this_hour = pv_to_load_this_hour
            pv_remaining = pv_available - pv_to_load_this_hour
            load_remaining = load_required - pv_to_load_this_hour

            # ====================================================================
            # Lorenzo Giannuzzo: FASE 2: VINCOLO XOR ESPLICITO
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
            # Lorenzo Giannuzzo: FASE 3: SCARICA BATTERIA
            # ====================================================================
            if action_mode == "DISCHARGE" and total_discharge_request > 0.001:
                actual_discharge_total = min(total_discharge_request, max_power_physical, max_discharge_soc)

                if actual_discharge_total > 0.001:
                    ratio_trading = discharge_request_trading / total_discharge_request
                    ratio_load = discharge_request_load / total_discharge_request

                    discharge_for_trading = actual_discharge_total * ratio_trading
                    discharge_for_load_raw = actual_discharge_total * ratio_load
                    discharge_for_load = min(discharge_for_load_raw, load_remaining)

                    # VINCOLO POD: Limita scarica trading
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

                    # ============================================================
                    # TRADING: Ricavo ESPLICITO vendita rete
                    # ============================================================
                    if discharge_for_trading > 0.001:
                        revenue_discharge = discharge_for_trading * price_sell
                        cumulative_profit += revenue_discharge
                        
                        # Lorenzo Giannuzzo: RIMOSSO costo degrado - già considerato nell'ottimizzazione PSO
                        # degradation_cost_trading = discharge_for_trading * DEGRADATION_COST_PER_MWH / (2 * self.battery.eol_cycles)
                        # cumulative_profit -= degradation_cost_trading
                        
                        trading_discharge_this_hour = discharge_for_trading
                        grid_injection_this_hour += discharge_for_trading

                    # ============================================================
                    # AUTOCONSUMO - CORREZIONE: Aggiungi risparmio batteria
                    # ============================================================
                    if discharge_for_load > 0.001:
                        savings_from_battery_to_load = discharge_for_load * price_buy
                        # cumulative_profit += savings_from_battery_to_load  # Lorenzo Giannuzzo: questo non ci deve essere nel simulate
                        
                        # Lorenzo Giannuzzo: RIMOSSO costo degrado - già considerato nell'ottimizzazione PSO
                        # degradation_cost_load = discharge_for_load * DEGRADATION_COST_PER_MWH / (2 * self.battery.eol_cycles)
                        # cumulative_profit -= degradation_cost_load

                        # Contabilità fisica
                        load_from_battery_this_hour = discharge_for_load
                        load_remaining -= discharge_for_load
                        battery_served_load = True
                        load_discharge_this_hour = discharge_for_load

            # ====================================================================
            # Lorenzo Giannuzzo: FASE 4: CARICO DALLA RETE
            # ====================================================================
            if load_remaining > 0.001:
                # CARICO USA POD PER PRIMO
                pod_available_for_withdrawal = POD_POWER_MW - grid_withdrawal_this_hour
                load_from_grid_allowed = min(load_remaining, pod_available_for_withdrawal)

                if load_from_grid_allowed > 0.001:
                    load_from_grid_this_hour = load_from_grid_allowed
                    grid_withdrawal_this_hour += load_from_grid_allowed

                    # Lorenzo Giannuzzo: COSTO: Acquisto energia dalla rete per carico
                    cost_grid_load = load_from_grid_this_hour * price_buy
                    cumulative_profit -= cost_grid_load

                    # Lorenzo Giannuzzo: Aggiorna carico residuo
                    load_remaining -= load_from_grid_allowed

                # Lorenzo Giannuzzo: Carico non servito causa POD
                if load_remaining > 0.001:
                    load_unserved_this_hour = load_remaining

                    # Lorenzo Giannuzzo: PENALITÀ - CORREZIONE: Usa stessa penalità dell'evaluate
                    penalty = load_unserved_this_hour * price_buy * 0000.0  # ✅ CORRETTO: Stessa penalità dell'evaluate
                    cumulative_profit -= penalty

                    if load_unserved_this_hour > 0.01:  # Log solo se significativo
                        print(
                            f"⚠️  Ora {current_hour}: Carico non servito {load_unserved_this_hour:.3f} MWh (POD limit)")
                        print(f"      → Penalità: {penalty:.2f} €")
                        print(f"      → POD usato: {grid_withdrawal_this_hour:.3f} MW / {POD_POWER_MW} MW")

            # Lorenzo Giannuzzo: Registra decisione batteria per carico
            if self.load_profile and (load_from_battery_this_hour > 0.001 or load_from_grid_this_hour > 0.001):
                self.load_profile.register_battery_decision(battery_served_load)

            # ====================================================================
            # Lorenzo Giannuzzo: FASE 5: CARICA BATTERIA
            # ====================================================================
            if action_mode == "CHARGE" and charge_request > 0.001:
                if self.battery.soc < self.battery.soc_max:
                    max_power_charge = max_energy_storable / (1.0 * self.battery.charge_efficiency)
                    actual_power = min(charge_request, max_power_charge, max_power_physical)

                    if actual_power > 0.01:
                        energy_needed = actual_power * 1.0
                        actual_charge_this_hour = actual_power

                        if pv_remaining > 0:
                            energy_from_pv = min(energy_needed, pv_remaining)
                            self.battery.charge(energy_from_pv / 1.0, dt=1.0, source='pv')
                            pv_to_battery_this_hour += energy_from_pv
                            pv_for_battery_charging += energy_from_pv
                            energy_needed -= energy_from_pv
                            pv_remaining -= energy_from_pv


                        if energy_needed > 0.01:

                            pod_available_for_withdrawal = POD_POWER_MW - grid_withdrawal_this_hour
                            energy_from_grid_allowed = min(energy_needed, pod_available_for_withdrawal)

                            if energy_from_grid_allowed > 0.01:
                                energy_from_grid = self.battery.charge(energy_from_grid_allowed / 1.0, dt=1.0,
                                                                       source='grid')
                                grid_to_battery_this_hour += energy_from_grid
                                grid_withdrawal_this_hour += energy_from_grid

                                # Lorenzo Giannuzzo: COSTO: Acquisto energia dalla rete
                                cost_grid_charge = energy_from_grid * price_buy
                                cumulative_profit -= cost_grid_charge

            # ====================================================================
            # Lorenzo Giannuzzo: FASE 6: VENDITA PV RESIDUO CON VINCOLO POD
            # ====================================================================
            if pv_remaining > 0.001:
                # Lorenzo Giannuzzo: VINCOLO POD: Limita immissione PV
                pod_available_for_injection = POD_POWER_MW - grid_injection_this_hour
                pv_to_grid_allowed = min(pv_remaining, pod_available_for_injection)

                if pv_to_grid_allowed > 0.001:
                    pv_to_grid_this_hour += pv_to_grid_allowed
                    grid_injection_this_hour += pv_to_grid_allowed

                    # Lorenzo Giannuzzo: RICAVO: Vendita PV alla rete
                    revenue_pv = pv_to_grid_allowed * price_sell
                    cumulative_profit += revenue_pv

                # Lorenzo Giannuzzo: PV curtailed causa POD
                pv_curtailed_this_hour = pv_remaining - pv_to_grid_allowed
                if pv_curtailed_this_hour > 0.001:
                    # Penalità leggera: energia persa ma non costo diretto (sempre 0 per il discorso dell'excel)
                    penalty = pv_curtailed_this_hour * price_sell * 0.0
                    cumulative_profit -= penalty
                    if pv_curtailed_this_hour > 0.1:
                        print(f"⚠️  Ora {current_hour}: PV curtailed {pv_curtailed_this_hour:.3f} MWh (POD limit)")

            # ====================================================================
            # Lorenzo Giannuzzo: TRACKING E REGISTRAZIONE
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

            # Lorenzo Giannuzzo: Converti scarica in negativo per compatibilità grafici
            net_trading_action = actual_charge_this_hour if actual_charge_this_hour > 0 else -actual_trading_discharge_this_hour

            # ====================================================================
            # ALorenzo Giannuzzo: PPEND RISULTATI
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
            pv_curtailed_history.append(pv_curtailed_this_hour)

            load_demand_history.append(load_demand[current_hour])
            load_from_pv_history.append(load_from_pv_this_hour)
            load_from_battery_history.append(load_from_battery_this_hour)
            load_from_grid_history.append(load_from_grid_this_hour)
            load_unserved_history.append(load_unserved_this_hour)

            energy_from_grid_to_battery_history.append(grid_to_battery_this_hour)
            energy_from_pv_to_battery_history.append(pv_for_battery_charging)
            price_buy_history.append(price_buy)

            battery_decision_history.append(1 if battery_served_load else 0)
            load_discharge_history.append(load_discharge_this_hour)
            trading_discharge_history.append(trading_discharge_this_hour)

            # Lorenzo Giannuzzo: POD tracking
            grid_withdrawal_history.append(grid_withdrawal_this_hour)
            grid_injection_history.append(grid_injection_this_hour)
            pod_violated = (grid_withdrawal_this_hour > POD_POWER_MW + 0.001) or (
                        grid_injection_this_hour > POD_POWER_MW + 0.001)
            pod_violation_history.append(1 if pod_violated else 0)

            current_hour += self.step_hours

        # ========================================================================
        # ALorenzo Giannuzzo: GGIORNAMENTO FINALE DEGRADO
        # ========================================================================
        self.battery.update_degradation()

        # Lorenzo Giannuzzo: Statistiche POD
        total_pod_violations = sum(pod_violation_history)
        total_pv_curtailed = sum(pv_curtailed_history)
        total_load_unserved = sum(load_unserved_history)

        print("\nSimulazione completata!")
        print(f"Profitto finale: {cumulative_profit:.2f} €")
        print(f"\nSTATISTICHE POD:")
        print(f"  • Violazioni POD: {total_pod_violations} ore su {len(pod_violation_history)}")
        print(f"  • PV curtailed: {total_pv_curtailed:.2f} MWh")
        print(f"  • Carico non servito: {total_load_unserved:.2f} MWh")
        if total_pod_violations > 0:
            print(f"  ⚠️  ATTENZIONE: Rilevate {total_pod_violations} violazioni POD")

        # ========================================================================
        # Lorenzo Giannuzzo: CREAZIONE DATAFRAME RISULTATI
        # ========================================================================
        results_df = prices_df.copy()
        pad_length = len(results_df) - len(actions_trading_effective)

        results_df['Prezzo_Acquisto_€/MWh'] = price_buy_history + [price_buy_history[-1]] * pad_length

        # Lorenzo Giannuzzo: Azioni PSO effettive
        results_df['Azione_Trading_MW'] = actions_trading_effective + [0] * pad_length
        results_df['Azione_Alpha_PV_Load'] = actions_alpha_pv + [0] * pad_length
        results_df['Azione_P_Batt_Load_MW'] = actions_p_load_effective + [0] * pad_length

        # Lorenzo Giannuzzo: Stati batteria
        results_df['SOC'] = soc_history + [soc_history[-1]] * pad_length
        results_df['Capacita_MWh'] = capacity_history + [capacity_history[-1]] * pad_length
        results_df['SOH_%'] = soh_history + [soh_history[-1]] * pad_length

        # Lorenzo Giannuzzo: Economia
        results_df['Profitto_Euro'] = profits_history + [profits_history[-1]] * pad_length
        results_df['MACSE_Availability'] = macse_availability_history + [macse_availability_history[-1]] * pad_length

        # Lorenzo Giannuzzo: PV
        results_df['PV_Production_MWh'] = pv_production_history + [0] * pad_length
        results_df['PV_to_Battery_MWh'] = pv_to_battery_history + [0] * pad_length
        results_df['PV_to_Grid_MWh'] = pv_to_grid_history + [0] * pad_length
        results_df['PV_to_Load_MWh'] = pv_to_load_history + [0] * pad_length
        results_df['PV_Curtailed_MWh'] = pv_curtailed_history + [0] * pad_length

        # Lorenzo Giannuzzo: Load
        results_df['Load_Demand_MWh'] = load_demand_history + [0] * pad_length
        results_df['Load_from_PV_MWh'] = load_from_pv_history + [0] * pad_length
        results_df['Load_from_Battery_MWh'] = load_from_battery_history + [0] * pad_length
        results_df['Load_from_Grid_MWh'] = load_from_grid_history + [0] * pad_length
        results_df['Load_Unserved_MWh'] = load_unserved_history + [0] * pad_length

        # Lorenzo Giannuzzo: Energy sources
        results_df['Energy_from_Grid_MWh'] = energy_from_grid_to_battery_history + [0] * pad_length
        results_df['Energy_from_PV_MWh'] = energy_from_pv_to_battery_history + [0] * pad_length

        # Lorenzo Giannuzzo: Decisioni
        results_df['Battery_Decision'] = battery_decision_history + [0] * pad_length
        results_df['Load_Discharge_MW'] = load_discharge_history + [0] * pad_length
        results_df['Trading_Discharge_MW'] = trading_discharge_history + [0] * pad_length

        # Lorenzo Giannuzzo: POD tracking
        results_df['Grid_Withdrawal_MW'] = grid_withdrawal_history + [0] * pad_length
        results_df['Grid_Injection_MW'] = grid_injection_history + [0] * pad_length
        results_df['POD_Violation'] = pod_violation_history + [0] * pad_length

        # Lorenzo Giannuzzo: Retrocompatibilità
        results_df['Azione_MW'] = actions_trading_effective + [0] * pad_length

        return results_df, cumulative_profit


# ========================================================================================================
# Lorenzo Giannuzzo: SEZIONE 8: MACSE E JSON EXPORT
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
                       macse_penalty, macse_bonus, battery_investment, simulation_time, baseline_scenario=None,
                       output_dir='results', output_filename=None):
    """
    Lorenzo Giannuzzo: Esporta risultati simulazione in formato JSON con confronto baseline
    """
    actions = results_df['Azione_Trading_MW'].values
    prices_sell = results_df['€/MWh'].values
    prices_buy = results_df['Prezzo_Acquisto_€/MWh'].values

    # ========================================================================
    # Lorenzo Giannuzzo: STATISTICHE PV
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
    # Lorenzo Giannuzzo: STATISTICHE CARICO
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
    # Lorenzo Giannuzzo: STATISTICHE TRADING
    # ========================================================================
    charge_hours = np.sum(actions > 0.01)
    discharge_hours = np.sum(actions < -0.01)
    idle_hours = len(actions) - charge_hours - discharge_hours
    total_energy_charged = np.sum(actions[actions > 0] * 1.0)
    total_energy_discharged = np.sum(np.abs(actions[actions < 0]) * 1.0)

    # ========================================================================
    # Lorenzo Giannuzzo: CALCOLI ECONOMICI
    # ========================================================================
    total_revenue = trading_profit + macse_revenue
    annual_profit = total_revenue
    roi_percent = (annual_profit / battery_investment) * 100 if battery_investment > 0 else 0
    payback_years = battery_investment / annual_profit if annual_profit > 0 else float('inf')

    # ========================================================================
    # Lorenzo Giannuzzo: CONFRONTO CON BASELINE
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
    # Lorenzo Giannuzzo: COSTRUZIONE JSON
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
        # Lorenzo Giannuzzo: SEZIONE CONFRONTO CON BASELINE
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
        # Lorenzo Giannuzzo: METADATI DECISIONI PSO
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
    # Lorenzo Giannuzzo: SALVATAGGIO JSON
    # ========================================================================
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lorenzo Giannuzzo: Usa nome custom se fornito
    if output_filename:
        json_file = os.path.join(output_dir, f'{output_filename}.json')
    else:
        json_file = os.path.join(output_dir,
                                 f'simulation_results_{battery.technology.lower().replace("-", "_")}_v270_autonomous.json')

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"✓ JSON salvato: {json_file}")

    return results_json


# ========================================================================================================
# Lorenzo Giannuzzo: SEZIONE 9: GRAFICI (placeholder - implementa come vuoi)
# ========================================================================================================
def create_detailed_monthly_pv_plots(results_df, battery, pv_system):
    """
    Lorenzo Giannuzzo: GRAFICI DETTAGLIATI MENSILI con IMPATTO PV
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

    # Lorenzo Giannuzzo: Per ogni mese, crea grafico dettagliato
    for mese in range(1, 13):
        df_mese = results_df[results_df['Mese'] == mese].copy()
        if len(df_mese) == 0:
            continue

        # Lorenzo Giannuzzo: Seleziona giorno rappresentativo
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

        # Lorenzo Giannuzzo: Crea figura con 4 subplot per analisi completa
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

        data_str = df_giorno['Data'].iloc[0].strftime('%d/%m/%Y') if len(df_giorno) > 0 else ''
        fig.suptitle(f'Analisi Dettagliata PV - {mesi_nomi[mese - 1]} {data_str}\n{battery.technology}',
                     fontsize=16, fontweight='bold', y=0.995)

        # Lorenzo Giannuzzo: Subplot 1: PRODUZIONE PV E ALLOCAZIONE
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

        # Lorenzo Giannuzzo: Subplot 2: FONTI DI CARICA BATTERIA
        ax2 = fig.add_subplot(gs[1, 0])

        # Lorenzo Giannuzzo: USA PV_to_Battery per coerenza con grafico sopra
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

        # Lorenzo Giannuzzo: Subplot 3: AZIONI BATTERIA
        ax3 = fig.add_subplot(gs[1, 1])
        ax3_twin = ax3.twinx()

        # Lorenzo Giannuzzo: USA DIRETTAMENTE LE COLONNE CORRETTE DAL DATAFRAME
        charge_pv_bars = []
        charge_grid_bars = []
        discharge_load_bars = []
        discharge_trading_bars = []

        for idx, row in df_giorno.iterrows():
            # Lorenzo Giannuzzo: Usa Azione_Trading_MW che contiene già l'azione trading corretta
            trading_action = row['Azione_Trading_MW']  # Positivo=carica, Negativo=scarica trading

            # Usa le colonne dedicate già presenti nel DataFrame
            load_discharge = row.get('Load_Discharge_MW', 0.0)
            trading_discharge = row.get('Trading_Discharge_MW', 0.0)

            # Lorenzo Giannuzzo: Carica (trading_action positivo)
            if trading_action > 0.001:
                # Recupera fonti di carica
                pv_to_batt = row.get('PV_to_Battery_MWh', 0.0)
                grid_to_batt = row.get('Energy_from_Grid_MWh', 0.0)

                charge_pv_bars.append(pv_to_batt)
                charge_grid_bars.append(grid_to_batt)
                discharge_load_bars.append(0.0)
                discharge_trading_bars.append(0.0)

            # Lorenzo Giannuzzo: Scarica trading (trading_action negativo)
            elif trading_action < -0.001:
                charge_pv_bars.append(0.0)
                charge_grid_bars.append(0.0)

                # Lorenzo Giannuzzo: Scarica trading (già registrata correttamente)
                discharge_trading_bars.append(-trading_discharge)  # Negativo per grafico

                # Lorenzo Giannuzzo: Scarica carico (può coesistere con trading se erano entrambi richiesti)
                if load_discharge > 0.001:
                    discharge_load_bars.append(-load_discharge)
                else:
                    discharge_load_bars.append(0.0)

            # Lorenzo Giannuzzo: Solo scarica per carico (nessun trading)
            elif load_discharge > 0.001:
                charge_pv_bars.append(0.0)
                charge_grid_bars.append(0.0)
                discharge_trading_bars.append(0.0)
                discharge_load_bars.append(-load_discharge)

            # Lorenzo Giannuzzo: IDLE
            else:
                charge_pv_bars.append(0.0)
                charge_grid_bars.append(0.0)
                discharge_load_bars.append(0.0)
                discharge_trading_bars.append(0.0)

        width = 0.8

        # Lorenzo Giannuzzo: Plotta SCARICA per TRADING
        ax3.bar(df_giorno['Ora'], discharge_trading_bars, width=width,
                color='#E63946', alpha=0.8, edgecolor='black', linewidth=1,
                label='Scarica Trading')

        # Lorenzo Giannuzzo: Plotta SCARICA per CARICO
        ax3.bar(df_giorno['Ora'], discharge_load_bars, width=width,
                bottom=discharge_trading_bars,
                color='#9D4EDD', alpha=0.8, edgecolor='black', linewidth=1,
                label='Scarica Carico')

        # Lorenzo Giannuzzo: Plotta CARICA da RETE
        ax3.bar(df_giorno['Ora'], charge_grid_bars, width=width,
                color='#F77F00', alpha=0.8, edgecolor='black', linewidth=1,
                label='Carica da Rete')

        # Lorenzo Giannuzzo: Plotta CARICA da PV
        ax3.bar(df_giorno['Ora'], charge_pv_bars, width=width,
                bottom=charge_grid_bars,
                color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1,
                label='Carica da PV')

        # Lorenzo Giannuzzo: Linea prezzo energia
        ax3_twin.plot(df_giorno['Ora'], df_giorno['€/MWh'],
                      color='#457B9D', linewidth=2.5, marker='s', markersize=5,
                      label='Prezzo Energia', zorder=10)

        # Lorenzo Giannuzzo: Linea zero
        ax3.axhline(y=0, color='black', linewidth=1.5, linestyle='-', zorder=5)

        # Lorenzo Giannuzzo: Etichette e formattazione
        ax3.set_ylabel('Potenza Batteria (MW)', fontsize=12, fontweight='bold')
        ax3_twin.set_ylabel('Prezzo (€/MWh)', fontsize=12, fontweight='bold', color='#457B9D')
        ax3_twin.tick_params(axis='y', labelcolor='#457B9D')
        ax3.set_xlabel('Ora', fontsize=11)
        ax3.set_title('Azioni Batteria (Carico vs Trading) vs Prezzo', fontsize=13, fontweight='bold')

        # Lorenzo Giannuzzo: Legends
        ax3.legend(fontsize=9, loc='upper left', framealpha=0.95)
        ax3_twin.legend(fontsize=9, loc='upper right', framealpha=0.95)

        # Lorenzo Giannuzzo: Grid e limiti
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_xticks(range(0, 24, 2))

        # Lorenzo Giannuzzo: Forza limiti corretti asse Y
        all_actions = (discharge_trading_bars + discharge_load_bars +
                       [sum(x) for x in zip(charge_pv_bars, charge_grid_bars)])
        if any(x != 0 for x in all_actions):
            y_max = max(abs(min(all_actions)), max(all_actions)) * 1.1
            ax3.set_ylim(-y_max, y_max)

        # Lorenzo Giannuzzo: Subplot 4: SOC E IMPATTO PV
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

        # Lorenzo Giannuzzo: Subplot 5: BILANCIO ECONOMICO ORARIO
        ax5 = fig.add_subplot(gs[2, 1])
        profitto_orario = []
        for idx, row in df_giorno.iterrows():
            profit_hour = 0
            # Lorenzo Giannuzzo: Costi: acquisto da rete
            if row['Energy_from_Grid_MWh'] > 0:
                profit_hour -= row['Energy_from_Grid_MWh'] * row.get('Prezzo_Acquisto_€/MWh', row['€/MWh'])
            if 'Load_from_Grid_MWh' in row and row['Load_from_Grid_MWh'] > 0:
                profit_hour -= row['Load_from_Grid_MWh'] * row.get('Prezzo_Acquisto_€/MWh', row['€/MWh'])

            # Lorenzo Giannuzzo: Ricavi: vendite a rete
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

        # Lorenzo Giannuzzo: Subplot 6: STATISTICHE GIORNALIERE
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

    # Lorenzo Giannuzzo: Subplot 1: Pie chart allocazione energia PV
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

    # Lorenzo Giannuzzo: Subplot 2: Grafico a barre energia PV per destinazione
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

    # Lorenzo Giannuzzo: Subplot 3: Timeline profitto cumulativo (stima contributo PV)
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

    # Lorenzo Giannuzzo: Subplot 4: Statistiche testuali PV
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

    # Lorenzo Giannuzzo: Grafico 1: Produzione PV oraria (sample 7 giorni)
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

    # Lorenzo Giannuzzo: Grafico 2: Confronto fonti carica batteria
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

    # Lorenzo Giannuzzo: Grafico 3: Utilizzo PV nel tempo (aggregato giornaliero)
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

    # Lorenzo Giannuzzo: Grafico 4: Impatto economico PV (stima ricavi)
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

    # Lorenzo Giannuzzo: Grafico 5: Percentuali utilizzo PV
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

    # Lorenzo Giannuzzo: Grafico 6: Indici prestazione PV
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

    # Lorenzo Giannuzzo: Subplot 1: Produzione PV e allocazione
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

    # Lorenzo Giannuzzo: Subplot 2: SOC batteria con evidenza carica da PV
    ax2 = axes[1]
    if 'SOC' in results_df.columns:
        ax2.plot(hours, results_df['SOC'].iloc[:sample_hours] * 100,
                 color='#457B9D', linewidth=2.5, label='SOC', zorder=3)
        ax2.fill_between(hours, battery.soc_min * 100,
                         results_df['SOC'].iloc[:sample_hours] * 100,
                         color='#457B9D', alpha=0.2)

        # Lorenzo Giannuzzo: Evidenzia ore con carica da PV
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

    # Lorenzo Giannuzzo: Subplot 3: Prezzo energia
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

    # Lorenzo Giannuzzo: Grafico 1: Pie chart fonti fornitura carico
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

    # Lorenzo Giannuzzo: Grafico 2: Barre energia per fonte
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

    # Lorenzo Giannuzzo: Grafico 3: Timeline fornitura carico (sample 7 giorni)
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

    # Lorenzo Giannuzzo: Grafico 4: Percentuali coverage
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

    # Lorenzo Giannuzzo: Grafico 5: Statistiche testuali
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
    Lorenzo Giannuzzo: Calcola scenario BASELINE senza batteria CON VINCOLO POD
    """
    total_cost_buy = 0.0
    total_revenue_sell = 0.0

    total_pv_to_load = 0.0
    total_pv_to_grid = 0.0
    total_load_from_grid = 0.0
    total_load_required = 0.0

    # Lorenzo Giannuzzo: tracking POD baseline
    total_pv_curtailed_baseline = 0.0
    total_load_unserved_baseline = 0.0

    n_hours = len(prices_sell)

    for h in range(n_hours):
        pv_available = pv_production[h] if PV_ENABLED else 0.0
        load_required = load_demand[h] if LOAD_ENABLED else 0.0
        price_sell = prices_sell[h]
        price_buy = prices_buy[h]

        total_load_required += load_required

        # Lorenzo Giannuzzo: LOGICA BASELINE: PV al carico prima
        pv_to_load = min(pv_available, load_required)
        pv_remaining = pv_available - pv_to_load
        load_remaining = load_required - pv_to_load

        total_pv_to_load += pv_to_load

        # Lorenzo Giannuzzo: PV eccesso venduto CON VINCOLO POD
        if pv_remaining > 0.001:
            pv_to_grid = min(pv_remaining, POD_POWER_MW)
            pv_curtailed = pv_remaining - pv_to_grid

            total_pv_to_grid += pv_to_grid
            total_revenue_sell += pv_to_grid * price_sell

            if pv_curtailed > 0.001:
                total_pv_curtailed_baseline += pv_curtailed
                # Penalità leggera per PV perso
                total_cost_buy += pv_curtailed * price_sell * 0.0

        # Lorenzo Giannuzzo: Carico residuo comprato da rete CON VINCOLO POD
        if load_remaining > 0.001:
            load_from_grid = min(load_remaining, POD_POWER_MW)
            load_unserved = load_remaining - load_from_grid

            total_load_from_grid += load_from_grid
            total_cost_buy += load_from_grid * price_buy

            if load_unserved > 0.001:
                total_load_unserved_baseline += load_unserved
                # Lorenzo Giannuzzo: Penalità pesante per carico non servito
                total_cost_buy += load_unserved * price_buy * 1000.0

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
        'pv_curtailed_baseline': total_pv_curtailed_baseline,
        'load_unserved_baseline': total_load_unserved_baseline
    }

def export_complete_results_to_json(results_df, battery, output_dir='results', output_filename=None):
    """
    Lorenzo Giannuzzo: Esporta DataFrame completo in JSON (equivalente Excel)
    UNICO JSON NECESSARIO - Contiene tutti i dati orari
    """
    print("\n📄 Esportazione JSON completo...")

    # Converti DataFrame in formato JSON-friendly
    results_dict = results_df.to_dict(orient='records')

    # Converti datetime e NaN in formati JSON validi
    for record in results_dict:
        if 'Data' in record and pd.notna(record['Data']):
            if isinstance(record['Data'], pd.Timestamp):
                record['Data'] = record['Data'].strftime('%Y-%m-%d %H:%M:%S')

        # Converti NaN in None e numpy types in Python types
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (np.int64, np.int32)):
                record[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                record[key] = float(value)

    # Crea JSON completo con metadati
    complete_json = {
        "metadata": {
            "version": "3.8.0-COMPLETE-DATA-POD",
            "description": "Complete hourly simulation data - equivalent to Excel export",
            "technology": battery.technology,
            "total_hours": len(results_df),
            "timestamp": datetime.now().isoformat(),
            "columns": list(results_df.columns),
            "pod_power_mw": POD_POWER_MW,
            "features": [
                "Custom efficiency support",
                "POD limit enforcement",
                "Autonomous load decisions",
                "PV integration",
                "Load profile management"
            ]
        },

        "battery_info": {
            "technology": battery.technology,
            "nominal_capacity_mwh": float(battery.nominal_capacity),
            "final_capacity_mwh": float(battery.capacity),
            "max_power_mw": float(battery.max_power),
            "max_c_rate": float(battery.max_c_rate),
            "final_soc": float(battery.get_soc()),
            "final_soh_percent": float(battery.get_soh()),
            "equivalent_cycles": float(battery.equivalent_cycles),
            "throughput_kwh": float(battery.throughput_kwh),
            "energy_from_grid_mwh": float(battery.energy_from_grid_mwh),
            "energy_from_pv_mwh": float(battery.energy_from_pv_mwh),
            "efficiency": {
                "charge": float(battery.charge_efficiency),
                "discharge": float(battery.discharge_efficiency),
                "roundtrip": float(battery.efficiency)
            }
        },

        "hourly_data": results_dict
    }

    # Salva JSON
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Usa nome custom se fornito, altrimenti automatico
    if output_filename:
        json_file = os.path.join(output_dir, f'{output_filename}.json')
    else:
        json_file = os.path.join(output_dir, f'complete_data_{battery.technology.lower().replace("-", "_")}_v380.json')

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(complete_json, f, indent=2, ensure_ascii=False)

    print(f"✓ JSON completo salvato: {json_file}")
    print(f"  • Record totali: {len(results_dict)}")
    print(f"  • Colonne: {len(results_df.columns)}")
    print(f"  • Dimensione file: {os.path.getsize(json_file) / 1024 / 1024:.2f} MB")

    return json_file

# ========================================================================================================
# Lorenzo Giannuzzo: FUNZIONE MAIN
# ========================================================================================================
def main():
    """
    Lorenzo Giannuzzo: Main function con argomenti da command line
    """
    # ========================================================================
    # PARSE ARGOMENTI
    # ========================================================================
    args = parse_arguments()

    # ========================================================================
    # Lorenzo Giannuzzo: OVERRIDE VARIABILI GLOBALI CON ARGOMENTI CLI
    # ========================================================================
    global POD_POWER_MW, BATTERY_TECHNOLOGY, BATTERY_CAPACITY_MWH, BATTERY_MAX_POWER_MW
    global BATTERY_MAX_C_RATE, PV_ENABLED, LOAD_ENABLED, PV_NOMINAL_POWER_KWP
    global LITHIUM_ION_SOC_MIN, LITHIUM_ION_SOC_MAX, GRAPHENE_SOC_MIN, GRAPHENE_SOC_MAX
    global SAVE_PLOTS, MACSE_ENABLED, MACSE_CAPACITY_MWH, MACSE_CONTRACT_YEARS, MACSE_PRICE_PER_MW_YEAR
    global ENABLE_MULTIPROCESSING, MULTIPROCESSING_CORES
    global ARBITRAGE_BIAS_MULTIPLIER, AUTOCONSUMO_BIAS_MULTIPLIER

    POD_POWER_MW = args.pod_limit
    BATTERY_TECHNOLOGY = args.battery_tech
    BATTERY_CAPACITY_MWH = args.battery_capacity
    BATTERY_MAX_POWER_MW = args.battery_power
    BATTERY_MAX_C_RATE = args.battery_c_rate

    LITHIUM_ION_SOC_MIN = args.lithium_soc_min
    LITHIUM_ION_SOC_MAX = args.lithium_soc_max
    GRAPHENE_SOC_MIN = args.graphene_soc_min
    GRAPHENE_SOC_MAX = args.graphene_soc_max

    PV_ENABLED = args.pv_enabled
    LOAD_ENABLED = args.load_enabled
    PV_NOMINAL_POWER_KWP = args.pv_nominal_power

    SAVE_PLOTS = args.save_plots

    MACSE_ENABLED = args.macse_enabled
    MACSE_CAPACITY_MWH = args.macse_capacity
    MACSE_CONTRACT_YEARS = args.macse_contract_years
    MACSE_PRICE_PER_MW_YEAR = args.macse_price_per_mw_year

    ENABLE_MULTIPROCESSING = not args.no_parallel
    MULTIPROCESSING_CORES = args.n_cores
    
    # Lorenzo Giannuzzo: Bias ottimizzazione
    ARBITRAGE_BIAS_MULTIPLIER = args.bias_arbitrage
    AUTOCONSUMO_BIAS_MULTIPLIER = args.bias_autoconsumo

    # ========================================================================
    # Lorenzo Giannuzzo: SETUP EFFICIENZA CUSTOM
    # ========================================================================
    custom_efficiency_dict = None
    if args.custom_efficiency:
        custom_efficiency_dict = {}

        if args.charge_efficiency is not None:
            custom_efficiency_dict['charge_efficiency'] = args.charge_efficiency

        if args.discharge_efficiency is not None:
            custom_efficiency_dict['discharge_efficiency'] = args.discharge_efficiency

        if args.roundtrip_efficiency is not None:
            custom_efficiency_dict['roundtrip_efficiency'] = args.roundtrip_efficiency

    # ========================================================================
    # Lorenzo Giannuzzo: PRINT CONFIGURAZIONE
    # ========================================================================
    print("=" * 80)
    print("BESS OPTIMIZATION v3.8 - COMMAND LINE MODE")
    print("=" * 80)
    print("\n📋 CONFIGURAZIONE:")
    print(f"  • Prezzi vendita:         {args.price_sell}")
    print(f"  • Prezzi acquisto:        {args.price_buy}")
    print(f"  • POD Limit:              {args.pod_limit} MW")
    print(f"  • Tecnologia batteria:    {args.battery_tech}")
    print(f"  • Capacità:               {args.battery_capacity} MWh")
    print(f"  • Potenza:                {args.battery_power} MW")
    print(f"  • C-rate:                 {args.battery_c_rate}")

    if args.battery_tech == "LITIO-IONE":
        print(f"  • SOC range:              {args.lithium_soc_min * 100:.0f}% - {args.lithium_soc_max * 100:.0f}%")
    else:
        print(f"  • SOC range:              {args.graphene_soc_min * 100:.0f}% - {args.graphene_soc_max * 100:.0f}%")

    if args.pv_enabled:
        print(f"  • PV abilitato:           SÌ ({args.pv_nominal_power} kWp)")
        print(f"  • File PV:                {args.pv_file}")
    else:
        print(f"  • PV abilitato:           NO")

    if args.load_enabled:
        print(f"  • Carico abilitato:       SÌ")
        print(f"  • File carico:            {args.load_file}")
    else:
        print(f"  • Carico abilitato:       NO")

    if args.macse_enabled:
        print(f"  • MACSE abilitato:        SÌ ({args.macse_capacity} MWh, {args.macse_contract_years} anni)")
    else:
        print(f"  • MACSE abilitato:        NO")

    print(f"  • Parallelizzazione:      {'SÌ' if not args.no_parallel else 'NO'}")
    if not args.no_parallel:
        print(f"  • Cores:                  {args.n_cores} ({'auto' if args.n_cores < 0 else 'manual'})")
    print(f"  • Particelle PSO:         {args.n_particles}")
    print(f"  • Iterazioni PSO:         {args.n_iterations}")
    print(f"  • Bias arbitraggio:       {args.bias_arbitrage:.2f}x")
    print(f"  • Bias autoconsumo:       {args.bias_autoconsumo:.2f}x")
    print(f"  • Salva grafici:          {'SÌ' if args.save_plots else 'NO'}")
    print(f"  • Directory output:       {args.output_dir}")
    print("=" * 80)

    # ========================================================================
    # Lorenzo Giannuzzo: CARICAMENTO DATI
    # ========================================================================
    try:
        df = pd.read_excel(args.price_sell)
        if df['€/MWh'].dtype == 'object':
            df['€/MWh'] = df['€/MWh'].astype(str).str.replace(',', '.').astype(float)
        print(f"\n✓ Prezzi vendita: {len(df)} righe, media {df['€/MWh'].mean():.2f} €/MWh")
    except Exception as e:
        print(f"❌ Errore caricamento prezzi vendita: {e}")
        sys.exit(1)

    try:
        df2 = pd.read_excel(args.price_buy)
        if df2['€/MWh'].dtype == 'object':
            df2['€/MWh'] = df2['€/MWh'].astype(str).str.replace(',', '.').astype(float)
        print(f"✓ Prezzi acquisto: {len(df2)} righe, media {df2['€/MWh'].mean():.2f} €/MWh")
    except Exception as e:
        print(f"❌ Errore caricamento prezzi acquisto: {e}")
        sys.exit(1)

    pv_df = None
    pv_system = None
    if PV_ENABLED and args.pv_file:
        try:
            pv_df = pd.read_csv(args.pv_file, sep=';')
            pv_system = PhotovoltaicSystem(nominal_power_kwp=args.pv_nominal_power)
            print(f"✓ PV: {len(pv_df)} righe caricato")
        except Exception as e:
            print(f"❌ Errore caricamento PV: {e}")
            pv_df = None
            pv_system = None

    load_df = None
    load_profile = None
    if LOAD_ENABLED and args.load_file:
        try:
            xls = pd.ExcelFile(args.load_file)
            first_sheet = xls.sheet_names[0]
            load_df_raw = pd.read_excel(args.load_file, sheet_name=first_sheet)

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
            print(f"✓ Carico: {len(load_df)} righe, media {load_df['value'].mean():.2f} kW")
        except Exception as e:
            print(f"❌ Errore caricamento carico: {e}")
            load_df = None
            load_profile = None

    # ========================================================================
    # Lorenzo Giannuzzo: SETUP BATTERIA E OTTIMIZZATORE MIGLIORATO
    # ========================================================================
    battery = Battery(
        technology=BATTERY_TECHNOLOGY,
        capacity_mwh=BATTERY_CAPACITY_MWH,
        max_power_mw=BATTERY_MAX_POWER_MW,
        max_c_rate=BATTERY_MAX_C_RATE,
        custom_efficiency=custom_efficiency_dict
    )

    # Usa PSO ottimizzato con parametri migliorati
    optimizer = PSOOptimizer(
        n_particles=args.n_particles,
        n_iterations=args.n_iterations,
        w_start=0.9,    # Migliorato da 0.95
        w_end=0.1,      # Migliorato da 0.005  
        c1=1.5,         # Migliorato da 2.0
        c2=2.5          # Migliorato da 2.0
    )
    
    # Configura ottimizzazioni in base agli argomenti
    if args.disable_optimizations:
        optimizer.configure_optimizations(intelligent_init=False, temporal_patterns=False,
                                        constraint_repair=False, adaptive_parameters=False)
        print(f"\n⚠️  OTTIMIZZAZIONI PSO DISABILITATE (modalità originale)")
    else:
        print(f"\n🚀 OTTIMIZZAZIONI PSO ATTIVE:")
        print(f"   • Inizializzazione intelligente: ✅")
        print(f"   • Constraint repair: ✅") 
        print(f"   • Parametri adattivi: ✅")
        print(f"   • Diversificazione dinamica: ✅")
    
    # ========================================================================
    # Lorenzo Giannuzzo: MODALITÀ CONFRONTO PSO (se richiesta)
    # ========================================================================
    if args.compare_pso:
        print(f"\n🔬 MODALITÀ CONFRONTO PSO ATTIVATA")
        
        # Prepara dati per test (sample ridotto per velocità)
        n_hours_test = min(48, len(df))  # Test su 48 ore
        prices_sell_test = df['€/MWh'].values[:n_hours_test]
        prices_buy_test = df2['€/MWh'].values[:n_hours_test]
        
        if PV_ENABLED and pv_df is not None:
            pv_test = pv_df['P'].values[:n_hours_test] / 1000.0
        else:
            pv_test = np.zeros(n_hours_test)
            
        if LOAD_ENABLED and load_df is not None:
            load_test = load_df['value'].values[:n_hours_test] / 1000.0
        else:
            load_test = np.zeros(n_hours_test)
        
        # Esegui confronto
        #comparison_results = compare_pso_versions(
        #    battery, prices_sell_test, prices_buy_test, pv_test, load_test,
        #    n_particles=min(30, args.n_particles), n_iterations=min(50, args.n_iterations)
        #)
        
        print(f"\n✅ Confronto completato. Usa --disable-optimizations per tornare alla versione originale.")
        return  # Esce dopo il confronto

    simulator = RollingHorizonSimulator(
        battery,
        optimizer,
        pv_system=pv_system,
        load_profile=load_profile
    )

    # ========================================================================
    # Lorenzo Giannuzzo: ESECUZIONE SIMULAZIONE
    # ========================================================================
    start_time = datetime.now()
    results_df, trading_profit = simulator.simulate(df, df2, pv_df, load_df)
    end_time = datetime.now()

    # ========================================================================
    # Lorenzo Giannuzzo: CALCOLI FINALI
    # ========================================================================
    macse_revenue, macse_base, macse_penalty, macse_bonus = calculate_macse_revenue(battery)
    total_system_profit = trading_profit + macse_revenue

    prices_sell = df['€/MWh'].values
    prices_buy = df2['€/MWh'].values

    if PV_ENABLED and pv_df is not None:
        pv_production = pv_df['P'].values / 1000.0
        if len(pv_production) < len(prices_sell):
            pv_production = np.pad(pv_production, (0, len(prices_sell) - len(pv_production)), 'constant')
        elif len(pv_production) > len(prices_sell):
            pv_production = pv_production[:len(prices_sell)]
    else:
        pv_production = np.zeros(len(prices_sell))

    if LOAD_ENABLED and load_df is not None:
        load_demand = load_df['value'].values / 1000.0
        if len(load_demand) < len(prices_sell):
            load_demand = np.pad(load_demand, (0, len(prices_sell) - len(load_demand)), 'constant')
        elif len(load_demand) > len(prices_sell):
            load_demand = load_demand[:len(prices_sell)]
    else:
        load_demand = np.zeros(len(prices_sell))

    baseline_scenario = calculate_baseline_scenario(prices_sell, prices_buy, pv_production, load_demand)

    # ========================================================================
    # Lorenzo Giannuzzo: STAMPA RISULTATI (usa la tua funzione print esistente)
    # ========================================================================
    print("\n" + "=" * 80)
    print("RISULTATI FINALI - CONFRONTO ECONOMICO")
    print("=" * 80)

    # ========================================================================
    # Lorenzo Giannuzzo: SALVATAGGIO RISULTATI - SOLO EXCEL + JSON COMPLETO
    # ========================================================================
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Determina nome file base
    if args.output_filename:
        base_filename = args.output_filename
    else:
        base_filename = f'risultati_{battery.technology.lower().replace("-", "_")}_cli'

    print("\n" + "=" * 80)
    print("💾 SALVATAGGIO RISULTATI")
    print("=" * 80)

    # ========================================================================
    # 1. SALVA EXCEL
    # ========================================================================
    output_file_excel = os.path.join(args.output_dir, f'{base_filename}.xlsx')
    with pd.ExcelWriter(output_file_excel, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Risultati', index=False)
    print(f"✓ Excel salvato: {output_file_excel}")

    # ========================================================================
    # 2. SALVA JSON COMPLETO (equivalente Excel)
    # ========================================================================
    export_complete_results_to_json(
        results_df,
        battery,
        output_dir=args.output_dir,
        output_filename=base_filename
    )

    print("\n✅ File di output creati:")
    print(f"  1. {base_filename}.xlsx   (Excel con tutti i dati)")
    print(f"  2. {base_filename}.json   (JSON con tutti i dati)")
    print("=" * 80)
    # ========================================================================
    # Lorenzo Giannuzzo: GRAFICI BASE (SEMPRE GENERATI se SAVE_PLOTS=True)
    # ========================================================================
    if SAVE_PLOTS:
        print("\n Generazione grafici...")

        # Crea directory visualization se non esiste
        viz_folder = 'visualization'
        if not os.path.exists(viz_folder):
            os.makedirs(viz_folder)

        # ====================================================================
        # Lorenzo Giannuzzo: GRAFICO 1: Overview Generale (4 subplot)
        # ====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'BESS Optimization Overview - {battery.technology}', fontsize=16, fontweight='bold')

        # Subplot 1: SOC nel tempo
        ax1 = axes[0, 0]
        hours = range(len(results_df))
        ax1.plot(hours, results_df['SOC'] * 100, color='#457B9D', linewidth=2, label='SOC')
        ax1.fill_between(hours, battery.soc_min * 100, results_df['SOC'] * 100, color='#457B9D', alpha=0.2)
        ax1.axhline(y=battery.soc_min * 100, color='red', linestyle='--', linewidth=1, alpha=0.7,
                    label=f'SOC min ({battery.soc_min * 100:.0f}%)')
        ax1.axhline(y=battery.soc_max * 100, color='green', linestyle='--', linewidth=1, alpha=0.7,
                    label=f'SOC max ({battery.soc_max * 100:.0f}%)')
        ax1.set_xlabel('Ora', fontsize=11)
        ax1.set_ylabel('SOC (%)', fontsize=11, fontweight='bold')
        ax1.set_title('State of Charge', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        # Lorenzo Giannuzzo: Subplot 2: Profitto Cumulativo
        ax2 = axes[0, 1]
        ax2.plot(hours, results_df['Profitto_Euro'], color='#06A77D', linewidth=2.5, label='Profitto Cumulativo')
        ax2.fill_between(hours, 0, results_df['Profitto_Euro'], color='#06A77D', alpha=0.2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Ora', fontsize=11)
        ax2.set_ylabel('Profitto (€)', fontsize=11, fontweight='bold')
        ax2.set_title('Profitto Cumulativo', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # SLorenzo Giannuzzo: ubplot 3: Azioni Batteria vs Prezzi
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()

        # Lorenzo Giannuzzo:  Azioni batteria
        colors_action = ['#06A77D' if a > 0 else '#E63946' if a < 0 else '#999999' for a in
                         results_df['Azione_Trading_MW']]
        ax3.bar(hours, results_df['Azione_Trading_MW'], color=colors_action, alpha=0.7, width=1, edgecolor='none')
        ax3.axhline(y=0, color='black', linewidth=1)
        ax3.set_xlabel('Ora', fontsize=11)
        ax3.set_ylabel('Potenza Batteria (MW)', fontsize=11, fontweight='bold')
        ax3.set_title('Azioni Batteria vs Prezzi Energia', fontsize=12, fontweight='bold')

        # Lorenzo Giannuzzo: Prezzi
        ax3_twin.plot(hours, results_df['€/MWh'], color='#457B9D', linewidth=1.5, alpha=0.8, label='Prezzo Vendita')
        if 'Prezzo_Acquisto_€/MWh' in results_df.columns:
            ax3_twin.plot(hours, results_df['Prezzo_Acquisto_€/MWh'], color='#8B0000', linewidth=1.5, alpha=0.6,
                          linestyle='--', label='Prezzo Acquisto')
        ax3_twin.set_ylabel('Prezzo (€/MWh)', fontsize=11, fontweight='bold', color='#457B9D')
        ax3_twin.tick_params(axis='y', labelcolor='#457B9D')
        ax3_twin.legend(fontsize=9, loc='upper right')

        # Lorenzo Giannuzzo: Subplot 4: SOH (Degradazione)
        ax4 = axes[1, 1]
        ax4.plot(hours, results_df['SOH_%'], color='#F77F00', linewidth=2.5, label='State of Health')
        ax4.fill_between(hours, 80, results_df['SOH_%'], where=(results_df['SOH_%'] >= 80), color='#06A77D', alpha=0.2,
                         label='SOH OK (>80%)')
        ax4.fill_between(hours, 0, results_df['SOH_%'], where=(results_df['SOH_%'] < 80), color='#E63946', alpha=0.2,
                         label='SOH Degradato (<80%)')
        ax4.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Soglia EOL (80%)')
        ax4.set_xlabel('Ora', fontsize=11)
        ax4.set_ylabel('SOH (%)', fontsize=11, fontweight='bold')
        ax4.set_title('State of Health (Degradazione)', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(75, 101)

        plt.tight_layout()
        overview_file = os.path.join(viz_folder, f'overview_{battery.technology.lower().replace("-", "_")}.png')
        plt.savefig(overview_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Salvato: {overview_file}")

        # ====================================================================
        # Lorenzo Giannuzzo: GRAFICO 2: POD Tracking (se LOAD_ENABLED)
        # ====================================================================
        if LOAD_ENABLED and 'Grid_Withdrawal_MW' in results_df.columns:
            fig, axes = plt.subplots(2, 1, figsize=(16, 10))
            fig.suptitle(f'POD Monitoring - Limite {POD_POWER_MW} MW', fontsize=16, fontweight='bold')

            # Subplot 1: Prelievo da rete
            ax1 = axes[0]
            ax1.plot(hours, results_df['Grid_Withdrawal_MW'], color='#E63946', linewidth=1.5, label='Prelievo Rete')
            ax1.fill_between(hours, 0, results_df['Grid_Withdrawal_MW'], color='#E63946', alpha=0.2)
            ax1.axhline(y=POD_POWER_MW, color='red', linestyle='--', linewidth=2,
                        label=f'POD Limit ({POD_POWER_MW} MW)')

            # Lorenzo Giannuzzo: Evidenzia violazioni
            violations_withdrawal = results_df['Grid_Withdrawal_MW'] > POD_POWER_MW
            if violations_withdrawal.any():
                ax1.fill_between(hours, POD_POWER_MW, results_df['Grid_Withdrawal_MW'],
                                 where=violations_withdrawal, color='red', alpha=0.5, label='Violazioni POD')

            ax1.set_ylabel('Prelievo (MW)', fontsize=11, fontweight='bold')
            ax1.set_title('Prelievo da Rete vs POD Limit', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Lorenzo Giannuzzo: Subplot 2: Immissione in rete
            ax2 = axes[1]
            ax2.plot(hours, results_df['Grid_Injection_MW'], color='#06A77D', linewidth=1.5, label='Immissione Rete')
            ax2.fill_between(hours, 0, results_df['Grid_Injection_MW'], color='#06A77D', alpha=0.2)
            ax2.axhline(y=POD_POWER_MW, color='green', linestyle='--', linewidth=2,
                        label=f'POD Limit ({POD_POWER_MW} MW)')

            # Lorenzo Giannuzzo: Evidenzia violazioni
            violations_injection = results_df['Grid_Injection_MW'] > POD_POWER_MW
            if violations_injection.any():
                ax2.fill_between(hours, POD_POWER_MW, results_df['Grid_Injection_MW'],
                                 where=violations_injection, color='red', alpha=0.5, label='Violazioni POD')

            ax2.set_xlabel('Ora', fontsize=11)
            ax2.set_ylabel('Immissione (MW)', fontsize=11, fontweight='bold')
            ax2.set_title('Immissione in Rete vs POD Limit', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            pod_file = os.path.join(viz_folder, f'pod_tracking_{battery.technology.lower().replace("-", "_")}.png')
            plt.savefig(pod_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Salvato: {pod_file}")

        # ====================================================================
        # Lorenzo Giannuzzo: GRAFICO 3: Analisi Economica
        # ====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Analisi Economica - {battery.technology}', fontsize=16, fontweight='bold')

        # Lorenzo Giannuzzo: Subplot 1: Profitto orario
        ax1 = axes[0, 0]
        profit_hourly = results_df['Profitto_Euro'].diff().fillna(results_df['Profitto_Euro'].iloc[0])
        colors_profit = ['#06A77D' if p >= 0 else '#E63946' for p in profit_hourly]
        ax1.bar(hours, profit_hourly, color=colors_profit, alpha=0.7, width=1, edgecolor='none')
        ax1.axhline(y=0, color='black', linewidth=1)
        ax1.set_xlabel('Ora', fontsize=11)
        ax1.set_ylabel('Profitto Orario (€)', fontsize=11, fontweight='bold')
        ax1.set_title('Profitto/Perdita Oraria', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Lorenzo Giannuzzo: Subplot 2: Breakdown costi/ricavi (sample 7 giorni)
        ax2 = axes[0, 1]
        sample_hours = min(168, len(results_df))
        hours_sample = range(sample_hours)

        # Lorenzo Giannuzzo: Calcola ricavi e costi per ogni ora
        revenues = []
        costs = []
        for idx in range(sample_hours):
            row = results_df.iloc[idx]

            # Lorenzo Giannuzzo: Ricavi: vendite batteria + vendite PV
            revenue = 0
            if row.get('Trading_Discharge_MW', 0) > 0:
                revenue += row['Trading_Discharge_MW'] * row['€/MWh']
            if row.get('PV_to_Grid_MWh', 0) > 0:
                revenue += row['PV_to_Grid_MWh'] * row['€/MWh']
            revenues.append(revenue)

            # Lorenzo Giannuzzo: Costi: acquisti rete
            cost = 0
            if row.get('Energy_from_Grid_MWh', 0) > 0:
                cost += row['Energy_from_Grid_MWh'] * row.get('Prezzo_Acquisto_€/MWh', row['€/MWh'])
            if row.get('Load_from_Grid_MWh', 0) > 0:
                cost += row['Load_from_Grid_MWh'] * row.get('Prezzo_Acquisto_€/MWh', row['€/MWh'])
            costs.append(cost)

        ax2.bar(hours_sample, revenues, color='#06A77D', alpha=0.7, label='Ricavi')
        ax2.bar(hours_sample, [-c for c in costs], color='#E63946', alpha=0.7, label='Costi')
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.set_xlabel('Ora (sample 7 giorni)', fontsize=11)
        ax2.set_ylabel('€', fontsize=11, fontweight='bold')
        ax2.set_title('Ricavi vs Costi', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        # Subplot 3: Distribuzione prezzi energia
        ax3 = axes[1, 0]
        ax3.hist(results_df['€/MWh'], bins=50, color='#457B9D', alpha=0.7, edgecolor='black')
        ax3.axvline(x=results_df['€/MWh'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Media: {results_df["€/MWh"].mean():.2f} €/MWh')
        ax3.axvline(x=results_df['€/MWh'].median(), color='green', linestyle='--', linewidth=2,
                    label=f'Mediana: {results_df["€/MWh"].median():.2f} €/MWh')
        ax3.set_xlabel('Prezzo (€/MWh)', fontsize=11)
        ax3.set_ylabel('Frequenza', fontsize=11, fontweight='bold')
        ax3.set_title('Distribuzione Prezzi Energia', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

        # Lorenzo Giannuzzo: Subplot 4: Statistiche testuali
        ax4 = axes[1, 1]
        ax4.axis('off')

        total_revenue = sum(revenues)
        total_cost = sum(costs)
        net_profit = total_revenue - total_cost

        stats_text = f"""
    STATISTICHE ECONOMICHE

    Ricavi Totali:
      • Vendita energia:        {total_revenue:.2f} €
      • Servizi MACSE:          {macse_revenue:.2f} €
      • TOTALE RICAVI:          {total_revenue + macse_revenue:.2f} €

    Costi Totali:
      • Acquisto energia:       {total_cost:.2f} €
      • Degrado batteria:       (incluso in calcolo)
      • TOTALE COSTI:           {total_cost:.2f} €

    Bilancio:
      • Profitto netto:         {trading_profit:.2f} €
      • ROI batteria:           {(trading_profit / 600000 * 100):.2f}%
      • Payback (stimato):      {(600000 / trading_profit if trading_profit > 0 else 999):.1f} anni

    Prezzi Energia:
      • Min:                    {results_df['€/MWh'].min():.2f} €/MWh
      • Max:                    {results_df['€/MWh'].max():.2f} €/MWh
      • Media:                  {results_df['€/MWh'].mean():.2f} €/MWh
      • Volatilità (std):       {results_df['€/MWh'].std():.2f} €/MWh
        """

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

        plt.tight_layout()
        economic_file = os.path.join(viz_folder,
                                     f'economic_analysis_{battery.technology.lower().replace("-", "_")}.png')
        plt.savefig(economic_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Salvato: {economic_file}")

        print("✓ Grafici base completati")

    # ========================================================================
    # GLorenzo Giannuzzo: GRAFICI PV E LOAD ì
    # ========================================================================
    if SAVE_PLOTS:
        if PV_ENABLED and pv_system:
            create_pv_impact_comparison(results_df, battery, pv_system)
            create_pv_impact_summary(results_df, battery, pv_system)
            create_pv_visualizations(results_df, battery, pv_system)
            create_detailed_monthly_pv_plots(results_df, battery, pv_system)

        if LOAD_ENABLED and load_profile:
            create_load_analysis_plots(results_df, battery, load_profile)

    # Grafici (se richiesti)
    if SAVE_PLOTS:
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

    # ========================================================================
    # STAMPA RISULTATI FINALI
    # ========================================================================
    print("\n" + "=" * 80)
    print("RISULTATI FINALI - ANALISI COMPLETA")
    print("=" * 80)

    # ========================================================================
    # 1. STATO FINALE BATTERIA
    # ========================================================================
    print("\n STATO FINALE BATTERIA:")
    print(f"  • Tecnologia:              {battery.technology}")
    print(f"  • Capacità nominale:       {battery.nominal_capacity:.2f} MWh")
    print(f"  • Capacità attuale:        {battery.capacity:.2f} MWh")
    print(f"  • SOC finale:              {battery.get_soc() * 100:.1f}%")
    print(f"  • SOH finale:              {battery.get_soh():.2f}%")
    print(f"  • Cicli equivalenti:       {battery.equivalent_cycles:.1f}")
    print(f"  • Throughput:              {battery.throughput_kwh / 1000:.2f} MWh")
    print(f"  • Energia da rete:         {battery.energy_from_grid_mwh:.2f} MWh")
    print(f"  • Energia da PV:           {battery.energy_from_pv_mwh:.2f} MWh")
    if battery.energy_from_grid_mwh + battery.energy_from_pv_mwh > 0:
        pv_percent = battery.energy_from_pv_mwh / (battery.energy_from_grid_mwh + battery.energy_from_pv_mwh) * 100
        print(f"  • % carica da PV:          {pv_percent:.1f}%")

    # ========================================================================
    # 2. RISULTATI ECONOMICI
    # ========================================================================
    print("\n💰 RISULTATI ECONOMICI:")
    print(f"  • Profitto trading:        {trading_profit:,.2f} €")
    if MACSE_ENABLED:
        print(f"  • Ricavi MACSE:")
        print(f"    - Base:                  {macse_base:,.2f} €")
        print(f"    - Bonus:                 {macse_bonus:,.2f} €")
        print(f"    - Penalità:              {macse_penalty:,.2f} €")
        print(f"    - Netto:                 {macse_revenue:,.2f} €")
        print(f"  • Profitto TOTALE:         {total_system_profit:,.2f} €")
    else:
        print(f"  • Profitto TOTALE:         {trading_profit:,.2f} €")

    # Prezzi energia
    print(f"\n  • Prezzo medio vendita:    {results_df['€/MWh'].mean():.2f} €/MWh")
    print(f"  • Prezzo medio acquisto:   {results_df['Prezzo_Acquisto_€/MWh'].mean():.2f} €/MWh")
    print(
        f"  • Spread medio:            {(results_df['€/MWh'].mean() - results_df['Prezzo_Acquisto_€/MWh'].mean()):.2f} €/MWh")

    # ========================================================================
    # 3. STATISTICHE FOTOVOLTAICO (se abilitato)
    # ========================================================================
    if PV_ENABLED and pv_system:
        pv_stats = pv_system.get_statistics()
        print("\n STATISTICHE FOTOVOLTAICO:")
        print(f"  • Produzione totale:       {pv_stats['total_production_mwh']:.2f} MWh")
        print(f"  • Allocazione energia:")
        print(
            f"    - A batteria:            {pv_stats['energy_to_battery_mwh']:.2f} MWh ({pv_stats['battery_utilization_percent']:.1f}%)")
        print(
            f"    - A rete (vendita):      {pv_stats['energy_to_grid_mwh']:.2f} MWh ({pv_stats['grid_sale_percent']:.1f}%)")
        print(
            f"    - A carico:              {pv_stats['energy_to_load_mwh']:.2f} MWh ({pv_stats['load_service_percent']:.1f}%)")
        if pv_stats['curtailed_energy_mwh'] > 0:
            print(
                f"    - Curtailed (POD):       {pv_stats['curtailed_energy_mwh']:.2f} MWh ({pv_stats['curtailment_percent']:.1f}%) ⚠️")

        # Valore economico PV stimato
        pv_value = (pv_stats['energy_to_grid_mwh'] * results_df['€/MWh'].mean() +
                    pv_stats['energy_to_battery_mwh'] * results_df['€/MWh'].mean() * 0.5 +
                    pv_stats['energy_to_load_mwh'] * results_df['Prezzo_Acquisto_€/MWh'].mean())
        print(f"  • Valore economico PV:     {pv_value:,.2f} €")

    # ========================================================================
    # 4. STATISTICHE CARICO (se abilitato)
    # ========================================================================
    if LOAD_ENABLED and load_profile:
        load_stats = load_profile.get_statistics()
        print("\n⚡ STATISTICHE CARICO:")
        print(f"  • Energia richiesta:       {load_stats['total_energy_required_mwh']:.2f} MWh")
        print(f"  • Fonti di fornitura:")
        print(
            f"    - Da PV:                 {load_stats['energy_from_pv_mwh']:.2f} MWh ({load_stats['pv_coverage_percent']:.1f}%)")
        print(
            f"    - Da batteria:           {load_stats['energy_from_battery_mwh']:.2f} MWh ({load_stats['battery_coverage_percent']:.1f}%)")
        print(
            f"    - Da rete:               {load_stats['energy_from_grid_mwh']:.2f} MWh ({load_stats['grid_dependency_percent']:.1f}%)")

        autosufficienza = 100 - load_stats['grid_dependency_percent']
        print(f"\n  • Autosufficienza:         {autosufficienza:.1f}%")
        print(f"  • Dipendenza rete:         {load_stats['grid_dependency_percent']:.1f}%")

        # Decisioni autonome batteria
        if load_stats.get('total_decisions', 0) > 0:
            print(f"\n  • Decisioni batteria:")
            print(
                f"    - Servito carico:        {load_stats['battery_served_count']} ore ({load_stats['battery_served_decisions_percent']:.1f}%)")
            print(
                f"    - Lasciato a rete:       {load_stats['grid_served_count']} ore ({load_stats['grid_served_decisions_percent']:.1f}%)")

    # ========================================================================
    # 5. STATISTICHE POD
    # ========================================================================
    if LOAD_ENABLED and 'POD_Violation' in results_df.columns:
        total_pod_violations = int(results_df['POD_Violation'].sum())
        total_pv_curtailed = results_df['PV_Curtailed_MWh'].sum() if 'PV_Curtailed_MWh' in results_df.columns else 0
        total_load_unserved = results_df['Load_Unserved_MWh'].sum() if 'Load_Unserved_MWh' in results_df.columns else 0
        max_withdrawal = results_df['Grid_Withdrawal_MW'].max() if 'Grid_Withdrawal_MW' in results_df.columns else 0
        max_injection = results_df['Grid_Injection_MW'].max() if 'Grid_Injection_MW' in results_df.columns else 0

        print(f"\n🔌 STATISTICHE POD (Limite: {POD_POWER_MW} MW):")
        print(
            f"  • Violazioni totali:       {total_pod_violations} ore ({total_pod_violations / len(results_df) * 100:.1f}%)")
        print(f"  • Max prelievo:            {max_withdrawal:.3f} MW")
        print(f"  • Max immissione:          {max_injection:.3f} MW")
        if total_pv_curtailed > 0:
            print(f"  • PV curtailed:            {total_pv_curtailed:.2f} MWh ⚠️")
        if total_load_unserved > 0:
            print(f"  • Carico non servito:      {total_load_unserved:.2f} MWh ⚠️")

        if total_pod_violations > 0 or total_load_unserved > 0.1:
            print(f"\n  ATTENZIONE: Considera aumentare POD limit o capacità batteria!")

    # ========================================================================
    # 6. CONFRONTO CON BASELINE (scenario senza batteria)
    # ========================================================================
    if baseline_scenario:
        print("\nCONFRONTO CON SCENARIO BASE (senza batteria):")
        print(f"  • Bilancio senza BESS:     {baseline_scenario['net_balance']:,.2f} €")
        print(f"  • Bilancio con BESS:       {total_system_profit:,.2f} €")

        battery_benefit = total_system_profit - baseline_scenario['net_balance']
        if baseline_scenario['net_balance'] != 0:
            benefit_percent = (battery_benefit / abs(baseline_scenario['net_balance']) * 100)
        else:
            benefit_percent = 0

        print(f"  • Beneficio batteria:      {battery_benefit:,.2f} € ({benefit_percent:+.1f}%)")

        if battery_benefit > 0:
            print(f"  SISTEMA PROFITTEVOLE")
        else:
            print(f"  SISTEMA NON PROFITTEVOLE (perdita: {abs(battery_benefit):,.2f} €)")

        # Autosufficienza
        baseline_autosufficienza = baseline_scenario['autosufficienza_percent']
        if LOAD_ENABLED and load_profile:
            current_autosufficienza = 100 - load_stats['grid_dependency_percent']
            delta_autosufficienza = current_autosufficienza - baseline_autosufficienza

            print(f"\n  • Autosufficienza base:    {baseline_autosufficienza:.1f}%")
            print(f"  • Autosufficienza BESS:    {current_autosufficienza:.1f}%")
            print(f"  • Miglioramento:           {delta_autosufficienza:+.1f} punti percentuali")

    # ========================================================================
    # 7. OPERAZIONI BATTERIA
    # ========================================================================
    charge_hours = np.sum(results_df['Azione_Trading_MW'] > 0.01)
    discharge_hours = np.sum(results_df['Azione_Trading_MW'] < -0.01)
    idle_hours = len(results_df) - charge_hours - discharge_hours
    utilization = (charge_hours + discharge_hours) / len(results_df) * 100

    print(f"\n🔄 OPERAZIONI BATTERIA:")
    print(f"  • Ore carica:              {charge_hours} ({charge_hours / len(results_df) * 100:.1f}%)")
    print(f"  • Ore scarica:             {discharge_hours} ({discharge_hours / len(results_df) * 100:.1f}%)")
    print(f"  • Ore idle:                {idle_hours} ({idle_hours / len(results_df) * 100:.1f}%)")
    print(f"  • Utilizzo totale:         {utilization:.1f}%")

    # ========================================================================
    # 8. TEMPO SIMULAZIONE
    # ========================================================================
    simulation_time = (end_time - start_time).total_seconds()
    print(f"\n  TEMPO SIMULAZIONE:")
    print(f"  • Durata:                  {simulation_time:.1f} secondi ({simulation_time / 60:.1f} minuti)")
    print(f"  • Ore simulate:            {len(results_df)}")
    print(f"  • Velocità:                {len(results_df) / simulation_time:.1f} ore/secondo")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()