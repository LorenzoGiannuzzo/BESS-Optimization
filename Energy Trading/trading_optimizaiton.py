import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==================== PARAMETRI CONFIGURABILI ====================
SAVE_PLOTS = True  # Flag per abilitare/disabilitare salvataggio grafici

# Costo degrado batteria (€/MWh throughput)
# Basato su: costo batteria / cicli vita utile / capacità
# Es: 600.000€ / 6.000 cicli / 4 MWh = 25 €/MWh
DEGRADATION_COST_PER_MWH = 25.0  # €/MWh di throughput
# ==================================================================

np.random.seed(42)


def degradation(cycle_num):
    """
    Funzione di degrado della batteria.
    Restituisce la capacità rimanente (%) in funzione del numero di cicli equivalenti.
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


class Battery:
    """Modello di batteria Litio-Ione con Coulomb Counting e degrado"""

    def __init__(self, capacity_mwh=4.0, efficiency=0.95, max_power_mw=2.0, max_c_rate=0.5):
        self.nominal_capacity = capacity_mwh
        self.capacity = capacity_mwh
        self.efficiency = efficiency
        self.max_power = max_power_mw
        self.max_c_rate = max_c_rate
        self.soc = 0.5
        self.soc_min = 0.1
        self.soc_max = 0.9

        # Parametri degrado
        self.equivalent_cycles = 0.0
        self.throughput_kwh = 0.0

    def charge(self, power_mw, dt=1.0):
        """Carica la batteria. power_mw positivo, dt in ore"""
        max_power_c_rate = self.capacity * self.max_c_rate
        power_mw = min(power_mw, max_power_c_rate)

        energy = power_mw * dt * self.efficiency
        new_soc = self.soc + (energy / self.capacity)
        new_soc = min(new_soc, self.soc_max)
        actual_energy = (new_soc - self.soc) * self.capacity
        self.soc = new_soc

        # Aggiorna throughput per degrado
        self.throughput_kwh += actual_energy * 1000

        return actual_energy / self.efficiency

    def discharge(self, power_mw, dt=1.0):
        """Scarica la batteria. power_mw positivo, dt in ore"""
        max_power_c_rate = self.capacity * self.max_c_rate
        power_mw = min(power_mw, max_power_c_rate)

        energy = power_mw * dt / self.efficiency
        new_soc = self.soc - (energy / self.capacity)
        new_soc = max(new_soc, self.soc_min)
        actual_energy = (self.soc - new_soc) * self.capacity
        self.soc = new_soc

        # Aggiorna throughput per degrado
        self.throughput_kwh += actual_energy * 1000

        return actual_energy * self.efficiency

    def update_degradation(self):
        """Aggiorna la capacità in base al degrado"""
        # CORREZIONE: 1 ciclo = throughput pari alla capacità nominale (non 2x!)
        self.equivalent_cycles = self.throughput_kwh / ( 2 * self.nominal_capacity * 1000)

        capacity_percentage = degradation(self.equivalent_cycles)
        self.capacity = self.nominal_capacity * (capacity_percentage / 100.0)

    def get_soh(self):
        """Ritorna lo State of Health (%)"""
        return (self.capacity / self.nominal_capacity) * 100.0

    def get_soc(self):
        return self.soc

    def copy(self):
        """Crea una copia della batteria per simulazioni"""
        b = Battery(self.nominal_capacity, self.efficiency, self.max_power, self.max_c_rate)
        b.capacity = self.capacity
        b.soc = self.soc
        b.equivalent_cycles = self.equivalent_cycles
        b.throughput_kwh = self.throughput_kwh
        return b

    def get_max_power_by_crate(self):
        """Ritorna la potenza massima consentita dal C-rate"""
        return self.capacity * self.max_c_rate


class PSOOptimizer:
    """Particle Swarm Optimization per batteria con convergenza migliorata"""

    def __init__(self, n_particles=50, n_iterations=150, w_start=0.95, w_end=0.2, c1=3.0, c2=3.0):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_start = w_start  # Inerzia iniziale (alta per esplorazione)
        self.w_end = w_end  # Inerzia finale (bassa per sfruttamento)
        self.c1 = c1  # Coefficiente cognitivo (aumentato)
        self.c2 = c2  # Coefficiente sociale (aumentato)
        self.stagnation_limit = 15  # Iterazioni senza miglioramento prima di reinizializzare

    def optimize(self, battery, prices, horizon_hours=18):
        """Ottimizza le azioni della batteria con PSO migliorato"""
        n_hours = min(horizon_hours, len(prices))
        max_power_limit = min(battery.max_power, battery.get_max_power_by_crate())

        # Inizializzazione SMART delle particelle
        # Strategia: carica quando prezzo basso, scarica quando alto
        positions = self._smart_initialization(battery, prices[:n_hours], max_power_limit)

        # Velocità iniziali più aggressive
        velocities = np.random.uniform(-1.0, 1.0, (self.n_particles, n_hours))

        # Best personali e globale
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self._evaluate(battery, p, prices[:n_hours])
                                         for p in positions])

        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        stagnation_counter = 0

        # Iterazioni PSO con inerzia decrescente
        for iteration in range(self.n_iterations):
            # Inerzia decrescente linearmente
            w = self.w_start - (self.w_start - self.w_end) * (iteration / self.n_iterations)

            for i in range(self.n_particles):
                # Update velocità con parametri più aggressivi
                r1, r2 = np.random.random(n_hours), np.random.random(n_hours)

                # Componente cognitiva e sociale
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (global_best_position - positions[i])

                velocities[i] = w * velocities[i] + cognitive + social

                # Limita velocità per evitare esplosioni
                max_velocity = max_power_limit * 0.5
                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

                # Update posizione
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], -max_power_limit, max_power_limit)

                # Valutazione
                score = self._evaluate(battery, positions[i], prices[:n_hours])

                # Update best personale
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                    # Update best globale
                    if score > global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()
                        stagnation_counter = 0  # Reset stagnazione

            stagnation_counter += 1

            # Reinizializza alcune particelle se stagnazione
            if stagnation_counter > self.stagnation_limit:
                n_reinit = self.n_particles // 4  # Reinizializza 25% delle particelle
                worst_indices = np.argsort(personal_best_scores)[:n_reinit]

                for idx in worst_indices:
                    # Reinizializza attorno al best globale con perturbazione
                    noise = np.random.uniform(-max_power_limit * 0.3, max_power_limit * 0.3, n_hours)
                    positions[idx] = np.clip(global_best_position + noise, -max_power_limit, max_power_limit)
                    velocities[idx] = np.random.uniform(-0.5, 0.5, n_hours)

                stagnation_counter = 0

        return global_best_position

    def _smart_initialization(self, battery, prices, max_power):
        """Inizializzazione intelligente basata sui prezzi"""
        n_hours = len(prices)
        positions = np.zeros((self.n_particles, n_hours))

        # Calcola percentili dei prezzi
        price_low = np.percentile(prices, 25)
        price_high = np.percentile(prices, 75)

        for i in range(self.n_particles):
            if i < self.n_particles // 3:
                # Strategia 1: Carica quando basso, scarica quando alto (aggressiva)
                for h in range(n_hours):
                    if prices[h] < price_low:
                        positions[i, h] = np.random.uniform(0.5 * max_power, max_power)
                    elif prices[h] > price_high:
                        positions[i, h] = np.random.uniform(-max_power, -0.5 * max_power)
                    else:
                        positions[i, h] = np.random.uniform(-0.3 * max_power, 0.3 * max_power)

            elif i < 2 * self.n_particles // 3:
                # Strategia 2: Più conservativa
                for h in range(n_hours):
                    if prices[h] < price_low:
                        positions[i, h] = np.random.uniform(0, 0.7 * max_power)
                    elif prices[h] > price_high:
                        positions[i, h] = np.random.uniform(-0.7 * max_power, 0)
                    else:
                        positions[i, h] = np.random.uniform(-0.2 * max_power, 0.2 * max_power)

            else:
                # Strategia 3: Completamente random per diversità
                positions[i] = np.random.uniform(-max_power, max_power, n_hours)

        return positions

    def _evaluate(self, battery, actions, prices):
        """Valuta una sequenza di azioni includendo il costo del degrado"""
        bat_sim = battery.copy()
        profit = 0.0

        for power, price in zip(actions, prices):
            if power > 0:  # Carica
                if bat_sim.soc >= bat_sim.soc_max:
                    continue

                max_energy_storable = (bat_sim.soc_max - bat_sim.soc) * bat_sim.capacity
                max_power_available = max_energy_storable / (1.0 * bat_sim.efficiency)
                actual_power = min(power, max_power_available)

                if actual_power > 0.01:
                    energy_from_grid = bat_sim.charge(actual_power, dt=1.0)
                    # Costo acquisto energia + costo degrado
                    profit -= energy_from_grid * price
                    profit -= energy_from_grid * DEGRADATION_COST_PER_MWH

            elif power < 0:  # Scarica
                if bat_sim.soc <= bat_sim.soc_min:
                    continue

                max_energy_available = (bat_sim.soc - bat_sim.soc_min) * bat_sim.capacity
                max_power_available = max_energy_available * bat_sim.efficiency / 1.0
                actual_power = min(-power, max_power_available)

                if actual_power > 0.01:
                    energy_to_grid = bat_sim.discharge(actual_power, dt=1.0)
                    # Ricavo vendita energia - costo degrado
                    profit += energy_to_grid * price
                    profit -= energy_to_grid * DEGRADATION_COST_PER_MWH

        return profit


class RollingHorizonSimulator:
    """Simulatore con rolling horizon"""

    def __init__(self, battery, optimizer, horizon_hours=18, step_hours=1):
        self.battery = battery
        self.optimizer = optimizer
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours

    def simulate(self, prices_df):
        """Esegue la simulazione"""
        prices = prices_df['€/MWh'].values
        n_hours = len(prices)

        # Storage risultati
        actions_taken = []
        soc_history = []
        capacity_history = []
        soh_history = []
        profits_history = []
        degradation_costs_history = []

        cumulative_profit = 0.0
        cumulative_degradation_cost = 0.0

        print("=" * 60)
        print("INIZIO SIMULAZIONE BATTERIA CON PSO E ROLLING HORIZON")
        print("=" * 60)
        print(f"Capacità batteria nominale: {self.battery.nominal_capacity} MWh")
        print(f"Potenza massima: {self.battery.max_power} MW")
        print(f"C-rate massimo: {self.battery.max_c_rate}C")
        print(f"Costo degrado: {DEGRADATION_COST_PER_MWH} €/MWh")
        print(f"Orizzonte temporale: {self.horizon_hours} ore")
        print(f"Ore totali da simulare: {n_hours}")
        print("=" * 60)
        print()

        print(f"⏳ Progresso: 0% - Inizio ottimizzazione...")

        current_hour = 0
        last_progress = 0
        degradation_update_interval = 24

        while current_hour < n_hours:
            # Progress tracking
            progress = int((current_hour / n_hours) * 100)
            if progress >= last_progress + 20:
                print(f"⏳ Progresso: {progress}% ({current_hour}/{n_hours} h) - SOH: {self.battery.get_soh():.2f}%")
                last_progress = progress

            # Aggiorna degrado periodicamente
            if current_hour % degradation_update_interval == 0 and current_hour > 0:
                self.battery.update_degradation()

            # Estrai finestra di prezzi
            end_hour = min(current_hour + self.horizon_hours, n_hours)
            window_prices = prices[current_hour:end_hour]

            # Ottimizza con PSO
            optimal_actions = self.optimizer.optimize(
                self.battery, window_prices, self.horizon_hours
            )

            # Esegui prima azione
            action = optimal_actions[0]
            price = prices[current_hour]

            energy_flow = 0.0  # MWh scambiate (per calcolo degrado)

            if action > 0.01:  # Carica
                if self.battery.soc < self.battery.soc_max:
                    max_energy_storable = (self.battery.soc_max - self.battery.soc) * self.battery.capacity
                    max_power_available = max_energy_storable / (1.0 * self.battery.efficiency)
                    actual_power = min(action, max_power_available)
                    action = actual_power

                    if actual_power > 0.01:
                        energy_from_grid = self.battery.charge(actual_power, dt=1.0)
                        energy_flow = energy_from_grid
                        profit = -energy_from_grid * price  # Costo acquisto
                    else:
                        profit = 0.0
                else:
                    profit = 0.0
                    action = 0.0

            elif action < -0.01:  # Scarica
                if self.battery.soc > self.battery.soc_min:
                    max_energy_available = (self.battery.soc - self.battery.soc_min) * self.battery.capacity
                    max_power_available = max_energy_available * self.battery.efficiency / 1.0
                    actual_power = min(-action, max_power_available)
                    action = -actual_power

                    if actual_power > 0.01:
                        energy_to_grid = self.battery.discharge(actual_power, dt=1.0)
                        energy_flow = energy_to_grid
                        profit = energy_to_grid * price  # Ricavo vendita
                    else:
                        profit = 0.0
                else:
                    profit = 0.0
                    action = 0.0
            else:
                profit = 0.0

            # Calcola costo degrado per questa azione
            degradation_cost = energy_flow * DEGRADATION_COST_PER_MWH

            cumulative_profit += profit
            cumulative_degradation_cost += degradation_cost

            # Profitto netto = profitto operativo - costo degrado
            net_profit = cumulative_profit - cumulative_degradation_cost

            # Salva risultati
            actions_taken.append(action)
            soc_history.append(self.battery.get_soc())
            capacity_history.append(self.battery.capacity)
            soh_history.append(self.battery.get_soh())
            profits_history.append(net_profit)
            degradation_costs_history.append(cumulative_degradation_cost)

            current_hour += self.step_hours

        # Aggiornamento finale degrado
        self.battery.update_degradation()

        print(f"⏳ Progresso: 100% completato ({n_hours}/{n_hours} ore)")
        print()

        # Crea DataFrame risultati
        results_df = prices_df.copy()
        pad_length = len(results_df) - len(actions_taken)

        results_df['Azione_MW'] = actions_taken + [0] * pad_length
        results_df['SOC'] = soc_history + [soc_history[-1]] * pad_length
        results_df['Capacità_MWh'] = capacity_history + [capacity_history[-1]] * pad_length
        results_df['SOH_%'] = soh_history + [soh_history[-1]] * pad_length
        results_df['Profitto_Netto_€'] = profits_history + [profits_history[-1]] * pad_length
        results_df['Costo_Degrado_€'] = degradation_costs_history + [degradation_costs_history[-1]] * pad_length

        return results_df, cumulative_profit, cumulative_degradation_cost


def create_visualizations(results_df):
    """Crea visualizzazioni se SAVE_PLOTS=True"""
    if not SAVE_PLOTS:
        print("\n⏭️  Salvataggio grafici disabilitato (SAVE_PLOTS=False)")
        return

    viz_folder = 'visualization'
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
        print(f"\n✓ Cartella '{viz_folder}' creata")

    print("\n" + "=" * 60)
    print("GENERAZIONE VISUALIZZAZIONI")
    print("=" * 60)

    # Converti Data in datetime
    if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
        results_df['Data'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')

    results_df['Mese'] = results_df['Data'].dt.month
    results_df['Anno'] = results_df['Data'].dt.year

    # Grafici mensili
    mesi_nomi = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno',
                 'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']

    for mese in range(1, 13):
        df_mese = results_df[results_df['Mese'] == mese].copy()
        if len(df_mese) == 0:
            continue

        giorni_disponibili = df_mese['Data'].dt.day.unique()
        giorno_target = 15 if 15 in giorni_disponibili else giorni_disponibili[len(giorni_disponibili) // 2]
        df_giorno = df_mese[df_mese['Data'].dt.day == giorno_target].copy()

        if len(df_giorno) == 0:
            continue

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f'Operazione Batteria - {mesi_nomi[mese - 1]} 2024 (Giorno {giorno_target})',
                     fontsize=16, fontweight='bold')

        ore = df_giorno['Ora'].values

        # Subplot 1: PUN
        ax1.plot(ore, df_giorno['€/MWh'].values, color='#2E86AB', linewidth=2, marker='o', markersize=4)
        ax1.set_ylabel('Prezzo (€/MWh)', fontsize=11, fontweight='bold')
        ax1.set_title('Prezzo Unico Nazionale (PUN)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0.5, 24.5)
        ax1.set_xticks(range(1, 25))

        prezzo_medio = df_giorno['€/MWh'].mean()
        ax1.axhline(y=prezzo_medio, color='red', linestyle='--', alpha=0.5,
                    label=f'Media: {prezzo_medio:.2f} €/MWh')
        ax1.legend(loc='upper right')

        # Subplot 2: Azioni Batteria
        azioni = df_giorno['Azione_MW'].values
        colori = ['green' if a > 0 else 'red' if a < 0 else 'gray' for a in azioni]

        ax2.bar(ore, azioni, color=colori, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Potenza (MW)', fontsize=11, fontweight='bold')
        ax2.set_title('Azioni Batteria', fontsize=12, fontweight='bold')
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_xlim(0.5, 24.5)
        ax2.set_xticks(range(1, 25))

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Carica'),
                           Patch(facecolor='red', alpha=0.7, label='Scarica'),
                           Patch(facecolor='gray', alpha=0.7, label='Idle')]
        ax2.legend(handles=legend_elements, loc='upper right')

        # Subplot 3: SOC
        ax3.plot(ore, df_giorno['SOC'].values * 100, color='#F77F00', linewidth=2.5, marker='s', markersize=5)
        ax3.fill_between(ore, 0, df_giorno['SOC'].values * 100, alpha=0.3, color='#F77F00')
        ax3.set_xlabel('Ora', fontsize=11, fontweight='bold')
        ax3.set_ylabel('SOC (%)', fontsize=11, fontweight='bold')
        ax3.set_title('State of Charge', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax3.axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(0.5, 24.5)
        ax3.set_xticks(range(1, 25))

        plt.tight_layout()

        filename = f'batteria_{mese:02d}_{mesi_nomi[mese - 1]}.png'
        filepath = os.path.join(viz_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Grafico salvato: {filename}")

    create_annual_summary(results_df, viz_folder)

    print("=" * 60)
    print(f"✓ Visualizzazioni salvate in '{viz_folder}/'")
    print("=" * 60)


def create_annual_summary(results_df, viz_folder):
    """Crea grafici riassuntivi annuali"""

    if not pd.api.types.is_datetime64_any_dtype(results_df['Data']):
        results_df['Data_dt'] = pd.to_datetime(results_df['Data'], format='%d/%m/%Y')
    else:
        results_df['Data_dt'] = results_df['Data']

    results_df['Mese'] = results_df['Data_dt'].dt.month
    df_daily = results_df.groupby(results_df['Data_dt'].dt.date, as_index=False).last()
    df_daily['Data_dt'] = pd.to_datetime(df_daily['Data_dt'])

    # 1. Profitto Netto
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_daily['Data_dt'], df_daily['Profitto_Netto_€'],
            color='#06A77D', linewidth=2, label='Profitto Netto')
    ax.fill_between(df_daily['Data_dt'], 0, df_daily['Profitto_Netto_€'],
                    alpha=0.3, color='#06A77D')
    ax.set_ylabel('Profitto (€)', fontsize=11, fontweight='bold')
    ax.set_title('Profitto Netto nel Tempo', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, '01_profitto_netto.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Salvato: 01_profitto_netto.png")

    # 2. Costo Degrado
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_daily['Data_dt'], df_daily['Costo_Degrado_€'],
            color='#E63946', linewidth=2)
    ax.fill_between(df_daily['Data_dt'], 0, df_daily['Costo_Degrado_€'],
                    alpha=0.3, color='#E63946')
    ax.set_ylabel('Costo Cumulativo (€)', fontsize=11, fontweight='bold')
    ax.set_title('Costo Degrado Batteria', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, '02_costo_degrado.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Salvato: 02_costo_degrado.png")

    # 3. SOH
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_daily['Data_dt'], df_daily['SOH_%'],
            color='#457B9D', linewidth=2)
    ax.set_ylabel('SOH (%)', fontsize=11, fontweight='bold')
    ax.set_title('State of Health nel Tempo', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Soglia 80%')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, '03_soh.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Salvato: 03_soh.png")

    print("\n✅ Tutti i grafici salvati!")


def main():
    """Funzione principale"""
    file_path = os.path.join('data', '20240101_20241231_PUN.xlsx')

    print("=" * 60)
    print("BATTERY ENERGY STORAGE SYSTEM (BESS) OPTIMIZATION")
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
        print(f"✗ Errore caricamento: {e}")
        return

    # Inizializza componenti con parametri migliorati
    battery = Battery(capacity_mwh=4.0, efficiency=0.95, max_power_mw=2.0, max_c_rate=0.5)

    # PSO con più particelle e parametri ottimizzati
    optimizer = PSOOptimizer(
        n_particles=50,  # Aumentato da 30 a 50
        n_iterations=100,  # Ridotto da 170 a 100 (più efficiente)
        w_start=0.9,  # Inerzia alta iniziale (esplorazione)
        w_end=0.4,  # Inerzia bassa finale (sfruttamento)
        c1=2.0,  # Aumentato da 1.5 a 2.0
        c2=2.0  # Aumentato da 1.5 a 2.0
    )

    simulator = RollingHorizonSimulator(battery, optimizer, horizon_hours=24, step_hours=1)

    # Esegui simulazione
    start_time = datetime.now()
    results_df, gross_profit, degradation_cost = simulator.simulate(df)
    end_time = datetime.now()

    net_profit = gross_profit - degradation_cost

    # Report finale
    print("=" * 60)
    print("RISULTATI SIMULAZIONE")
    print("=" * 60)
    print(f"💰 Profitto operativo (arbitraggio): €{gross_profit:,.2f}")
    print(f"⚙️  Costo degrado (solo tracking): €{degradation_cost:,.2f}")
    print(f"💵 Profitto netto (info): €{net_profit:,.2f}")
    print(f"📊 SOC finale: {battery.get_soc() * 100:.1f}%")
    print(f"🔋 Capacità finale: {battery.capacity:.3f} MWh (nominale: {battery.nominal_capacity:.3f} MWh)")
    print(f"📉 SOH finale: {battery.get_soh():.2f}%")
    print(f"♻️  Cicli equivalenti: {battery.equivalent_cycles:.1f}")
    print(f"⚡ Energia scaricata: {np.sum(results_df['Azione_MW'][results_df['Azione_MW'] < 0].abs()):.2f} MWh")
    print(f"🔌 Energia caricata: {np.sum(results_df['Azione_MW'][results_df['Azione_MW'] > 0]):.2f} MWh")
    print(f"⏱️  Tempo esecuzione: {(end_time - start_time).total_seconds():.1f} secondi")
    print("=" * 60)

    # Analisi economica (basata su profitto OPERATIVO, non netto)
    print("\n" + "=" * 60)
    print("ANALISI ECONOMICA")
    print("=" * 60)

    # Calcola throughput totale
    total_throughput_mwh = battery.throughput_kwh / 1000

    print(f"Throughput totale: {total_throughput_mwh:.2f} MWh")
    print(
        f"Margine medio arbitraggio: {(gross_profit / total_throughput_mwh) if total_throughput_mwh > 0 else 0:.2f} €/MWh")

    # Valore residuo batteria basato su SOH
    battery_investment = 600000
    residual_value = battery_investment * (battery.get_soh() / 100.0)

    print(f"\n💼 VALUTAZIONE INVESTIMENTO:")
    print(f"Investimento iniziale: €{battery_investment:,.0f}")
    print(f"Ricavi da arbitraggio: €{gross_profit:,.2f}")
    print(f"Valore residuo batteria ({battery.get_soh():.1f}% SOH): €{residual_value:,.2f}")

    # Profitto totale = ricavi + valore residuo - investimento iniziale
    total_return = gross_profit + residual_value - battery_investment
    roi_percentage = (total_return / battery_investment) * 100

    print(f"Ritorno totale (ricavi + residuo - investimento): €{total_return:,.2f}")
    print(f"ROI: {roi_percentage:.2f}%")

    if gross_profit > 0:
        # Payback considerando il degrado della batteria
        # Anni necessari per recuperare l'investimento
        years_to_payback = battery_investment / gross_profit
        print(f"Payback semplice: {years_to_payback:.1f} anni")

        # SOH dopo payback
        cycles_per_year = battery.equivalent_cycles
        total_cycles_at_payback = cycles_per_year * years_to_payback
        soh_at_payback = degradation(total_cycles_at_payback)
        print(f"SOH stimato a fine payback: {soh_at_payback:.1f}%")

    print(f"\n📝 NOTA: Il costo del degrado ({degradation_cost:,.2f}€) è usato dall'ottimizzatore")
    print(f"         per decidere se conviene operare, ma NON è una spesa reale.")
    print(f"         Il vero 'costo' è la perdita di valore della batteria,")
    print(f"         già contabilizzata nel valore residuo sopra.")
    print("=" * 60)

    # Salva risultati
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"\n✓ Cartella '{results_folder}' creata")

    output_file = os.path.join(results_folder, 'risultati_simulazione_batteria.xlsx')
    results_df.to_excel(output_file, index=False)
    print(f"✓ Risultati salvati in: {output_file}")

    # Genera visualizzazioni
    create_visualizations(results_df)

    print("\n" + "=" * 60)
    print("✅ SIMULAZIONE COMPLETATA CON SUCCESSO")
    print("=" * 60)


if __name__ == "__main__":
    main()


