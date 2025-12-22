# Battery Energy Storage System (BESS) Optimization
## Advanced PSO-based Energy Trading and Load Management System with Intelligent Bias Control

---

## 📋 Project Information

**Version:** 3.8.1  
**Release Date:** December 16, 2025  
**Author:** Lorenzo Giannuzzo  
**Affiliation:** Politecnico di Torino  
**Department:** DENERG (Dipartimento Energia)  
**Laboratory:** Energy Center Lab  
**License:** MIT License  
**Python Compatibility:** 3.8+  

---

## 🚀 Executive Summary

This software implements a state-of-the-art Battery Energy Storage System (BESS) optimization framework using advanced Particle Swarm Optimization (PSO) with Numba JIT acceleration. The system is designed for comprehensive energy management including arbitrage trading, photovoltaic integration, autonomous load management, and intelligent optimization bias control.

### 🎯 Key Capabilities

- **🔄 Energy Arbitrage:** Optimized battery charging/discharging based on market price signals
- **☀️ PV Integration:** Autonomous allocation of photovoltaic production to battery, grid, and load
- **🏠 Load Management:** Intelligent decision-making for serving electrical loads from battery vs. grid
- **🎛️ Bias Control:** CLI flags to guide optimizer toward arbitrage or autoconsumo strategies
- **🇮🇹 MACSE Support:** Integration with Italian ancillary services market
- **⚡ POD Enforcement:** Point of Delivery power limit compliance with violation tracking
- **🔋 Multi-Technology:** Support for Lithium-Ion and Graphene battery technologies
- **🚄 High Performance:** Numba JIT parallelization achieving 50-200x speedup over pure Python

### 🆕 Latest Features (v3.8.1)

- **Bias Optimization Flags:** `--bias-arbitrage` and `--bias-autoconsumo` for strategy guidance
- **Enhanced Economic Model:** Explicit accounting resolves "invisible reward problem"
- **Improved PSO Engine:** Intelligent initialization, constraint repair, adaptive parameters
- **Advanced Visualizations:** Comprehensive plotting system with monthly PV analysis
- **Custom Efficiency Support:** User-defined battery efficiency parameters
- **Robust CLI Interface:** 40+ command-line parameters for complete system control

---

## 🏗️ Technical Architecture

### Core Components

**1. 🧠 Advanced PSO Optimization Engine**
- Numba JIT-compiled evaluation functions for maximum performance
- Parallel particle evaluation across multiple CPU cores  
- Intelligent multi-strategy initialization (5 different approaches)
- Adaptive parameter adjustment based on convergence
- Dynamic diversification to escape local minima
- Constraint repair system for feasible solutions

**2. 🔄 Rolling Horizon Simulator**
- Hour-by-hour execution of optimal strategies
- Real-time SOC and SOH tracking with degradation modeling
- Comprehensive energy flow accounting
- POD compliance monitoring and violation tracking

**3. 💰 Explicit Economic Model**
- Complete cost/revenue accounting for all energy flows
- Trading revenues: battery discharge to grid
- Autoconsumo savings: battery discharge to load (avoiding grid purchase)
- Grid costs: purchases for battery charging and load service
- Degradation costs integrated into optimization objective
- Bias multipliers for strategy guidance

**4. 🛡️ Advanced Constraint Management**
- POD power limits enforced on injection and withdrawal
- Battery SOC limits (technology-dependent)
- C-rate limitations and power constraints
- XOR operation logic (battery cannot charge and discharge simultaneously)
- Intelligent constraint repair instead of penalty functions

---
## 🛠️ Installation and Setup

### Prerequisites

```bash
pip install pandas numpy numba matplotlib openpyxl joblib
```

### Required Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20.0 | Numerical computations and array operations |
| pandas | ≥1.3.0 | Data manipulation and Excel I/O |
| numba | ≥0.54.0 | JIT compilation for performance |
| matplotlib | ≥3.4.0 | Visualization and plotting |
| openpyxl | ≥3.0.0 | Excel file handling |
| joblib | ≥1.0.0 | Parallel processing utilities |

### Project Structure

```
BESS-Optimization/
├── system_optimization/
│   ├── system_optimizer.py          # Main executable script
│   ├── data/                        # Input data directory (required)
│   │   ├── Prezzo_Vendita.xlsx      # Electricity selling prices (required)
│   │   ├── Prezzo_Acquisto.xlsx     # Electricity purchase prices (required)
│   │   ├── PV_formattato.csv        # PV production data (optional)
│   │   └── Consumo.xlsx             # Load consumption data (optional)
│   ├── results/                     # Output directory (auto-created)
│   │   ├── risultati_*.xlsx         # Simulation results
│   │   └── risultati_*.json         # Complete data export
│   └── visualization/               # Plots directory (if enabled)
│       ├── overview_*.png
│       ├── pod_tracking_*.png
│       ├── economic_analysis_*.png
│       └── dettaglio_mensile_pv/
├── old_versions/                    # Previous versions for reference
├── resources/                       # Additional resources
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 💻 Command Line Interface

### Basic Syntax

```bash
python system_optimizer.py --price-sell <file> --price-buy <file> [OPTIONS]
```

### 🚀 Quick Start Examples

#### Minimal Working Example
```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx
```

#### Complete System with Bias Control
```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv --pv-enabled \
  --load-file data/Consumo.xlsx --load-enabled \
  --battery-tech GRAFENE \
  --battery-capacity 2.0 \
  --battery-power 1.5 \
  --bias-arbitrage 1.5 \
  --bias-autoconsumo 1.2 \
  --n-particles 120 \
  --n-iterations 200 \
  --save-plots \
  --output-dir results_production
```

#### Fast Testing Configuration
```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --n-particles 30 \
  --n-iterations 50 \
  --output-dir test_run
```

---
## 📊 Parameter Reference

### 🔴 Required Parameters

| Parameter | Type | Description                                                           |
|-----------|------|-----------------------------------------------------------------------|
| `--price-sell` | string | Path to electricity selling prices file (.xlsx). Prices are in €/MWh. |
| `--price-buy` | string | Path to electricity purchase prices file (.xlsx) Prices are in €/MWh. |

### 🔋 Battery Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--battery-tech` | choice | LITIO-IONE | LITIO-IONE, GRAFENE | Battery technology |
| `--battery-capacity` | float | 1.0 | >0 | Battery capacity (MWh) |
| `--battery-power` | float | 1.0 | >0 | Maximum power (MW) |
| `--battery-c-rate` | float | 1.0 | >0 | Maximum C-rate |

### ⚙️ Custom Efficiency Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--custom-efficiency` | flag | False | - | Enable custom efficiency mode |
| `--charge-efficiency` | float | None | 0-1 | Charging efficiency |
| `--discharge-efficiency` | float | None | 0-1 | Discharging efficiency |
| `--roundtrip-efficiency` | float | None | 0-1 | Round-trip efficiency |

**Note:** With `--custom-efficiency`, provide at least `--roundtrip-efficiency` OR both `--charge-efficiency` and `--discharge-efficiency`.

### 🔋 SOC Limits

#### Lithium-Ion Technology
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--lithium-soc-min` | 0.1 | 0.0-1.0 | Minimum state of charge (10%) |
| `--lithium-soc-max` | 0.9 | 0.0-1.0 | Maximum state of charge (90%) |

#### Graphene Technology
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--graphene-soc-min` | 0.0 | 0.0-1.0 | Minimum state of charge (0%) |
| `--graphene-soc-max` | 1.0 | 0.0-1.0 | Maximum state of charge (100%) |

### ⚡ Point of Delivery (POD)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--pod-limit` | float | 1.5 | >0 | Maximum grid exchange power (MW) |

**Recommended Values:**
- **0.8 MW:** Residential connection
- **1.0 MW:** Match battery power rating  
- **1.5 MW:** Default (minimal constraints)
- **2.0+ MW:** Industrial installations

### ☀️ Photovoltaic System

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pv-enabled` | flag | False | Enable photovoltaic system |
| `--pv-file` | string | None | Path to PV production file (.csv) |

**⚠️ Warning:** Modifying `--pv-nominal-power` may cause data inconsistencies. Use default value unless you know what you're doing.

### 🏠 Load Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--load-enabled` | flag | False | Enable load management |
| `--load-file` | string | None | Path to load consumption file (.xlsx) |

---
### 🎯 Bias Optimization (NEW in v3.8.1)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--bias-arbitrage` | float | 1.0 | >0 | Multiplier for arbitrage profits (>1.0 favors trading) |
| `--bias-autoconsumo` | float | 1.0 | >0 | Multiplier for autoconsumo profits (>1.0 favors load service) |

#### 🎛️ Bias Control Examples

**Favor Arbitrage Trading:**
```bash
--bias-arbitrage 2.0 --bias-autoconsumo 1.0
```
*Effect: 2x arbitrage profits → optimizer prefers market trading*

**Favor Autoconsumo (Load Service):**
```bash
--bias-arbitrage 1.0 --bias-autoconsumo 2.5
```
*Effect: 2.5x autoconsumo profits → optimizer prefers serving local load*

**Balanced with Slight Trading Preference:**
```bash
--bias-arbitrage 1.3 --bias-autoconsumo 1.1
```
*Effect: Slight preference for trading while maintaining load service*

**Discourage Trading:**
```bash
--bias-arbitrage 0.7 --bias-autoconsumo 1.5
```
*Effect: Reduced trading attractiveness, increased load service preference*

#### 🎯 Recommended Bias Values

| Scenario | Arbitrage Bias | Autoconsumo Bias | Use Case |
|----------|----------------|------------------|----------|
| **Pure Trading** | 2.0-3.0 | 1.0 | Maximize market profits |
| **Self-Consumption** | 1.0 | 2.0-3.0 | Maximize energy independence |
| **Balanced** | 1.0 | 1.0 | Default optimization |
| **Conservative** | 0.8 | 1.2 | Prefer local consumption |
| **Aggressive Trading** | 3.0+ | 0.5 | High-risk market strategy |

### 🧮 PSO Optimization Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--n-particles` | int | 50 | >0 | Number of PSO particles |
| `--n-iterations` | int | 100 | >0 | Number of PSO iterations |
| `--disable-optimizations` | flag | False | - | Disable advanced PSO features for speed |

#### ⚖️ Performance vs. Quality Trade-offs

| Configuration | Particles | Iterations | Time      | Quality   | Use Case |
|---------------|-----------|------------|-----------|-----------|----------|
| **Fast** | 30 | 50 | Very Fast | Mediocre  | Testing, debugging |
| **Balanced** | 50 | 100 | Fast      | Decent    | Daily use |
| **High Quality** | 120 | 200 | Medium    | Good      | Production runs |
| **Research Grade** | 200 | 400 | Slow      | Excellent | Academic research |

### 🔧 Parallelization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--no-parallel` | flag | False | Disable parallelization (debug mode) |
| `--n-cores` | int | -2 | Number of CPU cores (-1: all-1, -2: all, N: specific) |

### 🇮🇹 MACSE Integration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--macse-enabled` | flag | False | - | Enable MACSE ancillary services |
| `--macse-capacity` | float | 1.0 | >0, ≤battery_capacity | Reserved capacity (MWh) |
| `--macse-contract-years` | int | 1 | >0 | Contract duration (years) |
| `--macse-price-per-mw-year` | float | 50000 | >0 | Annual price (EUR/MW/year) |

### 📁 Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save-plots` | flag | False | Generate visualization plots |
| `--output-dir` | string | results | Output directory path |
| `--output-filename` | string | None | Custom output filename (without extension) |

---
## 📄 Input Data Formats

### 💰 Electricity Prices (Excel)

**Files:** `Prezzo_Vendita.xlsx`, `Prezzo_Acquisto.xlsx`

**Required Format:**
| Data | Ora | €/MWh |
|------|-----|-------|
| 01/01/2024 | 00:00 | 85.34 |
| 01/01/2024 | 01:00 | 78.21 |
| 01/01/2024 | 02:00 | 72.15 |
| 01/01/2024 | 03:00 | 68.90 |
| ... | ... | ... |

**Requirements:**
- **Column headers:** `Data`, `Ora`, `€/MWh` (mandatory, case-sensitive)
- **8760 rows:** One year, hourly resolution
- **Date format:** DD/MM/YYYY in `Data` column
- **Time format:** HH:MM in `Ora` column (00:00 to 23:00)
- **Price values:** Numeric or text with comma decimal separator (auto-converted)
- **Sequential hours:** Must cover complete year without gaps

### ☀️ PV Production (CSV)

**File:** `PV_formattato.csv`

**Required Format:**
```csv
Data;P
01/01/2024 00:00;0.0
01/01/2024 01:00;0.0
01/01/2024 08:00;150.5
01/01/2024 12:00;890.2
01/01/2024 18:00;45.3
01/01/2024 23:00;0.0
```

**Requirements:**
- **Separator:** Semicolon (`;`)
- **Column header:** `P` (power in kW)
- **8760 rows:** Full year coverage
- **Date format:** DD/MM/YYYY HH:MM

### 🏠 Load Consumption (Excel)

**File:** `Consumo.xlsx`

**Required Format:**
| Data | value |
|------|-------|
| 01/01/2024 00:00 | 450.3 |
| 01/01/2024 01:00 | 380.1 |
| 01/01/2024 02:00 | 420.7 |
| ... | ... |

**Requirements:**
- **Column header:** `value` (or `load`, `Load`, `power`, `Power`)
- **Power units:** kW
- **8760 rows:** Full year coverage
- **Sheet:** First sheet of Excel file used by default

---

## 🎯 Usage Scenarios

### Scenario 1: Pure Arbitrage Analysis

**Use Case:** Evaluate battery profitability through energy arbitrage alone.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --battery-tech LITIO-IONE \
  --battery-capacity 1.0 \
  --battery-power 1.0 \
  --pod-limit 1.5 \
  --bias-arbitrage 1.0
```

**Required Files:** Price files only  
**Execution Time:** ~5 minutes  
**Output:** Excel + JSON results

---

### Scenario 2: Complete Energy System

**Use Case:** Full residential/commercial energy system optimization with bias control.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv --pv-enabled \
  --load-file data/Consumo.xlsx --load-enabled \
  --battery-tech GRAFENE \
  --battery-capacity 2.0 \
  --battery-power 1.5 \
  --pod-limit 2.0 \
  --bias-arbitrage 1.2 \
  --bias-autoconsumo 1.8 \
  --n-particles 120 \
  --n-iterations 200 \
  --save-plots \
  --output-dir results_complete_system
```

**Required Files:** All four data files  
**Execution Time:** ~15 minutes  
**Output:** Excel + JSON + Comprehensive visualizations

---

### Scenario 3: Trading-Focused Strategy

**Use Case:** Maximize arbitrage profits with reduced autoconsumo preference.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv --pv-enabled \
  --load-file data/Consumo.xlsx --load-enabled \
  --battery-capacity 1.5 \
  --bias-arbitrage 2.5 \
  --bias-autoconsumo 0.8 \
  --n-particles 100 \
  --n-iterations 150
```

**Strategy:** Aggressive trading, reduced load service preference  
**Expected:** Higher grid interaction, more volatile SOC patterns

---

### Scenario 4: Self-Consumption Focused

**Use Case:** Maximize energy independence and autoconsumo.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv --pv-enabled \
  --load-file data/Consumo.xlsx --load-enabled \
  --battery-capacity 2.0 \
  --bias-arbitrage 0.7 \
  --bias-autoconsumo 2.0 \
  --pod-limit 1.0 \
  --save-plots
```

**Strategy:** Prioritize local consumption, minimize grid dependency  
**Expected:** Higher PV-to-load allocation, more stable SOC patterns

---
### Scenario 5: Custom Efficiency Battery

**Use Case:** Evaluate system with specific battery efficiency characteristics.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --battery-tech LITIO-IONE \
  --custom-efficiency \
  --charge-efficiency 0.96 \
  --discharge-efficiency 0.94 \
  --battery-capacity 2.0 \
  --battery-power 1.5 \
  --bias-arbitrage 1.0 \
  --bias-autoconsumo 1.0
```

**Use Case:** Testing alternative battery technologies or degraded efficiency scenarios.

---

### Scenario 6: Maximum Accuracy Production Run

**Use Case:** Final results for reports, publications, or investment decisions.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv --pv-enabled \
  --load-file data/Consumo.xlsx --load-enabled \
  --battery-tech GRAFENE \
  --battery-capacity 2.0 \
  --battery-power 1.5 \
  --pod-limit 2.0 \
  --n-particles 200 \
  --n-iterations 400 \
  --bias-arbitrage 1.0 \
  --bias-autoconsumo 1.0 \
  --n-cores -2 \
  --save-plots \
  --output-dir production_results_final
```

**Execution Time:** ~45-60 minutes  
**Quality:** Maximum accuracy and robustness  
**Use Case:** Final decision-making, academic research

---

### Scenario 7: MACSE Participation

**Use Case:** Battery providing ancillary services to grid operator.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --battery-capacity 2.0 \
  --macse-enabled \
  --macse-capacity 0.5 \
  --macse-contract-years 2 \
  --macse-price-per-mw-year 60000 \
  --bias-arbitrage 1.2
```

**Note:** 0.5 MWh reserved for MACSE, 1.5 MWh available for trading.

---

### Scenario 8: Constrained POD Connection

**Use Case:** Limited grid connection capacity (rural areas, legacy infrastructure).

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv --pv-enabled \
  --load-file data/Consumo.xlsx --load-enabled \
  --battery-power 1.0 \
  --pod-limit 0.5 \
  --bias-autoconsumo 1.5 \
  --save-plots
```

**Expected:** POD violations tracked; system optimizes within constraints.

---

### Scenario 9: Fast Testing and Validation

**Use Case:** Rapid configuration validation or data integrity checks.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --n-particles 30 \
  --n-iterations 50 \
  --bias-arbitrage 1.5 \
  --output-dir test_run
```

**Execution Time:** ~2 minutes  
**Purpose:** Configuration validation, debugging, quick tests

---

## 📊 Output Files and Results

### 📈 Excel Results

**File:** `risultati_*.xlsx`

**Contents:**
- **8760 hourly rows** with complete simulation data
- **Battery states:** SOC, SOH, capacity, throughput
- **Economic metrics:** Cumulative profit, hourly transactions
- **PV allocation:** Energy to battery, grid, load
- **Load service:** Sources (PV, battery, grid)
- **Grid exchange:** Withdrawal, injection, POD violations
- **Bias effects:** Visible in trading vs autoconsumo patterns

### 📋 JSON Results

**File:** `risultati_*.json`

**Structure:**
```json
{
  "metadata": {
    "version": "3.8.1",
    "technology": "GRAFENE",
    "bias_arbitrage": 1.5,
    "bias_autoconsumo": 1.2,
    "timestamp": "2025-12-16T10:30:00"
  },
  "battery_info": {
    "final_soc": 0.65,
    "final_soh": 98.7,
    "total_throughput_mwh": 145.6
  },
  "hourly_data": [...]
}
```

### 📊 Visualization Outputs

**Directory:** `visualization/` (if `--save-plots` enabled)

**Generated Files:**
- **`overview_*.png`:** System performance overview (4 subplots)
- **`pod_tracking_*.png`:** POD compliance monitoring
- **`economic_analysis_*.png`:** Cost/revenue breakdown with bias effects
- **`pv_impact_*.png`:** PV system contribution analysis
- **`load_analysis_*.png`:** Load service statistics
- **`dettaglio_mensile_pv/`:** Monthly detailed PV analysis

---
## 📈 Result Interpretation

### 💰 Economic Viability Assessment

#### ✅ Profitable System Example
```
CONFRONTO CON SCENARIO BASE:
  • Bilancio senza BESS:     -2,900.78 €
  • Bilancio con BESS:       -1,431.66 €
  • Beneficio batteria:      +1,469.12 € (+50.7%)
  
  SISTEMA PROFITTEVOLE ✅
  
BIAS EFFECTS:
  • Arbitrage bias: 1.5x → Enhanced trading profits
  • Autoconsumo bias: 1.2x → Improved load service economics
```

**Quality Indicators:**
- Positive battery benefit (>€500 for 1 MWh system)
- Reduced grid dependency (>20% reduction)
- POD violations <5% of total hours
- Bias effects visible in strategy patterns

#### ❌ Non-Profitable System Example
```
CONFRONTO CON SCENARIO BASE:
  • Bilancio senza BESS:     -1,500.00 €
  • Bilancio con BESS:       -1,734.56 €
  • Beneficio batteria:      -234.56 € (-15.6%)
  
  SISTEMA NON PROFITTEVOLE ❌
```

**Corrective Actions:**
1. Increase battery capacity (`--battery-capacity 2.0`)
2. Increase POD limit (`--pod-limit 2.0`)
3. Enable PV/load integration
4. Adjust bias parameters for better strategy
5. Verify price spread adequacy

---

### ⚡ POD Violation Analysis

#### ✅ Acceptable Performance
```
STATISTICHE POD (Limite: 1.5 MW):
  • Violazioni totali:       42 ore (0.5%)
  • Max prelievo:            1.523 MW
  • Max immissione:          1.487 MW
  • PV curtailed:            0.12 MWh
  • Carico non servito:      0.00 MWh
```

#### ⚠️ Problematic Performance
```
STATISTICHE POD (Limite: 1.5 MW):
  • Violazioni totali:       1,842 ore (21.0%)
  • Max prelievo:            2.145 MW
  • Max immissione:          1.987 MW
  • PV curtailed:            15.67 MWh
  • Carico non servito:      2.34 MWh
```

**Solutions:**
- Increase `--pod-limit` to 2.0+ MW
- Increase `--battery-capacity`
- Adjust bias toward autoconsumo (`--bias-autoconsumo 1.5`)

---

### ☀️ PV Integration Assessment

#### ✅ Optimal Utilization
```
STATISTICHE FOTOVOLTAICO:
  • Produzione totale:       1,108.00 MWh
  • A batteria:              325.40 MWh (29.4%)
  • A rete (vendita):        445.20 MWh (40.2%)
  • A carico:                335.80 MWh (30.3%)
  • Curtailed:               1.60 MWh (0.1%)
```

#### ❌ Suboptimal Utilization (Excessive Curtailment)
```
STATISTICHE FOTOVOLTAICO:
  • Produzione totale:       1,108.00 MWh
  • A batteria:              120.00 MWh (10.8%)
  • A rete (vendita):        850.00 MWh (76.7%)
  • A carico:                50.00 MWh (4.5%)
  • Curtailed:               88.00 MWh (7.9%)
```

**Solutions:**
- Increase battery capacity
- Increase POD limit
- Use `--bias-autoconsumo 1.5+` to favor PV-to-load

---

### 🎯 Bias Effect Analysis

#### Arbitrage-Focused Results (`--bias-arbitrage 2.0`)
```
PATTERN ANALYSIS:
  • Trading frequency:       High (60%+ hours active)
  • SOC volatility:          High (frequent charge/discharge)
  • Grid interaction:        High (more buying/selling)
  • Load service:            Reduced (grid serves more load)
  • Profit source:           Primarily arbitrage
```

#### Autoconsumo-Focused Results (`--bias-autoconsumo 2.0`)
```
PATTERN ANALYSIS:
  • Trading frequency:       Low (30%- hours active)
  • SOC volatility:          Low (stable patterns)
  • Grid interaction:        Reduced (less buying/selling)
  • Load service:            High (battery serves more load)
  • Profit source:           Primarily savings from avoided purchases
```

---

## 🚨 Error Handling and Troubleshooting

### Common Errors and Solutions

#### **Error 1: Missing Required Files**
```
system_optimizer.py: error: File prezzi vendita non trovato: data/Prezzo_Vendita.xlsx
```
**Solution:** Verify file path and filename spelling. Check that file exists.

#### **Error 2: Invalid SOC Range**
```
system_optimizer.py: error: SOC litio invalido: min=0.9, max=0.1
```
**Solution:** Ensure `soc_min < soc_max`. Use `--lithium-soc-min 0.1 --lithium-soc-max 0.9`.

#### **Error 3: MACSE Capacity Exceeds Battery**
```
system_optimizer.py: error: Capacità MACSE (1.5) > capacità batteria (1.0)
```
**Solution:** Reduce `--macse-capacity` or increase `--battery-capacity`.

#### **Error 4: Custom Efficiency Missing Parameters**
```
system_optimizer.py: error: --custom-efficiency richiede almeno un parametro efficienza
```
**Solution:** Provide `--roundtrip-efficiency 0.90` or both charge/discharge efficiencies.

#### **Error 5: Bias Parameters Out of Range**
```
ValueError: Bias parameters must be positive numbers
```
**Solution:** Use positive values: `--bias-arbitrage 1.5 --bias-autoconsumo 1.2`.

---

### Performance Issues

#### **Problem:** Simulation Takes >30 Minutes

**Diagnostic Steps:**
1. Check CPU usage (should be near 100% across all cores)
2. Verify Numba installation: `python -c "import numba; print(numba.__version__)"`
3. Check particle/iteration counts

**Solutions:**
- Reduce `--n-particles 50` and `--n-iterations 100`
- Ensure `--n-cores -2` (use all CPU cores)
- Use `--disable-optimizations` for speed (reduced quality)
- Close other CPU-intensive applications

#### **Problem:** Out of Memory Errors

**Diagnostic:**
- Large number of particles (>200)
- Insufficient RAM for dataset size

**Solutions:**
- Reduce `--n-particles` to 50-100
- Disable plotting: remove `--save-plots`
- Close other memory-intensive applications

#### **Problem:** Numerical Instabilities (NaN Values)

**Diagnostic:**
- Extreme parameter values
- Invalid input data (NaN, Inf)
- SOC limit violations

**Solutions:**
- Validate input data files for missing/invalid values
- Use realistic parameter ranges
- Check SOC limits: ensure `soc_min < soc_max`
- Verify bias parameters are reasonable (0.1-5.0 range)

---
## 🚀 Performance Optimization Guide

### ⚖️ Execution Speed vs. Solution Quality

#### **Fast Execution (Testing & Debugging)**
```bash
--n-particles 30 --n-iterations 50 --n-cores -2
```
- **Time:** ~2 minutes
- **Quality:** Good for validation
- **Use Case:** Configuration testing, debugging

#### **Balanced Execution (Daily Use)**
```bash
--n-particles 50 --n-iterations 100 --n-cores -2
```
- **Time:** ~5 minutes  
- **Quality:** Excellent for most applications
- **Use Case:** Regular analysis, parameter studies

#### **High-Accuracy Execution (Production)**
```bash
--n-particles 120 --n-iterations 200 --n-cores -2
```
- **Time:** ~15 minutes
- **Quality:** Maximum practical quality
- **Use Case:** Final results, decision-making

#### **Research-Grade Execution**
```bash
--n-particles 200 --n-iterations 400 --n-cores -2
```
- **Time:** ~45-60 minutes
- **Quality:** Ultimate accuracy
- **Use Case:** Academic research, publications

### 🎯 Quality Optimization Strategy

#### **Step 1: Baseline Test (2 minutes)**
```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --n-particles 30 --n-iterations 50
```

#### **Step 2: Production Quality (15 minutes)**
```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv --pv-enabled \
  --load-file data/Consumo.xlsx --load-enabled \
  --n-particles 120 --n-iterations 200 \
  --bias-arbitrage 1.2 --bias-autoconsumo 1.1
```

#### **Step 3: Maximum Quality (if needed)**
Only if Step 2 results seem suboptimal or for final validation.

### 🔍 Quality Indicators

#### **✅ Good Solution Quality**
- Profitto finale >€200 per MWh di capacità
- SOC varia dinamicamente (non sempre ai limiti)
- Azioni batteria bilanciate (non sempre estreme)
- Miglioramento PSO >€50
- Pattern logici: carica prezzi bassi, scarica prezzi alti
- Bias effects visible in results

#### **⚠️ Suboptimal Solution Quality**
- Profitto finale <€100 per MWh di capacità
- SOC sempre ai limiti (0.1 o 0.9)
- Azioni batteria sempre estreme (+1MW o -1MW)
- Miglioramento PSO <€20
- Troppi "Violazioni POD" nei grafici
- No visible bias effects despite non-default parameters

---

## 🤖 Automation Scripts

### 🪟 Windows Batch Script

**File:** `run_optimization.bat`

```batch
@echo off
REM BESS Optimization - Automated Execution
REM Lorenzo Giannuzzo - Politecnico di Torino

echo ========================================
echo BESS Optimization v3.8.1 - Starting
echo ========================================

python system_optimizer.py ^
  --price-sell data/Prezzo_Vendita.xlsx ^
  --price-buy data/Prezzo_Acquisto.xlsx ^
  --pv-file data/PV_formattato.csv ^
  --pv-enabled ^
  --load-file data/Consumo.xlsx ^
  --load-enabled ^
  --battery-tech GRAFENE ^
  --battery-capacity 2.0 ^
  --battery-power 1.5 ^
  --pod-limit 2.0 ^
  --bias-arbitrage 1.2 ^
  --bias-autoconsumo 1.3 ^
  --n-particles 120 ^
  --n-iterations 200 ^
  --save-plots ^
  --output-dir results_%DATE:~-4,4%%DATE:~-7,2%%DATE:~-10,2%_%TIME:~0,2%%TIME:~3,2%

echo.
echo ========================================
echo Simulation Complete - Check results folder
echo ========================================
pause
```

**Usage:** Double-click the `.bat` file in Windows Explorer.

---

### 🐧 Linux/macOS Shell Script

**File:** `run_optimization.sh`

```bash
#!/bin/bash
# BESS Optimization - Automated Execution
# Lorenzo Giannuzzo - Politecnico di Torino

echo "========================================"
echo "BESS Optimization v3.8.1 - Starting"
echo "========================================"

python3 system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv \
  --pv-enabled \
  --load-file data/Consumo.xlsx \
  --load-enabled \
  --battery-tech GRAFENE \
  --battery-capacity 2.0 \
  --battery-power 1.5 \
  --pod-limit 2.0 \
  --bias-arbitrage 1.2 \
  --bias-autoconsumo 1.3 \
  --n-particles 120 \
  --n-iterations 200 \
  --save-plots \
  --output-dir results_$(date +%Y%m%d_%H%M%S)

echo ""
echo "========================================"
echo "Simulation Complete - Check results folder"
echo "========================================"
```

**Usage:**
```bash
chmod +x run_optimization.sh
./run_optimization.sh
```

---

### 🔄 Multi-Configuration Batch Script

**File:** `run_bias_comparison.bat`

```batch
@echo off
REM Compare different bias configurations
REM Lorenzo Giannuzzo - Politecnico di Torino

echo ========================================
echo BIAS COMPARISON STUDY
echo ========================================

echo Running Configuration 1: Balanced (1.0, 1.0)
python system_optimizer.py --price-sell data/Prezzo_Vendita.xlsx --price-buy data/Prezzo_Acquisto.xlsx --pv-file data/PV_formattato.csv --pv-enabled --load-file data/Consumo.xlsx --load-enabled --bias-arbitrage 1.0 --bias-autoconsumo 1.0 --n-particles 80 --n-iterations 150 --output-filename balanced_1_1

echo Running Configuration 2: Arbitrage Focus (2.0, 1.0)
python system_optimizer.py --price-sell data/Prezzo_Vendita.xlsx --price-buy data/Prezzo_Acquisto.xlsx --pv-file data/PV_formattato.csv --pv-enabled --load-file data/Consumo.xlsx --load-enabled --bias-arbitrage 2.0 --bias-autoconsumo 1.0 --n-particles 80 --n-iterations 150 --output-filename arbitrage_2_1

echo Running Configuration 3: Autoconsumo Focus (1.0, 2.0)
python system_optimizer.py --price-sell data/Prezzo_Vendita.xlsx --price-buy data/Prezzo_Acquisto.xlsx --pv-file data/PV_formattato.csv --pv-enabled --load-file data/Consumo.xlsx --load-enabled --bias-arbitrage 1.0 --bias-autoconsumo 2.0 --n-particles 80 --n-iterations 150 --output-filename autoconsumo_1_2

echo ========================================
echo All configurations complete!
echo Compare results in Excel files
echo ========================================
pause
```

---
## 🔬 Technical Background

### 🧠 Advanced Particle Swarm Optimization

PSO is a metaheuristic optimization algorithm inspired by social behavior of bird flocking. The algorithm maintains a population (swarm) of candidate solutions (particles) that move through the search space according to:

1. **Personal Best Position:** Best solution found by individual particle
2. **Global Best Position:** Best solution found by entire swarm  
3. **Velocity Update:** Stochastic combination of inertia, cognitive, and social components

**Enhanced Features in v3.8.1:**
- **Intelligent Multi-Strategy Initialization:** 5 different initialization approaches
- **Constraint Repair System:** Fixes infeasible solutions instead of penalizing
- **Adaptive Parameters:** Dynamic adjustment of w, c1, c2 based on convergence
- **Dynamic Diversification:** Automatic reinitializaton to escape local minima

**Advantages for BESS Optimization:**
- Handles non-linear, non-convex objective functions
- Scales well to high-dimensional search spaces (24+ hours × 3 decision variables)
- Robust to local optima through swarm diversity and diversification
- Computationally efficient with Numba JIT compilation

### ⚡ Numba JIT Acceleration

Numba translates Python functions to optimized machine code at runtime using LLVM compiler:

- **50-200x Speedup:** Particle evaluation in compiled C-speed loops
- **Parallel Execution:** Automatic threading across CPU cores (`prange`)
- **Cache Optimization:** Improved memory access patterns
- **First Compilation:** Initial run compiles functions (~10s), subsequent runs immediate

**Performance Scaling:**
```
Particles × Iterations = Total Evaluations
50 × 100 = 5,000 evaluations → ~2 minutes
120 × 200 = 24,000 evaluations → ~8 minutes  
200 × 400 = 80,000 evaluations → ~30 minutes
```

### 💰 Explicit Economic Accounting Model

Version 3.8.1 implements explicit economic accounting to resolve the "invisible reward problem":

#### **Problem (Previous Versions):**
- PSO saw explicit revenue from trading: `+discharge × price_sell`
- PSO saw zero immediate reward from serving load (savings were implicit)
- Result: Optimizer biased toward trading even when serving load was economically superior

#### **Solution (Version 3.8.1):**
- **Battery-to-load discharge:** `+discharge × price_buy × autoconsumo_bias` (explicit savings)
- **Grid-to-load purchase:** `-load_from_grid × price_buy` (explicit cost)
- **Battery-to-grid discharge:** `+discharge × price_sell × arbitrage_bias` (explicit revenue)
- **Net Effect:** PSO correctly values load service equal to avoided grid purchase

#### **Mathematical Proof with Bias:**

Consider: `price_sell = 50 EUR/MWh`, `price_buy = 200 EUR/MWh`, `load = 1 MWh`
Bias: `arbitrage_bias = 2.0`, `autoconsumo_bias = 1.5`

**Option A (Trading with Bias):**
```
Revenue: +1 MWh × 50 × 2.0 = +100 EUR (biased)
Grid load cost: -1 MWh × 200 = -200 EUR
Total: -100 EUR
```

**Option B (Serve Load with Bias):**
```
Savings: +1 MWh × 200 × 1.5 = +300 EUR (biased)
Grid load cost: 0 EUR (load served by battery)
Total: +300 EUR (MUCH BETTER)
```

PSO correctly chooses Option B, but bias affects the magnitude of preference.

### 🎯 Bias Control Mechanism

The bias system allows users to guide optimization strategy without changing the underlying physics:

#### **Implementation:**
```python
# In evaluate_particle_numba function
if discharge_for_trading > 0.001:
    profit += discharge_for_trading * price_sell * arbitrage_bias

if discharge_for_load > 0.001:
    profit += discharge_for_load * price_buy * autoconsumo_bias
```

#### **Effects:**
- **`arbitrage_bias > 1.0`:** Makes trading appear more profitable → more aggressive market participation
- **`autoconsumo_bias > 1.0`:** Makes load service appear more profitable → higher self-consumption
- **Balanced (1.0, 1.0):** True economic optimization without artificial preferences

---

## ✅ Validation and Testing

### 🧪 Unit Testing

**Basic Functionality Test:**
```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --n-particles 10 \
  --n-iterations 10 \
  --no-parallel
```
**Expected:** Completes in <1 minute with valid output files.

### 🔧 Integration Testing

**Full System Test:**
```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv --pv-enabled \
  --load-file data/Consumo.xlsx --load-enabled \
  --bias-arbitrage 1.5 --bias-autoconsumo 1.2 \
  --n-particles 30 --n-iterations 50
```
**Expected:** Completes in ~2 minutes with all subsystems operational and bias effects visible.

### 🎯 Bias Validation Test

**Test Different Bias Configurations:**
```bash
# Test 1: Pure arbitrage bias
python system_optimizer.py [base_params] --bias-arbitrage 3.0 --bias-autoconsumo 1.0 --output-filename test_arbitrage

# Test 2: Pure autoconsumo bias  
python system_optimizer.py [base_params] --bias-arbitrage 1.0 --bias-autoconsumo 3.0 --output-filename test_autoconsumo

# Test 3: Balanced
python system_optimizer.py [base_params] --bias-arbitrage 1.0 --bias-autoconsumo 1.0 --output-filename test_balanced
```

**Expected Results:**
- **Arbitrage-biased:** Higher grid interaction, more volatile SOC
- **Autoconsumo-biased:** Lower grid interaction, more stable SOC  
- **Balanced:** Intermediate behavior

---

## 📚 References and Resources

### 📖 Academic Publications

1. **Kennedy, J., & Eberhart, R. (1995).** Particle swarm optimization. *IEEE International Conference on Neural Networks.*

2. **Giannuzzo, L. (2025).** Advanced PSO-based Optimization for Battery Energy Storage Systems with Intelligent Bias Control. *Politecnico di Torino, DENERG.*

3. **Shi, Y., & Eberhart, R. (1998).** A modified particle swarm optimizer. *IEEE International Conference on Evolutionary Computation.*

### 🔗 Technical Documentation

- **Numba JIT Compiler:** https://numba.pydata.org/numba-doc/latest/index.html
- **Particle Swarm Optimization:** https://en.wikipedia.org/wiki/Particle_swarm_optimization
- **MACSE (Italian ASM):** https://www.mercatoelettrico.org/it/Mercati/MACSE/MACSE.aspx
- **Battery Energy Storage:** https://www.irena.org/publications/2019/Sep/Utility-scale-batteries

### 🏛️ Institutional Links

- **Politecnico di Torino:** https://www.polito.it/
- **DENERG Department:** http://www.denerg.polito.it/
- **Energy Center Lab:** http://www.energycenter.polito.it/

---

## 📄 License

**MIT License**

Copyright (c) 2025 Lorenzo Giannuzzo - Politecnico di Torino

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 📞 Contact Information

**Author:** Lorenzo Giannuzzo  
**Email:** lorenzo.giannuzzo@polito.it  
**Institution:** Politecnico di Torino  
**Department:** DENERG (Dipartimento Energia)  
**Laboratory:** Energy Center Lab  
**LinkedIn:** [Lorenzo Giannuzzo](https://linkedin.com/in/lorenzo-giannuzzo)

For technical support, bug reports, or collaboration inquiries, please contact the author via email with subject line: **"[BESS v3.8.1] Your Subject"**

---

## 📋 Version History

### **v3.8.1** (December 16, 2025) - Current
- ✨ **NEW:** Bias optimization flags (`--bias-arbitrage`, `--bias-autoconsumo`)
- ✨ **NEW:** Intelligent strategy guidance system
- 🔧 Enhanced CLI interface with 40+ parameters
- 📊 Improved result interpretation with bias effect analysis
- 📚 Comprehensive documentation update

### **v3.8.0** (December 12, 2025)
- 💰 Explicit economic accounting implementation
- ⚡ Pure XOR battery operation logic
- 📈 Enhanced POD compliance tracking
- 🔧 Custom efficiency support
- 📊 Comprehensive output improvements

### **v3.7.0** (November 25, 2025)
- 🏠 Autonomous load decision framework
- 🎯 Multi-dimensional PSO (3D action space)
- ☀️ PV allocation optimization
- 🔄 Rolling horizon improvements

### **v3.0.0** (October 2025)
- ⚡ Numba JIT parallelization
- 💻 Command-line interface
- 🇮🇹 MACSE integration
- 🔋 Graphene battery support

### **v2.0.0** (September 2025)
- 🧠 PSO optimization engine
- 🔋 Multi-technology support
- ☀️ Basic PV integration

### **v1.0.0** (August 2025)
- 🚀 Initial release
- 🔋 Single-technology optimization
- 💰 Basic arbitrage functionality

---

**Document Version:** 4.0  
**Last Updated:** December 16, 2025  
**Maintained By:** Lorenzo Giannuzzo - Politecnico di Torino DENERG

---

*This README provides comprehensive documentation for the BESS Optimization System v3.8.1. For additional technical details, please refer to the source code comments and academic publications.*