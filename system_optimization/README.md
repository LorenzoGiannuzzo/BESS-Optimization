# Battery Energy Storage System (BESS) Optimization
## Advanced PSO-based Energy Trading and Load Management System

---

## Project Information

**Version:** 3.8.0  
**Release Date:** December 12, 2025  
**Author:** Lorenzo Giannuzzo  
**Affiliation:** Politecnico di Torino  
**Department:** DENERG (Dipartimento Energia)  
**Laboratory:** Energy Center Lab  
**License:** MIT License  
**Python Compatibility:** 3.8+  

---

## Executive Summary

This software implements a sophisticated Battery Energy Storage System (BESS) optimization framework using Particle Swarm Optimization (PSO) with Numba JIT acceleration. The system is designed for energy trading arbitrage, photovoltaic (PV) integration, and autonomous load management with explicit economic accounting.

### Key Capabilities

- **Energy Arbitrage:** Optimized battery charging and discharging based on market price signals
- **PV Integration:** Autonomous allocation of photovoltaic production to battery, grid, and load
- **Load Management:** Intelligent decision-making for serving electrical loads from battery vs. grid
- **MACSE Support:** Integration with Italian ancillary services market (Mercato dei Servizi Ancillari)
- **POD Enforcement:** Point of Delivery power limit compliance with violation tracking
- **Multi-Technology:** Support for Lithium-Ion and Graphene battery technologies
- **High Performance:** Numba JIT parallelization achieving 50-200x speedup over pure Python

---
## Technical Architecture

### Core Components

**1. PSO Optimization Engine**
- Numba JIT-compiled evaluation functions for maximum performance
- Parallel particle evaluation across multiple CPU cores
- Smart initialization based on price signals and system constraints
- Adaptive particle reinitialization to prevent stagnation

**2. Rolling Horizon Simulator**
- Hour-by-hour execution of optimal strategies
- Real-time SOC and SOH tracking
- Degradation modeling based on equivalent cycles
- Comprehensive energy flow accounting

**3. Economic Model**
- Explicit accounting of all costs and revenues
- Trading revenues: battery discharge to grid
- Savings: battery discharge to load (avoiding grid purchase)
- Costs: grid purchases for battery charging and load service
- Degradation costs integrated into optimization objective

**4. Constraint Management**
- POD power limits enforced on both injection and withdrawal
- Battery SOC limits (technology-dependent)
- C-rate limitations
- Physical power constraints

---

## Installation and Setup

### Prerequisites

```bash
pip install pandas numpy numba matplotlib openpyxl joblib
```

### Required Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20.0 | Numerical computations |
| pandas | ≥1.3.0 | Data manipulation |
| numba | ≥0.54.0 | JIT compilation |
| matplotlib | ≥3.4.0 | Visualization |
| openpyxl | ≥3.0.0 | Excel file handling |
| joblib | ≥1.0.0 | Parallel processing utilities |

### Project Structure

```
project_root/
├── system_optimizer.py          # Main executable script
├── data/                         # Input data directory (required)
│   ├── Prezzo_Vendita.xlsx      # Electricity selling prices (required)
│   ├── Prezzo_Acquisto.xlsx     # Electricity purchase prices (required)
│   ├── PV_formattato.csv        # PV production data (optional)
│   └── Consumo.xlsx             # Load consumption data (optional)
├── results/                      # Output directory (auto-created)
│   ├── risultati_*.xlsx         # Simulation results
│   └── risultati_*.json         # Complete data export
└── visualization/                # Plots directory (if enabled)
    ├── overview_*.png
    ├── pod_tracking_*.png
    └── dettaglio_mensile_pv/
```

---

## Command Line Interface

### Basic Syntax

```bash
python system_optimizer.py --price-sell <file> --price-buy <file> [OPTIONS]
```

### Minimal Working Example

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx
```

### Complete System Example

```bash
python system_optimizer.py \
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
  --n-particles 120 \
  --n-iterations 300 \
  --save-plots \
  --output-dir results_production
```

---

## Parameter Reference

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--price-sell` | string | Path to electricity selling prices file (.xlsx) |
| `--price-buy` | string | Path to electricity purchase prices file (.xlsx) |

### Battery Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--battery-tech` | choice | LITIO-IONE | LITIO-IONE, GRAFENE | Battery technology |
| `--battery-capacity` | float | 1.0 | >0 | Battery capacity (MWh) |
| `--battery-power` | float | 1.0 | >0 | Maximum power (MW) |
| `--battery-c-rate` | float | 1.0 | >0 | Maximum C-rate |

### Custom Efficiency Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--custom-efficiency` | flag | False | - | Enable custom efficiency mode |
| `--charge-efficiency` | float | None | 0-1 | Charging efficiency |
| `--discharge-efficiency` | float | None | 0-1 | Discharging efficiency |
| `--roundtrip-efficiency` | float | None | 0-1 | Round-trip efficiency |

**Note:** With `--custom-efficiency`, provide at least `--roundtrip-efficiency` OR both `--charge-efficiency` and `--discharge-efficiency`.

### SOC Limits

#### Lithium-Ion
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--lithium-soc-min` | 0.1 | 0.0-1.0 | Minimum state of charge (10%) |
| `--lithium-soc-max` | 0.9 | 0.0-1.0 | Maximum state of charge (90%) |

#### Graphene
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--graphene-soc-min` | 0.0 | 0.0-1.0 | Minimum state of charge (0%) |
| `--graphene-soc-max` | 1.0 | 0.0-1.0 | Maximum state of charge (100%) |

### Point of Delivery

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--pod-limit` | float | 1.5 | >0 | Maximum grid exchange power (MW) |

**Recommended Values:**
- 0.8 MW: Residential connection
- 1.0 MW: Match battery power rating
- 1.5 MW: Default (minimal constraints)
- 2.0+ MW: Industrial installations

### Photovoltaic System

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pv-enabled` | flag | False | Enable photovoltaic system |
| `--pv-file` | string | None | Path to PV production file (.csv) |
| `--pv-nominal-power` | float | 1.0 | Nominal PV power (kWp) |

**Warning:** Modifying `--pv-nominal-power` may cause data inconsistencies. Use default value.

### Load Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--load-enabled` | flag | False | Enable load management |
| `--load-file` | string | None | Path to load consumption file (.xlsx) |

### PSO Optimization

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--n-particles` | int | 50 | >0 | Number of PSO particles |
| `--n-iterations` | int | 100 | >0 | Number of PSO iterations |

**Performance vs. Quality Trade-offs:**

| Configuration | Particles | Iterations | Time                 | Quality |
|---------------|-----------|------------|----------------------|---------|
| Fast | 30 | 50 | very short (minutes) | Good |
| Balanced | 50 | 100 | short (30 min)       | Excellent |
| Accurate | 120 | 300 | long (30min >)       | Maximum |

### Parallelization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--no-parallel` | flag | False | Disable parallelization (debug mode) |
| `--n-cores` | int | -2 | Number of CPU cores (-1: all-1, -2: all, N: specific) |

### MACSE Integration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--macse-enabled` | flag | False | - | Enable MACSE ancillary services |
| `--macse-capacity` | float | 1.0 | >0, ≤battery_capacity | Reserved capacity (MWh) |
| `--macse-contract-years` | int | 1 | >0 | Contract duration (years) |
| `--macse-price-per-mw-year` | float | 50000 | >0 | Annual price (EUR/MW/year) |

### Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save-plots` | flag | False | Generate visualization plots |
| `--output-dir` | string | results | Output directory path |
| `--output-filename` | string | None | Custom output filename (without extension) |

---

## Input Data Formats

### Electricity Prices (Excel)

**Files:** `Prezzo_Vendita.xlsx`, `Prezzo_Acquisto.xlsx`

**Required Format:**
| Data | €/MWh |
|------|-------|
| 01/01/2024 00:00 | 85.34 |
| 01/01/2024 01:00 | 78.21 |
| ... | ... |

**Requirements:**
- Column header: `€/MWh` (mandatory)
- 8760 rows (one year, hourly resolution)
- Numeric values or text with comma decimal separator
- Date column optional (for reference only)

### PV Production (CSV)

**File:** `PV_formattato.csv`

**Required Format:**
```csv
Data;P
01/01/2024 00:00;0.0
01/01/2024 01:00;0.0
01/01/2024 08:00;150.5
01/01/2024 12:00;890.2
...
```

**Requirements:**
- Separator: semicolon (`;`)
- Column header: `P` (power in kW)
- 8760 rows
- Date column format: DD/MM/YYYY HH:MM

### Load Consumption (Excel)

**File:** `Consumo.xlsx`

**Required Format:**
| Data | value |
|------|-------|
| 01/01/2024 00:00 | 450.3 |
| 01/01/2024 01:00 | 380.1 |
| ... | ... |

**Requirements:**
- Column header: `value` (or `load`, `Load`, `power`, `Power`)
- Power in kW
- 8760 rows
- First sheet of Excel file used by default

---

## Usage Scenarios

### Scenario 1: Pure Arbitrage (No PV/Load)

**Use Case:** Evaluate battery profitability through energy arbitrage alone.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --battery-tech LITIO-IONE \
  --battery-capacity 1.0 \
  --battery-power 1.0 \
  --pod-limit 1.5
```

**Required Files:** Price files only  
**Execution Time:** Approximately 5 minutes  
**Output:** Excel + JSON results

---

### Scenario 2: Complete System (PV + Battery + Load)

**Use Case:** Full residential or commercial energy system optimization.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv \
  --pv-enabled \
  --load-file data/Consumo.xlsx \
  --load-enabled \
  --battery-tech GRAFENE \
  --battery-capacity 1.0 \
  --battery-power 1.0 \
  --pod-limit 1.5 \
  --save-plots \
  --output-dir results_complete
```

**Required Files:** All four data files  
**Execution Time:** Approximately 8 minutes  
**Output:** Excel + JSON + Visualizations

---

### Scenario 3: Custom Efficiency Battery

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
  --battery-power 1.5
```

**Use Case:** Testing alternative battery technologies or accounting for degraded efficiency.

---

### Scenario 4: Maximum Accuracy Production Run

**Use Case:** Final results for reports, publications, or decision-making.

```bash
python system_optimizer.py \
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
  --n-particles 120 \
  --n-iterations 300 \
  --n-cores -2 \
  --save-plots \
  --output-dir production_results
```

**Execution Time:** Approximately 20 minutes  
**Quality:** Maximum accuracy and robustness

---

### Scenario 5: MACSE Participation

**Use Case:** Battery providing ancillary services to grid operator.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --battery-capacity 2.0 \
  --macse-enabled \
  --macse-capacity 0.5 \
  --macse-contract-years 2 \
  --macse-price-per-mw-year 60000
```

**Note:** 0.5 MWh reserved for MACSE, 1.5 MWh available for trading.

---

### Scenario 6: Constrained POD Connection

**Use Case:** Limited grid connection capacity (e.g., rural area, legacy infrastructure).

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv \
  --pv-enabled \
  --load-file data/Consumo.xlsx \
  --load-enabled \
  --battery-power 1.0 \
  --pod-limit 0.5 \
  --save-plots
```

**Expected Outcome:** POD violations tracked; system optimizes within constraint.

---

### Scenario 7: Fast Testing

**Use Case:** Rapid validation of configuration or data integrity.

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --n-particles 30 \
  --n-iterations 50 \
  --output-dir test_run
```

**Execution Time:** Approximately 2 minutes  
**Purpose:** Configuration validation, debugging

---

## Output Files

### Excel Results

**File:** `risultati_*.xlsx`

**Contents:**
- Hourly simulation data (8760 rows)
- Battery state variables (SOC, SOH, capacity)
- Economic metrics (cumulative profit, hourly transactions)
- PV allocation (to battery, grid, load)
- Load service sources (PV, battery, grid)
- Grid exchange data (withdrawal, injection)
- POD violation flags

### JSON Results

**File:** `risultati_*.json`

**Structure:**
```json
{
  "metadata": {
    "version": "3.8.0",
    "technology": "...",
    "timestamp": "..."
  },
  "battery_info": {...},
  "hourly_data": [...]
}
```

**Contents:**
- Complete simulation metadata
- Battery specifications and final state
- Hour-by-hour data (equivalent to Excel)
- Programmatic access to all results

### Visualization Outputs

**Directory:** `visualization/`

**Generated Files (if `--save-plots` enabled):**
- `overview_*.png`: System performance overview (4 subplots)
- `pod_tracking_*.png`: POD compliance monitoring
- `economic_analysis_*.png`: Cost/revenue breakdown
- `pv_impact_*.png`: PV system contribution analysis
- `load_analysis_*.png`: Load service statistics
- `dettaglio_mensile_pv/`: Monthly detailed PV analysis

---

## Result Interpretation

### Economic Viability

**Profitable System:**
```
CONFRONTO CON SCENARIO BASE:
  • Bilancio senza BESS:     -2,900.78 €
  • Bilancio con BESS:       -1,431.66 €
  • Beneficio batteria:      +1,469.12 € (+50.7%)
  SISTEMA PROFITTEVOLE
```

**Indicators:**
- Positive battery benefit
- Reduced grid dependency
- POD violations <5%

**Non-Profitable System:**
```
CONFRONTO CON SCENARIO BASE:
  • Bilancio senza BESS:     -1,500.00 €
  • Bilancio con BESS:       -1,734.56 €
  • Beneficio batteria:      -234.56 € (-15.6%)
  SISTEMA NON PROFITTEVOLE
```

**Corrective Actions:**
1. Increase battery capacity
2. Increase POD limit
3. Enable PV/load integration
4. Verify price spread adequacy

---

### POD Violation Analysis

**Acceptable Performance:**
```
STATISTICHE POD (Limite: 1.5 MW):
  • Violazioni totali:       42 ore (0.5%)
  • Max prelievo:            1.523 MW
  • Max immissione:          1.487 MW
  • PV curtailed:            0.12 MWh
```

**Problematic Performance:**
```
STATISTICHE POD (Limite: 1.5 MW):
  • Violazioni totali:       1,842 ore (21.0%)
  • Max prelievo:            2.145 MW
  • Max immissione:          1.987 MW
  • PV curtailed:            15.67 MWh
  • Carico non servito:      2.34 MWh
```

**Solution:** Increase `--pod-limit` or `--battery-capacity`

---

### PV Integration Assessment

**Optimal Utilization:**
```
STATISTICHE FOTOVOLTAICO:
  • Produzione totale:       1,108.00 MWh
  • A batteria:              325.40 MWh (29.4%)
  • A rete (vendita):        445.20 MWh (40.2%)
  • A carico:                335.80 MWh (30.3%)
  • Curtailed:               1.60 MWh (0.1%)
```

**Suboptimal Utilization (excessive curtailment):**
```
STATISTICHE FOTOVOLTAICO:
  • Produzione totale:       1,108.00 MWh
  • A batteria:              120.00 MWh (10.8%)
  • A rete (vendita):        850.00 MWh (76.7%)
  • A carico:                50.00 MWh (4.5%)
  • Curtailed:               88.00 MWh (7.9%)
```

**Solution:** Increase battery capacity or POD limit

---

## Error Handling

### Common Errors and Solutions

**Error 1: Missing Required Files**
```
bess_cli.py: error: File prezzi vendita non trovato: data/Prezzo_Vendita.xlsx
```
**Solution:** Verify file path and filename spelling.

**Error 2: Invalid SOC Range**
```
bess_cli.py: error: SOC litio invalido: min=0.9, max=0.1
```
**Solution:** Ensure `soc_min < soc_max`.

**Error 3: MACSE Capacity Exceeds Battery Capacity**
```
bess_cli.py: error: Capacità MACSE (1.5) > capacità batteria (1.0)
```
**Solution:** Reduce `--macse-capacity` or increase `--battery-capacity`.

**Error 4: Custom Efficiency Without Values**
```
bess_cli.py: error: --custom-efficiency richiede almeno un parametro efficienza
```
**Solution:** Provide at least one efficiency parameter.

---

## Performance Optimization

### Execution Speed vs. Solution Quality

**Fast Execution (Testing):**
```bash
--n-particles 30 --n-iterations 50 --n-cores -2
```
Time: ~2 minutes | Quality: Good

**Balanced Execution (Daily Use):**
```bash
--n-particles 50 --n-iterations 100 --n-cores -2
```
Time: ~5 minutes | Quality: Excellent

**High-Accuracy Execution (Production):**
```bash
--n-particles 120 --n-iterations 300 --n-cores -2
```
Time: ~20 minutes | Quality: Maximum

---

## Automation Scripts

### Windows Batch Script

**File:** `run_optimization.bat`

```batch
@echo off
REM BESS Optimization - Automated Execution
REM Lorenzo Giannuzzo - Politecnico di Torino

echo ========================================
echo BESS Optimization - Starting
echo ========================================

python system_optimizer.py ^
  --price-sell data/Prezzo_Vendita.xlsx ^
  --price-buy data/Prezzo_Acquisto.xlsx ^
  --pv-file data/PV_formattato.csv ^
  --pv-enabled ^
  --load-file data/Consumo.xlsx ^
  --load-enabled ^
  --battery-tech GRAFENE ^
  --battery-capacity 1.0 ^
  --pod-limit 1.5 ^
  --n-particles 50 ^
  --n-iterations 100 ^
  --save-plots ^
  --output-dir results_%DATE:~-4,4%%DATE:~-7,2%%DATE:~-10,2%

echo.
echo ========================================
echo Simulation Complete
echo ========================================
pause
```

**Execution:** Double-click the `.bat` file.

---

### Linux/macOS Shell Script

**File:** `run_optimization.sh`

```bash
#!/bin/bash
# BESS Optimization - Automated Execution
# Lorenzo Giannuzzo - Politecnico di Torino

echo "========================================"
echo "BESS Optimization - Starting"
echo "========================================"

python3 system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv \
  --pv-enabled \
  --load-file data/Consumo.xlsx \
  --load-enabled \
  --battery-tech GRAFENE \
  --battery-capacity 1.0 \
  --pod-limit 1.5 \
  --n-particles 50 \
  --n-iterations 100 \
  --save-plots \
  --output-dir results_$(date +%Y%m%d_%H%M%S)

echo ""
echo "========================================"
echo "Simulation Complete"
echo "========================================"
```

**Execution:**
```bash
chmod +x run_optimization.sh
./run_optimization.sh
```

---

## Technical Background

### Particle Swarm Optimization

PSO is a metaheuristic optimization algorithm inspired by social behavior of bird flocking or fish schooling. The algorithm maintains a population (swarm) of candidate solutions (particles) that move through the search space according to:

1. **Personal best position:** Best solution found by individual particle
2. **Global best position:** Best solution found by entire swarm
3. **Velocity update:** Stochastic combination of inertia, cognitive, and social components

**Advantages for BESS optimization:**
- Handles non-linear, non-convex objective functions
- Scales well to high-dimensional search spaces (24+ hours × 3 decision variables)
- Robust to local optima through swarm diversity
- Computationally efficient with Numba JIT compilation

### Numba JIT Acceleration

Numba translates Python functions to optimized machine code at runtime using LLVM compiler. Key performance benefits:

- **50-200x speedup:** Particle evaluation in compiled C-speed loops
- **Parallel execution:** Automatic threading across CPU cores
- **Cache optimization:** Improved memory access patterns
- **First compilation overhead:** Initial run compiles functions (~10s), subsequent runs immediate

### Explicit Economic Accounting

Version 3.8.0 implements explicit economic accounting to resolve the "invisible reward problem" in previous versions:

**Problem (Previous Versions):**
- PSO saw explicit revenue from trading: `+discharge × price_sell`
- PSO saw zero immediate reward from serving load (savings were implicit)
- Result: Optimizer biased toward trading even when serving load was economically superior

**Solution (Version 3.8.0):**
- Battery-to-load discharge: `+discharge × price_buy` (explicit savings)
- Grid-to-load purchase: `-load_from_grid × price_buy` (explicit cost)
- Net effect: PSO correctly values load service equal to avoided grid purchase

**Mathematical Proof:**

Consider: `price_sell = 50 EUR/MWh`, `price_buy = 200 EUR/MWh`, `load = 10 MWh`

Option A (Trading): 
```
Revenue: +1 MWh × 50 = +50 EUR
Grid load cost: -9 MWh × 200 = -1,800 EUR
Total: -1,750 EUR
```

Option B (Serve Load):
```
Savings: +1 MWh × 200 = +200 EUR (explicit!)
Grid load cost: -9 MWh × 200 = -1,800 EUR
Total: -1,600 EUR (BETTER)
```

PSO now correctly chooses Option B.

---

## Validation and Testing

### Unit Testing

Run basic validation tests:

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --n-particles 10 \
  --n-iterations 10 \
  --no-parallel
```

Expected: Completes in <1 minute with valid output files.

### Integration Testing

Test full system with minimal dataset:

```bash
python system_optimizer.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv \
  --pv-enabled \
  --load-file data/Consumo.xlsx \
  --load-enabled \
  --n-particles 30 \
  --n-iterations 50
```

Expected: Completes in ~2 minutes with all subsystems operational.

---

## Troubleshooting

### Performance Issues

**Problem:** Simulation takes >30 minutes

**Diagnostic Steps:**
1. Check CPU usage (should be near 100% across all cores)
2. Verify Numba is installed: `python -c "import numba; print(numba.__version__)"`
3. Check particle/iteration counts

**Solutions:**
- Reduce `--n-particles` and `--n-iterations`
- Ensure `--n-cores -2` (use all CPU cores)
- Verify no background processes consuming CPU

---

### Memory Issues

**Problem:** Out of memory errors

**Diagnostic:**
- Large number of particles (>200)
- Insufficient RAM for dataset size

**Solutions:**
- Reduce `--n-particles` to 50-100
- Disable plotting: remove `--save-plots`
- Close other applications

---

### Numerical Instabilities

**Problem:** NaN values in output or divergent results

**Diagnostic:**
- Extreme parameter values
- Invalid input data (NaN, Inf)
- SOC limit violations

**Solutions:**
- Validate input data files
- Use realistic parameter ranges
- Check SOC limits: `soc_min < soc_max`

---

## References and Resources

### Academic Publications

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. IEEE International Conference on Neural Networks.

2. Giannuzzo, L. (2025). Advanced PSO-based Optimization for Battery Energy Storage Systems. Politecnico di Torino, DENERG.

### Technical Documentation

- **Numba JIT Compiler:** https://numba.pydata.org/numba-doc/latest/index.html
- **Particle Swarm Optimization:** https://en.wikipedia.org/wiki/Particle_swarm_optimization
- **MACSE (Italian ASM):** https://www.mercatoelettrico.org/it/Mercati/MACSE/MACSE.aspx

### Institutional Links

- **Politecnico di Torino:** https://www.polito.it/
- **DENERG Department:** http://www.denerg.polito.it/
- **Energy Center Lab:** http://www.energycenter.polito.it/

---

## License

Copyright (c) 2025 Lorenzo Giannuzzo - Politecnico di Torino

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Contact Information

**Author:** Lorenzo Giannuzzo  
**Email:** lorenzo.giannuzzo@polito.it  
**Institution:** Politecnico di Torino  
**Department:** DENERG (Dipartimento Energia)  
**Laboratory:** Energy Center Lab  

For technical support, bug reports, or collaboration inquiries, please contact the author via email with subject line: "[BESS v3.8.0] Your Subject"

---

## Version History

**v3.8.0** (December 12, 2025)
- Explicit economic accounting implementation
- Pure XOR battery operation logic
- Enhanced POD compliance tracking
- Custom efficiency support
- Comprehensive output improvements

**v3.7.0** (November 25, 2025)
- Autonomous load decision framework
- Multi-dimensional PSO (3D action space)
- PV allocation optimization
- Rolling horizon improvements

**v3.0.0** (October 2025)
- Numba JIT parallelization
- Command-line interface
- MACSE integration
- Graphene battery support

**v2.0.0** (September 2025)
- PSO optimization engine
- Multi-technology support
- Basic PV integration

**v1.0.0** (August 2025)
- Initial release
- Single-technology optimization
- Basic arbitrage functionality

---

**Document Version:** 3.0  
**Last Updated:** December 12, 2025  
**Maintained By:** Lorenzo Giannuzzo - Politecnico di Torino DENERG
