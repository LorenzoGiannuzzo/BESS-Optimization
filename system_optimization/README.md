# BESS Optimization - Guida Command Line Interface (CLI)

## Informazioni Generali

**Nome Progetto:** Battery Energy Storage System (BESS) Optimization CLI  
**Versione:** 3.8.0  
**Autore:** Lorenzo Giannuzzo  
**Affiliazione:** Politecnico di Torino - DENERG - Energy Center Lab  
**Data:** 28 Novembre 2025  

---

## Novità Versione CLI

✅ **Supporto completo argomenti da linea di comando**  
✅ **Configurazione parametri senza modificare codice**  
✅ **Validazione automatica input**  
✅ **Help integrato con esempi**  
✅ **Ottimizzazione PSO con Numba JIT (50-200x più veloce)**  
✅ **Parallelizzazione automatica**  

---

## Quick Start

### Esempio Base (Minimo)

```bash
python bess_cli.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx
```

### Esempio Completo (PV + Carico + Grafici)

```bash
python bess_cli.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv \
  --pv-enabled \
  --load-file data/Consumo.xlsx \
  --load-enabled \
  --save-plots \
  --output-dir results
```

### Esempio Batteria Grafene Custom

```bash
python bess_cli.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --battery-tech GRAFENE \
  --battery-capacity 2.0 \
  --battery-power 2.0 \
  --pod-limit 2.5 \
  --graphene-soc-min 0.0 \
  --graphene-soc-max 1.0
```

---

## Struttura Directory 

```
progetto/
├── bess_cli.py                    # Script principale
├── data/                          # ✅ CARTELLA OBBLIGATORIA
│   ├── Prezzo_Vendita.xlsx       # ✅ OBBLIGATORIO
│   ├── Prezzo_Acquisto.xlsx      # ✅ OBBLIGATORIO
│   ├── PV_formattato.csv         # ⚠️  Se --pv-enabled
│   └── Consumo.xlsx              # ⚠️  Se --load-enabled
├── results/                       # Creata automaticamente
└── visualization/                 # Se --save-plots
```

---

## Parametri Command Line

### **FILE PATHS (Obbligatori)**

| Parametro | Tipo | Descrizione | Esempio |
|-----------|------|-------------|---------|
| `--price-sell` | **REQUIRED** | File prezzi vendita (.xlsx) | `data/vendita.xlsx` |
| `--price-buy` | **REQUIRED** | File prezzi acquisto (.xlsx) | `data/acquisto.xlsx` |

**⚠IMPORTANTE:** Questi due parametri sono SEMPRE obbligatori!

---

### **FILE PATHS (Opzionali)**

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `--pv-file` | string | `None` | File produzione PV (.csv) |
| `--load-file` | string | `None` | File carico utente (.xlsx) |

---

###  **POINT OF DELIVERY (POD)**

| Parametro | Tipo | Default | Range | Descrizione |
|-----------|------|---------|-------|-------------|
| `--pod-limit` | float | `1.5` | > 0 | Potenza massima scambio rete [MW] |

**Valori Consigliati:**
- `0.8` → Connessione domestica
- `1.0` → Pari a potenza batteria
- `1.5` → Senza limitazioni (default)
- `>2.0` → Installazioni industriali

---

### **PARAMETRI BATTERIA**

| Parametro | Tipo | Default | Range | Descrizione |
|-----------|------|---------|-------|-------------|
| `--battery-tech` | choice | `LITIO-IONE` | `LITIO-IONE`, `GRAFENE` | Tecnologia batteria |
| `--battery-capacity` | float | `1.0` | > 0 | Capacità batteria [MWh] |
| `--battery-power` | float | `1.0` | > 0 | Potenza massima [MW] |
| `--battery-c-rate` | float | `1.0` | > 0 | C-rate massimo |

**Esempio:**
```bash
--battery-tech GRAFENE --battery-capacity 2.0 --battery-power 1.5
```

---

### **SOC LIMITS**

#### Litio-Ione

| Parametro | Default | Range | Descrizione |
|-----------|---------|-------|-------------|
| `--lithium-soc-min` | `0.1` | 0.0 - 1.0 | SOC minimo (10%) |
| `--lithium-soc-max` | `0.9` | 0.0 - 1.0 | SOC massimo (90%) |

#### Grafene

| Parametro | Default | Range | Descrizione |
|-----------|---------|-------|-------------|
| `--graphene-soc-min` | `0.0` | 0.0 - 1.0 | SOC minimo (0%) |
| `--graphene-soc-max` | `1.0` | 0.0 - 1.0 | SOC massimo (100%) |

** VALIDAZIONE:** `soc_min < soc_max` verificato automaticamente

---

### **FOTOVOLTAICO**

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `--pv-enabled` | flag | `False` | Abilita sistema fotovoltaico |
| `--pv-nominal-power` | float | `1.0` | Potenza nominale [kWp] |

** IMPORTANTE:** Non modificare `--pv-nominal-power` per evitare sbilanciamenti!

**Esempio:**
```bash
--pv-file data/pv.csv --pv-enabled
```

---

### **CARICO UTENTE**

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `--load-enabled` | flag | `False` | Abilita carico utente |

**Esempio:**
```bash
--load-file data/load.xlsx --load-enabled
```

---

### **PSO OPTIMIZATION**

| Parametro | Tipo | Default | Range | Descrizione |
|-----------|------|---------|-------|-------------|
| `--n-particles` | int | `50` | > 0 | Numero particelle PSO |
| `--n-iterations` | int | `100` | > 0 | Numero iterazioni PSO |

** Bilanciamento Velocità vs Qualità:**

| Config | Particles | Iterations | Tempo | Qualità |
|--------|-----------|------------|-------|---------|
| **Veloce** | 30 | 50 | ~2 min | Buona |
| **Bilanciata** | 50 | 100 | ~5 min | Ottima |
| **Accurata** | 120 | 300 | ~20 min | Eccellente |

**Esempio veloce:**
```bash
--n-particles 30 --n-iterations 50
```

**Esempio accurato:**
```bash
--n-particles 120 --n-iterations 300
```

---

### **PARALLELIZZAZIONE**

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `--no-parallel` | flag | `False` | Disabilita parallelizzazione |
| `--n-cores` | int | `-2` | Numero cores (-1=tutti-1, -2=tutti, N=specifico) |

**Esempi:**
```bash
--n-cores -2              # Usa tutti i core (massima velocità)
--n-cores -1              # Usa tutti i core meno uno
--n-cores 4               # Usa esattamente 4 core
--no-parallel             # Esecuzione sequenziale (debug)
```

** TIP:** Usa `-2` per massima velocità, usa `--no-parallel` solo per debug

---

### **MACSE (Servizi Ancillari)**

| Parametro | Tipo | Default | Range | Descrizione |
|-----------|------|---------|-------|-------------|
| `--macse-enabled` | flag | `False` | - | Abilita servizi MACSE |
| `--macse-capacity` | float | `1.0` | > 0, < battery_capacity | Capacità riservata [MWh] |
| `--macse-contract-years` | int | `1` | > 0 | Anni contratto |
| `--macse-price-per-mw-year` | float | `50000` | > 0 | Prezzo [€/MW/anno] |

** VALIDAZIONE:** `macse_capacity ≤ battery_capacity` verificato automaticamente

**Esempio:**
```bash
--macse-enabled --macse-capacity 0.5 --macse-contract-years 2
```

---

### 📤 **OUTPUT**

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `--save-plots` | flag | `False` | Salva grafici visualizzazione |
| `--output-dir` | string | `results` | Directory output risultati |

**Esempio:**
```bash
--save-plots --output-dir my_results
```

---

## Esempi Completi

### Scenario 1: Solo Arbitraggio (Senza PV/Carico)

```bash
python bess_cli.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --battery-tech LITIO-IONE \
  --battery-capacity 1.0 \
  --battery-power 1.0 \
  --pod-limit 1.5
```

**File Necessari:** Solo prezzi  
**Tempo Esecuzione:** ~5 minuti  

---

### Scenario 2: Sistema Completo (PV + Batteria + Carico)

```bash
python bess_cli.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --pv-file data/PV_formattato.csv \
  --pv-enabled \
  --load-file data/Consumo.xlsx \
  --load-enabled \
  --battery-tech GRAFENE \
  --battery-capacity 1.0 \
  --battery-power 1.0 \
  --pod-limit 0.8 \
  --save-plots \
  --output-dir results_completo
```

**File Necessari:** Tutti e 4  
**Tempo Esecuzione:** ~8 minuti  
**Output:** Excel + JSON + Grafici

---

### Scenario 3: Ottimizzazione Veloce (Test)

```bash
python bess_cli.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --n-particles 30 \
  --n-iterations 50 \
  --output-dir test_rapido
```

**Tempo Esecuzione:** ~2 minuti  
**Uso:** Test rapidi, debug  

---

### Scenario 4: Ottimizzazione Massima (Produzione)

```bash
python bess_cli.py \
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
  --output-dir production_run
```

**Tempo Esecuzione:** ~20 minuti  
**Uso:** Risultati finali, report  

---

### Scenario 5: Con MACSE

```bash
python bess_cli.py \
  --price-sell data/Prezzo_Vendita.xlsx \
  --price-buy data/Prezzo_Acquisto.xlsx \
  --battery-capacity 2.0 \
  --macse-enabled \
  --macse-capacity 0.5 \
  --macse-contract-years 2 \
  --macse-price-per-mw-year 60000
```

**Nota:** 0.5 MWh riservati MACSE, 1.5 MWh per trading

---

### Scenario 6: POD Limitato (Connessione Debole)

```bash
python bess_cli.py \
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

** Attenzione:** POD < potenza batteria → possibili limitazioni

---

## Help Integrato

### Visualizza Help Completo

```bash
python bess_cli.py --help
```

**Output:**
```
usage: bess_cli.py [-h] --price-sell PRICE_SELL --price-buy PRICE_BUY 
                   [--pv-file PV_FILE] [--load-file LOAD_FILE]
                   [--pod-limit POD_LIMIT]
                   [--battery-tech {LITIO-IONE,GRAFENE}]
                   [--battery-capacity BATTERY_CAPACITY]
                   ...

BESS Optimization with PSO - Command Line Interface

optional arguments:
  -h, --help            show this help message and exit

File Paths (Obbligatori):
  --price-sell PRICE_SELL
                        Path assoluto file prezzi vendita (.xlsx)
  --price-buy PRICE_BUY
                        Path assoluto file prezzi acquisto (.xlsx)

File Paths (Opzionali):
  --pv-file PV_FILE     Path assoluto file produzione PV (.csv)
  --load-file LOAD_FILE
                        Path assoluto file carico (.xlsx)

...

Esempi:
  # Simulazione base con batteria litio-ione
  python script.py --price-sell data/vendita.xlsx --price-buy data/acquisto.xlsx

  # Con PV e carico
  python script.py --price-sell data/vendita.xlsx --price-buy data/acquisto.xlsx \
                   --pv-file data/pv.csv --pv-enabled \
                   --load-file data/load.xlsx --load-enabled
```

---

## Output Generati

### File Excel

**Percorso:** `{output-dir}/risultati_{tecnologia}_cli.xlsx`

**Colonne Principali:**
- `Data`, `€/MWh`, `Prezzo_Acquisto_€/MWh`
- `Azione_Trading_MW` → Azione batteria trading
- `Azione_Alpha_PV_Load` → Frazione PV al carico
- `Azione_P_Batt_Load_MW` → Potenza batteria per carico
- `SOC`, `SOH_%`, `Profitto_Euro`
- `PV_Production_MWh`, `PV_to_Battery_MWh`, `PV_to_Grid_MWh`, `PV_to_Load_MWh`
- `Load_Demand_MWh`, `Load_from_PV_MWh`, `Load_from_Battery_MWh`, `Load_from_Grid_MWh`
- `Grid_Withdrawal_MW`, `Grid_Injection_MW`, `POD_Violation`

---

###  File JSON

**Percorso:** `{output-dir}/simulation_results_{tecnologia}_v270_autonomous.json`

**Sezioni:**
```json
{
  "simulation_info": { ... },
  "battery_parameters": { ... },
  "battery_state": { ... },
  "trading_operations": { ... },
  "photovoltaic_system": { ... },
  "load_profile": {
    "autonomous_decisions": {
      "battery_served_load_percent": ...,
      "grid_served_load_percent": ...
    }
  },
  "point_of_delivery": {
    "pod_power_mw": ...,
    "total_violations": ...,
    "pv_curtailed_mwh": ...,
    "load_unserved_mwh": ...
  },
  "baseline_comparison": {
    "scenario_without_battery": { ... },
    "scenario_with_battery": { ... },
    "battery_benefits": { ... }
  }
}
```

---

### Grafici (se `--save-plots`)

**Cartella:** `visualization/`

**Grafici Base:**
- `overview_{tecnologia}.png` → Overview generale (SOC, profitto, azioni, SOH)
- `pod_tracking_{tecnologia}.png` → Monitoraggio POD violations
- `economic_analysis_{tecnologia}.png` → Analisi economica dettagliata

**Grafici PV (se `--pv-enabled`):**
- `pv_impact_comparison_{tecnologia}.png`
- `pv_impact_summary_{tecnologia}.png`
- `pv_analysis_{tecnologia}.png`
- `dettaglio_mensile_pv/01_Gennaio_dettaglio_pv_{tecnologia}.png` (x12)

**Grafici Carico (se `--load-enabled`):**
- `load_analysis_{tecnologia}.png`

---

## Output Console

### Esempio Output Durante Esecuzione

```
================================================================================
BESS OPTIMIZATION v3.8 - COMMAND LINE MODE
================================================================================

CONFIGURAZIONE:
  • Prezzi vendita:         data/Prezzo_Vendita.xlsx
  • Prezzi acquisto:        data/Prezzo_Acquisto.xlsx
  • POD Limit:              1.5 MW
  • Tecnologia batteria:    GRAFENE
  • Capacità:               1.0 MWh
  • Potenza:                1.0 MW
  • PV abilitato:           SÌ (1.0 kWp)
  • Carico abilitato:       SÌ
  • Parallelizzazione:      SÌ
  • Particelle PSO:         50
  • Iterazioni PSO:         100
================================================================================

✓ Prezzi vendita: 8760 righe, media 85.34 €/MWh
✓ Prezzi acquisto: 8760 righe, media 92.15 €/MWh
✓ PV: 8760 righe caricato
✓ Carico: 8760 righe, media 0.45 kW

PSO NUMBA JIT PARALLELIZZATO
   • Particelle: 50
   • Iterazioni: 100
   • Numba parallel: TRUE
   • Threads Numba: 16
   • NOTA: Prima iterazione lenta (compilazione JIT), poi 50-200x più veloce

================================================================================
SIMULAZIONE BESS v3.8 NUMBA - CON VINCOLO POD
================================================================================
Progresso: 0% - SOH: 100.00% - SOC: 50.0%
Progresso: 20% - SOH: 100.00% - SOC: 55.3%
Progresso: 40% - SOH: 100.00% - SOC: 62.1%
Progresso: 60% - SOH: 100.00% - SOC: 48.7%
Progresso: 80% - SOH: 100.00% - SOC: 71.2%

Simulazione completata!
Profitto finale: 1234.56 €

STATISTICHE POD:
  • Violazioni POD: 42 ore su 8760
  • PV curtailed: 1.23 MWh
  • Carico non servito: 0.05 MWh

✓ Excel salvato: results/risultati_grafene_cli.xlsx
✓ JSON salvato: results/simulation_results_grafene_v270_autonomous.json

Generazione grafici...
  ✓ Salvato: visualization/overview_grafene.png
  ✓ Salvato: visualization/pod_tracking_grafene.png
  ✓ Salvato: visualization/economic_analysis_grafene.png
  ...

================================================================================
✓ SIMULAZIONE COMPLETATA
================================================================================

RISULTATI FINALI - ANALISI COMPLETA
================================================================================

STATO FINALE BATTERIA:
  • Tecnologia:              GRAFENE
  • SOC finale:              65.3%
  • SOH finale:              100.00%
  • % carica da PV:          45.2%

RISULTATI ECONOMICI:
  • Profitto trading:        1,234.56 €
  • Profitto TOTALE:         1,234.56 €
  • Prezzo medio vendita:    85.34 €/MWh

STATISTICHE FOTOVOLTAICO:
  • Produzione totale:       3.45 MWh
  • A batteria:              1.23 MWh (35.7%)
  • A rete:                  1.89 MWh (54.8%)
  • A carico:                0.33 MWh (9.5%)

STATISTICHE CARICO:
  • Energia richiesta:       3.94 MWh
  • Da PV:                   0.33 MWh (8.4%)
  • Da batteria:             1.45 MWh (36.8%)
  • Da rete:                 2.16 MWh (54.8%)
  • Autosufficienza:         45.2%

CONFRONTO CON SCENARIO BASE (senza batteria):
  • Bilancio senza BESS:     -234.56 €
  • Bilancio con BESS:       +1,234.56 €
  • Beneficio batteria:      +1,469.12 € (+626.3%)
SISTEMA PROFITTEVOLE

TEMPO SIMULAZIONE:
  • Durata:                  312.5 secondi (5.2 minuti)
  • Ore simulate:            8760
  • Velocità:                28.0 ore/secondo
```

---

## estione Errori

### Errore 1: File Non Trovato

```bash
$ python bess_cli.py --price-sell missing.xlsx --price-buy data/acquisto.xlsx
```

**Output:**
```
usage: bess_cli.py [-h] ...
bess_cli.py: error: File prezzi vendita non trovato: missing.xlsx
```

**Soluzione:** Verifica path file e esistenza

---

### Errore 2: SOC Invalido

```bash
$ python bess_cli.py ... --lithium-soc-min 0.9 --lithium-soc-max 0.1
```

**Output:**
```
bess_cli.py: error: SOC litio invalido: min=0.9, max=0.1
```

**Soluzione:** `soc_min < soc_max`

---

### Errore 3: MACSE Capacity Troppo Grande

```bash
$ python bess_cli.py ... --battery-capacity 1.0 --macse-capacity 1.5
```

**Output:**
```
bess_cli.py: error: Capacità MACSE (1.5) > capacità batteria (1.0)
```

**Soluzione:** Riduci `--macse-capacity` o aumenta `--battery-capacity`

---

### Errore 4: File PV Abilitato ma Non Fornito

```bash
$ python bess_cli.py ... --pv-enabled
```

**Output:**
```
Errore caricamento PV: [Errno 2] No such file or directory
```

**Soluzione:** Aggiungi `--pv-file data/pv.csv`

---

## Ottimizzazione Performance

### Esecuzione Veloce

```bash
python bess_cli.py \
  --price-sell data/vendita.xlsx \
  --price-buy data/acquisto.xlsx \
  --n-particles 30 \
  --n-iterations 50 \
  --n-cores -2
```

**Tempo:** ~2 minuti  
**Uso:** Test rapidi

---

### Esecuzione Bilanciata

```bash
python bess_cli.py \
  --price-sell data/vendita.xlsx \
  --price-buy data/acquisto.xlsx \
  --n-particles 50 \
  --n-iterations 100 \
  --n-cores -2
```

**Tempo:** ~5 minuti (default)  
**Uso:** Uso quotidiano

---

### Esecuzione Accurata

```bash
python bess_cli.py \
  --price-sell data/vendita.xlsx \
  --price-buy data/acquisto.xlsx \
  --pv-file data/pv.csv --pv-enabled \
  --load-file data/load.xlsx --load-enabled \
  --n-particles 120 \
  --n-iterations 300 \
  --n-cores -2 \
  --save-plots
```

**Tempo:** ~20 minuti  
**Uso:** Risultati finali, report

---

## Interpretazione Risultati

### Sistema Profittevole

**Indicatori:**
- `Beneficio batteria > 0`
- `Autosufficienza aumentata`
- `POD violations < 5%`

**Esempio Output:**
```
CONFRONTO CON SCENARIO BASE:
  • Beneficio batteria:      +1,469.12 € (+626.3%)
  SISTEMA PROFITTEVOLE
```

---

### Sistema Non Profittevole

**Indicatori:**
- `Beneficio batteria < 0`
- `Spread prezzi troppo basso`
- `POD violations > 20%`

**Esempio Output:**
```
CONFRONTO CON SCENARIO BASE:
  • Beneficio batteria:      -234.56 € (-15.2%)
  SISTEMA NON PROFITTEVOLE
```

**Azioni Correttive:**
1. Aumenta `--battery-capacity`
2. Aumenta `--pod-limit`
3. Verifica spread prezzi
4. Abilita PV/Carico

---

### POD Violations Eccessive

**Sintomo:**
```
STATISTICHE POD:
  • Violazioni POD: 1842 ore su 8760 (21.0%)
    ATTENZIONE: Considera aumentare POD limit!
```

**Soluzione:**
```bash
--pod-limit 2.0  # Aumenta da 1.5 a 2.0
```

---

### PV Curtailed Alto

**Sintomo:**
```
STATISTICHE FOTOVOLTAICO:
  • Curtailed (POD):         2.45 MWh (15.2%) ⚠️
```

**Soluzioni:**
1. Aumenta `--pod-limit`
2. Aumenta `--battery-capacity`
3. Abilita carico utente con `--load-enabled`

---

### Carico Non Servito

**Sintomo:**
```
STATISTICHE CARICO:
  • Carico non servito:      0.85 MWh ⚠️
```

**Soluzioni (CRITICHE):**
1. **Aumenta `--pod-limit` immediatamente**
2. Aumenta `--battery-power`
3. Verifica dati carico

---

## Formati File Input

### Prezzi Energia (Excel)

**File:** `Prezzo_Vendita.xlsx`, `Prezzo_Acquisto.xlsx`

**Formato:**
| Data | €/MWh |
|------|-------|
| 01/01/2024 00:00 | 85.34 |
| 01/01/2024 01:00 | 78.21 |
| ... | ... |

**Requisiti:**
- Colonna `€/MWh` obbligatoria
- Formato numerico o testo con virgola
- 8760 righe (anno completo)

---

###  Produzione PV (CSV)

**File:** `PV_formattato.csv`

**Formato:**
```csv
Data;P
01/01/2024 00:00;0.0
01/01/2024 01:00;0.0
01/01/2024 08:00;150.5
01/01/2024 12:00;890.2
...
```

**Requisiti:**
- Separatore: `;` (punto e virgola)
- Colonna `P` in kW
- 8760 righe

---

### ⚡ Carico Utente (Excel)

**File:** `Consumo.xlsx`

**Formato:**
| Data | value |
|------|-------|
| 01/01/2024 00:00 | 450.3 |
| 01/01/2024 01:00 | 380.1 |
| ... | ... |

**Requisiti:**
- Colonna `value` (o `load`, `power`) in kW
- 8760 righe
- Primo sheet Excel

---

## Script Batch per Automazione

### Windows (`.bat`)

```batch
@echo off
REM BESS Optimization - Esecuzione Automatica
REM Lorenzo Giannuzzo - Politecnico di Torino

echo ========================================
echo BESS Optimization Automation
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
echo Simulazione completata!
echo ========================================
pause
```

**Esecuzione:** Doppio click su file `.bat`

---

### Linux/Mac (`.sh`)

```bash
#!/bin/bash
# BESS Optimization - Esecuzione Automatica
# Lorenzo Giannuzzo - Politecnico di Torino

echo "========================================"
echo "BESS Optimization Automation"
echo "========================================"

python3 bess_cli.py \
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
  --output-dir results_$(date +%Y%m%d)

echo ""
echo "========================================"
echo "Simulazione completata!"
echo "========================================"
```

**Esecuzione:**
```bash
chmod +x run_simulation.sh
./run_simulation.sh
```

---

## 🧪 Testing e Debug

### Test Configurazione (Dry Run)

```bash
python bess_cli.py \
  --price-sell data/vendita.xlsx \
  --price-buy data/acquisto.xlsx \
  --n-particles 10 \
  --n-iterations 10 \
  --no-parallel
```

**Tempo:** <1 minuto  
**Uso:** Verifica setup

---

### Debug Mode

```bash
python bess_cli.py \
  --price-sell data/vendita.xlsx \
  --price-buy data/acquisto.xlsx \
  --no-parallel \
  --n-particles 20 \
  --n-iterations 20
```

**Uso:** Debugging PSO, no parallelizzazione

---

## Supporto

**Autore:** Lorenzo Giannuzzo  
**Email:** lorenzo.giannuzzo@polito.it
**Affiliazione:** Politecnico di Torino - DENERG - Energy Center Lab  

---

## Collegamenti Utili

- **Documentazione PSO:** https://en.wikipedia.org/wiki/Particle_swarm_optimization
- **MACSE (Mercato Servizi Ancillari):** https://www.mercatoelettrico.org/
- **Numba JIT:** https://numba.pydata.org/
- **Politecnico Torino DENERG:** http://www.denerg.polito.it/

---

**Ultima Modifica:** 28 Novembre 2025  
**Versione Guida:** 2.0 CLI  
**Compatibilità:** Python 3.8+
