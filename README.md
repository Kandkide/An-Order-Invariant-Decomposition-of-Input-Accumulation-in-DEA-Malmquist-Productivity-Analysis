# An-Order-Invariant-Decomposition-of-Input-Accumulation-in-DEA-Malmquist-Productivity-Analysis

This repository contains replication code for **“An Order-Invariant Decomposition of Input Accumulation in DEA–Malmquist Productivity Analysis.”**
It reproduces the tabulated outputs underlying **Tables 1–2** reported in the paper.

## 1. Overview

- **Repository URL**
  - https://github.com/Kandkide/An-Order-Invariant-Decomposition-of-Input-Accumulation-in-DEA-Malmquist-Productivity-Analysis.git

- **License**
  - MIT License

- **Recommended Operating Environment**
  - **OS**: Ubuntu 24.04.4 LTS
  - **Language**: Python 3.12
  - **IDE**: VS Code (Visual Studio Code) *(optional)*

---

## 2. Setup

- Create (or choose) a folder to clone this repository.
- Open the folder in VS Code (optional).
- Execute all commands below in a terminal (VS Code terminal or any shell).

### 2.1 Cloning the Repository

```bash
git clone https://github.com/Kandkide/An-Order-Invariant-Decomposition-of-Input-Accumulation-in-DEA-Malmquist-Productivity-Analysis.git .  
```

### 2.2 Creating a Virtual Environment

Initialize a Python 3.12 virtual environment and install required packages:

```bash
./init_python_venv.sh
```

### 2.3 Data Used (PWT 11.0)

This project uses **Penn World Table (PWT) 11.0** data.

- **Data Source**
  - PWT website: https://www.rug.nl/ggdc/productivity/pwt/
  - Dataverse (recommended, DOI): https://doi.org/10.34894/FABVLR

- **Target File**
  - `pwt110.xlsx` (PWT 11.0)

- **Location for Data File**
  - Create the folder below and place the file there:
    - `./data/`
    - `./data/pwt110.xlsx`

> Note: This repository does **not** redistribute the PWT data file.
> Please download it from the sources above.

---

## 3. Execution Method (Usage)

### 3.1 Activate the Python Virtual Environment

```bash
source ./startup-execution-commands.sh
```

### 3.2 Main Python Script

- `minimum_code_for_paper_2026_JPA.py`

### 3.3 Run from CLI (recommended for replication)

```bash
source ./startup-execution-commands.sh
python minimum_code_for_paper_2026_JPA.py
```

### 3.4 Outputs (what to expect)

`minimum_code_for_paper_2026_JPA.py` does **not** write output files by default.
Instead, it prints tabulated results to **standard output** (stdout) using `tabulate`.

Expected terminal output:
- A header line `Table 1`, followed by a tabulated table (one-input case)
- A header line `Table 2`, followed by a tabulated table (two-input case)

To save the printed tables for replication records, redirect stdout to a file:

```bash
source ./startup-execution-commands.sh
python minimum_code_for_paper_2026_JPA.py | tee replication_tables_2000_2010.txt
```
### 3.5 VS Code shortcut to run Python scripts (optional)

- **Start Debugging**: F5
- **Run Without Debugging**: Ctrl + F5

