# Heterogeneous-Sources-of-Productivity-Growth-along-DEA-Based-Hourly-Frontiers

This is a repository for releasing Python code to reproduce the estimation results from the paper "Heterogeneous Sources of Productivity Growth along DEA-Based Hourly Frontiers: A Path-Symmetrized Malmquist Decomposition of Physical and Human Capital Accumulation". It estimates hourly productivity frontiers using DEA (Data Envelopment Analysis).

## 1. Overview

- **Repository URL**

https://github.com/Kandkide/Heterogeneous-Sources-of-Productivity-Growth-along-DEA-Based-Hourly-Frontiers.git

- **License**

  MIT License

- **Recommended Operating Environment**

  - **OS**: Ubuntu 24.04.4 LTS

  - **Language**: Python 3.12

  - **IDE**: VS Code (Visual Studio Code)

---

## 2. Setup

 - Create the folder you want to clone.

 - Open the folder in VS Code.

 - Execute all commands in the following steps within the VS Code terminal.

### 2.1 Cloning the Repository

Obtain the repository using the following command.

```git clone https://github.com/Kandkide/Estimating-Hourly-Productivity-Frontiers-by-DEA.git .```

### 2.2 Creating a Virtual Environment

Execute the following command to initialize the Python 3.12 virtual environment.

```./init_python_venv.sh```

### 2.3 Data Used

This project uses PWT (Penn World Table) data.

- **Data Source**

  https://www.rug.nl/ggdc/productivity/pwt/

- **Target Files**

  - pwt56_forweb.xls

  - pwt110.xlsx

- **Location for Data Files**

  - ./data/

  - Create the folder yourself and place the files there.
---

## 3. Execution Method (Usage)

### 3.1 Activate the Python Virtual Environment

- **Command**

  ```source ./startup-execution-commands.sh```

### 3.2 Main Python Scripts

- **For KR (Kumar & Russell) Reproduction**

  python-scripts/minimum_code_for_paper_2026_replicate_KR.py

- **For Standard Execution**

  python-scripts/minimum_code_for_paper_2026.py

### 3.3 VS Code shortcut to run Python scripts

- **Start Debugging**: F5

- **Run Without Debugging**: Ctrl + F5

---