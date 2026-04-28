# Heterogeneous-Sources-of-Productivity-Growth-along-DEA-Based-Hourly-Frontiers

これは、論文「Heterogeneous Sources of Productivity Growth along DEA-Based Hourly Frontiers: A Path-Symmetrized Malmquist Decomposition of Physical and Human Capital Accumulation」の推定結果を再現するためのPythonコードを公開するリポジトリです。このリポジトリは、DEA（データ包絡分析）を用いて時間別生産性フロンティアを推定します。

## 1. 概要 (Overview)

- **リポジトリ URL**  
  
https://github.com/Kandkide/Heterogeneous-Sources-of-Productivity-Growth-along-DEA-Based-Hourly-Frontiers.git

- **ライセンス**  
  MIT License

- **推奨動作環境**
  - **OS**: Ubuntu 24.04.4 LTS
  - **Language**: Python 3.12
  - **IDE**: VS Code (Visual Studio Code)

---

## 2. セットアップ (Setup)

- クローンしたいフォルダを作成。
- VS Codeでフォルダを開く。
- 以下の手順中のコマンドは、すべてVS Codeのターミナル中で実行すること。

### 2.1 リポジトリのクローン

以下のコマンドでリポジトリを取得します。

```git clone https://github.com/Kandkide/Estimating-Hourly-Productivity-Frontiers-by-DEA.git .```

### 2.2 仮想環境の構築

以下のコマンドを実行して Python 3.12 の仮想環境を初期化します。

```./init_python_venv.sh```

### 2.3 使用データについて

本プロジェクトでは PWT (Penn World Table) のデータを使用します。

- **データ取得先**  
  https://www.rug.nl/ggdc/productivity/pwt/

- **対象ファイル**
  - pwt56_forweb.xls
  - pwt110.xlsx

- **データファイルを置く場所**
  - ./data/
  - フォルダは自分で作ってファイルを配置してください
---

## 3. 実行方法 (Usage)

### 3.1 python仮想環境をアクティベイト

- **コマンド**  
  ```source ./startup-execution-commands.sh```

### 3.2 主要な Python スクリプト

- **KR (Kumar & Russell) 再現用**  
  python-scripts/minimum_code_for_paper_2026_replicate_KR.py

- **標準実行用**  
  python-scripts/minimum_code_for_paper_2026.py

### 3.3 pythonスクリプトを実行するVS Codeのショートカット

- **デバッグ開始**: F5
- **デバッグなしで実行**: Ctrl + F5

---

