# @title 💎 DASHBOARD V28: FIX VARIABILI E LOGICA
# @markdown 1. Assicurati che il file sia caricato a sinistra.
# @markdown 2. Premi Play ▶️.

import sys
import subprocess
import os
import warnings

# Installazione silenziosa
try:
    import lifelines
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lifelines"])

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from lifelines import KaplanMeierFitter
from scipy.stats import poisson

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. TROVA IL FILE
# ==========================================
files_found = [f for f in os.listdir() if f.endswith('.csv') or f.endswith('.xlsx')]
if not files_found:
    files_found = ['Nessun file']

global_df = pd.DataFrame()

# ==========================================
# 2. INTERFACCIA
# ==========================================
style = {'description_width': 'initial'}
layout = widgets.Layout(width='98%')

w_file = widgets.Dropdown(options=files_found, description='📂 1. FILE:', style=style, layout=layout)
btn_load = widgets.Button(description="🔄 CARICA DATI", button_style='info', layout=layout)

w_paese = widgets.Dropdown(description='🌍 Paese:', style=style, layout=layout, disabled=True)
w_lega = widgets.Dropdown(description='🏆 Lega:', style=style, layout=layout, disabled=True)
w_home = widgets.Dropdown(description='🏠 Casa:', style=style, layout=layout, disabled=True)
w_away = widgets.Dropdown(description='✈️ Ospite:', style=style, layout=layout, disabled=True)

btn_run = widgets.Button(description="🚀 AVVIA ANALISI", button_style='success', layout=layout, disabled=True)

out_log = widgets.Output()
out_res = widgets.Output()

# ==========================================
# 3. LOGICA
# ==========================================
def load_data(b):
    global global_df
    with out_log:
        clear_output()
        fname = w_file.value
        if fname == 'Nessun file':
            print("❌ Carica un file prima!")
            return

        print(f"⏳ Leggo {fname}...")
        try:
            with open(fname, 'r', encoding='latin1', errors='replace') as f:
                line = f.readline()
                sep = ';' if line.count(';') > line.count(',') else ','

            df = pd.read_csv(fname, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False, header=None)
            
            header = df.iloc[0].astype(str).str.strip().str.upper().tolist()
            seen = {}
            unique_header = []
            for col in header:
                if col in seen:
                    seen[col] += 1
                    unique_header.append(f"{col}.{seen[col]}")
                else:
                    seen[col] = 0
                    unique_header.append(col)
            
            df = df.iloc[1:].copy()
            df.columns = unique_header
            
            col_map = {
                'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA'],
                'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE'],
                'LEGA': ['LEGA', 'LEAGUE'],
                'PAESE': ['PAESE', 'COUNTRY'],
                'CASA': ['CASA', 'HOME'],
                'OSPITE': ['OSPITE', 'AWAY']
            }
            for target, candidates in col_map.items():
                if target not in df.columns:
                    for cand in candidates:
                        found = next((c for c in df.columns if c == cand), None)
                        if found:
                            df.rename(columns={found: target}, inplace=True)
                            break
            
            for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
                if c in df.columns: df[c] = df[c].astype(str).str.strip()

            global_df = df
            print(f"✅ Caricato! Righe: {len(df)}")
            
            if 'PAESE' in df.columns:
                paesi = sorted(df['PAESE'].unique())
                w_paese.options = paesi
                if paesi: w_paese.value = paesi[0]
                
                w_paese.disabled = False
                w_lega.disabled = False
                w_home.disabled = False
                w_away.disabled = False
                btn_run.disabled = False
                
                update_leghe()
            else:
                print("❌ Colonne PAESE/LEGA non trovate.")

        except Exception as e:
            print(f"❌ Errore: {e}")

def update_leghe(*args):
    if not global_df.empty and w_paese.value:
        leghe = sorted(global_df[global_df['PAESE'] == w_paese.value]['LEGA'].unique())
        w_lega.options = leghe
        if leghe: w_lega.value = leghe[0]
        update_squadre()

def update_squadre(*args):
    if not global_df.empty and w_paese.value and w_lega.value:
        mask = (global_df['PAESE'] == w_paese.value) & (global_df['LEGA'] == w_lega.value)
        teams = sorted(pd.concat([global_df[mask]['CASA'], global_df[mask]['OSPITE']]).unique())
        w_home.options = teams
        w_away.options = teams
        if len(teams) > 1:
            w_home.value = teams[0]
            w_away.value = teams[1]

w_paese.observe(update_leghe, 'value')
w_lega.observe(update_squadre, 'value')
btn_load.on_click(load_data)

def run_analysis(b):
    with out_res:
        clear_output()
        
        # Recupera valori
        p, l, h, a = w_paese.value, w_lega.value, w_home.value, w_away.value
        
        if not p or not l or not h or not a:
            print("⚠️ Seleziona tutti i campi.")
            return

        print(f"⚙️ Analisi: {h} vs {a} ({l})...")
        
        # Filtra (Definisce df_sub QUI, così esiste sempre)
        df_sub = global_df[(global_df['PAESE'] == p) & (global_df['LEGA'] == l)].copy()
        
        if df_sub.empty:
            print("❌ Nessuna partita trovata con questi filtri.")
            return

        intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
        
        def get_mins(val):
            if pd.isna(val): return []
            s = str(val).replace(',', '.').replace(';', ' ').replace('"', '')
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            res = []
            for x in nums:
                try:
                    n = int(float(x))
                    if 0 <= n <= 130: res.append(n)
                except: pass
            return res
        
        goals_h = {'FT':0, 'HT':0, 'S_FT':0, 'S_HT':0}
        goals_a = {'FT':0, 'HT':0, 'S_FT':0, 'S_HT':0}
        match_h, match_a = 0, 0
        times_h, times_a, times_league = [], [], []
        
        # Heatmap Data
        hm_f = {h: {i:0 for i in intervals}, a: {i:0 for i in intervals}}
        hm_s = {h: {i:0 for i in intervals}, a: {i:0 for i in intervals}}
        
        c_mh = 'GOALMINH' if 'GOALMINH' in df_sub.columns else df_sub.columns[21] 
        c_ma = 'GOALMINA' if 'GOALMINA' in df_sub.columns else df_sub.columns[22]

        for _, row in df_sub.iterrows():
            th, ta = row['CASA'], row['OSPITE']
            mh = get_mins(row.get(c_mh))
            ma = get_mins(row.get(c_ma))
            
            # Media Lega
            if mh: times_league.append(min(mh))
            if ma: times_league.append(min(ma))

            # Heatmaps
            if th in [h, a]:
                target = th
                for m in mh:
                    idx = min(5, (m-1)//15)
                    if m > 45 and m <= 60 and idx < 3: idx = 3
                    hm_f[target][intervals[idx]] += 1
                for m in ma:
                    idx = min(5, (m-1)//15)
                    if m > 45 and m <= 60 and idx < 3: idx = 3
                    hm_s[target][intervals[idx]] += 1

            if ta in [h, a]:
