import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import warnings
import re

# CONFIGURAZIONE PAGINA
st.set_page_config(page_title="⚽ Dashboard Tattica Pro", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# TITOLO
st.title("⚽ Dashboard Analisi Calcio Pro")
st.markdown("**Analisi Tattica, Ritmo Gol & Previsioni Matematiche**")
st.divider()

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================
with st.sidebar:
    st.header("📂 Gestione Dati")
    
    # Opzione 1: Carica dal PC
    uploaded_file = st.file_uploader("Carica il file (CSV o Excel)", type=['csv', 'xlsx'])
    
    # Opzione 2: Usa file di default (se presente su GitHub)
    # Se hai caricato 'eng_tot_1.csv' su GitHub, puoi usarlo di default se non carichi nulla
    default_file = 'eng_tot_1.csv'
    use_default = False
    
    try:
        # Prova ad aprire il file di default solo per vedere se esiste
        with open(default_file):
            pass
        if uploaded_file is None:
            st.info(f"Uso il file predefinito: {default_file}")
            use_default = True
    except:
        pass

if uploaded_file is None and not use_default:
    st.warning("👈 Carica un file per iniziare.")
    st.stop()

@st.cache_data
def load_data(file_obj, is_path=False):
    try:
        # Rilevamento separatore (solo per CSV)
        sep = ',' # Default
        if is_path:
             with open(file_obj, 'r', encoding='latin1', errors='replace') as f:
                line = f.readline()
                sep = ';' if line.count(';') > line.count(',') else ','
             file_source = file_obj
        else:
             line = file_obj.readline().decode('latin1')
             file_obj.seek(0)
             sep = ';' if line.count(';') > line.count(',') else ','
             file_source = file_obj

        # Caricamento
        try:
            df = pd.read_csv(file_source, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False, header=None)
        except:
            # Fallback Excel
            df = pd.read_excel(file_source, header=None)

        # Header e Pulizia
        header = df.iloc[0].astype(str).str.strip().str.upper().tolist()
        
        # Nomi univoci
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

        # Mappatura Colonne
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY'],
            'CASA': ['CASA', 'HOME', 'TEAM1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TEAM2']
        }
        
        for target, candidates in col_map.items():
            if target not in df.columns:
                for candidate in candidates:
                    found = next((c for c in df.columns if c == candidate), None)
                    if found:
                        df.rename(columns={found: target}, inplace=True)
                        break
        
        # Pulizia Dati
        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        
        # ID Lega
        if 'PAESE' in df.columns:
            df['ID_LEGA'] = df['PAESE'] + " - " + df['LEGA']
        else:
            df['ID_LEGA'] = df['LEGA']

        return df

    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        return pd.DataFrame()

# Carica i dati
if uploaded_file:
    df = load_data(uploaded_file)
elif use_default:
    df = load_data(default_file, is_path=True)

if df.empty:
    st.error("Errore: Il file è vuoto o non leggibile.")
    st.stop()

st.sidebar.success(f"✅ Caricato: {len(df)} righe")

# ==========================================
# 2. SELEZIONE DATI
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    leghe = sorted(df['ID_LEGA'].unique())
    sel_lega = st.selectbox("🏆 Campionato", leghe)

df_league = df[df['ID_LEGA'] == sel_lega].copy()
teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())

with col2:
    sel_home = st.selectbox("🏠 Squadra Casa", teams, index=0)

with col3:
    idx_away = 1 if len(teams) > 1 else 0
    sel_away = st.selectbox("✈️ Squadra Ospite", teams, index=idx_away)

# ==========================================
# 3. AVVIO ANALISI
# ==========================================
if st.button("🚀 AVVIA ANALISI MATCH", type="primary"):
    st.divider()
    st.subheader(f"⚔️ Analisi: {sel_home} vs {sel_away}")

    intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    
    def get_minutes(val):
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

    # Setup Colonne
    c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else 'GOALMINCASA'
    c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else 'GOALMINOSPITE'

    # Accumulatori
    goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    match_h, match_a = 0, 0
    times_h, times_a, times_league = [], [], []
    
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        # Media Lega
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))

        # Heatmap Match
        if h in stats_match:
            for m in min_h:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[h]['F'][intervals[idx]] += 1
            for m in min_a:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[h]['S'][intervals[idx]] += 1
        
        if a in stats_match:
            for m in min_a:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[a]['F'][intervals[idx]] += 1
            for m in min_h:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[a]['S'][intervals[idx]] += 1

        # Stats Medie
        if h == sel_home:
            match_h += 1
            goals_h['FT'] += len(min_h)
            goals_h['HT'] += len([x for x in min_h if x <= 45])
            goals_h['S_FT'] += len(min_a)
            goals_h['S_HT'] += len([x for x in min_a if x <= 45])
            if min_h: times_h.append(min(min_h))
        
        if a == sel_away:
            match_a += 1
            goals_a['FT'] += len(min_a)
            goals_a['HT'] += len([x for x in min_a if x <= 45])
            goals_a['S_FT'] += len(min_h)
            goals_a['S_HT'] += len([x for x in min_h if x <= 45])
            if min_a: times_a.append(min(min_a))

    # --- OUTPUT STATISTICHE ---
    def safe_div(n, d): return n / d if d > 0 else 0

    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_conc_ft = safe_div(goals_h['S_FT'], match_h)
    avg_h_conc_ht = safe_div(goals_h['S_HT'], match_h)
