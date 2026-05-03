import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Drive Analytics", page_icon="🏎️", layout="wide")

try:
    from data_loader import CarDataStandardizer
except ImportError:
    st.error("Error inicial")
    st.stop()

# --- CSS 
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&display=swap');
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; text-transform: uppercase; color: #FFF; }
    div[data-testid="stMetric"] { background-color: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 15px; }
    div[data-testid="stMetricValue"] div { color: #38BDF8 !important; font-family: 'Rajdhani'; }
    .eco-warn { padding: 15px; background: rgba(239, 68, 68, 0.15); border-left: 4px solid #EF4444; margin-bottom: 10px; border-radius: 4px; font-size: 0.95rem;}
    .eco-good { padding: 15px; background: rgba(34, 197, 94, 0.15); border-left: 4px solid #22C55E; margin-bottom: 10px; border-radius: 4px; font-size: 0.95rem;}
    .health-tip { padding: 10px; background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3B82F6; margin-bottom: 5px; font-size: 0.9rem; }
    .ia-banner { border-left: 4px solid #10B981; padding: 15px; margin-bottom: 20px; background: rgba(16, 185, 129, 0.1); border-radius: 4px; }
    .ia-banner.warning { border-color: #F59E0B; background: rgba(245, 158, 11, 0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- CACHE ---
def invalidate_cache():
    if 'processed_data' in st.session_state: del st.session_state['processed_data']

@st.cache_resource
def load_ai_brain():
    path = 'robust_model.pkl'
    return joblib.load(path) if os.path.exists(path) else None

rf_model = load_ai_brain()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ANÁLISIS DE CONDUCCIÓN v5.0")
    uploaded_file = st.file_uploader("DATA INGEST", type=["csv"], on_change=invalidate_cache)

# --- PROCESAMIENTO ---
if uploaded_file and 'processed_data' not in st.session_state:
    with st.spinner("SINCRONIZANDO SENSORES Y DIAGNOSTICANDO..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            loader = CarDataStandardizer()
            df = loader.load_single_file(tmp_path)
            st.session_state['processed_data'] = df
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- DASHBOARD ---
if 'processed_data' in st.session_state and st.session_state['processed_data'] is not None:
    my_df = st.session_state['processed_data']
    
    # 0. DETECCIÓN DE MODO
    has_rpm = (my_df['rpm'] > 300).sum() > 20
    if has_rpm:
        thresh, mode, col_mode = (0.50, "MODO COMPLETO (OBD)", "#10B981")
        cls = ""
    else:
        thresh, mode, col_mode = (0.35, "MODO GPS (CINEMÁTICA)", "#F59E0B")
        cls = "warning"
        my_df['rpm'] = -1

    st.markdown(f"<div class='ia-banner {cls}' style='border-color:{col_mode};'><h3 style='margin:0; color:{col_mode};'>🧠 {mode}</h3><p style='margin:0; color:#AAA;'>Umbral IA: {thresh} | Lógica Experta Activa</p></div>", unsafe_allow_html=True)

    # 1. INFERENCIA
    feats = ['speed_kmh', 'rpm', 'accel_ms2', 'jerk_ms3', 'accel_std_3s', 'speed_std_3s', 'power_index', 'risk_index']
    for f in feats: 
        if f not in my_df.columns: my_df[f] = 0
    
    probs = rf_model.predict_proba(my_df[feats])[:, 0] if rf_model else np.zeros(len(my_df))
    my_df['score'] = pd.Series(probs).rolling(75, center=True).mean().fillna(pd.Series(probs))
    
    # [REGLA EXPERTA]
    SPEED_LIMIT = 140
    over_limit = my_df['speed_kmh'] > SPEED_LIMIT
    if over_limit.any():
        my_df.loc[over_limit, 'score'] = 1.0
        st.toast(f"🚨 Exceso Velocidad > {SPEED_LIMIT} km/h", icon="⚡")

    my_df['is_aggressive'] = my_df['score'] > thresh
    my_df['time_min'] = (my_df['timestamp_sec'] - my_df['timestamp_sec'].min()) / 60.0

    # 2. KPIS GLOBALES
    pct = my_df['is_aggressive'].mean() * 100
    verdict = "AGRESIVO" if pct > 40 else "DINÁMICO" if pct > 15 else "EFICIENTE"
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DURACIÓN", f"{my_df['time_min'].max():.1f} min")
    c2.metric("VEL. MAX", f"{my_df['speed_kmh'].max():.0f} km/h")
    c3.metric("FUERZA G", f"{my_df['accel_ms2'].max()/9.8:.2f} G")
    c4.metric("VEREDICTO IA", verdict, f"{pct:.1f}% Riesgo", delta_color="inverse" if pct>15 else "normal")

    # 3. PESTAÑAS
    t1, t2, t3, t4 = st.tabs(["📈 DINÁMICA", "🗺️ MAPA", "🍃 ECO-COACH", "🩺 SALUD"])

    with t1:
        # Gráfica Interactiva Plotly
        fig_dyn = go.Figure()
        fig_dyn.add_trace(go.Scatter(x=my_df['time_min'], y=my_df['speed_kmh'], mode='lines', name='Velocidad (km/h)', line=dict(color='#38BDF8', width=2)))
        agg_events = my_df[my_df['is_aggressive']]
        if not agg_events.empty:
            fig_dyn.add_trace(go.Scatter(x=agg_events['time_min'], y=agg_events['speed_kmh'], mode='markers', name='Evento Agresivo', marker=dict(color='#EF4444', size=6, symbol='circle')))

        fig_dyn.update_layout(title="Perfil de Velocidad y Detección de Riesgos", template="plotly_dark", height=400, hovermode="x unified", margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_dyn, use_container_width=True)

    with t2:
        if 'latitude' in my_df.columns and my_df['latitude'].notna().any():
            m_df = my_df.dropna(subset=['latitude', 'longitude'])
            if not m_df.empty:
                m = folium.Map(location=[m_df['latitude'].mean(), m_df['longitude'].mean()], zoom_start=13, tiles="cartodbdark_matter")
                folium.PolyLine(list(zip(m_df['latitude'], m_df['longitude'])), color="#38BDF8", weight=3).add_to(m)
                for _, r in m_df[m_df['is_aggressive']].iterrows():
                     folium.CircleMarker([r['latitude'], r['longitude']], radius=4, color='#EF4444', fill=True, fill_opacity=0.8).add_to(m)
                folium_static(m)
            else: st.warning("GPS sin datos.")
        else: st.info("Sin GPS.")

    with t3: # ECO-COACH EXPERTO
        total_fuel = (my_df['fuel_rate_lh'] * 0.068 / 3600).sum()
        idle_mask = (my_df['speed_kmh'] < 2) & (my_df['rpm'] > 0)
        idle_waste = (my_df.loc[idle_mask, 'fuel_rate_lh'] * 0.068 / 3600).sum()
        
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("CONSUMO TOTAL", f"{total_fuel:.2f} L")
        ec2.metric("DESPERDICIO RALENTÍ", f"{idle_waste:.2f} L", delta_color="inverse")
        ec3.metric("EFICIENCIA", f"{my_df['fuel_l100km'][my_df['speed_kmh']>10].mean():.1f} L/100km")
        
        st.divider()
        col_tips, col_chart = st.columns([1, 2])
        
        with col_tips:
            st.markdown("#### 💡 CONSEJOS")
            findings = []
            
            # Análisis de Marchas
            if has_rpm and 'engine_load' in my_df.columns:
                bad_gear = ((my_df['rpm'] > 2800) & (my_df['engine_load'] < 40) & (my_df['speed_kmh'] > 30)).sum()
                if bad_gear > 30: 
                    findings.append("⚠️ Gestión de Transmisión: Estás circulando en marchas cortas con poca carga. Sube de marcha para reducir la fricción interna del motor y mejorar el BSFC.")
            
            # Análisis de Inercia
            hard_brakes = (my_df['accel_ms2'] < -2.5).sum()
            if hard_brakes > 5:
                findings.append(f"🛑 Gestión de Inercia: Se detectaron {hard_brakes} frenadas fuertes. Estás disipando energía cinética en calor (frenos) en lugar de aprovechar el *Fuel Cut-Off* (freno motor). Anticipa más.")
            
            # Start/Stop
            if idle_waste > 0.10:
                findings.append(f"🔥 Eficiencia Estática: Has gastado {idle_waste:.2f}L completamente parado. En paradas >30s, apagar el motor ahorra más combustible que el coste de arranque.")

            # Aerodinámica
            if (my_df['speed_kmh'] > 120).sum() > 60:
                findings.append("💨 Resistencia Aerodinámica: A partir de 120 km/h, la potencia necesaria para vencer el aire crece al cubo. Reducir a 110 km/h puede ahorrarte un 15%.")

            if findings:
                for f in findings: st.markdown(f"<div class='eco-warn'>{f}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='eco-good'>✅ Conducción Óptimao.</div>", unsafe_allow_html=True)

        with col_chart:
            fig = go.Figure(go.Scatter(x=my_df['speed_kmh'], y=my_df['fuel_l100km'], mode='markers', marker=dict(color=my_df['accel_ms2'], colorscale='RdYlGn_r', showscale=True)))
            fig.update_layout(title="Mapa de Eficiencia (Color=Aceleración)", template="plotly_dark", height=350, yaxis_range=[0,30], margin=dict(l=0,r=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with t4: # SALUD PROFUNDA
        st.markdown("#### 🩺 DIAGNÓSTICO DE SISTEMAS")
        
        # Extracción segura
        volt = my_df['voltage'].mean() if 'voltage' in my_df.columns else 0
        temp = my_df['coolant_temp'].max() if 'coolant_temp' in my_df.columns else 0
        stft = my_df['stft'].mean() if 'stft' in my_df.columns else 0
        load_avg = my_df['engine_load'].mean() if 'engine_load' in my_df.columns else 0
        duration_min = my_df['time_min'].max()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("BATERÍA", f"{volt:.1f} V", "Alternador OK" if volt > 13.5 else "Bajo" if volt > 1 else "N/A")
        k2.metric("TEMP MAX", f"{temp:.0f} °C", "Normal" if temp < 105 else "Alerta", delta_color="inverse" if temp>105 else "normal")
        k3.metric("MEZCLA (STFT)", f"{stft:.1f} %", "Fuga Vacío" if stft>10 else "Goteo" if stft<-10 else "Ideal")
        k4.metric("CARGA MEDIA", f"{load_avg:.1f} %", f"Pico: {my_df['engine_load'].max() if 'engine_load' in my_df.columns else 0:.0f}%")

        st.divider()

        c_diag, c_graph = st.columns([1, 2])
        
        with c_diag:
            st.markdown("##### 🛠️ ALERTAS DE TALLER")
            health_issues = []
            
            # Lógica Experta de Salud
            if volt > 0 and volt < 13.2:
                health_issues.append("⚡ Sistema Eléctrico: Voltaje bajo (<13.2V) en marcha. Posible fallo del regulador del alternador o correa de accesorios destensada.")
            
            if temp > 0:
                if temp < 75 and duration_min > 15:
                    health_issues.append("🌡️ Termostato: El motor no alcanza temperatura de servicio tras 15 min. Posible termostato atascado en abierto (aumenta consumo y desgaste).")
                elif temp > 105:
                    health_issues.append("🔥 Refrigeración: Temperatura crítica (>105°C). Verificar nivel de refrigerante y funcionamiento del electroventilador.")
            
            if stft != 0:
                if stft > 10:
                    health_issues.append("💨 Admisión: STFT positivo (>10%). La ECU está inyectando extra. Busca fugas de vacío en el colector o baja presión de combustible.")
                elif stft < -10:
                    health_issues.append("💧 Inyección: STFT negativo (<-10%). La ECU recorta combustible. Posible inyector goteando o filtro de aire obstruido.")
            
            if not health_issues:
                st.markdown("<div class='eco-good'>✅ <b>Sistemas Nominales:</b> Todos los parámetros operativos están dentro de rango.</div>", unsafe_allow_html=True)
            else:
                for issue in health_issues:
                    st.markdown(f"<div class='eco-warn'>{issue}</div>", unsafe_allow_html=True)
            
            # El texto que te gustaba, restaurado
            with st.expander("ℹ️ GUÍA DE DIAGNÓSTICO MECÁNICO"):
                st.markdown("""
                * **Carga del Motor (Engine Load):** Indica cuánto "esfuerzo" hace el motor respecto a su capacidad máxima actual.
                    * *Diagnóstico:* >60% constante aumenta el desgaste. <15% en movimiento puede indicar retención.
                * **Sistema Eléctrico:** <13.5V en marcha = Posible fallo de alternador.
                * **Refrigerante:** <75°C tras 15 min = Termostato abierto (ineficiencia).
                * **STFT:** Desviaciones >10% indican fugas de vacío (+) o goteo de inyectores (-).
                """)

        with c_graph:
            # Gráfica Sincronizada
            if 'coolant_temp' in my_df.columns and 'engine_load' in my_df.columns:
                chart_df = my_df[['timestamp_sec', 'coolant_temp', 'engine_load']].copy()
                chart_df = chart_df.set_index('timestamp_sec').sort_index().interpolate(method='index', limit_direction='both').reset_index()
                chart_df['time_min'] = (chart_df['timestamp_sec'] - chart_df['timestamp_sec'].min()) / 60.0

                fig_health = make_subplots(specs=[[{"secondary_y": True}]])
                fig_health.add_trace(go.Scatter(x=chart_df['time_min'], y=chart_df['engine_load'], mode='lines', name='Carga (%)', fill='tozeroy', line=dict(color='#38BDF8', width=1)), secondary_y=False)
                fig_health.add_trace(go.Scatter(x=chart_df['time_min'], y=chart_df['coolant_temp'], mode='lines', name='Temp (°C)', line=dict(color='#EF4444', width=3)), secondary_y=True)

                fig_health.update_layout(title="Carga Motor vs Temperatura (Sincronizado)", template="plotly_dark", height=350, hovermode="x unified", legend=dict(orientation="h", y=1.1))
                fig_health.update_yaxes(title_text="Carga (%)", secondary_y=False, range=[0, 100])
                fig_health.update_yaxes(title_text="Temp (°C)", secondary_y=True, range=[0, 120])
                st.plotly_chart(fig_health, use_container_width=True)
            else:
                st.warning("Faltan sensores para la gráfica cruzada.")
else:
    # Landing Page
    st.markdown("""<div style='text-align: center; margin-top: 100px;'><h1 style='font-size: 4rem; text-shadow: 0 0 30px rgba(56, 189, 248, 0.8);'>DAT <span style='color:#38BDF8'>ANALYTICS</span></h1></div>""", unsafe_allow_html=True)