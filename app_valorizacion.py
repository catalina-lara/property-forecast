# app_valorizacion.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import Rectangle, CircleMarker
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
import plotly.express as px
import sys
import os

# Agregar la ruta del archivo actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import pickle
# importar la función predict (asumiendo archivo predict.py en la misma carpeta)
from predict import predict_batch

# -------------------------
# Configuración básica
# -------------------------
st.set_page_config(page_title="PROTOTIPO FUNCIONALIDAD VALORIZACION", layout="wide")
st.title("PROTOTIPO FUNCIONALIDAD VALORIZACION")

# -------------------------
# Rutas de archivos
# -------------------------
BASE_DIR = r"C:\Users\jorge.marin\NODRIVE\COURSERA\2025-5 Despliegue\Proyecto\Entrega2-jmm"
PATH_DATA_MODEL = os.path.join(BASE_DIR, "datalimpio_modelo_final.csv")
PATH_DATAT = os.path.join(BASE_DIR, "dataTlimpio.csv")

# -------------------------
# Cargar datos
# -------------------------
@st.cache_data(show_spinner=False)
def load_data_model(path):
    df = pd.read_csv(path)
    df['zona'] = df['zona'].astype(str)
    df['mes_cont'] = df['mes_cont'].astype(int)
    if 'price_per_m2' not in df.columns:
        df['price_per_m2'] = df['price'] / df['surface_total']
    return df

@st.cache_data(show_spinner=False)
def load_dataT(path):
    try:
        df = pd.read_csv(path, parse_dates=['start_date'], low_memory=False)
    except Exception:
        df = pd.read_csv(path, low_memory=False)
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    # ensure numeric where expected
    for c in ['price','surface_total','bedrooms','lat','lon']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # ensure zona as str
    if 'zona' in df.columns:
        df['zona'] = df['zona'].astype(str)
    return df

try:
    data_model = load_data_model(PATH_DATA_MODEL)
    dataTlimpio = load_dataT(PATH_DATAT)
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# -------------------------
# Build tablas por zona (mes_cont vs avg price_per_m2)
# -------------------------
@st.cache_data(show_spinner=False)
def build_zone_tables(df_model):
    grouped = df_model.groupby(['zona', 'mes_cont'])['price_per_m2'].mean().reset_index()
    zone_tables = {}
    for zona, g in grouped.groupby('zona'):
        zone_tables[zona] = g.sort_values('mes_cont').reset_index(drop=True)[['mes_cont', 'price_per_m2']]
    return zone_tables

zone_tables = build_zone_tables(data_model)

# -------------------------
# Cálculo valorización por zona (val = (B-A)/A) — usado para mapa
# -------------------------
def compute_zone_val_total(zone_table):
    if zone_table is None or zone_table.empty:
        return None
    A = zone_table.iloc[0]['price_per_m2']
    B = zone_table.iloc[-1]['price_per_m2']
    if pd.isna(A) or A == 0 or pd.isna(B):
        return None
    return (B - A) / A  # proporción (ej. 0.2 -> 20%)

zona_rows = []
for z, table in zone_tables.items():
    val = compute_zone_val_total(table)
    if val is None:
        continue
    # get zona_lat/zona_lon if present in dataTlimpio, else try to parse zona "i_j"
    if 'zona_lat' in dataTlimpio.columns and 'zona_lon' in dataTlimpio.columns:
        # take mean indices for that zona
        subset = dataTlimpio[dataTlimpio['zona'] == z]
        if not subset.empty:
            zona_lat = int(subset['zona_lat'].mode()[0])
            zona_lon = int(subset['zona_lon'].mode()[0])
        else:
            # fallback parse
            try:
                zona_lat, zona_lon = [int(x) for x in z.split('_')]
            except Exception:
                zona_lat = zona_lon = 0
    else:
        try:
            zona_lat, zona_lon = [int(x) for x in z.split('_')]
        except Exception:
            zona_lat = zona_lon = 0
    zona_rows.append({'zona': z, 'zona_lat': zona_lat, 'zona_lon': zona_lon, 'val_total': val})

zona_prom = pd.DataFrame(zona_rows)

# -------------------------
# INTERFAZ: filtros con unidades
# -------------------------
st.header("FILTROS DE USUARIO")

col1, col2, col3 = st.columns(3)
with col1:
    precio_min = st.number_input("Precio mínimo (USD)", min_value=0.0, value=float(dataTlimpio['price'].min() if dataTlimpio['price'].notna().any() else 0.0), step=100.0, format="%.0f")
    area_min = st.number_input("Área mínima (m²)", min_value=0.0, value=float(dataTlimpio['surface_total'].min() if dataTlimpio['surface_total'].notna().any() else 0.0), step=1.0, format="%.1f")
with col2:
    precio_max = st.number_input("Precio máximo (USD)", min_value=0.0, value=float(dataTlimpio['price'].max() if dataTlimpio['price'].notna().any() else 1000000.0), step=100.0, format="%.0f")
    area_max = st.number_input("Área máxima (m²)", min_value=0.0, value=float(dataTlimpio['surface_total'].max() if dataTlimpio['surface_total'].notna().any() else 500.0), step=1.0, format="%.1f")
with col3:
    dormitorios = st.number_input("Número de dormitorios (mínimo)", min_value=0, value=int(dataTlimpio['bedrooms'].min() if dataTlimpio['bedrooms'].notna().any() else 0), step=1)

# -------------------------
# Session state para mantener selección única
# -------------------------
if 'selected_results' not in st.session_state:
    st.session_state['selected_results'] = None

buscar, nueva = st.columns([1,1])
with buscar:
    if st.button("Buscar inmuebles"):
        # filtrar inicialmente por area y dormitorios (trabajar con TODO dataTlimpio)
        dff = dataTlimpio.dropna(subset=['surface_total','bedrooms','lat','lon','zona']).copy()
        dff = dff[
            (dff['surface_total'] >= area_min) &
            (dff['surface_total'] <= area_max) &
            (dff['bedrooms'] >= dormitorios)
        ]
        if dff.empty:
            st.warning("No se encontraron inmuebles dentro de area/dormitorios.")
            st.session_state['selected_results'] = pd.DataFrame()
        else:
            # 1) Predecir price_per_m2 para TODOS los registros filtrados (batch)
            data_tmp = predict_batch(dff)  # retorna price_per_m2_pred y price_pred_tmp

            # 2) Filtrar por precio pronostico entre precio_min y precio_max
            data_tmp_filtered = data_tmp[
                (data_tmp['price_pred_tmp'] >= precio_min) &
                (data_tmp['price_pred_tmp'] <= precio_max)
            ].copy()

            n_found = len(data_tmp_filtered)
            if n_found == 0:
                st.warning("No se encontró ningún inmueble con los criterios de búsqueda, por favor repite la búsqueda.")
                # limpiar estado y regresar al modo inicial
                st.session_state['selected_results'] = None
                st.experimental_rerun()
            else:
                # seleccionar los 4 aleatorios (o menos si no hay 4)
                sel = data_tmp_filtered.sample(n=min(4, n_found), random_state=None).copy()

                # calcular valorizacion por inmueble usando tabla de zona (igual que antes)
                rows = []
                for _, r in sel.iterrows():
                    z = str(r['zona'])
                    table_z = zone_tables.get(z)
                    if table_z is None or table_z.empty:
                        val_pct = np.nan
                    else:
                        v = compute_zone_val_total(table_z)
                        val_pct = v * 100 if v is not None else np.nan

                    rows.append({
                        'id': r.get('id', None),
                        'valorizacion_anual_%': val_pct,
                        'zona': z,
                        'n_dormitorios': int(r['bedrooms']) if pd.notna(r['bedrooms']) else np.nan,
                        'surface_total': r['surface_total'],
                        'lat': r['lat'],
                        'lon': r['lon'],
                        'Precio pronostico': r['price_pred_tmp']  # nueva columna
                    })
                st.session_state['selected_results'] = pd.DataFrame(rows)


with nueva:
    if st.button("Nueva búsqueda (limpiar)"):
        st.session_state['selected_results'] = None
        st.experimental_rerun()

# -------------------------
# Parte 3: mostrar tabla de resultados (sin columna id)
# -------------------------
st.header("RESULTADOS DE LA BÚSQUEDA")
if st.session_state['selected_results'] is None:
    st.info("Aún no hay búsqueda. Ingresá filtros y presioná 'Buscar inmuebles'.")
elif st.session_state['selected_results'].empty:
    st.warning("La búsqueda no devolvió inmuebles.")
else:
    results_df = st.session_state['selected_results'].sort_values('valorizacion_anual_%', ascending=False).reset_index(drop=True)
    # quitar columna id para mostrar
    mostrar = results_df.drop(columns=['id'])
    # formatear valorizacion
    mostrar['valorizacion_anual_%'] = mostrar['valorizacion_anual_%'].round(2)
    st.dataframe(mostrar.style.background_gradient(subset=['valorizacion_anual_%'], cmap='RdYlGn', vmin=-50, vmax=50))

    # -------------------------
    # Parte 4: Gráfica de valorización por zona (mes_cont vs price_per_m2)
    # -------------------------
    st.header("COMPORTAMIENTO DE VALORIZACION")
    fig = px.line(title="Evolución price_per_m2 por mes_cont (zona de cada inmueble)")
    for z in results_df['zona'].unique():
        t = zone_tables.get(z)
        if t is None or t.empty:
            continue
        fig.add_scatter(x=t['mes_cont'], y=t['price_per_m2'], mode='lines+markers', name=f"{z}")
    fig.update_layout(xaxis_title='mes_cont (meses cont.)', yaxis_title='price_per_m2 (USD / m²)', legend_title='zona')
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Parte 5: MAPA DE VALORIZACION (areas sólidas por zona) + pins visibles
    # -------------------------
    st.header("MAPA DE VALORIZACION")
    # prepare grid bounds
    lat_min = dataTlimpio['lat'].min()
    lat_max = dataTlimpio['lat'].max()
    lon_min = dataTlimpio['lon'].min()
    lon_max = dataTlimpio['lon'].max()

    # determine number of zones in grid
    if 'zona_lat' in dataTlimpio.columns and 'zona_lon' in dataTlimpio.columns:
        n_lat = int(dataTlimpio['zona_lat'].max() + 1)
        n_lon = int(dataTlimpio['zona_lon'].max() + 1)
    elif not zona_prom.empty:
        n_lat = int(zona_prom['zona_lat'].max() + 1)
        n_lon = int(zona_prom['zona_lon'].max() + 1)
    else:
        n_lat = n_lon = 6

    tam_lat = (lat_max - lat_min) / n_lat if n_lat>0 else 0
    tam_lon = (lon_max - lon_min) / n_lon if n_lon>0 else 0

    # create folium map
    centro = [(lat_min+lat_max)/2, (lon_min+lon_max)/2]
    m = folium.Map(location=centro, zoom_start=12)

    # compute color scale for val_total (in percent)
    if not zona_prom.empty:
        # percent values
        zona_prom['val_pct'] = zona_prom['val_total'] * 100
        vmin = float(zona_prom['val_pct'].quantile(0.05))
        vmax = float(zona_prom['val_pct'].quantile(0.95))
        # use green for high positive, red for negative: RdYlGn reversed? We'll create colormap from red->yellow->green
        colormap = LinearColormap(['red','yellow','green'], vmin=vmin, vmax=vmax, caption='Valorización (%)')

        for _, zr in zona_prom.iterrows():
            zlat = int(zr['zona_lat'])
            zlon = int(zr['zona_lon'])
            valp = float(zr['val_total'] * 100)
            # clip to vmin/vmax for color mapping
            val_for_color = np.clip(valp, vmin, vmax)
            color_hex = colormap(val_for_color)  # returns color hex
            # rectangle coords for the zone
            lat0 = lat_min + zlat * tam_lat
            lat1 = lat_min + (zlat + 1) * tam_lat
            lon0 = lon_min + zlon * tam_lon
            lon1 = lon_min + (zlon + 1) * tam_lon
            Rectangle(bounds=[[lat0, lon0], [lat1, lon1]],
                      color='black', weight=0.5, fill=True, fill_color=color_hex, fill_opacity=0.8,
                      popup=f"Zona {zr['zona']}: {valp:.2f}%").add_to(m)
        colormap.add_to(m)
    else:
        # no zona_prom: just add a neutral basemap
        pass

    # add prominent pins for selected properties (big, high-contrast)
    for _, r in results_df.iterrows():
        CircleMarker(location=[r['lat'], r['lon']],
                     radius=9,
                     color='black',
                     weight=1,
                     fill=True,
                     fill_color='#0000FF',  # bright blue
                     fill_opacity=0.9,
                     popup=f"id: {r['id']}<br>zona: {r['zona']}<br>valorización: {r['valorizacion_anual_%']:.2f}%").add_to(m)

    # show map
    st_data = st_folium(m, width=900, height=520)

