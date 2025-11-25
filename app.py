"""
UAM Vertiport Demand/Capacity Simulation Tool
A beautiful Streamlit interface for vertiport simulation analysis
with dynamic population data from WorldPop/GHS-POP
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
from streamlit_folium import st_folium
import requests
from io import BytesIO
import utm
from numba import njit, prange
import time
import base64
import os
import zipfile
import tempfile
import rasterio
from rasterio.windows import Window

# Page configuration
st.set_page_config(
    page_title="UAM Vertiport Simulator",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Data source badge */
    .data-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 25px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Location picker card */
    .location-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-loading {
        color: #f39c12;
        font-weight: 600;
    }
    
    .status-success {
        color: #27ae60;
        font-weight: 600;
    }
    
    .status-error {
        color: #e74c3c;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ========================
# Population Data Functions - GHS-POP from Copernicus
# ========================

# GHS-POP tile configuration
# Tiles are 10x10 degrees in EPSG:4326
# Row 1 starts at 90°N, Column 1 starts at 180°W
# Resolution: 3 arc-seconds (~100m at equator)
GHS_POP_BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_3ss/V1-0/tiles/"
GHS_POP_TILE_SIZE = 10  # degrees


def get_tile_indices(lat, lon):
    """
    Calculate GHS-POP tile row and column indices from lat/lon.
    Tiles are 10x10 degrees, starting from 90°N and 180°W.
    """
    # Row: starts from top (90°N), going down
    # Row 1: 90°N to 80°N, Row 2: 80°N to 70°N, etc.
    row = int((90 - lat) // GHS_POP_TILE_SIZE) + 1
    
    # Column: starts from left (-180°), going right  
    # Col 1: -180° to -170°, Col 2: -170° to -160°, etc.
    col = int((lon + 180) // GHS_POP_TILE_SIZE) + 1
    
    # Clamp to valid ranges
    row = max(1, min(row, 18))  # 18 rows cover 90°N to -90°S
    col = max(1, min(col, 36))  # 36 columns cover -180° to 180°
    
    return row, col


def download_ghspop_tile(row, col, cache_dir=None):
    """
    Download a GHS-POP tile from JRC/Copernicus servers.
    Returns the path to the downloaded TIF file.
    """
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "ghspop_cache")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Construct filename
    tile_name = f"GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}"
    zip_filename = f"{tile_name}.zip"
    tif_filename = f"{tile_name}.tif"
    
    zip_path = os.path.join(cache_dir, zip_filename)
    tif_path = os.path.join(cache_dir, tif_filename)
    
    # Check if already cached
    if os.path.exists(tif_path):
        return tif_path
    
    # Download the zip file
    url = f"{GHS_POP_BASE_URL}{zip_filename}"
    
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        # Save zip file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the TIF file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find the TIF file in the archive
            tif_files = [f for f in zip_ref.namelist() if f.endswith('.tif')]
            if tif_files:
                # Extract and rename
                zip_ref.extract(tif_files[0], cache_dir)
                extracted_path = os.path.join(cache_dir, tif_files[0])
                if extracted_path != tif_path:
                    os.rename(extracted_path, tif_path)
        
        # Clean up zip file
        os.remove(zip_path)
        
        return tif_path
        
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not download tile R{row}_C{col}: {e}")
        return None
    except zipfile.BadZipFile as e:
        st.warning(f"Invalid zip file for tile R{row}_C{col}: {e}")
        return None


def read_population_from_tile(tif_path, lat, lon, radius_km):
    """
    Read population data from a GHS-POP GeoTIFF file for an area around lat/lon.
    Returns a dictionary with the population grid data.
    """
    with rasterio.open(tif_path) as src:
        # Calculate bounding box in degrees
        lat_buffer = radius_km / 111.0  # ~111 km per degree latitude
        lon_buffer = radius_km / (111.0 * np.cos(np.radians(lat)))
        
        min_lon = lon - lon_buffer
        max_lon = lon + lon_buffer
        min_lat = lat - lat_buffer
        max_lat = lat + lat_buffer
        
        # Get pixel coordinates for the bounding box
        # GeoTIFF uses (lon, lat) order for transform
        row_start, col_start = src.index(min_lon, max_lat)  # top-left
        row_end, col_end = src.index(max_lon, min_lat)  # bottom-right
        
        # Ensure valid window
        row_start = max(0, min(row_start, src.height - 1))
        row_end = max(0, min(row_end, src.height - 1))
        col_start = max(0, min(col_start, src.width - 1))
        col_end = max(0, min(col_end, src.width - 1))
        
        if row_start > row_end:
            row_start, row_end = row_end, row_start
        if col_start > col_end:
            col_start, col_end = col_end, col_start
        
        # Add buffer and ensure minimum size
        row_start = max(0, row_start - 10)
        row_end = min(src.height, row_end + 10)
        col_start = max(0, col_start - 10)
        col_end = min(src.width, col_end + 10)
        
        # Read the window
        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
        population = src.read(1, window=window)
        
        # Handle nodata values
        if src.nodata is not None:
            population = np.where(population == src.nodata, 0, population)
        population = np.maximum(population, 0)  # No negative populations
        
        # Get the transform for this window
        window_transform = src.window_transform(window)
        
        # Create coordinate arrays
        rows, cols = population.shape
        col_indices = np.arange(cols)
        row_indices = np.arange(rows)
        
        # Transform pixel coordinates to geographic coordinates
        lons = window_transform.c + col_indices * window_transform.a
        lats = window_transform.f + row_indices * window_transform.e
        
        # Get pixel size in meters (approximate)
        pixel_size_deg = abs(window_transform.a)  # degrees per pixel
        pixel_size_m = pixel_size_deg * 111000 * np.cos(np.radians(lat))  # approximate meters
        
        # Convert population counts to density (people per km²)
        cell_area_km2 = (pixel_size_m / 1000) ** 2
        if cell_area_km2 > 0:
            density = population / cell_area_km2
        else:
            density = population
        
        # Convert to UTM for distance calculations
        try:
            utm_center = utm.from_latlon(lat, lon)
            center_x, center_y = utm_center[0], utm_center[1]
            utm_zone = utm_center[2]
            
            # Create UTM coordinate grids
            LON, LAT = np.meshgrid(lons, lats)
            x_utm = np.zeros_like(LON)
            y_utm = np.zeros_like(LAT)
            
            # Convert each point to UTM (vectorized approximation)
            for i in range(rows):
                for j in range(cols):
                    try:
                        utm_point = utm.from_latlon(LAT[i, j], LON[i, j], force_zone_number=utm_zone)
                        x_utm[i, j] = utm_point[0]
                        y_utm[i, j] = utm_point[1]
                    except:
                        x_utm[i, j] = center_x
                        y_utm[i, j] = center_y
            
            x_coords = x_utm[0, :]
            y_coords = y_utm[:, 0]
        except:
            # Fallback: approximate UTM coordinates
            center_x = lon * 111000 * np.cos(np.radians(lat))
            center_y = lat * 111000
            x_coords = lons * 111000 * np.cos(np.radians(lat))
            y_coords = lats * 111000
        
        # Calculate total population within radius
        LON_grid, LAT_grid = np.meshgrid(lons, lats)
        dist_deg = np.sqrt((LON_grid - lon)**2 + (LAT_grid - lat)**2 * (111/111)**2)
        dist_km = dist_deg * 111  # Approximate distance in km
        
        mask = dist_km <= radius_km
        total_pop = np.sum(population[mask])
        
        return {
            'x': x_coords if 'x_coords' in dir() else lons * 111000,
            'y': y_coords if 'y_coords' in dir() else lats * 111000,
            'lat': lats,
            'lon': lons,
            'density': density,
            'population': population,
            'total_population': total_pop,
            'resolution': pixel_size_m,
            'center_x': center_x if 'center_x' in dir() else lon * 111000,
            'center_y': center_y if 'center_y' in dir() else lat * 111000
        }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ghsl_population(lat, lon, radius_km):
    """
    Fetch real GHS-POP population data from Copernicus JRC.
    Downloads the appropriate tile and extracts data for the specified area.
    """
    # Get tile indices
    row, col = get_tile_indices(lat, lon)
    
    # Download the tile
    tif_path = download_ghspop_tile(row, col)
    
    if tif_path is None or not os.path.exists(tif_path):
        st.error(f"Could not download GHS-POP tile for location ({lat:.4f}, {lon:.4f})")
        return None
    
    # Read population data
    try:
        pop_data = read_population_from_tile(tif_path, lat, lon, radius_km)
        return pop_data
    except Exception as e:
        st.error(f"Error reading population data: {e}")
        return None


# ========================
# Simulation Functions
# ========================

@njit()
def generate_random_demand(num_passengers, time_limit):
    """Generate random passenger arrival times."""
    return np.random.randint(0, time_limit, num_passengers)


@njit(fastmath=True, parallel=True)
def compute_failure_rate(population_area, num_platforms, num_parkings, N_simulations,
                         num_vehicles_initial, fraction_of_passengers, taxi_time,
                         max_wait_platform, max_wait_parking, max_wait_departure):
    """
    Compute the failure rate of a vertiport simulation.
    """
    num_passengers = int(population_area * fraction_of_passengers)
    num_passengers_arr = num_passengers // 2
    num_passengers_dep = num_passengers // 2
    time_limit = 60

    failure_array = np.zeros(N_simulations)

    for sim in prange(N_simulations):
        num_vehicles = num_vehicles_initial
        platforms_available = np.zeros(num_platforms)

        arrival_times = generate_random_demand(num_passengers_arr, time_limit)
        departure_times = generate_random_demand(num_passengers_dep, time_limit)
        operations = np.append(arrival_times, departure_times)
        operation_type = np.append(np.ones(num_passengers_arr), np.zeros(num_passengers_dep))
        sort_idx = np.argsort(operations)
        operations = operations[sort_idx]
        operation_type = operation_type[sort_idx]

        for i in range(len(operations)):
            current_time = operations[i]
            current_type = operation_type[i]

            earliest_platform = np.min(platforms_available)
            actual_time = max(current_time, earliest_platform)
            platform_wait = actual_time - current_time

            if platform_wait > max_wait_platform:
                failure_array[sim] = 1
                break

            platform_idx = np.argmin(platforms_available)
            platforms_available[platform_idx] = actual_time + taxi_time

            if current_type == 1:
                if num_vehicles + 1 > num_parkings:
                    found = False
                    for j in range(i + 1, len(operations)):
                        if operation_type[j] == 0:
                            dep_time = operations[j]
                            wait_time = dep_time - actual_time
                            if wait_time <= max_wait_parking:
                                found = True
                                break
                    if not found:
                        failure_array[sim] = 1
                        break
                num_vehicles += 1
            else:
                if num_vehicles - 1 < 0:
                    found = False
                    for j in range(i + 1, len(operations)):
                        if operation_type[j] == 1:
                            arr_time = operations[j]
                            wait_time = arr_time - actual_time
                            if wait_time <= max_wait_departure:
                                found = True
                                break
                    if not found:
                        failure_array[sim] = 1
                        break
                num_vehicles -= 1

    return np.float32(np.mean(failure_array))


def population_within_circle(x_coords, y_coords, pop_density, center_x, center_y, radius):
    """Calculate total population within a circular area."""
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
    distances = (X - center_x) ** 2 + (Y - center_y) ** 2
    inside_circle = distances <= radius ** 2
    
    # Calculate area per cell in km²
    dx = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 100
    dy = abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 100
    cell_area_km2 = (dx * dy) / 1e6
    
    # Sum population (density * area)
    inhabitants = np.sum(pop_density[inside_circle]) * cell_area_km2
    return inhabitants


# ========================
# UI Components
# ========================

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🚁 UAM Vertiport Simulator</h1>
        <p>Advanced Demand/Capacity Simulation for Urban Air Mobility</p>
        <div class="data-badge">📊 Powered by GHS-POP Population Data</div>
    </div>
    """, unsafe_allow_html=True)


def create_folium_map(lat, lon, radius_km):
    """Create an interactive Folium map with free-form clicking."""
    # Create map centered on location
    m = folium.Map(
        location=[lat, lon],
        zoom_start=10,
        tiles='cartodbpositron'
    )
    
    # Add influence radius circle
    folium.Circle(
        location=[lat, lon],
        radius=radius_km * 1000,  # Convert km to meters
        color='#667eea',
        weight=3,
        fill=True,
        fill_color='#667eea',
        fill_opacity=0.2,
        popup=f'Influence Radius: {radius_km} km'
    ).add_to(m)
    
    # Add vertiport marker with simple icon
    folium.Marker(
        location=[lat, lon],
        popup=f'Vertiport - Lat: {lat:.6f}, Lon: {lon:.6f}',
        tooltip='Vertiport Location',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    return m


def create_population_heatmap(pop_data, lat, lon, radius_km):
    """Create an interactive population density heatmap."""
    fig = go.Figure()
    
    # Downsample for performance
    step = max(1, len(pop_data['lat']) // 100)
    lat_small = pop_data['lat'][::step]
    lon_small = pop_data['lon'][::step]
    density_small = pop_data['density'][::step, ::step]
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=density_small,
        x=lon_small,
        y=lat_small,
        colorscale='YlOrRd',
        colorbar=dict(
            title=dict(text='Population<br>Density<br>(per km²)', font=dict(size=12))
        ),
        hovertemplate='Lon: %{x:.4f}<br>Lat: %{y:.4f}<br>Density: %{z:.0f}/km²<extra></extra>'
    ))
    
    # Add vertiport marker
    fig.add_trace(go.Scatter(
        x=[lon],
        y=[lat],
        mode='markers+text',
        marker=dict(size=20, color='#667eea', symbol='star', 
                   line=dict(width=2, color='white')),
        text=['🚁 Vertiport'],
        textposition='top center',
        textfont=dict(size=12, color='#667eea'),
        name='Vertiport',
        showlegend=True
    ))
    
    # Add influence circle
    theta = np.linspace(0, 2 * np.pi, 100)
    radius_deg_lat = radius_km / 111
    radius_deg_lon = radius_km / (111 * np.cos(np.radians(lat)))
    circle_lon = lon + radius_deg_lon * np.cos(theta)
    circle_lat = lat + radius_deg_lat * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=circle_lon,
        y=circle_lat,
        mode='lines',
        line=dict(color='#667eea', width=3, dash='dash'),
        name=f'Influence Radius ({radius_km} km)',
        showlegend=True
    ))
    
    fig.update_layout(
        title=dict(
            text='Population Density Map',
            font=dict(size=18, color='#333')
        ),
        title_x=0.5,
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(scaleanchor='x'),
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig


def create_heatmap(failure_matrix, num_platforms_vec, num_parkings_vec):
    """Create an interactive heatmap of failure rates."""
    fig = go.Figure(data=go.Heatmap(
        z=failure_matrix * 100,
        x=num_parkings_vec,
        y=num_platforms_vec,
        colorscale=[
            [0, '#2ecc71'],
            [0.25, '#f1c40f'],
            [0.5, '#e67e22'],
            [0.75, '#e74c3c'],
            [1, '#8e44ad']
        ],
        colorbar=dict(
            title=dict(text='Failure Rate (%)', font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        hovertemplate='Platforms: %{y}<br>Parking Spots: %{x}<br>Failure Rate: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Failure Rate Analysis', font=dict(size=20, color='#333')),
        title_x=0.5,
        xaxis=dict(title=dict(text='Number of Parking Spots', font=dict(size=14)), tickfont=dict(size=12)),
        yaxis=dict(title=dict(text='Number of Landing Platforms', font=dict(size=14)), tickfont=dict(size=12)),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_3d_surface(failure_matrix, num_platforms_vec, num_parkings_vec):
    """Create a 3D surface plot of failure rates."""
    X, Y = np.meshgrid(num_parkings_vec, num_platforms_vec)
    
    fig = go.Figure(data=[go.Surface(
        z=failure_matrix * 100,
        x=X,
        y=Y,
        colorscale='Viridis',
        colorbar=dict(title='Failure Rate (%)')
    )])
    
    fig.update_layout(
        title=dict(text='3D Failure Rate Surface', font=dict(size=20, color='#333')),
        title_x=0.5,
        scene=dict(
            xaxis_title='Parking Spots',
            yaxis_title='Platforms',
            zaxis_title='Failure Rate (%)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_contour_plot(failure_matrix, num_platforms_vec, num_parkings_vec, target_rate):
    """Create contour plot with target failure rate highlighted."""
    fig = go.Figure()
    
    fig.add_trace(go.Contour(
        z=failure_matrix * 100,
        x=num_parkings_vec,
        y=num_platforms_vec,
        colorscale='RdYlGn_r',
        contours=dict(showlabels=True, labelfont=dict(size=12, color='white')),
        colorbar=dict(title='Failure Rate (%)'),
        hovertemplate='Platforms: %{y}<br>Parking Spots: %{x}<br>Failure Rate: %{z:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Contour(
        z=failure_matrix * 100,
        x=num_parkings_vec,
        y=num_platforms_vec,
        contours=dict(start=target_rate, end=target_rate, size=0.1, coloring='lines'),
        line=dict(width=4, color='blue', dash='dash'),
        showscale=False,
        name=f'Target ({target_rate}%)'
    ))
    
    fig.update_layout(
        title=dict(text=f'Contour Analysis (Target: {target_rate}% failure rate)', 
                  font=dict(size=20, color='#333')),
        title_x=0.5,
        xaxis_title='Number of Parking Spots',
        yaxis_title='Number of Landing Platforms',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_line_analysis(failure_matrix, num_platforms_vec, num_parkings_vec):
    """Create line plots for specific configurations."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('By Parking Spots', 'By Platforms'))
    
    colors = px.colors.qualitative.Set2
    
    for i, platforms in enumerate([1, 3, 5, 10]):
        if platforms <= len(num_platforms_vec):
            idx = platforms - 1
            fig.add_trace(
                go.Scatter(
                    x=num_parkings_vec,
                    y=failure_matrix[idx, :] * 100,
                    mode='lines+markers',
                    name=f'{platforms} Platform(s)',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
    
    for i, parkings in enumerate([5, 10, 20, 30]):
        if parkings <= len(num_parkings_vec):
            idx = parkings - 1
            fig.add_trace(
                go.Scatter(
                    x=num_platforms_vec,
                    y=failure_matrix[:, idx] * 100,
                    mode='lines+markers',
                    name=f'{parkings} Parking(s)',
                    line=dict(color=colors[(i + 4) % len(colors)], width=2),
                    marker=dict(size=6)
                ),
                row=1, col=2
            )
    
    fig.update_xaxes(title_text='Number of Parking Spots', row=1, col=1)
    fig.update_xaxes(title_text='Number of Platforms', row=1, col=2)
    fig.update_yaxes(title_text='Failure Rate (%)', row=1, col=1)
    fig.update_yaxes(title_text='Failure Rate (%)', row=1, col=2)
    
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


# ========================
# Preset Locations
# ========================

PRESET_LOCATIONS = {
    "São Paulo, Brazil": {"lat": -23.5505, "lon": -46.6333},
    "New York City, USA": {"lat": 40.7128, "lon": -74.0060},
    "London, UK": {"lat": 51.5074, "lon": -0.1278},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503},
    "Paris, France": {"lat": 48.8566, "lon": 2.3522},
    "Singapore": {"lat": 1.3521, "lon": 103.8198},
    "Dubai, UAE": {"lat": 25.2048, "lon": 55.2708},
    "Los Angeles, USA": {"lat": 34.0522, "lon": -118.2437},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777},
    "Custom Location": {"lat": 0.0, "lon": 0.0}
}


# ========================
# Main Application
# ========================

def main():
    render_header()
    
    # Initialize session state
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False
    if 'failure_matrix' not in st.session_state:
        st.session_state.failure_matrix = None
    if 'pop_data' not in st.session_state:
        st.session_state.pop_data = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'custom_lat' not in st.session_state:
        st.session_state.custom_lat = None
    if 'custom_lon' not in st.session_state:
        st.session_state.custom_lon = None
    if 'use_clicked_location' not in st.session_state:
        st.session_state.use_clicked_location = False
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # Location Selection
        st.markdown("### 📍 Vertiport Location")
        
        # Show clicked location if available
        if st.session_state.use_clicked_location and st.session_state.custom_lat is not None:
            st.success(f"📍 Using clicked location")
            st.markdown(f"**Lat:** {st.session_state.custom_lat:.6f}")
            st.markdown(f"**Lon:** {st.session_state.custom_lon:.6f}")
            
            if st.button("🔄 Reset to Preset", use_container_width=True):
                st.session_state.use_clicked_location = False
                st.session_state.custom_lat = None
                st.session_state.custom_lon = None
                st.rerun()
            
            lat_vertiport = st.session_state.custom_lat
            lon_vertiport = st.session_state.custom_lon
            selected_preset = "Custom (Clicked)"
        else:
            # Preset locations dropdown
            selected_preset = st.selectbox(
                "Select City",
                options=list(PRESET_LOCATIONS.keys()),
                index=0,
                help="Choose a preset city or click on the map to select custom location"
            )
            
            if selected_preset == "Custom Location":
                lat_vertiport = st.number_input(
                    "Latitude",
                    value=0.0,
                    min_value=-90.0,
                    max_value=90.0,
                    format="%.6f",
                    help="Latitude of the vertiport location"
                )
                lon_vertiport = st.number_input(
                    "Longitude",
                    value=0.0,
                    min_value=-180.0,
                    max_value=180.0,
                    format="%.6f",
                    help="Longitude of the vertiport location"
                )
            else:
                lat_vertiport = PRESET_LOCATIONS[selected_preset]["lat"]
                lon_vertiport = PRESET_LOCATIONS[selected_preset]["lon"]
                st.info(f"📍 Lat: {lat_vertiport:.4f}, Lon: {lon_vertiport:.4f}")
            
            st.caption("💡 Or click on the map to select a location")
        
        st.markdown("---")
        
        # Influence Area
        st.markdown("### 🎯 Influence Area")
        radius_km = st.slider(
            "Influence Radius (km)",
            min_value=1.0,
            max_value=25.0,
            value=9.0,
            step=0.5,
            help="Radius of the area from which passengers are drawn"
        )
        
        st.markdown("---")
        
        # Demand Parameters
        st.markdown("### 👥 Demand Parameters")
        fraction_of_passengers = st.slider(
            "Population Fraction (×10⁻⁵)",
            min_value=1.0,
            max_value=50.0,
            value=6.0,
            step=0.5,
            help="Fraction of population using vertiport per hour"
        ) / 100000
        
        st.markdown("---")
        
        # Operational Parameters
        st.markdown("### ⏱️ Operational Parameters")
        taxi_time = st.slider(
            "Taxi Time (min)",
            min_value=1,
            max_value=10,
            value=2,
            help="Minimum time between consecutive platform operations"
        )
        
        col_wait1, col_wait2 = st.columns(2)
        with col_wait1:
            max_wait_platform = st.number_input(
                "Max Platform Wait",
                min_value=1,
                max_value=30,
                value=1,
                help="Minutes"
            )
        with col_wait2:
            max_wait_parking = st.number_input(
                "Max Parking Wait",
                min_value=1,
                max_value=30,
                value=1,
                help="Minutes"
            )
        
        max_wait_departure = st.number_input(
            "Max Departure Wait (min)",
            min_value=1,
            max_value=30,
            value=1,
            help="Maximum acceptable waiting time for departure vehicle"
        )
        
        st.markdown("---")
        
        # Simulation Range
        st.markdown("### 📏 Simulation Range")
        col_range1, col_range2 = st.columns(2)
        with col_range1:
            max_platforms = st.number_input(
                "Max Platforms",
                min_value=5,
                max_value=50,
                value=19,
                help="Maximum platforms to test"
            )
        with col_range2:
            max_parkings = st.number_input(
                "Max Parkings",
                min_value=10,
                max_value=150,
                value=49,
                help="Maximum parking spots to test"
            )
        
        n_simulations = st.select_slider(
            "Simulations",
            options=[1000, 2500, 5000, 10000, 25000],
            value=5000,
            help="Higher values = more accurate but slower"
        )
    
    # Main content area
    tab_main, tab_map, tab_results = st.tabs(["🎮 Simulation", "🗺️ Population Map", "📊 Results"])
    
    with tab_main:
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("### 🌍 Select Vertiport Location")
            st.info("👆 **Click anywhere on the map** to select a vertiport location freely!")
            
            # Create and display Folium map with free-form clicking
            folium_map = create_folium_map(lat_vertiport, lon_vertiport, radius_km)
            
            # Display map and capture clicks - minimal parameters to avoid serialization issues
            map_data = st_folium(
                folium_map,
                height=400,
                width=700
            )
            
            # Handle map click - FREE FORM, any point!
            if map_data and map_data.get("last_clicked"):
                clicked = map_data["last_clicked"]
                new_lat = clicked["lat"]
                new_lon = clicked["lng"]
                # Only update if location actually changed significantly
                if (abs(new_lat - lat_vertiport) > 0.0001 or 
                    abs(new_lon - lon_vertiport) > 0.0001):
                    st.session_state.custom_lat = new_lat
                    st.session_state.custom_lon = new_lon
                    st.session_state.use_clicked_location = True
                    st.rerun()
            
            # Manual coordinate input
            st.markdown("##### 📝 Or Enter Coordinates Manually:")
            col_coord1, col_coord2, col_coord3 = st.columns([2, 2, 1])
            with col_coord1:
                manual_lat = st.number_input(
                    "Latitude",
                    value=lat_vertiport,
                    min_value=-90.0,
                    max_value=90.0,
                    format="%.6f",
                    key="manual_lat_input"
                )
            with col_coord2:
                manual_lon = st.number_input(
                    "Longitude", 
                    value=lon_vertiport,
                    min_value=-180.0,
                    max_value=180.0,
                    format="%.6f",
                    key="manual_lon_input"
                )
            with col_coord3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("📍 Set Location", use_container_width=True):
                    st.session_state.custom_lat = manual_lat
                    st.session_state.custom_lon = manual_lon
                    st.session_state.use_clicked_location = True
                    st.rerun()
        
        with col_right:
            st.markdown("### 📋 Configuration Summary")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                        padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <p><strong>📍 Location:</strong> {selected_preset}</p>
                <p><strong>🎯 Radius:</strong> {radius_km} km</p>
                <p><strong>⏱️ Taxi Time:</strong> {taxi_time} min</p>
                <p><strong>📊 Simulations:</strong> {n_simulations:,}</p>
                <p><strong>🛬 Platforms:</strong> 1-{max_platforms}</p>
                <p><strong>🅿️ Parkings:</strong> 1-{max_parkings}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Target failure rate
            target_failure_rate = st.slider(
                "🎯 Target Failure Rate (%)",
                min_value=0.5,
                max_value=25.0,
                value=5.0,
                step=0.5,
                help="Highlight configurations meeting this target"
            )
        
        st.markdown("---")
        
        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            load_data_btn = st.button("📥 Load Population Data", use_container_width=True)
        
        with col_btn2:
            run_sim_btn = st.button("🚀 Run Simulation", use_container_width=True, 
                                    disabled=not st.session_state.data_loaded)
        
        with col_btn3:
            clear_btn = st.button("🗑️ Clear Results", use_container_width=True)
        
        if clear_btn:
            st.session_state.simulation_run = False
            st.session_state.failure_matrix = None
            st.session_state.pop_data = None
            st.session_state.data_loaded = False
            st.rerun()
        
        # Load population data
        if load_data_btn:
            with st.spinner("🌐 Fetching population data from GHS-POP model..."):
                pop_data = fetch_ghsl_population(lat_vertiport, lon_vertiport, radius_km)
                
                if pop_data is not None:
                    st.session_state.pop_data = pop_data
                    st.session_state.data_loaded = True
                    st.session_state.lat_vertiport = lat_vertiport
                    st.session_state.lon_vertiport = lon_vertiport
                    st.session_state.radius_km = radius_km
                    
                    total_pop = pop_data['total_population']
                    st.success(f"✅ Population data loaded! Estimated population in area: **{int(total_pop):,}**")
                else:
                    st.error("❌ Failed to load population data")
        
        # Display population info if loaded
        if st.session_state.data_loaded and st.session_state.pop_data is not None:
            pop_data = st.session_state.pop_data
            total_pop = pop_data['total_population']
            expected_passengers = int(total_pop * fraction_of_passengers)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("🏘️ Population in Area", f"{int(total_pop):,}")
            with col_m2:
                st.metric("👥 Expected Passengers/hr", f"{expected_passengers:,}")
            with col_m3:
                st.metric("📏 Area Size", f"{np.pi * radius_km**2:.1f} km²")
            with col_m4:
                avg_density = total_pop / (np.pi * radius_km**2)
                st.metric("📊 Avg Density", f"{int(avg_density):,}/km²")
        
        # Run simulation
        if run_sim_btn and st.session_state.data_loaded:
            pop_data = st.session_state.pop_data
            
            # Get population within radius
            inhabitants_inside = population_within_circle(
                pop_data['x'], pop_data['y'], pop_data['density'],
                pop_data['center_x'], pop_data['center_y'],
                radius_km * 1000
            )
            
            num_passengers = int(inhabitants_inside * fraction_of_passengers)
            
            st.info(f"📊 Running simulation with **{num_passengers}** expected passengers/hour...")
            
            # Setup simulation arrays
            num_platforms_vec = np.arange(1, max_platforms + 1).astype(int)
            num_parkings_vec = np.arange(1, max_parkings + 1).astype(int)
            
            # Compile function (dummy run)
            with st.spinner("⚙️ Compiling simulation engine..."):
                _ = compute_failure_rate(
                    inhabitants_inside, 1, 1, np.int32(1), 1,
                    fraction_of_passengers, taxi_time,
                    max_wait_platform, max_wait_parking, max_wait_departure
                )
            
            # Run main simulation
            failure_matrix = np.zeros((len(num_platforms_vec), len(num_parkings_vec)))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_iterations = len(num_platforms_vec) * len(num_parkings_vec)
            current_iteration = 0
            
            start_time = time.time()
            
            for i, num_platforms in enumerate(num_platforms_vec):
                for j, num_parkings in enumerate(num_parkings_vec):
                    num_vehicles_initial = np.int32(num_parkings / 2)
                    failure_matrix[i, j] = compute_failure_rate(
                        inhabitants_inside, num_platforms, num_parkings,
                        np.int32(n_simulations), num_vehicles_initial,
                        fraction_of_passengers, taxi_time,
                        max_wait_platform, max_wait_parking, max_wait_departure
                    )
                    
                    current_iteration += 1
                    progress = current_iteration / total_iterations
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time
                    eta = (elapsed / current_iteration) * (total_iterations - current_iteration)
                    status_text.text(f"Progress: {current_iteration}/{total_iterations} | ETA: {eta:.0f}s")
            
            progress_bar.empty()
            status_text.empty()
            
            elapsed_total = time.time() - start_time
            st.success(f"✅ Simulation completed in {elapsed_total:.1f} seconds!")
            
            # Store results
            st.session_state.simulation_run = True
            st.session_state.failure_matrix = failure_matrix
            st.session_state.num_platforms_vec = num_platforms_vec
            st.session_state.num_parkings_vec = num_parkings_vec
            st.session_state.inhabitants = inhabitants_inside
            st.session_state.num_passengers = num_passengers
            st.session_state.target_failure_rate = target_failure_rate
    
    with tab_map:
        st.markdown("### 🗺️ Population Density Visualization")
        
        if st.session_state.data_loaded and st.session_state.pop_data is not None:
            pop_data = st.session_state.pop_data
            fig = create_population_heatmap(
                pop_data,
                st.session_state.lat_vertiport,
                st.session_state.lon_vertiport,
                st.session_state.radius_km
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>ℹ️ About the Data:</strong><br>
                Population density data is downloaded from the <strong>GHS-POP R2023A</strong> dataset (Global Human Settlement 
                Population Grid) provided by the European Commission Joint Research Centre (JRC) via Copernicus. 
                The data has a spatial resolution of <strong>3 arc-seconds (~100m)</strong> and represents population 
                estimates for the year 2020.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Click **Load Population Data** in the Simulation tab to visualize population density.")
            st.markdown("""
            <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>📡 Data Source:</strong><br>
                This app downloads real population data from the <strong>GHS-POP R2023A</strong> dataset 
                (European Commission / JRC / Copernicus). First load may take 1-2 minutes as the 
                population tile is downloaded from the server.
            </div>
            """)
    
    with tab_results:
        if st.session_state.simulation_run and st.session_state.failure_matrix is not None:
            st.markdown("## 📊 Simulation Results")
            
            failure_matrix = st.session_state.failure_matrix
            num_platforms_vec = st.session_state.num_platforms_vec
            num_parkings_vec = st.session_state.num_parkings_vec
            target_rate = st.session_state.get('target_failure_rate', 5.0)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                min_failure = np.min(failure_matrix) * 100
                st.metric("Min Failure Rate", f"{min_failure:.2f}%", 
                         delta=f"{min_failure - target_rate:.1f}%" if min_failure < target_rate else None,
                         delta_color="normal")
            
            with col2:
                max_failure = np.max(failure_matrix) * 100
                st.metric("Max Failure Rate", f"{max_failure:.2f}%")
            
            with col3:
                avg_failure = np.mean(failure_matrix) * 100
                st.metric("Average Failure Rate", f"{avg_failure:.2f}%")
            
            with col4:
                target = target_rate / 100
                meeting_target = failure_matrix <= target
                if np.any(meeting_target):
                    configs = np.argwhere(meeting_target)
                    min_resources = float('inf')
                    optimal = None
                    for cfg in configs:
                        total = num_platforms_vec[cfg[0]] + num_parkings_vec[cfg[1]]
                        if total < min_resources:
                            min_resources = total
                            optimal = cfg
                    st.metric("🏆 Optimal Config", 
                             f"{num_platforms_vec[optimal[0]]}P / {num_parkings_vec[optimal[1]]}S")
                else:
                    st.metric("Optimal Config", "None found")
            
            # Visualization tabs
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "🗺️ Heatmap", "📈 3D Surface", "🎯 Contour", "📊 Analysis"
            ])
            
            with viz_tab1:
                fig = create_heatmap(failure_matrix, num_platforms_vec, num_parkings_vec)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab2:
                fig = create_3d_surface(failure_matrix, num_platforms_vec, num_parkings_vec)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab3:
                fig = create_contour_plot(failure_matrix, num_platforms_vec, num_parkings_vec, target_rate)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab4:
                fig = create_line_analysis(failure_matrix, num_platforms_vec, num_parkings_vec)
                st.plotly_chart(fig, use_container_width=True)
            
            # Export and Configuration Finder
            st.markdown("---")
            col_export, col_finder = st.columns(2)
            
            with col_export:
                st.markdown("### 💾 Export Results")
                
                df = pd.DataFrame(
                    failure_matrix * 100,
                    index=[f"{p} Platforms" for p in num_platforms_vec],
                    columns=[f"{s} Spots" for s in num_parkings_vec]
                )
                csv = df.to_csv()
                
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="failure_matrix.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_finder:
                st.markdown("### 🔍 Configuration Finder")
                
                target_input = st.number_input(
                    "Target Max Failure Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=target_rate,
                    step=0.5
                )
                
                target = target_input / 100
                meeting_target = failure_matrix <= target
                
                if np.any(meeting_target):
                    configs = np.argwhere(meeting_target)
                    results = []
                    
                    for cfg in configs:
                        platforms = num_platforms_vec[cfg[0]]
                        parkings = num_parkings_vec[cfg[1]]
                        rate = failure_matrix[cfg[0], cfg[1]] * 100
                        results.append({
                            'Platforms': platforms,
                            'Parkings': parkings,
                            'Total': platforms + parkings,
                            'Failure (%)': round(rate, 2)
                        })
                    
                    results_df = pd.DataFrame(results).sort_values('Total')
                    
                    st.success(f"Found **{len(results_df)}** configurations meeting target!")
                    st.dataframe(results_df.head(10), use_container_width=True, hide_index=True)
                else:
                    st.warning(f"No configuration meets {target_input}% target.")
        
        else:
            st.info("👆 Run a simulation first to see results here.")
            
            # Show demo results
            st.markdown("### 📈 Sample Results Preview")
            
            demo_platforms = np.arange(1, 15)
            demo_parkings = np.arange(1, 30)
            X, Y = np.meshgrid(demo_parkings, demo_platforms)
            np.random.seed(42)
            demo_failure = 1 / (1 + 0.3 * X + 0.5 * Y) + np.random.random(X.shape) * 0.05
            
            fig = create_heatmap(demo_failure, demo_platforms, demo_parkings)
            fig.update_layout(title='Sample Heatmap (Demo Data)')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("*This is demo data. Run a simulation to see actual results.*")
    
    # Footer with author info
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                padding: 1.5rem; border-radius: 12px; margin-top: 1rem;">
        <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 20px;">
            <div style="text-align: center;">
                <p style="margin: 0; font-size: 1rem; color: #333; font-weight: 600;">
                    👨‍💻 Developed by <strong>Enrique Aldao Pensado</strong>
                </p>
                <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem; color: #666;">
                    PhD Researcher | University of Vigo
                </p>
            </div>
        </div>
        <div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #ddd;">
            <p style="margin: 0; font-size: 0.85rem; color: #888;">
                🚁 UAM Vertiport Simulator | Powered by GHS-POP Population Model | 
                <a href="https://www.uvigo.gal" target="_blank" style="color: #667eea;">University of Vigo</a>
            </p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.8rem; color: #aaa;">
                © 2025 All rights reserved
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
