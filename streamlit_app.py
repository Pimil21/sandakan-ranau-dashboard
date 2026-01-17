import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from scipy import stats
import branca.colormap as cm
from shapely.geometry import Point
import ast
import re

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Sandakan-Ranau Death Marches Analysis",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONFIGURATION ====================
class Config:
    # Emotion data paths
    HYBRID_EMOTIONS = "data/hybrid_emotion_analysis.csv"
    LOCATION_EMOTIONS = "data/location_emotions_hybrid.csv"
    
    # Spatial data paths - 2D shapefiles
    POI_M1 = "2d-spatial-data/POI_M1.shp"
    POI_M2 = "2d-spatial-data/POI_M2.shp"
    POI_M3 = "2d-spatial-data/POI_M3.shp"
    
    ROUTE1 = "2d-spatial-data/March1.shp"
    ROUTE2 = "2d-spatial-data/March2.shp"
    ROUTE3 = "2d-spatial-data/March3.shp"

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    """Load emotion analysis data"""
    try:
        hybrid_df = pd.read_csv(Config.HYBRID_EMOTIONS)
        location_df = pd.read_csv(Config.LOCATION_EMOTIONS)
        return hybrid_df, location_df
    except Exception as e:
        st.error(f"Error loading emotion data: {str(e)}")
        return None, None

@st.cache_data
def load_spatial_data():
    """Load 2D shapefiles"""
    try:
        # Load POI locations
        poi_m1 = gpd.read_file(Config.POI_M1)
        poi_m2 = gpd.read_file(Config.POI_M2)
        poi_m3 = gpd.read_file(Config.POI_M3)
        
        # Add march identifiers
        poi_m1['march_id'] = 1
        poi_m2['march_id'] = 2
        poi_m3['march_id'] = 3
        
        poi_m1['march_name'] = 'First March'
        poi_m2['march_name'] = 'Second March'
        poi_m3['march_name'] = 'Third March'
        
        # Combine all POIs
        poi_gdf = pd.concat([poi_m1, poi_m2, poi_m3], ignore_index=True)
        
        # Load route polylines
        route1_gdf = gpd.read_file(Config.ROUTE1)
        route2_gdf = gpd.read_file(Config.ROUTE2)
        route3_gdf = gpd.read_file(Config.ROUTE3)
        
        # Ensure WGS84 projection for web mapping
        poi_gdf = poi_gdf.to_crs(epsg=4326)
        route1_gdf = route1_gdf.to_crs(epsg=4326)
        route2_gdf = route2_gdf.to_crs(epsg=4326)
        route3_gdf = route3_gdf.to_crs(epsg=4326)
        
        st.success(f"✅ Loaded {len(poi_gdf)} POI locations and {len(route1_gdf)+len(route2_gdf)+len(route3_gdf)} route segments")
        
        return poi_gdf, route1_gdf, route2_gdf, route3_gdf
        
    except Exception as e:
        st.error(f"⚠️ Error loading spatial data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None

# ==================== SMART COLUMN DETECTION ====================
def analyze_data_structure(emotion_df, poi_gdf):
    """
    Automatically analyze data structure and identify matching columns
    Returns a mapping dictionary for column names
    NOTE: Not cached because it accepts GeoDataFrame (unhashable)
    """
    analysis_results = {
        'emotion_columns': list(emotion_df.columns),
        'spatial_columns': list(poi_gdf.columns),
        'location_column': None,
        'poi_name_column': None,
        'death_count_column': None,
        'emotion_score_columns': [],
        'sentiment_column': None,
        'temporal_column': None,
        'sample_locations': [],
        'emotion_format': 'unknown'  # 'numeric' or 'dictionary'
    }
    
    # 1. Find POI name column in spatial data
    poi_name_candidates = ['POI_Name', 'poi_name', 'name', 'Name', 'location_name', 'place_name']
    for col in poi_name_candidates:
        if col in poi_gdf.columns:
            analysis_results['poi_name_column'] = col
            # Get sample location names
            analysis_results['sample_locations'] = poi_gdf[col].dropna().unique()[:5].tolist()
            break
    
    # 2. Find death count column in spatial data
    death_count_candidates = ['size_M1', 'deaths', 'death_count', 'casualties', 'pow_deaths']
    for col in death_count_candidates:
        if col in poi_gdf.columns:
            analysis_results['death_count_column'] = col
            break
    
    # 3. Find location column in emotion data (smart matching)
    location_candidates = [
        'location_name', 'entity_text', 'location', 'Location', 'poi_name', 'POI_Name', 
        'place', 'Place', 'name', 'Name', 'entity', 'named_entity', 'location_text', 'place_name'
    ]
    
    # First: Try exact matches
    for col in location_candidates:
        if col in emotion_df.columns:
            analysis_results['location_column'] = col
            break
    
    # Second: Try fuzzy matching
    if analysis_results['location_column'] is None:
        for col in emotion_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['location', 'place', 'poi', 'entity', 'site']):
                if emotion_df[col].dtype == 'object':
                    analysis_results['location_column'] = col
                    break
    
    # Third: Smart content-based detection
    if analysis_results['location_column'] is None and len(analysis_results['sample_locations']) > 0:
        sample_poi = analysis_results['sample_locations'][0].split()[0]
        
        for col in emotion_df.columns:
            if emotion_df[col].dtype == 'object':
                try:
                    matches = emotion_df[col].str.contains(sample_poi, case=False, na=False).sum()
                    if matches > 0:
                        analysis_results['location_column'] = col
                        break
                except:
                    continue
    
    # 4. Find emotion score columns - Enhanced for dictionary format
    emotion_score_columns = []
    
    # First: Check for traditional numeric emotion columns
    emotion_keywords = ['anger', 'fear', 'sadness', 'joy', 'surprise', 'disgust', 
                       'neutral', 'hunger', 'despair', 'cruelty']
    
    for col in emotion_df.columns:
        if any(keyword in col.lower() for keyword in emotion_keywords):
            if pd.api.types.is_numeric_dtype(emotion_df[col]):
                emotion_score_columns.append(col)
                analysis_results['emotion_format'] = 'numeric'
    
    # Second: Check for dictionary-based emotion columns
    dict_emotion_columns = ['emotions', 'bert_emotions', 'combined_emotions', 
                           'emotion_dict', 'emotion_data', 'bert_top_emotions']
    
    for col in dict_emotion_columns:
        if col in emotion_df.columns:
            sample_value = emotion_df[col].dropna().iloc[0] if len(emotion_df[col].dropna()) > 0 else None
            if sample_value and isinstance(sample_value, str):
                if sample_value.strip().startswith('{') or sample_value.strip().startswith('['):
                    emotion_score_columns.append(col)
                    analysis_results['emotion_format'] = 'dictionary'
    
    analysis_results['emotion_score_columns'] = emotion_score_columns
    
    # 5. Find sentiment column
    sentiment_candidates = ['sentiment_score', 'sentiment', 'sentiment_value', 'polarity', 'sentiment_label']
    for col in sentiment_candidates:
        if col in emotion_df.columns:
            if pd.api.types.is_numeric_dtype(emotion_df[col]):
                analysis_results['sentiment_column'] = col
                break
    
    # 6. Find temporal column
    temporal_candidates = ['march_name', 'march_phase', 'march_id', 'date', 'time', 'period', 'phase', 'month', 'day']
    for col in temporal_candidates:
        if col in emotion_df.columns:
            analysis_results['temporal_column'] = col
            break
    
    return analysis_results

def display_data_analysis(analysis_results):
    """Display the automatic data structure analysis results"""
    with st.expander("🔍 Auto-Detected Data Structure", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Emotion Data")
            st.write(f"**Total columns:** {len(analysis_results['emotion_columns'])}")
            st.write(f"**Location column:** `{analysis_results['location_column']}`")
            st.write(f"**Sentiment column:** `{analysis_results['sentiment_column']}`")
            st.write(f"**Temporal column:** `{analysis_results['temporal_column']}`")
            st.write(f"**Emotion format:** `{analysis_results['emotion_format']}`")
            st.write(f"**Emotion columns:** {len(analysis_results['emotion_score_columns'])}")
            if analysis_results['emotion_score_columns']:
                st.code(", ".join(analysis_results['emotion_score_columns']))
        
        with col2:
            st.markdown("### 🗺️ Spatial Data")
            st.write(f"**Total columns:** {len(analysis_results['spatial_columns'])}")
            st.write(f"**POI name column:** `{analysis_results['poi_name_column']}`")
            st.write(f"**Death count column:** `{analysis_results['death_count_column']}`")
            st.write(f"**Sample locations:**")
            if analysis_results['sample_locations']:
                for loc in analysis_results['sample_locations'][:5]:
                    st.write(f"  • {loc}")
        
        with st.expander("📋 All Columns Detail"):
            st.write("**Emotion CSV columns:**")
            st.code(", ".join(analysis_results['emotion_columns']))
            st.write("**Spatial shapefile columns:**")
            st.code(", ".join(analysis_results['spatial_columns']))

# ==================== EMOTION PARSING FUNCTIONS ====================
def parse_emotion_column(emotion_text):
    """
    Parse emotion data from text-based dictionary/list format
    Returns a dictionary of emotion: score pairs
    """
    if pd.isna(emotion_text) or not emotion_text:
        return {}
    
    try:
        # Try direct AST parsing (safest method)
        parsed = ast.literal_eval(str(emotion_text))
        
        # Handle different formats
        if isinstance(parsed, dict):
            result = {}
            for key, value in parsed.items():
                if isinstance(value, (int, float)):
                    result[key] = float(value)
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        result[key] = float(value[0])
                    else:
                        result[key] = 1.0
            return result
        
        elif isinstance(parsed, list):
            return {emotion: 1.0 for emotion in parsed if isinstance(emotion, str)}
        
    except:
        # Fallback: regex pattern matching
        try:
            emotion_words = ['anger', 'fear', 'sadness', 'joy', 'surprise', 
                           'disgust', 'neutral', 'hunger', 'despair', 'cruelty']
            result = {}
            text_lower = str(emotion_text).lower()
            
            for emotion in emotion_words:
                if emotion in text_lower:
                    pattern = f"'{emotion}':\s*([\d.]+)"
                    match = re.search(pattern, text_lower)
                    if match:
                        result[emotion] = float(match.group(1))
                    else:
                        result[emotion] = 1.0
            
            return result
        except:
            return {}
    
    return {}

def aggregate_emotions_from_dataframe(location_data, emotion_columns):
    """
    Aggregate emotion scores from dictionary-format columns
    Returns a Series with emotion counts/scores
    """
    all_emotions = {}
    
    for col in emotion_columns:
        if col not in location_data.columns:
            continue
        
        for emotion_text in location_data[col]:
            parsed_emotions = parse_emotion_column(emotion_text)
            for emotion, score in parsed_emotions.items():
                if emotion not in all_emotions:
                    all_emotions[emotion] = 0
                all_emotions[emotion] += score
    
    if not all_emotions:
        return pd.Series()
    
    return pd.Series(all_emotions).sort_values(ascending=False)

# ==================== SMART HELPER FUNCTIONS ====================
def find_location_column(df, column_mapping):
    """Find the location column using smart mapping"""
    return column_mapping.get('location_column', None)

def safe_get_value(row, column_name, default=0, as_type='int'):
    """Safely get value from row with NaN handling"""
    value = row.get(column_name, default)
    
    if pd.isna(value):
        return default
    
    if as_type == 'int':
        try:
            return int(value)
        except:
            return default
    elif as_type == 'str':
        return str(value)
    elif as_type == 'float':
        try:
            return float(value)
        except:
            return default
    else:
        return value

# ==================== MAP CREATION ====================
def create_interactive_map(poi_gdf, route1_gdf, route2_gdf, route3_gdf, emotion_df, column_mapping):
    """Create Folium map with POI, routes, and synchronized emotions with layer control"""
    
    # Calculate map center
    center_lat = poi_gdf.geometry.centroid.y.mean()
    center_lon = poi_gdf.geometry.centroid.x.mean()
    
    # Initialize map with default OSM tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='OpenStreetMap',
        attr='© OpenStreetMap contributors'
    )
    
    # Add alternate basemap with proper attribution
    folium.TileLayer(
        tiles='CartoDB positron',
        name='Light Map',
        attr='© CartoDB',
        overlay=False,
        control=True
    ).add_to(m)
    
    # ========== ROUTE CONFIGURATION ==========
    route_configs = [
        {'gdf': route1_gdf, 'color': '#8B0000', 'name': 'First March Route', 'label': 'First March (Jan-Feb 1945)'},
        {'gdf': route2_gdf, 'color': '#DC143C', 'name': 'Second March Route', 'label': 'Second March (May 1945)'},
        {'gdf': route3_gdf, 'color': '#B22222', 'name': 'Third March Route', 'label': 'Third March (Jun 1945)'}
    ]
    
    poi_name_col = column_mapping['poi_name_column']
    death_count_col = column_mapping['death_count_column']
    
    # ========== ADD ROUTE LINES WITH FEATURE GROUPS ==========
    for config in route_configs:
        route_gdf = config['gdf']
        if route_gdf is None or route_gdf.empty:
            continue
        
        # Create FeatureGroup for this route (enables layer control)
        fg = folium.FeatureGroup(name=config['name'], show=True)
        
        for idx, row in route_gdf.iterrows():
            death_count = safe_get_value(row, death_count_col, default=0, as_type='int')
            line_weight = max(3, min(death_count / 50, 12))
            poi_name = safe_get_value(row, poi_name_col, default='Unknown', as_type='str')
            
            route_popup_html = f"""
            <div style="font-family: Arial; width: 220px; padding: 10px;">
                <h4 style="color: #8B0000; margin: 0 0 8px 0;">{config['label']}</h4>
                <p style="margin: 4px 0;"><strong>Location:</strong> {poi_name}</p>
                <p style="margin: 4px 0;"><strong>Deaths:</strong> {death_count} POWs</p>
                <p style="margin: 4px 0;"><strong>Month:</strong> {safe_get_value(row, 'Base_month', 'N/A', 'str')}</p>
                <p style="margin: 4px 0;"><strong>Segment:</strong> {safe_get_value(row, 'segment', 'N/A', 'str')}</p>
            </div>
            """
            
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, w=line_weight, c=config['color']: {
                    'color': c,
                    'weight': w,
                    'opacity': 0.8
                },
                popup=folium.Popup(route_popup_html, max_width=250),
                tooltip=poi_name
            ).add_to(fg)
        
        fg.add_to(m)
    
    # ========== ADD POI MARKERS WITH FEATURE GROUPS BY MARCH ==========
    location_col = column_mapping['location_column']
    emotion_cols = column_mapping['emotion_score_columns']
    emotion_format = column_mapping['emotion_format']
    
    # Create separate FeatureGroups for each march's POIs
    poi_feature_groups = {
        1: folium.FeatureGroup(name='First March POIs', show=True),
        2: folium.FeatureGroup(name='Second March POIs', show=True),
        3: folium.FeatureGroup(name='Third March POIs', show=True)
    }
    
    for idx, row in poi_gdf.iterrows():
        poi_name = safe_get_value(row, poi_name_col, default='Unknown', as_type='str')
        march_name = safe_get_value(row, 'march_name', default='Unknown March', as_type='str')
        death_count = safe_get_value(row, death_count_col, default=0, as_type='int')
        base_month = safe_get_value(row, 'Base_month', default='Unknown', as_type='str')
        march_id = safe_get_value(row, 'march_id', default=1, as_type='int')
        
        # Match with emotion data
        location_emotions = pd.DataFrame()
        if location_col and location_col in emotion_df.columns:
            try:
                search_term = poi_name.split()[0] if poi_name != 'Unknown' else poi_name
                location_emotions = emotion_df[
                    emotion_df[location_col].str.contains(search_term, case=False, na=False)
                ]
            except Exception as e:
                location_emotions = pd.DataFrame()
        
        # Determine dominant emotion and marker color
        if not location_emotions.empty and emotion_cols:
            available_cols = [col for col in emotion_cols if col in location_emotions.columns]
            
            if available_cols:
                if emotion_format == 'dictionary':
                    emotion_sums = aggregate_emotions_from_dataframe(location_emotions, available_cols)
                else:
                    emotion_sums = location_emotions[available_cols].sum()
                
                if len(emotion_sums) > 0:
                    dominant_emotion = emotion_sums.idxmax()
                    total_emotions = len(location_emotions)
                    
                    emotion_colors = {
                        'anger': 'red', 'fear': 'darkpurple', 'sadness': 'blue',
                        'joy': 'green', 'surprise': 'orange', 'disgust': 'darkred',
                        'neutral': 'gray', 'hunger': 'lightred', 'despair': 'black',
                        'cruelty': 'purple', 'death': 'darkred'
                    }
                    
                    marker_color = emotion_colors.get(dominant_emotion, 'gray')
                else:
                    dominant_emotion = 'unknown'
                    total_emotions = len(location_emotions)
                    marker_color = 'gray'
            else:
                dominant_emotion = 'unknown'
                total_emotions = len(location_emotions)
                marker_color = 'gray'
        else:
            dominant_emotion = 'No data'
            total_emotions = 0
            marker_color = 'lightgray'
        
        # Icon by march phase
        icon_map = {1: 'star', 2: 'flag', 3: 'certificate'}
        icon = icon_map.get(march_id, 'info-sign')
        
        # Create popup
        popup_html = f"""
        <div style="font-family: Arial; width: 240px; padding: 10px; background: #f5f5f5; border: 2px solid #8B7355; border-radius: 5px;">
            <h4 style="margin: 0 0 8px 0; color: #3E2723; border-bottom: 2px solid #8B0000;">{poi_name}</h4>
            
            <div style="background: #4A5D3F; color: white; padding: 6px; margin: 6px 0; border-radius: 3px;">
                <strong>{march_name}</strong><br>
                Month: {base_month} | Segment: {safe_get_value(row, 'segment', 'N/A', 'str')}
            </div>
            
            <div style="background: #2C1810; color: #FFD700; padding: 6px; margin: 6px 0; border-radius: 3px;">
                <strong>POW Deaths:</strong> {death_count}<br>
                <strong>Emotions:</strong> {total_emotions} records<br>
                <strong>Dominant:</strong> {dominant_emotion.upper()}
            </div>
            
            <div style="background: #5D4E37; color: #C3B091; padding: 5px; margin: 6px 0; border-radius: 3px; font-size: 10px;">
                Lat: {row.geometry.y:.5f} | Lon: {row.geometry.x:.5f}
            </div>
            
            <small style="color: #666;">Click for detailed analysis</small>
        </div>
        """
        
        # Add marker to the appropriate FeatureGroup
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{poi_name} ({march_name}) - {death_count} deaths",
            icon=folium.Icon(color=marker_color, icon=icon, prefix='glyphicon')
        ).add_to(poi_feature_groups[march_id])
    
    # Add all POI FeatureGroups to map
    for fg in poi_feature_groups.values():
        fg.add_to(m)
    
    # ========== ENHANCED LEGEND ==========
    legend_html = '''
    <div style="position: fixed; top: 10px; right: 10px; width: 220px; 
                background: rgba(255,255,255,0.98); border: 3px solid #8B7355; 
                border-radius: 8px; padding: 12px; font-size: 11px; z-index: 9999;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);">
        <h4 style="margin: 0 0 10px 0; color: #3E2723; border-bottom: 2px solid #8B0000; padding-bottom: 5px;">
            🗺️ Map Legend
        </h4>
        
        <p style="margin: 8px 0 4px 0; font-weight: bold; color: #3E2723; font-size: 10px;">
            📍 MARKER COLORS (Emotions):
        </p>
        <div style="display: flex; flex-direction: column; gap: 3px; margin-left: 5px;">
            <div style="display: flex; align-items: center;">
                <span style="width: 12px; height: 12px; background: red; display: inline-block; margin-right: 6px; border-radius: 2px;"></span>
                <span style="font-size: 10px;">Red - Anger/Disgust</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="width: 12px; height: 12px; background: blue; display: inline-block; margin-right: 6px; border-radius: 2px;"></span>
                <span style="font-size: 10px;">Blue - Sadness</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="width: 12px; height: 12px; background: purple; display: inline-block; margin-right: 6px; border-radius: 2px;"></span>
                <span style="font-size: 10px;">Purple - Fear</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="width: 12px; height: 12px; background: gray; display: inline-block; margin-right: 6px; border-radius: 2px;"></span>
                <span style="font-size: 10px;">Gray - Neutral</span>
            </div>
        </div>
        
        <p style="margin: 10px 0 4px 0; font-weight: bold; color: #3E2723; font-size: 10px;">
            🎖️ MARCH PHASE ICONS:
        </p>
        <div style="margin-left: 5px; font-size: 10px; line-height: 1.4;">
            <p style="margin: 2px 0;">⭐ Star - First March</p>
            <p style="margin: 2px 0;">🚩 Flag - Second March</p>
            <p style="margin: 2px 0;">🏅 Medal - Third March</p>
        </div>
        
        <p style="margin: 10px 0 4px 0; font-weight: bold; color: #3E2723; font-size: 10px;">
            📏 ROUTE LINE THICKNESS:
        </p>
        <p style="margin: 2px 0 0 5px; font-size: 10px;">
            Proportional to death count
        </p>
        
        <div style="margin-top: 10px; padding: 6px; background: #FFF8DC; border: 1px solid #DEB887; border-radius: 4px;">
            <p style="margin: 0; font-size: 9px; color: #8B7355; text-align: center;">
                💡 Use layer control (bottom-left)<br>to toggle routes and POIs
            </p>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # ========== LAYER CONTROL ==========
    # Add enhanced layer control with all layers
    folium.LayerControl(
        position='bottomleft',
        collapsed=False,
        autoZIndex=True
    ).add_to(m)
    
    return m

# ==================== ANALYSIS FUNCTIONS ====================
def analyze_location_emotions(location_name, hybrid_df, poi_row, column_mapping):
    """Analyze emotions for selected location"""
    
    location_col = column_mapping['location_column']
    emotion_cols = column_mapping['emotion_score_columns']
    emotion_format = column_mapping['emotion_format']
    sentiment_col = column_mapping['sentiment_column']
    temporal_col = column_mapping['temporal_column']
    
    if location_col is None or location_col not in hybrid_df.columns:
        return None
    
    # Try to match location name
    try:
        search_term = location_name.split()[0]
        location_data = hybrid_df[
            hybrid_df[location_col].str.contains(search_term, case=False, na=False)
        ]
    except Exception as e:
        return None
    
    if location_data.empty:
        return None
    
    if not emotion_cols:
        return None
    
    # Calculate statistics
    available_emotion_cols = [col for col in emotion_cols if col in location_data.columns]
    if not available_emotion_cols:
        return None
    
    # Aggregate emotions based on format
    if emotion_format == 'dictionary':
        emotion_counts = aggregate_emotions_from_dataframe(location_data, available_emotion_cols)
    else:
        emotion_counts = location_data[available_emotion_cols].sum().sort_values(ascending=False)
    
    if emotion_counts.empty:
        return None
    
    # Temporal analysis if available
    temporal_emotions = None
    if temporal_col and temporal_col in location_data.columns:
        if emotion_format == 'dictionary':
            temporal_emotions = {}
            for phase in location_data[temporal_col].unique():
                phase_data = location_data[location_data[temporal_col] == phase]
                temporal_emotions[phase] = aggregate_emotions_from_dataframe(phase_data, available_emotion_cols)
            temporal_emotions = pd.DataFrame(temporal_emotions).T
        else:
            temporal_emotions = location_data.groupby(temporal_col)[available_emotion_cols].sum()
    
    # Sentiment statistics
    avg_sentiment = None
    sentiment_std = None
    if sentiment_col and sentiment_col in location_data.columns:
        avg_sentiment = location_data[sentiment_col].mean()
        sentiment_std = location_data[sentiment_col].std()
    
    return {
        'total_records': len(location_data),
        'emotion_counts': emotion_counts,
        'temporal_emotions': temporal_emotions,
        'avg_sentiment': avg_sentiment,
        'sentiment_std': sentiment_std,
        'data': location_data,
        'poi_info': poi_row
    }

def create_location_charts(analysis_results, column_mapping):
    """Create Plotly charts for selected location"""
    charts = []
    
    poi_name_col = column_mapping['poi_name_column']
    poi_name = safe_get_value(analysis_results['poi_info'], poi_name_col, 'Location', 'str')
    
    # 1. Emotion Distribution Pie Chart
    fig_pie = px.pie(
        values=analysis_results['emotion_counts'].values,
        names=analysis_results['emotion_counts'].index,
        title=f"Emotion Distribution - {poi_name}",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.3
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=400)
    charts.append(fig_pie)
    
    # 2. Emotion Bar Chart
    fig_bar = px.bar(
        x=analysis_results['emotion_counts'].index,
        y=analysis_results['emotion_counts'].values,
        title="Emotion Frequency Count",
        labels={'x': 'Emotion Type', 'y': 'Count'},
        color=analysis_results['emotion_counts'].values,
        color_continuous_scale='Reds'
    )
    fig_bar.update_layout(showlegend=False, height=350)
    charts.append(fig_bar)
    
    # 3. Temporal Emotion Evolution (if available)
    if analysis_results['temporal_emotions'] is not None and not analysis_results['temporal_emotions'].empty:
        fig_temporal = px.line(
            analysis_results['temporal_emotions'],
            title="Emotion Evolution Over March Phases",
            markers=True,
            labels={'value': 'Emotion Count', 'variable': 'Emotion Type'}
        )
        fig_temporal.update_layout(hovermode='x unified', height=350)
        charts.append(fig_temporal)
    
    # 4. Sentiment Distribution
    if analysis_results['avg_sentiment'] is not None:
        sentiment_col = column_mapping['sentiment_column']
        if sentiment_col in analysis_results['data'].columns:
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Histogram(
                x=analysis_results['data'][sentiment_col],
                nbinsx=25,
                name='Sentiment Distribution',
                marker_color='indianred'
            ))
            fig_sent.add_vline(
                x=analysis_results['avg_sentiment'],
                line_dash="dash",
                line_color="darkred",
                annotation_text=f"Mean: {analysis_results['avg_sentiment']:.3f}",
                annotation_position="top"
            )
            fig_sent.update_layout(
                title="Sentiment Score Distribution",
                xaxis_title="Sentiment Score",
                yaxis_title="Frequency",
                showlegend=False,
                height=350
            )
            charts.append(fig_sent)
    
    return charts

# ==================== MAIN APP ====================
def main():
    # WWII-themed header
    st.markdown("""
    <div style="text-align: center; padding: 25px; 
                background: linear-gradient(135deg, #3E2723 0%, #5D4E37 100%); 
                color: #FFD700; border-radius: 10px; border: 3px solid #8B7355; 
                margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        <h1 style="margin: 0; font-family: 'Courier New', monospace; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            SANDAKAN-RANAU DEATH MARCHES
        </h1>
        <h3 style="color: #D7CCC8; margin: 10px 0; font-weight: 400;">
            Interactive 2D Emotional and Sentiment Pattern Analysis Dashboard (X, Y + Emotion)
        </h3>
        <p style="color: #C3B091; font-style: italic; margin: 5px 0;">
            Research Objective 2: Spatial-Emotional-Sentiment Analysis Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading emotion data and spatial layers..."):
        hybrid_df, location_df = load_data()
        poi_gdf, route1_gdf, route2_gdf, route3_gdf = load_spatial_data()
    
    # Check if data loaded successfully
    if poi_gdf is None or hybrid_df is None:
        st.error("⚠️ Failed to load required data. Please check file paths and data integrity.")
        st.stop()
    
    # ========== SMART AUTO-DETECTION ==========
    with st.spinner("🧠 Analyzing data structure and auto-detecting columns..."):
        column_mapping = analyze_data_structure(hybrid_df, poi_gdf)
    
    # Display detection results
    display_data_analysis(column_mapping)
    
    # Warning if critical columns not found
    if column_mapping['location_column'] is None:
        st.warning("⚠️ **No location column auto-detected in emotion data.** Emotion analysis will be limited.")
    
    if not column_mapping['emotion_score_columns']:
        st.warning("⚠️ **No emotion score columns detected.** Emotion visualization will be unavailable.")
    else:
        st.success(f"✅ Detected {len(column_mapping['emotion_score_columns'])} emotion columns in '{column_mapping['emotion_format']}' format")
    
    # Create two-column layout
    col_map, col_analysis = st.columns([1.3, 1])
    
    with col_map:
        st.subheader("🗺️ Interactive Death March Route Map")
        st.markdown("*Click on POI markers to view detailed emotion analysis in the right panel →*")
        
        # Create and display map
        death_march_map = create_interactive_map(
            poi_gdf, route1_gdf, route2_gdf, route3_gdf, hybrid_df, column_mapping
        )
        
        # Display map with interaction tracking
        map_data = st_folium(
            death_march_map,
            width=None,
            height=650,
            returned_objects=["last_object_clicked"]
        )
        
        # Detect clicked location
        clicked_location = None
        if map_data and map_data.get("last_object_clicked"):
            coords = map_data["last_object_clicked"]
            clicked_point = Point(coords['lng'], coords['lat'])
            distances = poi_gdf.geometry.distance(clicked_point)
            nearest_idx = distances.idxmin()
            poi_name_col = column_mapping['poi_name_column']
            clicked_location = poi_gdf.iloc[nearest_idx][poi_name_col]
    
    with col_analysis:
        st.subheader("📊 Location-Specific Emotion Analysis")
        
        # Location selector with map synchronization
        poi_name_col = column_mapping['poi_name_column']
        location_options = sorted(poi_gdf[poi_name_col].unique())
        default_idx = 0
        
        if clicked_location and clicked_location in location_options:
            default_idx = location_options.index(clicked_location)
        
        selected_location = st.selectbox(
            "🎯 Select Location:",
            options=location_options,
            index=default_idx,
            help="Click map markers or select from dropdown"
        )
        
        if selected_location:
            # Get POI information
            poi_row = poi_gdf[poi_gdf[poi_name_col] == selected_location].iloc[0]
            
            # Display key metrics with safe value extraction
            death_count_col = column_mapping['death_count_column']
            col1, col2, col3 = st.columns(3)
            with col1:
                death_value = safe_get_value(poi_row, death_count_col, 0, 'int')
                st.metric("Deaths", death_value)
            with col2:
                month_value = safe_get_value(poi_row, 'Base_month', 'N/A', 'str')
                st.metric("Month", month_value)
            with col3:
                march_name = safe_get_value(poi_row, 'march_name', 'Unknown', 'str')
                march_name = march_name.replace(' March', '')
                st.metric("March", march_name)
            
            # Analyze emotions for location
            analysis = analyze_location_emotions(selected_location, hybrid_df, poi_row, column_mapping)
            
            if analysis and analysis['total_records'] > 0:
                # Display emotion statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Emotion Records", analysis['total_records'])
                with col2:
                    if analysis['avg_sentiment'] is not None:
                        st.metric("Avg Sentiment", f"{analysis['avg_sentiment']:.3f}")
                
                # Display charts
                st.markdown("---")
                charts = create_location_charts(analysis, column_mapping)
                for chart in charts:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Raw data expander
                with st.expander("📄 View Raw Emotion Data"):
                    display_cols = []
                    potential_cols = ['sentence', 'text', 'entity_text', 
                                     column_mapping['sentiment_column'], 
                                     column_mapping['location_column']]
                    
                    for col in potential_cols:
                        if col and col in analysis['data'].columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        st.dataframe(
                            analysis['data'][display_cols].head(20),
                            use_container_width=True
                        )
                    else:
                        st.dataframe(analysis['data'].head(20), use_container_width=True)
            else:
                st.warning(f"⚠️ No emotion data available for **{selected_location}**")
                st.info("📌 This location exists in spatial data but no matching records found in emotion analysis dataset.")
                st.markdown("**Possible reasons:**")
                st.markdown("- Location name mismatch between spatial and emotion data")
                st.markdown("- No narrative text available for this location")
                st.markdown("- Data was filtered during preprocessing")

if __name__ == "__main__":
    main()


