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
    """Load 2D shapefiles - should work perfectly now!"""
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

# ==================== MAP CREATION ====================
def create_interactive_map(poi_gdf, route1_gdf, route2_gdf, route3_gdf, emotion_df):
    """Create Folium map with POI, routes, and synchronized emotions"""
    
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
    
    # Add alternate basemaps with proper attribution
    folium.TileLayer(
        tiles='CartoDB positron',
        name='Light Map',
        attr='© CartoDB',
        overlay=False,
        control=True
    ).add_to(m)
    
    # ========== ADD ROUTE LINES ==========
    route_configs = [
        {'gdf': route1_gdf, 'color': '#8B0000', 'name': 'First March (Jan-Feb 1945)'},
        {'gdf': route2_gdf, 'color': '#DC143C', 'name': 'Second March (May 1945)'},
        {'gdf': route3_gdf, 'color': '#B22222', 'name': 'Third March (Jun 1945)'}
    ]
    
    for config in route_configs:
        route_gdf = config['gdf']
        if route_gdf is None or route_gdf.empty:
            continue
        
        fg = folium.FeatureGroup(name=config['name'])
        
        for idx, row in route_gdf.iterrows():
            # Get death count for line thickness
            death_count = row.get('size_M1', 10)
            line_weight = max(3, min(death_count / 50, 12))
            
            # Popup content
            popup_html = f"""
            <div style="font-family: Arial; width: 220px; padding: 10px;">
                <h4 style="color: #8B0000; margin: 0 0 8px 0;">{config['name']}</h4>
                <p style="margin: 4px 0;"><strong>Location:</strong> {row.get('POI_Name', 'Unknown')}</p>
                <p style="margin: 4px 0;"><strong>💀 Deaths:</strong> {int(death_count)}</p>
                <p style="margin: 4px 0;"><strong>📅 Month:</strong> {row.get('Base_month', 'N/A')}</p>
                <p style="margin: 4px 0;"><strong>Segment:</strong> {row.get('segment', 'N/A')}</p>
                <p style="margin: 4px 0;"><strong>Days:</strong> {row.get('Start_day', '?')}-{row.get('End_day', '?')}</p>
            </div>
            """
            
            # Add polyline
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, w=line_weight, c=config['color']: {
                    'color': c,
                    'weight': w,
                    'opacity': 0.8
                },
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=row.get('POI_Name', 'Route segment')
            ).add_to(fg)
        
        fg.add_to(m)
    
    # ========== ADD POI MARKERS ==========
    for idx, row in poi_gdf.iterrows():
        poi_name = row['POI_Name']
        march_name = row['march_name']
        death_count = row.get('size_M1', 0)
        base_month = row.get('Base_month', 'Unknown')
        
        # Match with emotion data
        location_emotions = emotion_df[
            emotion_df['location'].str.contains(poi_name.split()[0], case=False, na=False)
        ]
        
        if not location_emotions.empty:
            emotion_cols = ['anger', 'fear', 'sadness', 'joy', 'surprise', 'disgust', 'neutral']
            available_cols = [col for col in emotion_cols if col in location_emotions.columns]
            
            if available_cols:
                emotion_sums = location_emotions[available_cols].sum()
                dominant_emotion = emotion_sums.idxmax()
                total_emotions = len(location_emotions)
                
                emotion_colors = {
                    'anger': 'red', 'fear': 'darkpurple', 'sadness': 'blue',
                    'joy': 'green', 'surprise': 'orange', 'disgust': 'darkred',
                    'neutral': 'gray', 'hunger': 'lightred', 'despair': 'black'
                }
                marker_color = emotion_colors.get(dominant_emotion, 'gray')
            else:
                dominant_emotion = 'unknown'
                total_emotions = len(location_emotions)
                marker_color = 'gray'
        else:
            dominant_emotion = 'No data'
            total_emotions = 0
            marker_color = 'lightgray'
        
        # Icon by march
        icon_map = {1: 'star', 2: 'flag', 3: 'certificate'}
        icon = icon_map.get(row['march_id'], 'info-sign')
        
        # Popup
        popup_html = f"""
        <div style="font-family: 'Courier New', monospace; width: 260px; padding: 12px; 
                    background: #E8DCC4; border: 3px solid #8B7355; border-radius: 5px;">
            <h3 style="margin: 0 0 10px 0; color: #3E2723; border-bottom: 2px solid #8B0000;">
                📍 {poi_name}
            </h3>
            
            <div style="background: #4A5D3F; color: #E8DCC4; padding: 8px; border-radius: 3px; margin: 8px 0;">
                <strong>🎖️ {march_name}</strong><br>
                <strong>📅 Month:</strong> {base_month}<br>
                <strong>🔢 Segment:</strong> {row.get('segment', 'N/A')}
            </div>
            
            <div style="background: #2C1810; color: #FFD700; padding: 8px; border-radius: 3px; margin: 8px 0;">
                <strong>💀 POW Deaths:</strong> {int(death_count)}<br>
                <strong>📊 Emotion Records:</strong> {total_emotions}<br>
                <strong>😔 Dominant:</strong> <span style="color: {marker_color}; text-transform: uppercase;">{dominant_emotion}</span>
            </div>
            
            <small style="color: #5D4E37; font-style: italic;">
                ⬅️ Click to view analysis in panel
            </small>
        </div>
        """
        
        # Add marker
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{poi_name} - {int(death_count)} deaths",
            icon=folium.Icon(color=marker_color, icon=icon, prefix='glyphicon')
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; top: 10px; right: 10px; width: 200px; 
                background: rgba(255,255,255,0.9); border: 2px solid #8B7355; 
                border-radius: 5px; padding: 10px; font-size: 11px; z-index: 9999;">
        <h4 style="margin: 0 0 8px 0; color: #3E2723;">🎖️ Legend</h4>
        <p style="margin: 3px 0;"><b>Emotions:</b></p>
        <p style="margin: 2px 0;">⚫ Red - Anger</p>
        <p style="margin: 2px 0;">⚫ Blue - Sadness</p>
        <p style="margin: 2px 0;">⚫ Purple - Fear</p>
        <p style="margin: 6px 0 3px 0;"><b>March Icons:</b></p>
        <p style="margin: 2px 0;">⭐ First | 🚩 Second | 🎖️ Third</p>
        <p style="margin: 6px 0 3px 0;"><b>Line Thickness:</b></p>
        <p style="margin: 2px 0;">Proportional to deaths</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Layer control
    folium.LayerControl(position='bottomleft').add_to(m)
    
    return m

    # ========== ADD ROUTE LINES ==========
    route_configs = [
        {'gdf': route1_gdf, 'color': '#8B0000', 'name': 'First March (Jan-Feb 1945)'},
        {'gdf': route2_gdf, 'color': '#DC143C', 'name': 'Second March (May 1945)'},
        {'gdf': route3_gdf, 'color': '#B22222', 'name': 'Third March (Jun 1945)'}
    ]
    
    for config in route_configs:
        route_gdf = config['gdf']
        if route_gdf is None or route_gdf.empty:
            continue
        
        fg = folium.FeatureGroup(name=config['name'])
        
        for idx, row in route_gdf.iterrows():
            # Get death count for line thickness
            death_count = row.get('size_M1', 10)
            line_weight = max(3, min(death_count / 50, 12))
            
            # Popup content
            popup_html = f"""
            <div style="font-family: Arial; width: 220px; padding: 10px;">
                <h4 style="color: #8B0000; margin: 0 0 8px 0;">{config['name']}</h4>
                <p style="margin: 4px 0;"><strong>Location:</strong> {row.get('POI_Name', 'Unknown')}</p>
                <p style="margin: 4px 0;"><strong>💀 Deaths:</strong> {int(death_count)}</p>
                <p style="margin: 4px 0;"><strong>📅 Month:</strong> {row.get('Base_month', 'N/A')}</p>
                <p style="margin: 4px 0;"><strong>Segment:</strong> {row.get('segment', 'N/A')}</p>
                <p style="margin: 4px 0;"><strong>Days:</strong> {row.get('Start_day', '?')}-{row.get('End_day', '?')}</p>
            </div>
            """
            
            # Add polyline
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, w=line_weight, c=config['color']: {
                    'color': c,
                    'weight': w,
                    'opacity': 0.8
                },
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=row.get('POI_Name', 'Route segment')
            ).add_to(fg)
        
        fg.add_to(m)
    
    # ========== ADD POI MARKERS ==========
    for idx, row in poi_gdf.iterrows():
        poi_name = row['POI_Name']
        march_name = row['march_name']
        death_count = row.get('size_M1', 0)
        base_month = row.get('Base_month', 'Unknown')
        
        # Match with emotion data
        location_emotions = emotion_df[
            emotion_df['location'].str.contains(poi_name.split()[0], case=False, na=False)
        ]
        
        if not location_emotions.empty:
            # Calculate dominant emotion
            emotion_cols = ['anger', 'fear', 'sadness', 'joy', 'surprise', 'disgust', 'neutral']
            available_cols = [col for col in emotion_cols if col in location_emotions.columns]
            
            if available_cols:
                emotion_sums = location_emotions[available_cols].sum()
                dominant_emotion = emotion_sums.idxmax()
                total_emotions = len(location_emotions)
                
                # Color mapping
                emotion_colors = {
                    'anger': 'red', 'fear': 'darkpurple', 'sadness': 'blue',
                    'joy': 'green', 'surprise': 'orange', 'disgust': 'darkred',
                    'neutral': 'gray', 'hunger': 'lightred', 'despair': 'black'
                }
                marker_color = emotion_colors.get(dominant_emotion, 'gray')
            else:
                dominant_emotion = 'unknown'
                total_emotions = len(location_emotions)
                marker_color = 'gray'
        else:
            dominant_emotion = 'No data'
            total_emotions = 0
            marker_color = 'lightgray'
        
        # Icon by march
        icon_map = {1: 'star', 2: 'flag', 3: 'certificate'}
        icon = icon_map.get(row['march_id'], 'info-sign')
        
        # Popup
        popup_html = f"""
        <div style="font-family: 'Courier New', monospace; width: 260px; padding: 12px; 
                    background: #E8DCC4; border: 3px solid #8B7355; border-radius: 5px;">
            <h3 style="margin: 0 0 10px 0; color: #3E2723; border-bottom: 2px solid #8B0000;">
                📍 {poi_name}
            </h3>
            
            <div style="background: #4A5D3F; color: #E8DCC4; padding: 8px; border-radius: 3px; margin: 8px 0;">
                <strong>🎖️ {march_name}</strong><br>
                <strong>📅 Month:</strong> {base_month}<br>
                <strong>🔢 Segment:</strong> {row.get('segment', 'N/A')}
            </div>
            
            <div style="background: #2C1810; color: #FFD700; padding: 8px; border-radius: 3px; margin: 8px 0;">
                <strong>💀 POW Deaths:</strong> {int(death_count)}<br>
                <strong>📊 Emotion Records:</strong> {total_emotions}<br>
                <strong>😔 Dominant:</strong> <span style="color: {marker_color}; text-transform: uppercase;">{dominant_emotion}</span>
            </div>
            
            <small style="color: #5D4E37; font-style: italic;">
                ⬅️ Click to view analysis in panel
            </small>
        </div>
        """
        
        # Add marker
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{poi_name} - {int(death_count)} deaths",
            icon=folium.Icon(color=marker_color, icon=icon, prefix='glyphicon')
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; top: 10px; right: 10px; width: 200px; 
                background: rgba(255,255,255,0.9); border: 2px solid #8B7355; 
                border-radius: 5px; padding: 10px; font-size: 11px; z-index: 9999;">
        <h4 style="margin: 0 0 8px 0; color: #3E2723;">🎖️ Legend</h4>
        <p style="margin: 3px 0;"><b>Emotions:</b></p>
        <p style="margin: 2px 0;">⚫ Red - Anger</p>
        <p style="margin: 2px 0;">⚫ Blue - Sadness</p>
        <p style="margin: 2px 0;">⚫ Purple - Fear</p>
        <p style="margin: 6px 0 3px 0;"><b>March Icons:</b></p>
        <p style="margin: 2px 0;">⭐ First | 🚩 Second | 🎖️ Third</p>
        <p style="margin: 6px 0 3px 0;"><b>Line Thickness:</b></p>
        <p style="margin: 2px 0;">Proportional to deaths</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Layer control
    folium.LayerControl(position='bottomleft').add_to(m)
    folium.plugins.Fullscreen().add_to(m)
    
    return m

# ==================== ANALYSIS ====================
def analyze_location_emotions(location_name, hybrid_df, poi_row):
    """Analyze emotions for selected location"""
    location_data = hybrid_df[
        hybrid_df['location'].str.contains(location_name.split()[0], case=False, na=False)
    ]
    
    if location_data.empty:
        return None
    
    emotion_cols = [col for col in ['anger', 'fear', 'sadness', 'joy', 'surprise', 
                                     'disgust', 'neutral', 'hunger', 'despair'] 
                    if col in location_data.columns]
    
    emotion_counts = location_data[emotion_cols].sum().sort_values(ascending=False)
    
    avg_sentiment = location_data['sentiment_score'].mean() if 'sentiment_score' in location_data.columns else None
    
    return {
        'total_records': len(location_data),
        'emotion_counts': emotion_counts,
        'avg_sentiment': avg_sentiment,
        'data': location_data,
        'poi_info': poi_row
    }

def create_location_charts(analysis_results):
    """Create charts for location"""
    charts = []
    
    # Pie chart
    fig_pie = px.pie(
        values=analysis_results['emotion_counts'].values,
        names=analysis_results['emotion_counts'].index,
        title=f"Emotions - {analysis_results['poi_info']['POI_Name']}",
        hole=0.3
    )
    charts.append(fig_pie)
    
    # Bar chart
    fig_bar = px.bar(
        x=analysis_results['emotion_counts'].index,
        y=analysis_results['emotion_counts'].values,
        title="Emotion Frequency",
        labels={'x': 'Emotion', 'y': 'Count'},
        color=analysis_results['emotion_counts'].values,
        color_continuous_scale='Reds'
    )
    charts.append(fig_bar)
    
    # Sentiment histogram
    if analysis_results['avg_sentiment'] is not None:
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Histogram(
            x=analysis_results['data']['sentiment_score'],
            nbinsx=25,
            marker_color='indianred'
        ))
        fig_sent.add_vline(
            x=analysis_results['avg_sentiment'],
            line_dash="dash",
            annotation_text=f"Mean: {analysis_results['avg_sentiment']:.3f}"
        )
        fig_sent.update_layout(title="Sentiment Distribution", xaxis_title="Score", yaxis_title="Frequency")
        charts.append(fig_sent)
    
    return charts

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 25px; 
                background: linear-gradient(135deg, #3E2723 0%, #5D4E37 100%); 
                color: #FFD700; border-radius: 10px; border: 3px solid #8B7355; margin-bottom: 20px;">
        <h1 style="margin: 0;">🎖️ SANDAKAN-RANAU DEATH MARCHES 🎖️</h1>
        <h3 style="color: #D7CCC8; margin: 10px 0;">⚔️ Interactive 4D Geovisualization (X, Y, T + Emotion) ⚔️</h3>
        <p style="color: #C3B091;">📊 Research Objective 3: Spatial-Temporal-Emotional Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        hybrid_df, location_df = load_data()
        poi_gdf, route1_gdf, route2_gdf, route3_gdf = load_spatial_data()
    
    if poi_gdf is None or hybrid_df is None:
        st.stop()
    
    # Layout
    col_map, col_analysis = st.columns([1.3, 1])
    
    with col_map:
        st.subheader("🗺️ Interactive Map")
        
        death_march_map = create_interactive_map(
            poi_gdf, route1_gdf, route2_gdf, route3_gdf, hybrid_df
        )
        
        map_data = st_folium(death_march_map, width=None, height=650)
        
        # Detect clicks
        clicked_location = None
        if map_data and map_data.get("last_object_clicked"):
            coords = map_data["last_object_clicked"]
            from shapely.geometry import Point
            clicked_point = Point(coords['lng'], coords['lat'])
            distances = poi_gdf.geometry.distance(clicked_point)
            clicked_location = poi_gdf.iloc[distances.idxmin()]['POI_Name']
    
    with col_analysis:
        st.subheader("📊 Location Analysis")
        
        locations = sorted(poi_gdf['POI_Name'].unique())
        default_idx = locations.index(clicked_location) if clicked_location and clicked_location in locations else 0
        
        selected_location = st.selectbox("Select Location:", locations, index=default_idx)
        
        if selected_location:
            poi_row = poi_gdf[poi_gdf['POI_Name'] == selected_location].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("💀 Deaths", int(poi_row.get('size_M1', 0)))
            col2.metric("📅 Month", poi_row.get('Base_month', 'N/A'))
            col3.metric("🎖️ March", poi_row['march_name'])
            
            analysis = analyze_location_emotions(selected_location, hybrid_df, poi_row)
            
            if analysis:
                st.metric("📝 Emotion Records", analysis['total_records'])
                if analysis['avg_sentiment']:
                    st.metric("💭 Avg Sentiment", f"{analysis['avg_sentiment']:.3f}")
                
                for chart in create_location_charts(analysis):
                    st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning(f"No emotion data for {selected_location}")

if __name__ == "__main__":
    main()

