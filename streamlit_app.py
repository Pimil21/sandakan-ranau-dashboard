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
    # Data paths
    HYBRID_EMOTIONS = "data/hybrid_emotion_analysis.csv"
    LOCATION_EMOTIONS = "data/location_emotions_hybrid.csv"
    
    # Spatial data paths - POI files
    POI_M1 = "spatial-data/POI_M1_3D.shp"
    POI_M2 = "spatial-data/POI_M2_3D.shp"
    POI_M3 = "spatial-data/POI_M3_3D.shp"
    
    # Spatial data paths - Route files
    ROUTE1 = "spatial-data/March1_Multipatch.shp"
    ROUTE2 = "spatial-data/March2_Multipatch.shp"
    ROUTE3 = "spatial-data/March3_Multipatch.shp"

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
@st.cache_data
def load_spatial_data():
    """Load shapefiles with 2D geometry extraction"""
    try:
        import fiona
        from shapely.geometry import shape, Point, mapping
        
        def read_as_2d(filepath):
            """Read shapefile and force 2D geometry"""
            features = []
            with fiona.open(filepath) as src:
                for feature in src:
                    # Get properties
                    props = feature['properties']
                    
                    # Get geometry and force 2D
                    geom = shape(feature['geometry'])
                    
                    # Convert to 2D
                    if hasattr(geom, 'coords'):
                        # Point or LineString
                        coords_2d = [(x, y) for x, y, *z in geom.coords] if geom.has_z else list(geom.coords)
                        if geom.geom_type == 'Point':
                            geom_2d = Point(coords_2d[0])
                        else:
                            geom_2d = type(geom)(coords_2d)
                    else:
                        geom_2d = geom
                    
                    features.append({'geometry': geom_2d, **props})
            
            return gpd.GeoDataFrame(features, crs='EPSG:4326')
        
        # Load POIs
        poi_m1 = read_as_2d(Config.POI_M1)
        poi_m2 = read_as_2d(Config.POI_M2)
        poi_m3 = read_as_2d(Config.POI_M3)
        
        # Add march info
        poi_m1['march_id'] = 1
        poi_m2['march_id'] = 2
        poi_m3['march_id'] = 3
        
        poi_m1['march_name'] = 'First March'
        poi_m2['march_name'] = 'Second March'
        poi_m3['march_name'] = 'Third March'
        
        # Combine
        poi_gdf = pd.concat([poi_m1, poi_m2, poi_m3], ignore_index=True)
        
        # Load routes
        route1_gdf = read_as_2d(Config.ROUTE1)
        route2_gdf = read_as_2d(Config.ROUTE2)
        route3_gdf = read_as_2d(Config.ROUTE3)
        
        st.success(f"✅ Loaded {len(poi_gdf)} POI locations (3D → 2D)")
        
        return poi_gdf, route1_gdf, route2_gdf, route3_gdf
        
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        return None, None, None, None

# ==================== MAP CREATION ====================
def create_interactive_map(poi_gdf, route1_gdf, route2_gdf, route3_gdf, emotion_df):
    """Create Folium map with POI, routes, and death statistics"""
    
    # Calculate map center
    center_lat = poi_gdf.geometry.centroid.y.mean()
    center_lon = poi_gdf.geometry.centroid.x.mean()
    
    # Initialize map with historical tile
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='OpenStreetMap',
        attr='Sandakan-Ranau Death Marches 1945'
    )
    
    # Add alternate basemaps
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
    
    # ========== ADD ROUTE LINES WITH DEATH STATISTICS ==========
    route_configs = [
        {
            'gdf': route1_gdf,
            'color': '#8B0000',  # Dark red
            'name': 'First March (Jan-Feb 1945)',
            'size_field': 'size_M1'
        },
        {
            'gdf': route2_gdf,
            'color': '#DC143C',  # Crimson
            'name': 'Second March (May 1945)',
            'size_field': 'size_M1'  # Adjust if different
        },
        {
            'gdf': route3_gdf,
            'color': '#B22222',  # Fire brick
            'name': 'Third March (Jun 1945)',
            'size_field': 'size_M1'  # Adjust if different
        }
    ]
    
    for config in route_configs:
        route_gdf = config['gdf']
        if route_gdf is not None and not route_gdf.empty:
            # Create feature group for this march
            fg = folium.FeatureGroup(name=config['name'])
            
            for idx, row in route_gdf.iterrows():
                # Get death count for line weight
                death_count = row.get(config['size_field'], 10)
                line_weight = max(3, min(death_count / 50, 15))  # Scale between 3-15
                
                # Create popup with segment info
                popup_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4 style="color: #8B0000; margin: 0 0 10px 0;">{config['name']}</h4>
                    <p style="margin: 5px 0;"><strong>Location:</strong> {row.get('POI_Name', 'Unknown')}</p>
                    <p style="margin: 5px 0;"><strong>💀 Deaths:</strong> {int(death_count)}</p>
                    <p style="margin: 5px 0;"><strong>📅 Base Month:</strong> {row.get('Base_month', 'N/A')}</p>
                    <p style="margin: 5px 0;"><strong>Segment:</strong> {row.get('segment', 'N/A')}</p>
                </div>
                """
                
                # Add line with varying thickness
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x, w=line_weight, c=config['color']: {
                        'color': c,
                        'weight': w,
                        'opacity': 0.7,
                        'lineJoin': 'round'
                    },
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{row.get('POI_Name', 'Route segment')}"
                ).add_to(fg)
            
            fg.add_to(m)
    
    # ========== ADD POI MARKERS WITH EMOTIONS ==========
    # Create colormap for death counts
    max_deaths = poi_gdf['size_M1'].max() if 'size_M1' in poi_gdf.columns else 100
    
    for idx, row in poi_gdf.iterrows():
        poi_name = row['POI_Name']
        march_name = row['march_name']
        death_count = row.get('size_M1', 0)
        base_month = row.get('Base_month', 'Unknown')
        
        # Try to match with emotion data (fuzzy matching)
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
                
                # Color by dominant emotion
                emotion_colors = {
                    'anger': 'red',
                    'fear': 'darkpurple',
                    'sadness': 'blue',
                    'joy': 'green',
                    'surprise': 'orange',
                    'disgust': 'darkred',
                    'neutral': 'gray',
                    'hunger': 'lightred',
                    'despair': 'black',
                    'cruelty': 'purple'
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
        
        # Icon shape by march
        icon_shapes = {1: 'star', 2: 'flag', 3: 'certificate'}
        icon = icon_shapes.get(row['march_id'], 'info-sign')
        
        # Create detailed popup
        popup_html = f"""
        <div style="font-family: 'Courier New', monospace; width: 260px; padding: 12px; 
                    background: linear-gradient(135deg, #E8DCC4, #D7CCC8); 
                    border: 3px solid #8B7355; border-radius: 5px;">
            <h3 style="margin: 0 0 12px 0; color: #3E2723; border-bottom: 2px solid #8B0000; padding-bottom: 5px;">
                📍 {poi_name}
            </h3>
            
            <div style="background: #4A5D3F; color: #E8DCC4; padding: 8px; border-radius: 3px; margin: 10px 0;">
                <strong>🎖️ March Phase:</strong> {march_name}<br>
                <strong>📅 Month:</strong> {base_month}<br>
                <strong>🔢 Segment:</strong> {row.get('segment', 'N/A')}
            </div>
            
            <div style="background: #2C1810; color: #FFD700; padding: 8px; border-radius: 3px; margin: 10px 0;">
                <strong>💀 POW Deaths:</strong> {int(death_count)} prisoners<br>
                <strong>📊 Emotion Records:</strong> {total_emotions}<br>
                <strong>😔 Dominant Emotion:</strong> 
                <span style="color: {marker_color}; font-weight: bold; text-transform: uppercase;">
                    {dominant_emotion}
                </span>
            </div>
            
            <div style="background: #5D4E37; color: #C3B091; padding: 6px; border-radius: 3px; font-size: 11px;">
                <strong>Coordinates:</strong><br>
                Lat: {row.geometry.y:.5f}°<br>
                Lon: {row.geometry.x:.5f}°
            </div>
            
            <hr style="border-color: #8B7355; margin: 8px 0;">
            <small style="color: #5D4E37; font-style: italic;">
                ⬅️ Click marker to view detailed analysis in dashboard panel
            </small>
        </div>
        """
        
        # Add marker
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{poi_name} ({march_name}) - {int(death_count)} deaths",
            icon=folium.Icon(
                color=marker_color,
                icon=icon,
                prefix='glyphicon'
            )
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; top: 10px; right: 10px; width: 220px; 
                background: rgba(227, 220, 196, 0.95); 
                border: 3px solid #8B7355; border-radius: 8px; 
                padding: 15px; font-family: 'Courier New', monospace; 
                font-size: 12px; z-index: 9999;">
        <h4 style="margin: 0 0 10px 0; color: #3E2723; border-bottom: 2px solid #8B0000;">
            🎖️ Map Legend
        </h4>
        <p style="margin: 5px 0;"><b>Marker Colors (Emotions):</b></p>
        <p style="margin: 3px 0; color: red;">⚫ Red - Anger/Disgust</p>
        <p style="margin: 3px 0; color: blue;">⚫ Blue - Sadness</p>
        <p style="margin: 3px 0; color: purple;">⚫ Purple - Fear</p>
        <p style="margin: 3px 0; color: gray;">⚫ Gray - Neutral</p>
        
        <p style="margin: 10px 0 5px 0;"><b>Marker Shapes:</b></p>
        <p style="margin: 3px 0;">⭐ Star - First March</p>
        <p style="margin: 3px 0;">🚩 Flag - Second March</p>
        <p style="margin: 3px 0;">🎖️ Medal - Third March</p>
        
        <p style="margin: 10px 0 5px 0;"><b>Line Thickness:</b></p>
        <p style="margin: 3px 0;">Proportional to death count</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl(position='bottomleft').add_to(m)
    
    # Add fullscreen option
    folium.plugins.Fullscreen(position='topleft').add_to(m)
    
    return m

# ==================== ANALYSIS FUNCTIONS ====================
def analyze_location_emotions(location_name, hybrid_df, poi_row):
    """Generate comprehensive emotion statistics for selected location"""
    
    # Fuzzy match location names
    location_data = hybrid_df[
        hybrid_df['location'].str.contains(location_name.split()[0], case=False, na=False)
    ]
    
    if location_data.empty:
        return None
    
    # Get emotion columns
    emotion_cols = [col for col in ['anger', 'fear', 'sadness', 'joy', 'surprise', 
                                     'disgust', 'neutral', 'hunger', 'despair', 'cruelty'] 
                    if col in location_data.columns]
    
    # Calculate statistics
    emotion_counts = location_data[emotion_cols].sum().sort_values(ascending=False)
    
    # Temporal analysis if available
    temporal_col = None
    if 'march_phase' in location_data.columns:
        temporal_col = 'march_phase'
    elif 'date' in location_data.columns:
        temporal_col = 'date'
    
    temporal_emotions = None
    if temporal_col:
        temporal_emotions = location_data.groupby(temporal_col)[emotion_cols].sum()
    
    # Sentiment statistics
    avg_sentiment = location_data['sentiment_score'].mean() if 'sentiment_score' in location_data.columns else None
    sentiment_std = location_data['sentiment_score'].std() if 'sentiment_score' in location_data.columns else None
    
    return {
        'total_records': len(location_data),
        'emotion_counts': emotion_counts,
        'temporal_emotions': temporal_emotions,
        'avg_sentiment': avg_sentiment,
        'sentiment_std': sentiment_std,
        'data': location_data,
        'poi_info': poi_row
    }

def create_location_charts(analysis_results):
    """Create detailed Plotly charts for selected location"""
    charts = []
    
    # 1. Emotion Distribution Pie Chart
    fig_pie = px.pie(
        values=analysis_results['emotion_counts'].values,
        names=analysis_results['emotion_counts'].index,
        title=f"Emotion Distribution - {analysis_results['poi_info']['POI_Name']}",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.3
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
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
    charts.append(fig_bar)
    
    # 3. Temporal Evolution (if available)
    if analysis_results['temporal_emotions'] is not None:
        fig_temporal = px.line(
            analysis_results['temporal_emotions'],
            title="Emotion Evolution Over March Phases",
            markers=True,
            labels={'value': 'Emotion Count', 'variable': 'Emotion Type'}
        )
        fig_temporal.update_layout(hovermode='x unified')
        charts.append(fig_temporal)
    
    # 4. Sentiment Distribution
    if analysis_results['avg_sentiment'] is not None:
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Histogram(
            x=analysis_results['data']['sentiment_score'],
            nbinsx=30,
            name='Sentiment Distribution',
            marker_color='indianred'
        ))
        fig_sent.add_vline(
            x=analysis_results['avg_sentiment'],
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"Mean: {analysis_results['avg_sentiment']:.3f}"
        )
        fig_sent.update_layout(
            title="Sentiment Score Distribution",
            xaxis_title="Sentiment Score",
            yaxis_title="Frequency",
            showlegend=False
        )
        charts.append(fig_sent)
    
    return charts

# ==================== MAIN APP ====================
def main():
    # Apply CSS theme
    st.markdown("""
    <style>
    .main {
        background-color: #E8DCC4;
    }
    .stMetric {
        background: linear-gradient(135deg, #2C1810 0%, #3E2723 100%);
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #8B7355;
        color: #FFD700 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 25px; 
                background: linear-gradient(135deg, #3E2723 0%, #5D4E37 100%); 
                color: #E8DCC4; border-radius: 10px; margin-bottom: 30px;
                border: 3px solid #8B7355; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        <h1 style="color: #FFD700; font-family: 'Courier New', monospace; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            🎖️ SANDAKAN-RANAU DEATH MARCHES: SPATIAL-EMOTION ANALYSIS 🎖️
        </h1>
        <h3 style="color: #D7CCC8; font-weight: 400;">
            ⚔️ Interactive 4D Geovisualization (X, Y, T + Emotion) ⚔️
        </h3>
        <p style="color: #C3B091; font-style: italic;">
            🗺️ Research Objective 3: Interactive GIS storytelling with synchronized spatial-temporal-emotional analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading emotion data and spatial layers..."):
        hybrid_df, location_df = load_data()
        poi_gdf, route1_gdf, route2_gdf, route3_gdf = load_spatial_data()
    
    if poi_gdf is None or hybrid_df is None:
        st.error("⚠️ Failed to load required data. Please check file paths.")
        return
    
    # Create layout
    col_map, col_analysis = st.columns([1.3, 1])
    
    with col_map:
        st.subheader("🗺️ Interactive Death March Route Map")
        st.markdown("*Click on POI markers to view detailed emotion analysis →*")
        
        # Create and display map
        death_march_map = create_interactive_map(
            poi_gdf, route1_gdf, route2_gdf, route3_gdf, hybrid_df
        )
        
        # Display map with interaction
        map_data = st_folium(
            death_march_map,
            width=None,
            height=650,
            returned_objects=["last_object_clicked"]
        )
        
        # Detect clicked location
        clicked_location = None
        if map_data and map_data.get("last_object_clicked"):
            clicked_coords = map_data["last_object_clicked"]
            clicked_point = gpd.points_from_xy([clicked_coords['lng']], [clicked_coords['lat']])[0]
            distances = poi_gdf.geometry.distance(clicked_point)
            nearest_idx = distances.idxmin()
            clicked_location = poi_gdf.iloc[nearest_idx]['POI_Name']
    
    with col_analysis:
        st.subheader("📊 Location-Specific Emotion Analysis")
        
        # Location selector
        location_options = sorted(poi_gdf['POI_Name'].unique())
        
        if clicked_location and clicked_location in location_options:
            default_idx = location_options.index(clicked_location)
        else:
            default_idx = 0
        
        selected_location = st.selectbox(
            "🎯 Select Location:",
            options=location_options,
            index=default_idx,
            help="Click map markers or select from dropdown"
        )
        
        if selected_location:
            poi_row = poi_gdf[poi_gdf['POI_Name'] == selected_location].iloc[0]
            
            # Display POI info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💀 Deaths", int(poi_row.get('size_M1', 0)))
            with col2:
                st.metric("🗓️ Month", poi_row.get('Base_month', 'N/A'))
            with col3:
                st.metric("🎖️ March", poi_row['march_name'])
            
            # Analyze emotions
            analysis = analyze_location_emotions(selected_location, hybrid_df, poi_row)
            
            if analysis and analysis['total_records'] > 0:
                st.metric("📝 Emotion Records", analysis['total_records'])
                
                if analysis['avg_sentiment'] is not None:
                    st.metric("💭 Avg Sentiment", f"{analysis['avg_sentiment']:.3f}")
                
                # Display charts
                charts = create_location_charts(analysis)
                for chart in charts:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Raw data expander
                with st.expander("📄 View Raw Emotion Data"):
                    st.dataframe(analysis['data'][['location', 'sentence', 'dominant_emotion', 'sentiment_score']], use_container_width=True)
            else:
                st.warning(f"⚠️ No emotion data available for **{selected_location}**")
                st.info("This location exists in spatial data but no matching records found in emotion analysis.")

if __name__ == "__main__":
    main()


