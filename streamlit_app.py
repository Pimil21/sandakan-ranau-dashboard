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

# ==================== HELPER FUNCTIONS ====================
def find_location_column(df):
    """Find the location column name in DataFrame"""
    possible_names = ['location', 'Location', 'poi_name', 'POI_Name', 'place', 'Place', 'name', 'Name']
    for col_name in possible_names:
        if col_name in df.columns:
            return col_name
    return None

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
    
    # Add alternate basemap with proper attribution
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
            # Handle NaN values
        if pd.isna(death_count):
            death_count = 0
            line_weight = max(3, min(death_count / 50, 12))
           
            # Popup content for routes
            route_popup_html = f"""
            <div style="font-family: Arial; width: 220px; padding: 10px;">
                <h4 style="color: #8B0000; margin: 0 0 8px 0;">{config['name']}</h4>
                <p style="margin: 4px 0;"><strong>Location:</strong> {row.get('POI_Name', 'Unknown')}</p>
                <p style="margin: 4px 0;"><strong>Deaths:</strong> {int(death_count)} POWs</p>
                <p style="margin: 4px 0;"><strong>Month:</strong> {row.get('Base_month', 'N/A')}</p>
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
                popup=folium.Popup(route_popup_html, max_width=250),
                tooltip=row.get('POI_Name', 'Route segment')
            ).add_to(fg)
        
        fg.add_to(m)
    
        # ========== ADD POI MARKERS ==========
    # Find location column name in emotion data
    location_col = find_location_column(emotion_df)
    
    for idx, row in poi_gdf.iterrows():
        poi_name = row['POI_Name']
        march_name = row['march_name']
        death_count = row.get('size_M1', 0) if pd.notna(row.get('size_M1')) else 0
        
        # Handle NaN values
        if pd.isna(death_count):
            death_count = 0
        
        base_month = row.get('Base_month', 'Unknown')
        
        # Match with emotion data
        location_emotions = pd.DataFrame()
        if location_col:
            try:
                # Try to match location name (fuzzy matching by first word)
                location_emotions = emotion_df[
                    emotion_df[location_col].str.contains(poi_name.split()[0], case=False, na=False)
                ]
            except Exception as e:
                location_emotions = pd.DataFrame()
        
        # Determine dominant emotion and marker color
        if not location_emotions.empty:
            emotion_cols = ['anger', 'fear', 'sadness', 'joy', 'surprise', 'disgust', 'neutral', 
                          'hunger', 'despair', 'cruelty']
            available_cols = [col for col in emotion_cols if col in location_emotions.columns]
            
            if available_cols:
                emotion_sums = location_emotions[available_cols].sum()
                dominant_emotion = emotion_sums.idxmax()
                total_emotions = len(location_emotions)
                
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
        
        # Icon by march phase
        icon_map = {1: 'star', 2: 'flag', 3: 'certificate'}
        icon = icon_map.get(row['march_id'], 'info-sign')
        
        # Create popup - simplified without problematic emojis
        popup_html = f"""
        <div style="font-family: Arial; width: 240px; padding: 10px; background: #f5f5f5; border: 2px solid #8B7355; border-radius: 5px;">
            <h4 style="margin: 0 0 8px 0; color: #3E2723; border-bottom: 2px solid #8B0000;">{poi_name}</h4>
            
            <div style="background: #4A5D3F; color: white; padding: 6px; margin: 6px 0; border-radius: 3px;">
                <strong>{march_name}</strong><br>
                Month: {base_month} | Segment: {row.get('segment', 'N/A')}
            </div>
            
            <div style="background: #2C1810; color: #FFD700; padding: 6px; margin: 6px 0; border-radius: 3px;">
                <strong>POW Deaths:</strong> {int(death_count)}<br>
                <strong>Emotions:</strong> {total_emotions} records<br>
                <strong>Dominant:</strong> {dominant_emotion.upper()}
            </div>
            
            <div style="background: #5D4E37; color: #C3B091; padding: 5px; margin: 6px 0; border-radius: 3px; font-size: 10px;">
                Lat: {row.geometry.y:.5f} | Lon: {row.geometry.x:.5f}
            </div>
            
            <small style="color: #666;">Click for detailed analysis</small>
        </div>
        """
        
        # Add marker to map
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{poi_name} ({march_name}) - {int(death_count)} deaths",
            icon=folium.Icon(color=marker_color, icon=icon, prefix='glyphicon')
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; top: 10px; right: 10px; width: 200px; 
                background: rgba(255,255,255,0.95); border: 2px solid #8B7355; 
                border-radius: 5px; padding: 10px; font-size: 11px; z-index: 9999;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 8px 0; color: #3E2723; border-bottom: 2px solid #8B0000;">
            Map Legend
        </h4>
        
        <p style="margin: 5px 0 3px 0; font-weight: bold;">Marker Colors (Emotions):</p>
        <p style="margin: 2px 0; color: red;">Red - Anger/Disgust</p>
        <p style="margin: 2px 0; color: blue;">Blue - Sadness</p>
        <p style="margin: 2px 0; color: purple;">Purple - Fear</p>
        <p style="margin: 2px 0; color: gray;">Gray - Neutral</p>
        
        <p style="margin: 8px 0 3px 0; font-weight: bold;">March Phase Icons:</p>
        <p style="margin: 2px 0;">Star - First March</p>
        <p style="margin: 2px 0;">Flag - Second March</p>
        <p style="margin: 2px 0;">Medal - Third March</p>
        
        <p style="margin: 8px 0 3px 0; font-weight: bold;">Route Line Thickness:</p>
        <p style="margin: 2px 0;">Proportional to death count</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Layer control
    folium.LayerControl(position='bottomleft').add_to(m)
    
    return m

# ==================== ANALYSIS FUNCTIONS ====================
def analyze_location_emotions(location_name, hybrid_df, poi_row):
    """Analyze emotions for selected location"""
    
    # Find location column
    location_col = find_location_column(hybrid_df)
    
    if location_col is None:
        st.error(f"⚠️ No location column found in emotion data. Available columns: {hybrid_df.columns.tolist()}")
        return None
    
    # Try to match location name (fuzzy matching)
    try:
        location_data = hybrid_df[
            hybrid_df[location_col].str.contains(location_name.split()[0], case=False, na=False)
        ]
    except Exception as e:
        st.error(f"Error matching location: {str(e)}")
        return None
    
    if location_data.empty:
        return None
    
    # Get emotion columns
    emotion_cols = [col for col in ['anger', 'fear', 'sadness', 'joy', 'surprise', 
                                     'disgust', 'neutral', 'hunger', 'despair', 'cruelty'] 
                    if col in location_data.columns]
    
    if not emotion_cols:
        st.warning("No emotion columns found in data")
        return None
    
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
    """Create Plotly charts for selected location"""
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
    if analysis_results['temporal_emotions'] is not None:
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
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Histogram(
            x=analysis_results['data']['sentiment_score'],
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
            Interactive 4D Geovisualization (X, Y, T + Emotion)
        </h3>
        <p style="color: #C3B091; font-style: italic; margin: 5px 0;">
            Research Objective 3: Spatial-Temporal-Emotional Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading emotion data and spatial layers..."):
        hybrid_df, location_df = load_data()
        poi_gdf, route1_gdf, route2_gdf, route3_gdf = load_spatial_data()
    
    # Check if data loaded successfully
    if poi_gdf is None or hybrid_df is None:
        st.error("⚠️ Failed to load required data. Please check file paths and data integrity.")
        st.stop()
    
    # Create two-column layout
    col_map, col_analysis = st.columns([1.3, 1])
    
    with col_map:
        st.subheader("🗺️ Interactive Death March Route Map")
        st.markdown("*Click on POI markers to view detailed emotion analysis in the right panel →*")
        
        # Create and display map
        death_march_map = create_interactive_map(
            poi_gdf, route1_gdf, route2_gdf, route3_gdf, hybrid_df
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
            clicked_location = poi_gdf.iloc[nearest_idx]['POI_Name']
    
    with col_analysis:
        st.subheader("📊 Location-Specific Emotion Analysis")
        
        # Location selector with map synchronization
        location_options = sorted(poi_gdf['POI_Name'].unique())
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
            poi_row = poi_gdf[poi_gdf['POI_Name'] == selected_location].iloc[0]
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Deaths", int(poi_row.get('size_M1', 0)))
            with col2:
                st.metric("Month", poi_row.get('Base_month', 'N/A'))
            with col3:
                st.metric("March", poi_row['march_name'].replace(' March', ''))
            
            # Analyze emotions for location
            analysis = analyze_location_emotions(selected_location, hybrid_df, poi_row)
            
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
                charts = create_location_charts(analysis)
                for chart in charts:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Raw data expander
                with st.expander("📄 View Raw Emotion Data"):
                    display_cols = ['sentence', 'dominant_emotion', 'sentiment_score']
                    available_display_cols = [col for col in display_cols if col in analysis['data'].columns]
                    
                    if available_display_cols:
                        st.dataframe(
                            analysis['data'][available_display_cols].head(20),
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


