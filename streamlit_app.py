#!/usr/bin/env python3
"""
SOLUTION C: RO2 STREAMLIT DASHBOARD - CLOUD DEPLOYABLE
Sandakan-Ranau Death Marches Emotion Analysis
Deploy for FREE on Streamlit Community Cloud!

Author: Your Name
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
# Remove scipy import - use numpy instead
# from scipy.stats import pearsonr
from pathlib import Path
from collections import Counter

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Sandakan-Ranau Death Marches Emotion Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Use relative paths for Streamlit Cloud deployment
    HYBRID_EMOTIONS = "data/hybrid_emotion_analysis.csv"
    LOCATION_EMOTIONS = "data/location_emotions_hybrid.csv"

# ============================================================================
# COLORS
# ============================================================================

INSIDE_OUT_COLORS = {
    'joy': '#F8D547', 'sadness': '#5499C7', 'anger': '#E74C3C',
    'fear': '#9B59B6', 'disgust': '#27AE60', 'trust': '#3498DB',
    'anticipation': '#F39C12', 'surprise': '#E91E63',
    'suffering': '#8B4513', 'death': '#2C3E50', 'exhaustion': '#7F8C8D',
    'courage': '#16A085', 'despair': '#34495E', 'no_data': '#95A5A6'
}

# ============================================================================
# CACHING DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load and process emotion data (cached for performance)"""
    
    try:
        hybrid_emotions = pd.read_csv(Config.HYBRID_EMOTIONS)
        location_emotions = pd.read_csv(Config.LOCATION_EMOTIONS)
        
        # Parse emotion lists
        for col in ['emotions', 'bert_top_emotions', 'combined_emotions']:
            if col in hybrid_emotions.columns:
                hybrid_emotions[col] = hybrid_emotions[col].apply(
                    lambda x: eval(x) if pd.notna(x) and isinstance(x, str) and x.startswith('[') else []
                )
        
        # Create RO2 compatible dataframe
        records = []
        
        for idx, row in hybrid_emotions.iterrows():
            location_name = row['entity_text']
            
            loc_info = location_emotions[location_emotions['location_name'] == location_name]
            if len(loc_info) > 0:
                lat = loc_info.iloc[0]['latitude']
                lon = loc_info.iloc[0]['longitude']
            else:
                lat, lon = None, None
            
            combined_emotions = row.get('combined_emotions', [])
            if combined_emotions and len(combined_emotions) > 0:
                emotion_counts = Counter(combined_emotions)
                dominant_emotion = emotion_counts.most_common(1)[0][0]
                emotion_intensity = len(combined_emotions) / 10
            else:
                dominant_emotion = 'unknown'
                emotion_intensity = 0
            
            sentiment_polarity = row.get('sentiment_score', 0)
            sentiment_label = row.get('sentiment_label', 'neutral')
            
            emotion_scores = {}
            for emotion in combined_emotions:
                emotion_scores[f'{emotion}_score'] = 1
            
            march_number = assign_march_number(location_name)
            
            record = {
                'location_name': location_name,
                'latitude': lat,
                'longitude': lon,
                'dominant_emotion': dominant_emotion,
                'emotion_intensity': emotion_intensity,
                'sentiment_polarity': sentiment_polarity,
                'sentiment_label': sentiment_label,
                'march_number': march_number,
                'distance_from_start_km': 0,
                **emotion_scores
            }
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df['distance_from_start_km'] = df['distance_from_start_km'].fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def assign_march_number(location_name):
    """Assign march number based on location"""
    loc_lower = str(location_name).lower()
    
    if any(x in loc_lower for x in ['sandakan', 'mile 8', 'beluran']):
        return 1
    elif any(x in loc_lower for x in ['paginatan', 'telupid']):
        return 2
    elif any(x in loc_lower for x in ['boto']):
        return 3
    elif 'ranau' in loc_lower:
        return 1
    else:
        return 1

# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def calculate_moran_i(df):
    """Calculate Moran's I"""
    emotions = df.groupby('location_name')['emotion_intensity'].mean()
    n = len(emotions)
    
    if n < 2:
        return 0.0
    
    mean_emotion = emotions.mean()
    locations = df[['location_name', 'latitude', 'longitude']].drop_duplicates()
    locations = locations[locations['latitude'].notna() & locations['longitude'].notna()]
    
    if len(locations) < 2:
        return 0.0
    
    numerator = 0
    denominator = 0
    weights_sum = 0
    
    for i, row1 in locations.iterrows():
        for j, row2 in locations.iterrows():
            if i != j:
                dist = np.sqrt((row1['latitude'] - row2['latitude'])**2 + 
                             (row1['longitude'] - row2['longitude'])**2)
                weight = 1 / dist if dist > 0 else 0
                weights_sum += weight
                
                val1 = emotions.get(row1['location_name'], 0)
                val2 = emotions.get(row2['location_name'], 0)
                numerator += weight * (val1 - mean_emotion) * (val2 - mean_emotion)
    
    for val in emotions:
        denominator += (val - mean_emotion)**2
    
    if denominator > 0 and weights_sum > 0:
        moran_i = (n / weights_sum) * (numerator / denominator)
    else:
        moran_i = 0
    
    return moran_i

def get_statistical_summary(df):
    """Get statistical summary"""
    valid_data = df[['emotion_intensity', 'sentiment_polarity']].dropna()
    
    if len(valid_data) > 1:
        # Use numpy correlation instead of scipy
        corr = np.corrcoef(valid_data['emotion_intensity'], 
                          valid_data['sentiment_polarity'])[0, 1]
        # Simple p-value approximation
        n = len(valid_data)
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2)
        from math import erfc
        p_value = erfc(abs(t_stat) / np.sqrt(2))
    else:
        corr, p_value = 0, 1.0
    
    moran_i = calculate_moran_i(df)
    interpretation = "Dispersed" if moran_i < 0 else "Clustered" if moran_i > 0.3 else "Random"
    
    dominant = df['dominant_emotion'].mode()[0] if len(df['dominant_emotion'].mode()) > 0 else 'N/A'
    
    return {
        'Total Emotion Records': len(df),
        'Unique POI Locations': df['location_name'].nunique(),
        "Moran's I": f"{moran_i:.4f}",
        'Interpretation': interpretation,
        'Avg Emotion Score': f"{df['emotion_intensity'].mean():.3f}",
        'Avg Sentiment Polarity': f"{df['sentiment_polarity'].mean():.3f}",
        'Emotion-Sentiment Corr': f"{corr:.3f}",
        'Correlation P-value': f"{p_value:.4f}",
        'Dominant Emotion': dominant.capitalize()
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_emotion_distribution(df):
    """Chart 1: Emotion Distribution"""
    df_filtered = df[df['march_number'].isin([1, 2, 3])].copy()
    
    if len(df_filtered) == 0:
        return None
    
    march_labels = {1: 'First March', 2: 'Second March', 3: 'Third March'}
    df_filtered['march_phase'] = df_filtered['march_number'].map(march_labels)
    
    phase_emotion = df_filtered.groupby(['march_phase', 'dominant_emotion']).size().reset_index(name='count')
    
    fig = px.bar(phase_emotion, x='march_phase', y='count',
                 color='dominant_emotion',
                 title='Emotion Distribution by March Phase',
                 color_discrete_map=INSIDE_OUT_COLORS)
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig

def create_emotion_boxplot(df):
    """Chart 2: Emotion Boxplot"""
    emotion_cols = [col for col in df.columns if col.endswith('_score')]
    
    plot_data = []
    for col in emotion_cols:
        emotion_name = col.replace('_score', '').capitalize()
        values = df[col].dropna().values
        plot_data.extend([{'Emotion': emotion_name, 'Score': val} 
                         for val in values if val > 0])
    
    if len(plot_data) == 0:
        return None
    
    plot_df = pd.DataFrame(plot_data)
    color_map = {e.capitalize(): INSIDE_OUT_COLORS.get(e, '#95A5A6') for e in INSIDE_OUT_COLORS.keys()}
    
    fig = px.box(plot_df, x='Emotion', y='Score',
                 title='Emotion Score Distribution',
                 color='Emotion', color_discrete_map=color_map)
    
    fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
    return fig

def create_scatter(df):
    """Chart 3: Emotion vs Sentiment"""
    fig = px.scatter(df, x='emotion_intensity', y='sentiment_polarity',
                    color='dominant_emotion', hover_data=['location_name'],
                    title='Emotion Score vs Sentiment Polarity',
                    color_discrete_map=INSIDE_OUT_COLORS)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400)
    return fig

def create_distance_decay(df):
    """Chart 4: Distance Decay"""
    fear_data = df[(df['dominant_emotion'] == 'fear') & (df['distance_from_start_km'] > 0)]
    
    if len(fear_data) < 3:
        fear_data = df[df['distance_from_start_km'] > 0]
        if len(fear_data) == 0:
            return None
        
        fig = px.scatter(fear_data, x='distance_from_start_km', y='emotion_intensity',
                        color='dominant_emotion', color_discrete_map=INSIDE_OUT_COLORS,
                        hover_data=['location_name'],
                        title='Distance Decay: All Emotions')
    else:
        fig = px.scatter(fear_data, x='distance_from_start_km', y='emotion_intensity',
                        hover_data=['location_name'],
                        color_discrete_sequence=[INSIDE_OUT_COLORS['fear']],
                        title='Distance Decay: Fear Emotion')
    
    fig.update_layout(height=400)
    return fig

def create_avg_by_phase(df):
    """Chart 5: Average by Phase"""
    df_filtered = df[df['march_number'].isin([1, 2, 3])].copy()
    
    if len(df_filtered) == 0:
        return None
    
    march_labels = {1: 'First March', 2: 'Second March', 3: 'Third March'}
    df_filtered['march_phase'] = df_filtered['march_number'].map(march_labels)
    
    phase_avg = df_filtered.groupby('march_phase')['emotion_intensity'].mean().reset_index()
    
    fig = px.bar(phase_avg, x='march_phase', y='emotion_intensity',
                title='Avg Emotion Intensity by March Phase',
                color_discrete_sequence=['#FA8072'])
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .info-box {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Statistical Spatial Analysis: Sandakan-Ranau Death Marches</h1>
        <h3>Enhanced with Hybrid Emotion Detection (BERT + Lexicon)</h3>
        <p>Research Objective 2: Analyze spatial-temporal sentiment patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading emotion analysis data...'):
        df = load_data()
    
    if df.empty:
        st.error("❌ Could not load data. Please check data files.")
        return
    
    # Get statistics
    summary = get_statistical_summary(df)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <h4>🔬 Methodology & Dataset</h4>
        <p><strong>Data Source:</strong> Step 5 Hybrid Emotion Analysis (BERT Transformer + Lexicon-based detection)</p>
        <p><strong>Improvement:</strong> 1,831 emotions detected (417% increase over lexicon-only approach)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📚 Total Records", f"{summary['Total Emotion Records']:,}")
    with col2:
        st.metric("📍 Unique Locations", summary['Unique POI Locations'])
    with col3:
        st.metric("🧮 Moran's I", summary["Moran's I"])
    with col4:
        st.metric("📊 Correlation", summary['Emotion-Sentiment Corr'])
    with col5:
        st.metric("😢 Dominant Emotion", summary['Dominant Emotion'])
    
    st.markdown("---")
    
    # Charts in 3x2 grid
    st.subheader("📈 Statistical Visualizations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig1 = create_emotion_distribution(df)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = create_emotion_boxplot(df)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        fig3 = create_scatter(df)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        fig4 = create_distance_decay(df)
        if fig4:
            st.plotly_chart(fig4, use_container_width=True)
    
    with col5:
        fig5 = create_avg_by_phase(df)
        if fig5:
            st.plotly_chart(fig5, use_container_width=True)
    
    with col6:
        st.markdown("### 📋 Statistical Summary")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total Records | {summary['Total Emotion Records']} |
        | Unique Locations | {summary['Unique POI Locations']} |
        | Moran's I | {summary["Moran's I"]} |
        | Interpretation | {summary['Interpretation']} |
        | Avg Emotion Score | {summary['Avg Emotion Score']} |
        | Avg Sentiment | {summary['Avg Sentiment Polarity']} |
        | Correlation | {summary['Emotion-Sentiment Corr']} |
        | P-value | {summary['Correlation P-value']} |
        | Dominant Emotion | {summary['Dominant Emotion']} |
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p><strong>🎓 Sandakan-Ranau Death Marches Emotion Mapping Project</strong></p>
        <p>GES722 Geospatial Research | Python 3.13 | spaCy 3.8.7 | Transformers 4.49.0 | Streamlit</p>
        <p><em>✨ Deployed on Streamlit Community Cloud</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


