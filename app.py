import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
from PIL import Image
import time

from utils.video_processor import VideoProcessor
from utils.data_analyzer import analyze_suspension_data
from utils.recommendations import get_suspension_recommendations

# Set page configuration
st.set_page_config(
    page_title="Motorcycle Suspension Analyzer",
    page_icon="🏍️",
    layout="wide"
)

# Application title and introduction
st.title("Motorcycle Suspension Analysis Tool")

st.markdown("""
This application analyzes motorcycle suspension movement by tracking dots in videos.
Upload a video showing your motorcycle's suspension in action, and get detailed metrics, 
visualizations, and recommendations for optimal suspension settings.
""")

# Instructions for optimal video recording
with st.expander("📋 Instructions for optimal video recording"):
    st.markdown("""
    ### Recording Instructions
    
    For best results, follow these guidelines:
    
    1. **Dot Placement**:
       - Place two black circular dots (approx. 1-2 cm in diameter) on the fork or shock
       - One dot at the top (fixed part) and one at the bottom (moving part)
       - Ensure dots have high contrast against background
    
    2. **Camera Setup**:
       - Position camera perpendicular to the suspension components
       - Maintain fixed camera position throughout recording
       - Avoid shadows or changing lighting conditions
       - Record at 30fps or higher if possible
    
    3. **Recording Conditions**:
       - Record during typical riding conditions
       - Include footage of compression and rebound movements
       - 10-30 seconds of footage is usually sufficient
    
    4. **Video Format**:
       - MP4, AVI, or MOV formats work best
       - Resolution of 720p or higher recommended
    """)

# File uploader for video
st.subheader("Upload Video")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

# Processing parameters
with st.expander("Advanced Processing Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        min_dot_size = st.slider("Minimum dot size (pixels)", 3, 20, 5)
        max_dot_size = st.slider("Maximum dot size (pixels)", min_dot_size + 1, 50, 20)
        
    with col2:
        threshold_value = st.slider("Dot detection threshold", 0, 255, 100)
        sample_rate = st.slider("Analysis sample rate (frames)", 1, 10, 3, 
                             help="Analyze every Nth frame. Higher values speed up processing but reduce precision.")

# Process video if uploaded
if uploaded_file is not None:
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    # Video processing
    st.subheader("Video Processing")
    
    # Initialize the video processor
    video_processor = VideoProcessor(
        min_dot_size=min_dot_size,
        max_dot_size=max_dot_size,
        threshold=threshold_value,
        sample_rate=sample_rate
    )
    
    # Progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process the video
    try:
        status_text.text("Starting video processing...")
        
        # Extract video information
        video_info = video_processor.get_video_info(video_path)
        
        # Display video information
        cols = st.columns(4)
        cols[0].metric("Duration", f"{video_info['duration']:.2f} sec")
        cols[1].metric("Frame Rate", f"{video_info['fps']:.1f} FPS")
        cols[2].metric("Resolution", f"{video_info['width']}x{video_info['height']}")
        cols[3].metric("Total Frames", f"{video_info['frame_count']}")
        
        # Track the dots in the video
        status_text.text("Tracking suspension movement...")
        
        # Process the video with progress updates
        suspension_data, sample_frames = video_processor.process_video(
            video_path, 
            progress_callback=lambda p: progress_bar.progress(p)
        )
        
        # Check if tracking was successful
        if suspension_data is None or len(suspension_data) < 10:
            st.error("Dot tracking failed. Please ensure your video has two distinct black dots visible throughout the clip.")
            if os.path.exists(video_path):
                os.unlink(video_path)
            st.stop()
            
        status_text.text("Processing complete!")
        progress_bar.progress(100)
        
        # Convert tracking data to DataFrame
        df = pd.DataFrame(suspension_data)
        
        # Display sample frames with tracking
        st.subheader("Suspension Tracking Visualization")
        st.write("Sample frames from the video with dot tracking:")
        
        # Display up to 3 sample frames in a row
        sample_cols = st.columns(min(len(sample_frames), 3))
        for i, (frame, frame_num) in enumerate(sample_frames[:3]):
            pil_image = Image.fromarray(frame)
            sample_cols[i].image(pil_image, caption=f"Frame #{frame_num}", use_column_width=True)
        
        # Analyze the suspension data
        st.subheader("Suspension Analysis")
        analysis_results = analyze_suspension_data(df)
        
        # Display metrics
        metric_cols = st.columns(4)
        metric_cols[0].metric("Avg. Travel", f"{analysis_results['avg_travel']:.2f} mm")
        metric_cols[1].metric("Max Travel", f"{analysis_results['max_travel']:.2f} mm")
        metric_cols[2].metric("Avg. Compression Speed", f"{analysis_results['avg_compression_speed']:.2f} mm/s")
        metric_cols[3].metric("Avg. Rebound Speed", f"{analysis_results['avg_rebound_speed']:.2f} mm/s")
        
        # Display suspension movement chart
        st.subheader("Suspension Movement Over Time")
        
        fig = px.line(
            df, 
            x='time', 
            y='distance', 
            title='Suspension Travel vs Time',
            labels={'time': 'Time (seconds)', 'distance': 'Suspension Travel (mm)'}
        )
        fig.update_layout(
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display compression/rebound speeds
        st.subheader("Compression/Rebound Analysis")
        
        # Create two charts side by side
        col1, col2 = st.columns(2)
        
        # Compression/Rebound Speed Distribution
        with col1:
            # Filter for compression (negative velocities) and rebound (positive velocities)
            compression_speeds = df['velocity'][df['velocity'] < 0].abs()
            rebound_speeds = df['velocity'][df['velocity'] > 0]
            
            # Create Histogram for speeds
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Histogram(
                x=compression_speeds,
                name='Compression',
                marker_color='red',
                opacity=0.7,
                nbinsx=20
            ))
            
            fig_hist.add_trace(go.Histogram(
                x=rebound_speeds,
                name='Rebound',
                marker_color='blue',
                opacity=0.7,
                nbinsx=20
            ))
            
            fig_hist.update_layout(
                title='Speed Distribution',
                xaxis_title='Speed (mm/s)',
                yaxis_title='Count',
                barmode='overlay',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Velocity vs Position
        with col2:
            fig_vel = px.scatter(
                df, 
                x='distance', 
                y='velocity',
                color_discrete_sequence=['green'],
                opacity=0.6,
                title='Velocity vs Position',
                labels={'distance': 'Suspension Travel (mm)', 'velocity': 'Velocity (mm/s)'}
            )
            
            fig_vel.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_vel.update_layout(
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            
            st.plotly_chart(fig_vel, use_container_width=True)
        
        # Show detailed data table
        with st.expander("View Detailed Data"):
            st.dataframe(df)
            
            # Download link for the data
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "suspension_data.csv",
                "text/csv",
                key='download-csv'
            )
        
        # Get suspension recommendations
        st.subheader("Suspension Recommendations")
        recommendations = get_suspension_recommendations(analysis_results)
        
        # Display recommendations
        for category, rec in recommendations.items():
            with st.expander(f"{category} Setting Recommendations"):
                st.markdown(f"**Current Performance Analysis**: {rec['analysis']}")
                st.markdown(f"**Recommendation**: {rec['recommendation']}")
                st.markdown(f"**Expected Improvement**: {rec['improvement']}")
        
        # Clean up the temporary file
        if os.path.exists(video_path):
            os.unlink(video_path)
            
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        # Clean up the temporary file
        if os.path.exists(video_path):
            os.unlink(video_path)

# Footer
st.markdown("---")
st.markdown("""
**Motorcycle Suspension Analyzer** - A tool for analyzing suspension performance through video processing.
""")
