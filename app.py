import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import cv2
from PIL import Image
import time
import datetime

from utils.video_processor_new import VideoProcessor
from utils.data_analyzer import analyze_suspension_data
from utils.recommendations import get_suspension_recommendations
from utils.database import (save_analysis_session, get_analysis_sessions, 
                           get_analysis_session_by_id, delete_analysis_session)

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

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analysis History"], key="nav_radio")

# Set the current page based on sidebar selection
st.session_state.page = page.lower()

# Only show main content if we're on the home page
if st.session_state.page == 'home':
    # Instructions for optimal video recording
    with st.expander("📋 Instructions for optimal video recording"):
        st.markdown("""
        ### Recording Instructions
        
        For best results, follow these guidelines:
        
        1. **Dot Placement**:
           - Place one red and one green circular dot (approx. 1-2 cm in diameter) on the fork or shock
           - One dot at the top (fixed part) and one at the bottom (moving part)
           - Ensure dots have good color contrast against the background
        
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
    
    # Add motorcycle info input for database storage
    motorcycle_info = st.text_input("Motorcycle Information (optional)", 
                                 placeholder="e.g., 2023 Honda CBR650R, stock suspension")
elif st.session_state.page == 'analysis history':
    st.header("Analysis History")
    st.write("View and load previous suspension analysis sessions.")
    
    # Fetch all analysis sessions from the database
    try:
        sessions = get_analysis_sessions()
        
        if not sessions:
            st.info("No analysis sessions found. Run an analysis to save it to the database.")
        else:
            # Display sessions in a table
            session_df = pd.DataFrame(sessions)
            # Format the created_at date
            session_df['created_at'] = pd.to_datetime(session_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            session_df.columns = ['ID', 'Session Name', 'Motorcycle', 'Date', 'Video Filename']
            
            st.dataframe(session_df)
            
            # Select a session to view
            selected_id = st.selectbox(
                "Select a session to view:", 
                options=[session['id'] for session in sessions],
                format_func=lambda x: next((s['session_name'] for s in sessions if s['id'] == x), str(x))
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Load Selected Session"):
                    # Set session data to be loaded
                    st.session_state.load_session_id = selected_id
                    st.session_state.page = 'home'  # Navigate back to home
                    st.rerun()
            
            with col2:
                if st.button("Delete Selected Session", type="secondary"):
                    if delete_analysis_session(selected_id):
                        st.success("Session deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete session.")
    
    except Exception as e:
        st.error(f"Error accessing analysis history: {str(e)}")
        st.info("Make sure the database is properly configured.")

# Initialize session loading
if 'load_session_id' in st.session_state and st.session_state.page == 'home':
    session_id = st.session_state.load_session_id
    st.info(f"Loading analysis session {session_id}...")
    
    try:
        # Load session data from database
        session_data = get_analysis_session_by_id(session_id)
        
        if not session_data:
            st.error(f"Session {session_id} not found or could not be loaded.")
        else:
            # Display loaded session information
            st.subheader(f"Loaded Analysis: {session_data['session']['session_name']}")
            
            # Display session metadata
            info_cols = st.columns(3)
            info_cols[0].info(f"Motorcycle: {session_data['session']['motorcycle_info'] or 'Not specified'}")
            info_cols[1].info(f"Date: {session_data['session']['created_at'].strftime('%Y-%m-%d %H:%M') if isinstance(session_data['session']['created_at'], datetime.datetime) else session_data['session']['created_at']}")
            info_cols[2].info(f"Video: {session_data['session']['video_filename'] or 'Unknown'}")
            
            # Convert data points to DataFrame
            df = pd.DataFrame(session_data['data_points'])
            
            # Display metrics
            metrics = session_data['metrics']
            metric_cols = st.columns(4)
            metric_cols[0].metric("Avg. Travel", f"{metrics.get('avg_travel', 0):.2f} mm")
            metric_cols[1].metric("Max Travel", f"{metrics.get('max_travel', 0):.2f} mm")
            metric_cols[2].metric("Avg. Compression Speed", f"{metrics.get('avg_compression_speed', 0):.2f} mm/s")
            metric_cols[3].metric("Avg. Rebound Speed", f"{metrics.get('avg_rebound_speed', 0):.2f} mm/s")
            
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
                    key='download-csv-loaded'
                )
            
            # Display recommendations
            st.subheader("Suspension Recommendations")
            
            for component, rec in session_data['recommendations'].items():
                with st.expander(f"{component} Setting Recommendations"):
                    st.markdown(f"**Current Performance Analysis**: {rec['analysis']}")
                    st.markdown(f"**Recommendation**: {rec['recommendation']}")
                    st.markdown(f"**Expected Improvement**: {rec['improvement']}")
                    
            # Clear the loaded session after displaying
            if st.button("Clear Loaded Session"):
                del st.session_state.load_session_id
                st.rerun()
                
    except Exception as e:
        st.error(f"Error loading session: {str(e)}")
        st.button("Clear Error", on_click=lambda: setattr(st.session_state, 'load_session_id', None))

# Processing parameters
with st.expander("Advanced Processing Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        min_dot_size = st.slider("Minimum dot size (pixels)", 3, 20, 5)
        max_dot_size = st.slider("Maximum dot size (pixels)", min_dot_size + 1, 50, 20)
        initial_distance = st.number_input(
            "Initial distance between dots (mm)", 
            min_value=10.0, 
            max_value=1000.0, 
            value=100.0,
            step=1.0,
            help="The actual measured distance between the dots in the first frame (mm). This is used for accurate calibration."
        )
        
    with col2:
        threshold_value = st.slider("Dot detection threshold", 0, 255, 100)
        sample_rate = st.slider("Analysis sample rate (frames)", 1, 10, 3, 
                             help="Analyze every Nth frame. Higher values speed up processing but reduce precision.")

# Set uploaded_file to None initially to avoid NameError
if 'uploaded_file' not in locals():
    uploaded_file = None

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
        
        # Allow manual selection of dots in the first frame
        status_text.text("Loading first frame for manual dot selection...")
        
        # Get the first frame of the video
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if not ret:
            st.error("Could not read the first frame of the video.")
            if os.path.exists(video_path):
                os.unlink(video_path)
            st.stop()
        
        # Convert the frame from BGR to RGB for display
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame for dot selection
        st.subheader("Manual Marker Selection")
        st.write("Use the sliders to position the markers precisely on the red and green squares:")
        
        # Display the first frame
        st.image(first_frame_rgb, channels="RGB", use_container_width=True)
        
        # Create containers for the top and bottom marker selection
        st.write("### Top Marker (fixed part)")
        top_col1, top_col2 = st.columns(2)
        with top_col1:
            dot1_x = st.slider("X Position", 0, first_frame.shape[1]-1, first_frame.shape[1]//4, 1)
        with top_col2:
            dot1_y = st.slider("Y Position", 0, first_frame.shape[0]-1, first_frame.shape[0]//3, 1)
        
        st.write("### Bottom Marker (moving part)")
        bottom_col1, bottom_col2 = st.columns(2)
        with bottom_col1:
            dot2_x = st.slider("X Position", 0, first_frame.shape[1]-1, first_frame.shape[1]//4, 1)
        with bottom_col2:
            dot2_y = st.slider("Y Position", 0, first_frame.shape[0]-1, 2*first_frame.shape[0]//3, 1)
        
        # Create a real-time visualization that updates as sliders change
        confirmation_frame = first_frame_rgb.copy()
        
        # Draw the markers with larger size for better visibility
        cv2.drawMarker(confirmation_frame, (dot1_x, dot1_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 3)  # Green cross for top marker
        cv2.drawMarker(confirmation_frame, (dot2_x, dot2_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 3)  # Red cross for bottom marker
        
        # Draw the selected dots with circles
        cv2.circle(confirmation_frame, (dot1_x, dot1_y), 10, (0, 255, 0), 2)  # Green circle for top marker
        cv2.circle(confirmation_frame, (dot2_x, dot2_y), 10, (255, 0, 0), 2)  # Red circle for bottom marker
        
        # Draw line connecting dots
        cv2.line(confirmation_frame, (dot1_x, dot1_y), (dot2_x, dot2_y), (255, 255, 0), 2)
        
        st.subheader("Preview with Selected Markers")
        st.write("This preview updates in real-time as you adjust the sliders. Green represents the top (fixed) marker and red represents the bottom (moving) marker.")
        st.image(confirmation_frame, channels="RGB", use_container_width=True)
        
        # Confirm and proceed
        if st.button("Confirm and Process Video"):
            manual_dots = [(dot1_x, dot1_y), (dot2_x, dot2_y)]
            
            # Track the dots in the video
            status_text.text("Tracking suspension movement...")
            
            # Process the video with progress updates
            suspension_data, sample_frames = video_processor.process_video(
                video_path,
                initial_distance_mm=initial_distance,
                manual_dots=manual_dots,
                progress_callback=lambda p: progress_bar.progress(p)
            )
        else:
            st.warning("Please confirm your dot selection to process the video.")
            st.stop()
        
        # Check if tracking was successful
        if suspension_data is None or len(suspension_data) < 10:
            st.error("Dot tracking failed. Please ensure your video has one red and one green dot visible throughout the clip.")
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
            sample_cols[i].image(pil_image, caption=f"Frame #{frame_num}", use_container_width=True)
        
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
        
        # Save analysis to database section
        st.subheader("Save Analysis")
        save_col1, save_col2 = st.columns([3, 1])
        
        with save_col1:
            session_name = st.text_input(
                "Session Name", 
                value=f"Suspension Analysis {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                key="session_name"
            )
            
        with save_col2:
            if st.button("Save to Database", type="primary", key="save_btn"):
                try:
                    # Save to database
                    session_id = save_analysis_session(
                        session_name=session_name,
                        motorcycle_info=motorcycle_info,
                        video_filename=uploaded_file.name if uploaded_file else None,
                        video_info=video_info,
                        analysis_results=analysis_results,
                        suspension_data=suspension_data,
                        recommendations=recommendations
                    )
                    
                    if session_id:
                        st.success(f"Analysis saved to database with ID: {session_id}")
                    else:
                        st.error("Failed to save analysis to database.")
                except Exception as e:
                    st.error(f"Error saving to database: {str(e)}")
        
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
