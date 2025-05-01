import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Callable, Any
import random

class VideoProcessor:
    """
    Processes videos to track dots representing motorcycle suspension components.
    """
    
    def __init__(self, min_dot_size: int = 5, max_dot_size: int = 20, 
                 threshold: int = 100, sample_rate: int = 3):
        """
        Initialize the VideoProcessor with configuration parameters.
        
        Args:
            min_dot_size: Minimum dot radius in pixels
            max_dot_size: Maximum dot radius in pixels
            threshold: Threshold for binary image conversion (0-255)
            sample_rate: Process every Nth frame
        """
        self.min_dot_size = min_dot_size
        self.max_dot_size = max_dot_size
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.pixel_to_mm_ratio = 1.0  # Default conversion (will be calibrated)
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Extract basic information from the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open the video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Release the video capture object
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        }
    
    def detect_dots(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect red and green dots in a frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of (x, y) coordinates of detected dots
        """
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for red and green
        # Red is tricky in HSV as it wraps around 0/180, so we need two ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        
        # Create masks for each color
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine masks to get all dots
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for contour in contours:
            # Filter contours by size
            area = cv2.contourArea(contour)
            min_area = np.pi * (self.min_dot_size ** 2)
            max_area = np.pi * (self.max_dot_size ** 2)
            
            if min_area <= area <= max_area:
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy))
        
        # We need exactly 2 dots
        if len(dots) >= 2:
            # If more than 2 dots found, take the 2 most distant ones
            if len(dots) > 2:
                max_dist = 0
                final_dots = None
                
                for i in range(len(dots)):
                    for j in range(i + 1, len(dots)):
                        dist = np.sqrt((dots[i][0] - dots[j][0])**2 + (dots[i][1] - dots[j][1])**2)
                        if dist > max_dist:
                            max_dist = dist
                            final_dots = [dots[i], dots[j]]
                
                return final_dots
            return dots
        
        return []
    
    def process_video(self, video_path: str, 
                      initial_distance_mm: float = 100.0,
                      progress_callback: Optional[Callable[[float], None]] = None) -> Tuple[List[Dict[str, float]], List[Tuple[np.ndarray, int]]]:
        """
        Process the video to track suspension movement.
        
        Args:
            video_path: Path to the video file
            initial_distance_mm: Actual distance between dots in mm for calibration
            progress_callback: Callback function to report progress (0.0 to 1.0)
            
        Returns:
            Tuple containing:
            - List of dictionaries with tracking data
            - List of sample frames with tracking visualization
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open the video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize tracking data
        suspension_data = []
        sample_frames = []
        
        # Set pixel to mm conversion (can be adjusted based on known measurements)
        # For now we'll use a reasonable default value
        self.pixel_to_mm_ratio = 0.5  # 1 pixel = 0.5 mm (will be calibrated below)
        
        # Process the first frame to establish dot positions
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, []
        
        # Detect dots in the first frame
        dots = self.detect_dots(frame)
        
        # Need at least 2 dots to track
        if len(dots) < 2:
            cap.release()
            return None, []
        
        # Sort dots by y-coordinate (top dot first, bottom dot second)
        dots.sort(key=lambda p: p[1])
        
        # Calculate initial distance for calibration
        initial_distance_px = np.sqrt((dots[0][0] - dots[1][0])**2 + (dots[0][1] - dots[1][1])**2)
        
        # Use the provided initial distance for calibration
        self.pixel_to_mm_ratio = initial_distance_mm / initial_distance_px
        
        # Initialize previous positions and time
        prev_dots = dots
        prev_time = 0
        prev_distance = initial_distance_px * self.pixel_to_mm_ratio
        
        # Process frames
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            frame_idx += 1
            
            # Process only every sample_rate frames
            if frame_idx % self.sample_rate != 0:
                continue
                
            processed_count += 1
            current_time = frame_idx / fps
            
            # Detect dots
            dots = self.detect_dots(frame)
            
            # If we found 2 dots, update tracking data
            if len(dots) == 2:
                # Sort dots by y-coordinate
                dots.sort(key=lambda p: p[1])
                
                # Calculate distance
                distance_px = np.sqrt((dots[0][0] - dots[1][0])**2 + (dots[0][1] - dots[1][1])**2)
                distance_mm = distance_px * self.pixel_to_mm_ratio
                
                # Calculate velocity (mm/s)
                time_delta = current_time - prev_time
                if time_delta > 0:
                    velocity = (distance_mm - prev_distance) / time_delta
                else:
                    velocity = 0
                
                # Record data
                suspension_data.append({
                    'time': current_time,
                    'distance': distance_mm,
                    'velocity': velocity
                })
                
                # Save reference to previous values
                prev_dots = dots
                prev_time = current_time
                prev_distance = distance_mm
                
                # Draw tracking visualization
                draw_frame = frame.copy()
                
                # Draw dots
                for i, (x, y) in enumerate(dots):
                    color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for top, Red for bottom
                    cv2.circle(draw_frame, (x, y), 5, color, -1)
                
                # Draw line connecting dots
                cv2.line(draw_frame, dots[0], dots[1], (255, 255, 0), 2)
                
                # Add distance text
                midpoint = ((dots[0][0] + dots[1][0]) // 2, (dots[0][1] + dots[1][1]) // 2)
                cv2.putText(draw_frame, f"{distance_mm:.1f}mm", 
                            (midpoint[0] + 10, midpoint[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save sample frames (first, last, and some random frames in between)
                if processed_count == 1 or frame_idx >= frame_count - self.sample_rate or random.random() < 0.05:
                    sample_frames.append((draw_frame, frame_idx))
            else:
                # If dots not found, use previous dots if available
                if prev_dots:
                    # Draw tracking visualization with previous dots
                    draw_frame = frame.copy()
                    
                    # Draw previous dots (in yellow to indicate they're from previous frame)
                    for i, (x, y) in enumerate(prev_dots):
                        cv2.circle(draw_frame, (x, y), 5, (0, 255, 255), -1)
                    
                    # Draw line connecting dots
                    cv2.line(draw_frame, prev_dots[0], prev_dots[1], (0, 255, 255), 2)
                    
                    # Add text indicating tracking loss
                    cv2.putText(draw_frame, "Tracking lost", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Save as sample frame if it's one of the first frames with tracking loss
                    if len(sample_frames) < 6:
                        sample_frames.append((draw_frame, frame_idx))
            
            # Update progress
            if progress_callback and frame_count > 0:
                progress_callback(frame_idx / frame_count)
        
        # Release the video capture object
        cap.release()
        
        return suspension_data, sample_frames
