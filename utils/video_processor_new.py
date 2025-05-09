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
        Detect dots in a frame using multiple methods for robustness.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of (x, y) coordinates of detected dots
        """
        # Try all detection methods
        dots_from_methods = []
        
        # Try detecting by color (red and green)
        dots_color = self.detect_dots_by_color(frame)
        if dots_color:
            dots_from_methods.append(dots_color)
        
        # Try detecting square shapes
        dots_square = self.detect_square_markers(frame)
        if dots_square:
            dots_from_methods.append(dots_square)
        
        # Try blob detection as a fallback
        dots_blob = self.detect_dots_by_blob(frame)
        if dots_blob:
            dots_from_methods.append(dots_blob)
        
        # Choose the best method's results (one with exactly 2 dots, or most promising)
        for method_dots in dots_from_methods:
            if len(method_dots) == 2:
                return method_dots
        
        # If no method got exactly 2 dots, combine and filter
        all_dots = []
        for method_dots in dots_from_methods:
            all_dots.extend(method_dots)
        
        # Remove duplicates
        unique_dots = []
        for dot in all_dots:
            is_duplicate = False
            for existing_dot in unique_dots:
                # If dots are very close, consider them duplicates
                dist = np.sqrt((dot[0] - existing_dot[0])**2 + (dot[1] - existing_dot[1])**2)
                if dist < self.min_dot_size:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_dots.append(dot)
        
        # If we have more than 2 dots, take the best pair (most distant for better measurements)
        if len(unique_dots) > 2:
            max_dist = 0
            best_pair = None
            
            for i in range(len(unique_dots)):
                for j in range(i + 1, len(unique_dots)):
                    dist = np.sqrt((unique_dots[i][0] - unique_dots[j][0])**2 + 
                                   (unique_dots[i][1] - unique_dots[j][1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = [unique_dots[i], unique_dots[j]]
            
            if best_pair:
                return best_pair
        
        return unique_dots
    
    def detect_dots_by_color(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect red and green dots using color thresholding.
        """
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for red and green with very lenient thresholds
        # Red is tricky in HSV as it wraps around 0/180, so we need two ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        #Blue
        lower_blue = np.array([100,50,50])
        upper_blue = np.array([140,255,255])

        #Green
        lower_green = np.array([36, 0, 0])
        upper_green = np.array([86, 255, 255])
        
        # Create masks for each color
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        blue_mask = cv2.inRange(hsv,lower_blue,upper_blue)
        
        # Process each mask separately to find red and green dots
        red_dots = self.find_dots_in_mask(blue_mask) #changed to blue mask
        green_dots = self.find_dots_in_mask(green_mask)
        
        # Combine the dots
        all_dots = red_dots + green_dots
        
        # If we have exactly 2 dots, one red and one green, return them
        if len(red_dots) == 1 and len(green_dots) == 1:
            return all_dots
            
        # Otherwise, if we have more dots, select the most likely pair
        if len(all_dots) > 2:
            # Find the dots that are most distant from each other
            max_dist = 0
            final_dots = []
            
            for i in range(len(all_dots)):
                for j in range(i + 1, len(all_dots)):
                    dist = np.sqrt((all_dots[i][0] - all_dots[j][0])**2 + (all_dots[i][1] - all_dots[j][1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        final_dots = [all_dots[i], all_dots[j]]
            
            if final_dots:
                return final_dots
        
        return all_dots
        
    def detect_square_markers(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect square-shaped markers of red and green color.
        """
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for red and green with very lenient thresholds
        # Red is tricky in HSV as it wraps around 0/180, so we need two ranges
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        # Create masks for each color
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine masks if needed
        all_mask = cv2.bitwise_or(red_mask, green_mask)
        
        # Clean up masks with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(all_mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for square-like contours
        square_centers = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            min_area = 25  # Minimum area of squares to detect (5x5 pixels)
            max_area = 400  # Maximum area (20x20 pixels)
            
            if min_area <= area <= max_area:
                # Approximate the contour to a polygon
                epsilon = 0.1 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If it has 4 vertices, it might be a square or rectangle
                if len(approx) == 4:
                    # Check if it's somewhat square (not a long rectangle)
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    
                    if 0.7 <= aspect_ratio <= 1.3:  # Allow some tolerance from perfect square
                        # Calculate centroid
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Check which color this square is
                            # We'll just add it to the results and later pick the best pair
                            square_centers.append((cx, cy))
        
        return square_centers
    
    def find_dots_in_mask(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find dots in a binary mask.
        """
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for contour in contours:
            # Filter contours by size
            area = cv2.contourArea(contour)
            min_area = np.pi * (self.min_dot_size ** 2) * 0.5  # More lenient
            max_area = np.pi * (self.max_dot_size ** 2) * 2.0  # More lenient
            
            if min_area <= area <= max_area:
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy))
                    
        return dots
        
    def detect_dots_by_blob(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect dots using blob detection for more general detection.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try different thresholds for better detection
        dots = []
        
        # Try regular thresholding first
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        dots.extend(self.find_dots_in_mask(binary))
        
        # If not enough dots, try adaptive thresholding
        if len(dots) < 2:
            adaptive_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
            dots.extend(self.find_dots_in_mask(adaptive_binary))
            
        # Remove duplicates
        unique_dots = []
        for dot in dots:
            is_duplicate = False
            for existing_dot in unique_dots:
                # If dots are very close, consider them duplicates
                dist = np.sqrt((dot[0] - existing_dot[0])**2 + (dot[1] - existing_dot[1])**2)
                if dist < self.min_dot_size:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_dots.append(dot)
                
        return unique_dots
    
    def process_video(self, video_path: str, 
                      initial_distance_mm: float = 100.0,
                      manual_dots: Optional[List[Tuple[int, int]]] = None,
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
        
        # Set initial pixel to mm ratio (will be calibrated below)
        self.pixel_to_mm_ratio = 0.5
        
        # Process the first frame to establish dot positions
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return [], []
        
        # Use manual dots if provided, otherwise detect automatically
        if manual_dots and len(manual_dots) >= 2:
            dots = manual_dots[:2]  # Use only the first two dots if more were provided
        else:
            # Try to automatically detect dots in the first frame
            dots = self.detect_dots(frame)
            
            # Need at least 2 dots to track
            if len(dots) < 2:
                cap.release()
                return [], []
        
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
