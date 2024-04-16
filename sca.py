import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to process the captured frame and find document contours
def process_frame(frame):
    # Convert frame to grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    
    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour
    max_contour = max(contours, key=cv2.contourArea)
    
    # Get the perimeter of the contour
    perimeter = cv2.arcLength(max_contour, True)
    
    # Approximate the contour by a polygon
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # Ensure the contour is a quadrilateral
    if len(approx) == 4:
        # Reshape the vertices of the quadril        
        pts = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
        
        # Define the destination points for perspective transformation
        width = 500
        height = 700
        dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        # Perform perspective transformation
        matrix = cv2.getPerspectiveTransform(pts, dst)
        scanned_image = cv2.warpPerspective(frame, matrix, (width, height))
        
        return scanned_image
    else:
        return frame
count = 0
def main():
    st.title("Document Scanner")
    st.write("Use the button below to start capturing the document.")
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        st.error("Error: Unable to open the camera.")
        return
    
    # Start capturing the document when the button is clicked
    if st.button("Start Capture", key = 'hello'):
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Display the captured frame
            st.image(process_frame(frame), channels="BGR", use_column_width=True)
            
            
            # Check for stop condition
            st.button("Quit", key = count+1)
            break

    # Release the camera and close OpenCV windows
    cap.release()

if __name__ == "__main__":
    main()
