import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
from face_encode import encode_face, save_encodings, load_encodings
from utils import init_db, mark_attendance, get_today_attendance, get_attendance_history, export_to_csv, recognize_face

# Initialize DB and load encodings
init_db()
encodings = load_encodings()

# Custom CSS for modern, mobile-friendly UI with improved font visibility
st.markdown("""
    <style>
    body, .main, .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader, .stButton, .stTextInput, .stFileUploader {
        font-family: Arial, sans-serif !important;  /* Web-safe font for better visibility */
        font-size: 16px;  /* Ensure readable size */
    }
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stTextInput, .stFileUploader {border-radius: 5px;}
    .mobile-friendly {max-width: 100%; overflow-x: auto;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Sidebar navigation (updated with Admin Reports)
st.sidebar.title("Attendance System")
pages = ["Dashboard", "Register Student", "Live Attendance", "Attendance History", "Export Attendance", "Admin Reports"]
page = st.sidebar.radio("Navigate", pages)

# Login check for Admin Reports
if page == "Admin Reports":
    if not st.session_state.logged_in:
        st.sidebar.subheader("Admin Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username == "admin" and password == "password":  # Change credentials in production
                st.session_state.logged_in = True
                st.sidebar.success("Logged in successfully!")
                st.rerun()  # Refresh to show the page
            else:
                st.sidebar.error("Invalid credentials.")
        st.stop()  # Stop here if not logged in

# Main page logic (only show if logged in for Admin Reports or always for others)
if page == "Dashboard":
    st.title("Dashboard")
    st.subheader("Today's Attendance")
    df_today = get_today_attendance()
    if not df_today.empty:
        st.dataframe(df_today, use_container_width=True)
        st.metric("Total Present Today", len(df_today))
    else:
        st.info("No attendance marked today yet.")

elif page == "Register Student":
    st.title("Register Student")
    with st.form("register_form"):
        student_id = st.text_input("Student ID")
        student_name = st.text_input("Student Name")
        uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])
        submitted = st.form_submit_button("Register")
        if submitted:
            if student_id in encodings:
                st.error("Student ID already exists.")
            elif not uploaded_file:
                st.error("Please upload an image.")
            else:
                image = Image.open(uploaded_file)
                encoding, error = encode_face(image, student_id)
                if error:
                    st.error(error)
                else:
                    encodings[student_id] = {"name": student_name, "encoding": encoding}
                    save_encodings(encodings)
                    st.success("Student registered successfully!")

elif page == "Live Attendance":
    st.title("Live Attendance")
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    
    # WebRTC configuration for better compatibility
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    class VideoProcessor:
        def __init__(self):
            self.encodings = load_encodings()  # Load encodings
            self.last_recognized = None
            self.last_time = 0
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Process every 1 second to avoid overload
            current_time = time.time()
            if current_time - self.last_time > 1:
                self.last_time = current_time
                student_id, student_name, error = recognize_face(img, self.encodings)
                if error:
                    st.session_state.status = f"❌ {error}"
                elif student_id and student_id != self.last_recognized:
                    success, msg = mark_attendance(student_id, student_name)
                    if success:
                        st.session_state.status = f"✅ {msg}"
                        self.last_recognized = student_id
                    else:
                        st.session_state.status = f"⚠️ {msg}"
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    ctx = webrtc_streamer(
        key="attendance",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Display status and table
    if "status" in st.session_state:
        st.write(st.session_state.status)
    df_today = get_today_attendance()
    st.dataframe(df_today, use_container_width=True)

elif page == "Attendance History":
    st.title("Attendance History")
    col1, col2, col3 = st.columns(3)
    with col1:
        date_filter = st.date_input("Filter by Date", value=None)
    with col2:
        id_filter = st.text_input("Filter by Student ID")
    with col3:
        name_filter = st.text_input("Filter by Student Name")
    
    df_history = get_attendance_history(
        date=date_filter.strftime("%Y-%m-%d") if date_filter else None,
        student_id=id_filter if id_filter else None,
        student_name=name_filter if name_filter else None
    )
    st.dataframe(df_history, use_container_width=True)

elif page == "Export Attendance":
    st.title("Export Attendance")
    if st.button("Export to CSV"):
        csv_file = export_to_csv()
        st.success(f"Exported to {csv_file}")
        with open(csv_file, "rb") as f:
            st.download_button("Download CSV", f, file_name="attendance.csv")

elif page == "Admin Reports":
    # This is only shown if logged in (checked above)
    st.title("Admin Reports")
    st.subheader("Advanced Attendance Statistics")
    # Example: Total attendance count, etc. Expand as needed.
    import sqlite3  # Ensure it's imported if not already
    conn = sqlite3.connect("attendance.db")
    df_all = pd.read_sql_query("SELECT * FROM attendance", conn)
    conn.close()
    if not df_all.empty:
        st.metric("Total Attendance Records", len(df_all))
        st.dataframe(df_all, use_container_width=True)
    else:
        st.info("No attendance records yet.")
    # Add logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
