import streamlit as st
import os

st.title("Minimal Extension Test")

uploaded_file = st.file_uploader("Upload WAV or PKF", type=["wav", "pkf"])
if uploaded_file is not None:
    # Extract the extension from the uploaded file name
    original_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Create a temporary file name with the original extension
    temp_test_file = f"temp_test_file{original_ext}"
    
    with open(temp_test_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Now check the extension of the temp file
    ext = os.path.splitext(temp_test_file)[1].lower()
    st.write(f"DEBUG: The extension I see is: {ext}")
    
    if ext not in [".wav", ".pkf"]:
        st.error("Unsupported file type.")
    else:
        st.success("File extension recognized as .wav or .pkf")
