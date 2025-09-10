import streamlit as st
def display_results(caption, segmentation_result):
    st.write("Caption:", caption)
    st.write("Segmentation Classes:", segmentation_result.get("classes", []))
    st.write("Scores:", segmentation_result.get("scores", []))
