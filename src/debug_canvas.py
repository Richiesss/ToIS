import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.title("Canvas Debug")

st.markdown("Below should be a 280x280 black canvas.")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="debug_canvas",
)

if canvas_result.image_data is not None:
    st.write("Drawing data exists.")
else:
    st.write("No drawing data.")
