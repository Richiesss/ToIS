export LD_LIBRARY_PATH=$(find .venv/lib/python3.10/site-packages/nvidia -name lib -type d | paste -sd ":" -):$LD_LIBRARY_PATH
streamlit run src/app.py
