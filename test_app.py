#!/usr/bin/env python3
"""
Test script to verify app.py can be imported and dependencies are available
"""

print("Testing app.py dependencies...")

try:
    import gradio as gr
    print("✓ gradio")
except ImportError as e:
    print(f"✗ gradio: {e}")

try:
    import pandas as pd
    print("✓ pandas")
except ImportError as e:
    print(f"✗ pandas: {e}")

try:
    import plotly.express as px
    print("✓ plotly")
except ImportError as e:
    print(f"✗ plotly: {e}")

try:
    from xgboost import XGBRegressor
    print("✓ xgboost")
except ImportError as e:
    print(f"✗ xgboost: {e}")

try:
    import tensorflow as tf
    print("✓ tensorflow")
except ImportError as e:
    print(f"✗ tensorflow: {e}")

try:
    import torch
    print("✓ torch")
except ImportError as e:
    print(f"✗ torch: {e}")

try:
    from langchain_community.vectorstores import FAISS
    print("✓ langchain_community")
except ImportError as e:
    print(f"✗ langchain_community: {e}")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✓ langchain_huggingface")
except ImportError as e:
    print(f"✗ langchain_huggingface: {e}")

try:
    from transformers import pipeline
    print("✓ transformers")
except ImportError as e:
    print(f"✗ transformers: {e}")

print("\nAll dependencies checked!")
print("\nTo run the app:")
print("  1. Make sure all dependencies are installed (pip install -r requirements.txt)")
print("  2. (Optional) Set HF_TOKEN environment variable for API access:")
print("     set HF_TOKEN=your_token_here")
print("  3. Run: python app.py")
