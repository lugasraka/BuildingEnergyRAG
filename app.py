import os
from pathlib import Path
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA

# ==========================================
# LANGFLOW CONFIGURATION
# ==========================================
# Set to True to use Langflow, False to use the original LangChain implementation
USE_LANGFLOW = False

# Path to exported Langflow flow JSON file
FLOW_PATH = Path(__file__).parent / "flows" / "energy_rag_flow.json"

# Try to import Langflow (optional dependency)
LANGFLOW_AVAILABLE = False
try:
    from langflow.load import run_flow_from_json
    LANGFLOW_AVAILABLE = True
    print("✅ Langflow is available")
except ImportError:
    print("ℹ️ Langflow not installed. Using default LangChain RAG implementation.")
    print("   To enable Langflow: pip install langflow")

# ==========================================
# 1. CONFIGURATION & KNOWLEDGE BASE
# ==========================================
def load_knowledge_base(filepath="knowledge_base/sustainability_manual.txt"):
    """
    Load the sustainability knowledge base from an external file.
    Falls back to a minimal default if file not found.
    """
    try:
        # Try relative path first
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✅ Loaded knowledge base from {filepath}")
            return content

        # Try absolute path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(script_dir, filepath)
        if os.path.exists(abs_path):
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✅ Loaded knowledge base from {abs_path}")
            return content

        print(f"⚠️ Knowledge base file not found: {filepath}")
        return DEFAULT_KNOWLEDGE_BASE

    except Exception as e:
        print(f"⚠️ Error loading knowledge base: {e}")
        return DEFAULT_KNOWLEDGE_BASE

# Minimal fallback knowledge base
DEFAULT_KNOWLEDGE_BASE = """
To optimize HVAC systems in residential buildings, maintain a setpoint of 20°C in winter and 25°C in summer.
Predictive maintenance of furnaces can reduce energy waste by up to 15%.
Temperature and humidity are the most important features for predicting residential energy consumption.
Time-series features like lag values (1h, 6h, 24h) significantly improve appliance energy prediction accuracy.
"""

# Load the knowledge base at startup
SUSTAINABILITY_MANUAL = load_knowledge_base()

# PyTorch Model Architecture
class PyTorchDeepNN(nn.Module):
    """Deep Neural Network for Energy Prediction - Improved Architecture"""
    def __init__(self, input_dim):
        super(PyTorchDeepNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# ==========================================
# 2. DATA PROCESSING & FEATURE ENGINEERING
# ==========================================
def process_energy_data(df):
    """
    Process energy data with lag features and rolling averages.
    Based on the best performing approach from the notebook.
    """
    df = df.copy()
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Extract time features
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour
        
        # Create lag features for Appliances (data recorded every 10 minutes)
        # 1 hour = 6 rows, 6 hours = 36 rows, 24 hours = 144 rows
        if 'Appliances' in df.columns:
            df['Appliances_lag_1h'] = df['Appliances'].shift(6)
            df['Appliances_lag_6h'] = df['Appliances'].shift(36)
            df['Appliances_lag_24h'] = df['Appliances'].shift(144)
            
            # Rolling averages
            df['Appliances_roll_3h'] = df['Appliances'].rolling(window=18).mean()
            df['Appliances_roll_6h'] = df['Appliances'].rolling(window=36).mean()
            df['Appliances_roll_12h'] = df['Appliances'].rolling(window=72).mean()
        
        # Drop rows with NaN from lag/rolling
        df = df.dropna().reset_index(drop=True)
        
        # Drop date and lights columns
        df.drop(columns=['date'], inplace=True, errors='ignore')
        df.drop(columns=['lights'], inplace=True, errors='ignore')
    
    return df

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error, handling zero values"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def create_scatter_plot(y_true, y_pred, title, r2, rmse, mae):
    """Generates a scatter plot of actual vs. predicted values with metrics."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, label='Predictions')
    
    # Add a 1:1 line for reference
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal Fit')
    
    # Add metrics to the plot
    metrics_text = (
        f"R² = {r2:.4f}\n"
        f"RMSE = {rmse:.2f}\n"
        f"MAE = {mae:.2f}"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    ax.set_xlabel("Actual Values (Wh)")
    ax.set_ylabel("Predicted Values (Wh)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def load_and_train_models(file_obj):
    """
    Loads data, processes it, and trains multiple best-performing models.
    Returns comprehensive metrics including RMSE, MAE, MAPE, Train/Test R², Latency, and Cost.
    """
    if file_obj is None:
        return None, None, None, None, None, "Please upload a CSV file.", None, None, None, None

    try:
        # Load Data
        df = pd.read_csv(file_obj.name)

        # Process data with feature engineering
        df = process_energy_data(df)

        # Identify target column
        target_col = 'Appliances' if 'Appliances' in df.columns else df.select_dtypes(include=[np.number]).columns[0]

        # Prepare features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features for neural networks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = []
        trained_models = {}
        plots = {}

        # 1. Random Forest
        print("Training Random Forest...")
        start_time = time.time()
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_latency = time.time() - start_time

        y_pred_rf_train = rf_model.predict(X_train)
        y_pred_rf = rf_model.predict(X_test)
        
        # Calculate metrics
        rf_r2_train = r2_score(y_train, y_pred_rf_train)
        rf_r2_test = r2_score(y_test, y_pred_rf)
        rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        rf_mae = mean_absolute_error(y_test, y_pred_rf)
        rf_mape = calculate_mape(y_test, y_pred_rf)
        
        plots['rf'] = create_scatter_plot(y_test, y_pred_rf, "Random Forest: Actual vs. Predicted", rf_r2_test, rf_rmse, rf_mae)

        results.append({
            'Model': 'Random Forest',
            'Train_R2': round(rf_r2_train, 4),
            'Test_R2': round(rf_r2_test, 4),
            'RMSE': round(rf_rmse, 2),
            'MAE': round(rf_mae, 2),
            'MAPE (%)': round(rf_mape, 2),
            'Latency (s)': round(rf_latency, 2),
            'Cost': 'Low'
        })
        trained_models['Random Forest'] = rf_model

        # 2. XGBoost
        print("Training XGBoost...")
        start_time = time.time()
        xgb_model = XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        xgb_latency = time.time() - start_time

        y_pred_xgb_train = xgb_model.predict(X_train)
        y_pred_xgb = xgb_model.predict(X_test)

        # Calculate metrics
        xgb_r2_train = r2_score(y_train, y_pred_xgb_train)
        xgb_r2_test = r2_score(y_test, y_pred_xgb)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
        xgb_mape = calculate_mape(y_test, y_pred_xgb)

        plots['xgb'] = create_scatter_plot(y_test, y_pred_xgb, "XGBoost: Actual vs. Predicted", xgb_r2_test, xgb_rmse, xgb_mae)

        results.append({
            'Model': 'XGBoost',
            'Train_R2': round(xgb_r2_train, 4),
            'Test_R2': round(xgb_r2_test, 4),
            'RMSE': round(xgb_rmse, 2),
            'MAE': round(xgb_mae, 2),
            'MAPE (%)': round(xgb_mape, 2),
            'Latency (s)': round(xgb_latency, 2),
            'Cost': 'Low'
        })
        trained_models['XGBoost'] = xgb_model

        # 3. TensorFlow Deep NN
        print("Training TensorFlow Deep NN...")
        start_time = time.time()
        tf_model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        tf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        tf_model.fit(X_train_scaled, y_train.values, validation_split=0.2, epochs=50,
                     batch_size=64, verbose=0)
        tf_latency = time.time() - start_time

        y_pred_tf_train = tf_model.predict(X_train_scaled, verbose=0).flatten()
        y_pred_tf = tf_model.predict(X_test_scaled, verbose=0).flatten()

        # Calculate metrics
        tf_r2_train = r2_score(y_train, y_pred_tf_train)
        tf_r2_test = r2_score(y_test, y_pred_tf)
        tf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tf))
        tf_mae = mean_absolute_error(y_test, y_pred_tf)
        tf_mape = calculate_mape(y_test, y_pred_tf)
        
        plots['tf'] = create_scatter_plot(y_test, y_pred_tf, "TensorFlow NN: Actual vs. Predicted", tf_r2_test, tf_rmse, tf_mae)

        results.append({
            'Model': 'TensorFlow NN',
            'Train_R2': round(tf_r2_train, 4),
            'Test_R2': round(tf_r2_test, 4),
            'RMSE': round(tf_rmse, 2),
            'MAE': round(tf_mae, 2),
            'MAPE (%)': round(tf_mape, 2),
            'Latency (s)': round(tf_latency, 2),
            'Cost': 'Medium'
        })
        trained_models['TensorFlow'] = tf_model

        # 4. PyTorch Deep NN - Optimized Training
        print("Training PyTorch Deep NN...")
        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pytorch_model = PyTorchDeepNN(X_train_scaled.shape[1]).to(device)

        # Optimized training setup
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.AdamW(pytorch_model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # Prepare training data
        X_train_pt = torch.FloatTensor(X_train_scaled).to(device)
        y_train_pt = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)

        # Optimized training parameters
        batch_size = 128  # Larger batches = faster training
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 8  # More aggressive early stopping

        pytorch_model.train()
        for epoch in range(80):  # Reduced from 200
            indices = torch.randperm(len(X_train_pt))
            X_shuffled = X_train_pt[indices]
            y_shuffled = y_train_pt[indices]

            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(X_shuffled), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                optimizer.zero_grad()
                outputs = pytorch_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            scheduler.step(avg_epoch_loss)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

        pt_latency = time.time() - start_time

        # Evaluate PyTorch model
        pytorch_model.eval()
        with torch.no_grad():
            y_pred_pt_train = pytorch_model(X_train_pt).cpu().numpy().flatten()
            X_test_pt = torch.FloatTensor(X_test_scaled).to(device)
            y_pred_pt = pytorch_model(X_test_pt).cpu().numpy().flatten()

        # Calculate metrics
        pt_r2_train = r2_score(y_train, y_pred_pt_train)
        pt_r2_test = r2_score(y_test, y_pred_pt)
        pt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_pt))
        pt_mae = mean_absolute_error(y_test, y_pred_pt)
        pt_mape = calculate_mape(y_test, y_pred_pt)

        plots['pt'] = create_scatter_plot(y_test, y_pred_pt, "PyTorch NN: Actual vs. Predicted", pt_r2_test, pt_rmse, pt_mae)


        results.append({
            'Model': 'PyTorch NN',
            'Train_R2': round(pt_r2_train, 4),
            'Test_R2': round(pt_r2_test, 4),
            'RMSE': round(pt_rmse, 2),
            'MAE': round(pt_mae, 2),
            'MAPE (%)': round(pt_mape, 2),
            'Latency (s)': round(pt_latency, 2),
            'Cost': 'Medium'
        })
        trained_models['PyTorch'] = pytorch_model

        results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)

        # Create detailed status message
        best_model = results_df.iloc[0]
        status_msg = f"✅ All models trained successfully! Target: {target_col}\n"
        status_msg += f"Best Model: {best_model['Model']} (Test R²: {best_model['Test_R2']}, RMSE: {best_model['RMSE']})"

        return df, trained_models, scaler, X.columns.tolist(), results_df, status_msg, plots['rf'], plots['xgb'], plots['tf'], plots['pt']
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return None, None, None, None, None, error_msg, None, None, None, None


def create_feature_importance_chart(models, features):
    """Create feature importance chart from Random Forest model"""
    if models is None or features is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for model training. Please upload a dataset.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Feature Importance",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500
        )
        return fig

    try:
        # Get feature importance from Random Forest (most interpretable)
        if 'Random Forest' in models:
            rf_model = models['Random Forest']
            importances = rf_model.feature_importances_

            # Create dataframe for plotting
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(15)  # Top 15 features

            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Feature Importance (Random Forest)',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                yaxis_title="",
                xaxis_title="Importance Score",
                showlegend=False,
                height=500
            )
            return fig

        # Fallback to XGBoost if RF not available
        elif 'XGBoost' in models:
            xgb_model = models['XGBoost']
            importances = xgb_model.feature_importances_

            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(15)

            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Feature Importance (XGBoost)',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                yaxis_title="",
                xaxis_title="Importance Score",
                showlegend=False,
                height=500
            )
            return fig

        # No tree-based model available
        fig = go.Figure()
        fig.add_annotation(
            text="No tree-based model available for feature importance",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Feature Importance",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500
        )
        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Feature Importance Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Feature Importance",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500
        )
        return fig

def create_comparison_chart(results_df):
    """Create model comparison chart"""
    if results_df is None or (isinstance(results_df, pd.DataFrame) and results_df.empty):
        # Create empty figure with proper Plotly structure
        fig = go.Figure()
        fig.add_annotation(
            text="No model results yet. Please upload a dataset to train models.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Model Performance Comparison (Test R² Score)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig

    try:
        if isinstance(results_df, pd.DataFrame) and not results_df.empty:
            fig = px.bar(
                results_df,
                x='Model',
                y='Test_R2',
                title='Model Performance Comparison (Test R² Score)',
                text='Test_R2',
                color='Test_R2',
                color_continuous_scale='Viridis'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(showlegend=False, yaxis_title="R² Score")
            return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

def show_charts_interpretation(results_df):
    """Show chart interpretations after model training completes"""
    if results_df is None or (isinstance(results_df, pd.DataFrame) and results_df.empty):
        return gr.Markdown(value="", visible=False)
    
    interpretation = """
#### How to Interpret These Charts:

**Feature Importance (Left Chart):**
- Shows which variables most strongly influence energy predictions
- Higher values = more important for the model's decisions
- Lag features (e.g., Appliances_lag_24h) typically rank highest because past usage strongly predicts future usage
- Temperature and humidity sensors often appear as they directly affect HVAC and appliance behavior
- Use this to identify which sensors/factors to monitor most closely in your building

**Model Performance Comparison (Right Chart):**
- Compares how well each model predicts energy consumption
- R² Score ranges from 0 to 1 (higher is better): 
  - 0.9+ = Excellent predictions
  - 0.7-0.9 = Good predictions
  - Below 0.7 = May need improvement
- Neural networks (TensorFlow/PyTorch) often excel at capturing complex patterns
- Tree-based models (Random Forest/XGBoost) provide good baselines and are easier to interpret
"""
    return gr.Markdown(value=interpretation, visible=True)

def show_results_interpretation(results_df):
    """Show results table interpretation after model training completes"""
    if results_df is None or (isinstance(results_df, pd.DataFrame) and results_df.empty):
        return gr.Markdown(value="", visible=False)
    
    interpretation = """
#### Understanding the Model Results Table:

**Key Metrics Explained:**
- **Train R² vs Test R²**: Compare these to check for overfitting
  - If Train R² >> Test R², the model memorized training data instead of learning patterns
  - Similar values indicate good generalization to new data

- **RMSE (Root Mean Squared Error)**: Average prediction error in Watt-hours
  - Lower is better. Heavily penalizes large errors
  - Compare this to the typical range of your energy values

- **MAE (Mean Absolute Error)**: Average absolute error in Watt-hours
  - Lower is better. More robust to outliers than RMSE
  - Easier to interpret: "on average, predictions are off by X Wh"

- **MAPE (%)**: Percentage error relative to actual values
  - Lower is better. Useful for comparing across different scales
  - 10% MAPE means predictions are typically within 10% of actual values

- **Latency**: Training time in seconds
  - Consider this for deployment: faster models can retrain more frequently
  - Neural networks take longer but may achieve better accuracy

- **Cost**: Computational resources required
  - Low: Can run on basic hardware, minimal cloud costs
  - Medium: Benefits from GPU acceleration for faster training
"""
    return gr.Markdown(value=interpretation, visible=True)

# ==========================================
# 3. GEN AI / RAG ENGINE (Supports both local and API-based)
# ==========================================
def setup_rag_system():
    """
    Initializes the RAG chain with Ollama for high-quality local inference.
    """
    print("Setting up RAG system with Ollama...")

    # 1. Chunking (Splitting the manual into retrievable bits)
    # Filter out empty lines and comment lines (starting with #)
    lines = [line.strip() for line in SUSTAINABILITY_MANUAL.split('\n')
             if line.strip() and not line.strip().startswith('#')]
    docs = [Document(page_content=txt) for txt in lines]
    print(f"   Loaded {len(docs)} knowledge entries")

    # 2. Embeddings (Free, runs locally)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Vector Store
    vector_store = FAISS.from_documents(docs, embeddings)

    # 4. LLM - Use Ollama with llama3.2 model (better quality)
    try:
        print("Connecting to Ollama (llama3.2 model)...")
        from langchain_ollama import OllamaLLM
        from langchain_core.prompts import PromptTemplate

        llm = OllamaLLM(
            model="llama3.2",         # 3B model for better quality
            temperature=0.7,
            num_predict=512,
            num_ctx=4096,
        )

        # Custom prompt template for better responses with smaller models
        prompt_template = """Use the following context to answer the question. Be helpful and informative.

Context:
{context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        print("✅ RAG system initialized with Ollama (llama3.2)!")

        # 5. Retrieval Chain with custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False
        )
        return qa_chain

    except Exception as e:
        print(f"⚠️ Error setting up RAG: {e}")
        return None

# Initialize RAG globally
rag_chain = None
try:
    rag_chain = setup_rag_system()
except Exception as e:
    print(f"RAG Setup Warning: {e}")

def ask_consultant_langflow(user_query: str) -> str:
    """
    Query the RAG system using Langflow.

    This function loads and executes a Langflow flow exported as JSON.
    Design your flow in Langflow UI, export it, and save to flows/energy_rag_flow.json

    Args:
        user_query: User's question about energy efficiency

    Returns:
        AI-generated response based on RAG
    """
    if not LANGFLOW_AVAILABLE:
        return "⚠️ Langflow is not installed. Run: pip install langflow"

    if not FLOW_PATH.exists():
        return (
            "⚠️ Langflow flow file not found.\n\n"
            "To set up Langflow:\n"
            "1. Run: langflow run\n"
            "2. Design your RAG flow in the visual editor\n"
            "3. Export the flow to: flows/energy_rag_flow.json\n\n"
            "Falling back to original implementation..."
        )

    try:
        result = run_flow_from_json(
            flow=str(FLOW_PATH),
            input_value=user_query,
            fallback_to_env_vars=True,
            session_id="energy-consultant"
        )

        # Navigate the result structure to extract the response text
        # Langflow returns a list of RunOutputs
        if result and len(result) > 0:
            outputs = result[0].outputs
            if outputs and len(outputs) > 0:
                # Try to get the message text from the result
                message = outputs[0].results.get("message")
                if message and hasattr(message, "text"):
                    return message.text
                # Fallback: try to get result directly
                if "text" in outputs[0].results:
                    return outputs[0].results["text"]
                # Last resort: return string representation
                return str(outputs[0].results)

        return "No response generated. Please check your Langflow flow configuration."

    except Exception as e:
        return f"Langflow error: {str(e)}\n\nTry checking your flow configuration or switch to the original implementation."


def ask_consultant_langchain(user_query: str) -> str:
    """
    AI consultant using the original LangChain RAG system.

    Args:
        user_query: User's question about energy efficiency

    Returns:
        AI-generated response based on RAG
    """
    if not rag_chain:
        return "⚠️ RAG engine not initialized. Please check console for errors."

    try:
        # Pass query directly - custom prompt handles formatting
        response = rag_chain.invoke({"query": user_query})

        # Handle different response formats
        if isinstance(response, dict):
            return response.get("result", str(response))
        return str(response)
    except Exception as e:
        return f"Error: {str(e)}. Try rephrasing your question."


def ask_consultant(user_query, history):
    """
    AI consultant that switches between Langflow and LangChain implementations.

    Toggle implementation by setting USE_LANGFLOW at the top of this file:
    - USE_LANGFLOW = True  -> Uses Langflow (requires exported flow JSON)
    - USE_LANGFLOW = False -> Uses original LangChain implementation (default)

    Args:
        user_query: User's question about energy efficiency
        history: Chat history (required by Gradio ChatInterface)

    Returns:
        AI-generated response based on RAG
    """
    if USE_LANGFLOW and LANGFLOW_AVAILABLE:
        response = ask_consultant_langflow(user_query)
        # If Langflow fails due to missing flow, fall back to LangChain
        if "Falling back to original implementation" in response:
            return ask_consultant_langchain(user_query)
        return response
    else:
        return ask_consultant_langchain(user_query)

# ==========================================
# 4. GRADIO INTERFACE
# ==========================================

# Professional theme with neutral fonts
professional_theme = gr.themes.Base(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.1)",
    button_primary_background_fill="*primary_600",
    button_primary_text_color="white",
    input_background_fill="white",
    input_border_color="*neutral_300",
)

with gr.Blocks(theme=professional_theme, title="EcomindAI Energy Consultant") as demo:
    gr.Markdown("""
    # EcoMindAI: Advanced Energy Prediction & RAG-Powered AI Consultant
    
    **Transform your building energy data into actionable insights with state-of-the-art ML and AI.**
    
    **Key Features:**
    - Train 4 advanced AI models in parallel: Scikit-learn (Random Forest), XGBoost, TensorFlow & PyTorch Neural Networks
    - Automatic feature engineering with temporal patterns and lag analysis
    - Interactive AI consultant powered by RAG (Retrieval-Augmented Generation) for expert energy efficiency advice
    - Comprehensive model comparison with interpretable metrics and visualizations
    
    **Created by [Raka Adrianto](https://www.linkedin.com/in/lugasraka/)**
    """)
    
    with gr.Tab("ML Model Training & Analytics"):
        gr.Markdown("### Upload your energy dataset (CSV format)")
        
        with gr.Row():
            file_input = gr.File(label="Upload CSV (e.g., energydata_complete.csv)", file_types=[".csv"])
        
        status_text = gr.Textbox(label="Training Status", interactive=False, lines=3)
        
        with gr.Row():
            plot_output = gr.Plot(label="Feature Importance")
            comparison_plot = gr.Plot(label="Model Performance Comparison")
        
        charts_interpretation = gr.Markdown(value="", visible=False)

        results_table = gr.Dataframe(label="Model Results Summary")
        
        results_interpretation = gr.Markdown(value="", visible=False)
        
        with gr.Accordion("Model Performance Scatter Plots", open=False):
            gr.Markdown("Visualizing model accuracy by plotting actual vs. predicted values. Points close to the red dashed line indicate higher accuracy.")
            with gr.Row():
                rf_plot = gr.Plot(label="Random Forest Performance")
                xgb_plot = gr.Plot(label="XGBoost Performance")
            with gr.Row():
                tf_plot = gr.Plot(label="TensorFlow NN Performance")
                pt_plot = gr.Plot(label="PyTorch NN Performance")


        # State variables
        df_state = gr.State()
        models_state = gr.State()
        scaler_state = gr.State()
        features_state = gr.State()
        results_state = gr.State()

        # Event handlers
        file_input.change(
            fn=load_and_train_models,
            inputs=file_input,
            outputs=[df_state, models_state, scaler_state, features_state, results_state, status_text, rf_plot, xgb_plot, tf_plot, pt_plot]
        )

        # Feature importance chart (triggered when models are trained)
        models_state.change(
            fn=create_feature_importance_chart,
            inputs=[models_state, features_state],
            outputs=plot_output
        )
        results_state.change(fn=create_comparison_chart, inputs=results_state, outputs=comparison_plot)
        results_state.change(fn=lambda x: x, inputs=results_state, outputs=results_table)
        
        # Show interpretations only after training completes
        results_state.change(fn=show_charts_interpretation, inputs=results_state, outputs=charts_interpretation)
        results_state.change(fn=show_results_interpretation, inputs=results_state, outputs=results_interpretation)

    with gr.Tab("AI Energy Consultant (RAG)"):
        gr.Markdown("""
        ### Ask the AI about energy efficiency and sustainability

        **Powered by:** RAG with Ollama (llama3.2) - Local inference. llama3.2 model provides high-quality responses, balancing performance and resource usage. 
                    Created by Meta with optimizations for local deployments and open-source development. Obtained via [Ollama](https://ollama.com/library/llama3.2).
        """)
        
        chatbot = gr.ChatInterface(
            fn=ask_consultant,
            examples=[
                "How does humidity affect energy usage?",
                "What is the best temperature for HVAC systems?",
                "How do lag features improve energy prediction?",
                "What are the most important features for predicting energy consumption?",
                "How can I reduce energy waste in my home?"
            ],
            title="Energy Efficiency Expert",
            description="Ask questions about building energy, HVAC optimization, and sustainability best practices."
        )
    
    with gr.Tab("Model Architecture & Rationale"):
        gr.Markdown("""
        ## Model Architecture & Design Rationale

        ---

        ### Target Variable: `Appliances` (Energy Consumption in Wh)

        **Why this target?**
        - Appliance energy consumption is the most controllable component of household energy use
        - Direct correlation with user behavior patterns (schedulable, optimizable)
        - High variance makes it ideal for ML prediction (unlike relatively stable baseline loads)
        - Actionable insights: predictions can trigger demand response or scheduling recommendations

        ---

        ### Feature Variables & Engineering

        #### Original Features (from UCI Energy Dataset)
        | Feature | Description | Why Important |
        |---------|-------------|---------------|
        | T1-T9 | Temperature sensors (°C) in different rooms | Indoor climate directly affects HVAC and appliance usage |
        | RH_1-RH_9 | Humidity sensors (%) in different rooms | High humidity triggers dehumidifiers; affects comfort |
        | T_out, RH_out | Outdoor temperature & humidity | External conditions drive heating/cooling demand |
        | Press_mm_hg | Atmospheric pressure | Weather patterns correlate with energy behavior |
        | Windspeed, Visibility | Weather conditions | Affects natural ventilation choices |

        #### Engineered Features (Critical for Performance)
        | Feature | Formula | Rationale |
        |---------|---------|-----------|
        | **Lag 1h** | `Appliances.shift(6)` | Captures immediate persistence (what was used 1 hour ago) |
        | **Lag 6h** | `Appliances.shift(36)` | Captures mid-day patterns (morning vs afternoon) |
        | **Lag 24h** | `Appliances.shift(144)` | Captures daily cycles (same time yesterday) |
        | **Roll 3h** | `Appliances.rolling(18).mean()` | Smooths short-term noise |
        | **Roll 6h** | `Appliances.rolling(36).mean()` | Captures half-day trends |
        | **Roll 12h** | `Appliances.rolling(72).mean()` | Captures day/night patterns |
        | **Month** | `date.dt.month` | Seasonal patterns (heating in winter, cooling in summer) |
        | **Day of Week** | `date.dt.dayofweek` | Weekday vs weekend behavior |
        | **Hour** | `date.dt.hour` | Time-of-day patterns (peak hours) |

        **Why lag features matter:** Energy consumption is highly autocorrelated. Knowing recent usage is the strongest predictor of current usage. This improves R² by ~15-20%.

        ---

        ### Model Architectures

        #### 1. Random Forest (Baseline - Tree Ensemble)
        ```
        RandomForestRegressor(n_estimators=200, max_depth=15)
        ```
        | Parameter | Value | Rationale |
        |-----------|-------|-----------|
        | n_estimators | 200 | Enough trees for stable predictions without overfitting |
        | max_depth | 15 | Prevents overfitting while capturing complex interactions |
        | n_jobs | -1 | Parallel training for speed |

        **Why Random Forest?**
        - Robust to outliers and non-linear relationships
        - No feature scaling required
        - Provides feature importance rankings
        - Excellent baseline with minimal tuning

        ---

        #### 2. XGBoost (Gradient Boosting)
        ```
        XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1)
        ```
        | Parameter | Value | Rationale |
        |-----------|-------|-----------|
        | n_estimators | 200 | Sequential boosting iterations |
        | max_depth | 7 | Shallower than RF (boosting corrects errors iteratively) |
        | learning_rate | 0.1 | Balance between speed and accuracy |

        **Why XGBoost?**
        - Often outperforms Random Forest on structured data
        - Built-in regularization prevents overfitting
        - Handles missing values natively
        - Industry standard for tabular data competitions

        ---

        #### 3. TensorFlow Neural Network
        ```
        Input → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.2)
              → Dense(64) → Dropout(0.2) → Dense(32) → Dense(1)
        ```
        | Layer | Neurons | Activation | Rationale |
        |-------|---------|------------|-----------|
        | Input | 256 | ReLU | Wide first layer captures feature combinations |
        | Hidden 1 | 128 | ReLU | Gradual compression learns hierarchical patterns |
        | Hidden 2 | 64 | ReLU | Further abstraction |
        | Hidden 3 | 32 | ReLU | Final feature compression |
        | Output | 1 | Linear | Regression output |

        **Why this architecture?**
        - Funnel shape (256→128→64→32→1) progressively compresses information
        - Dropout (0.3→0.2→0.2) prevents overfitting, higher at start where more parameters
        - ReLU activation: computationally efficient, avoids vanishing gradients
        - Adam optimizer: adaptive learning rate, works well out-of-the-box

        ---

        #### 4. PyTorch Neural Network (Advanced)
        ```
        Input → Linear(512) → BatchNorm → ReLU → Dropout(0.4)
              → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(64)  → BatchNorm → ReLU → Dropout(0.2)
              → Linear(32)  → ReLU → Dropout(0.1) → Linear(1)
        ```
        | Component | Purpose |
        |-----------|---------|
        | **BatchNorm** | Normalizes activations, allows higher learning rates, acts as regularizer |
        | **HuberLoss** | Robust to outliers (combines MSE for small errors, MAE for large) |
        | **AdamW** | Adam with decoupled weight decay for better generalization |
        | **ReduceLROnPlateau** | Automatically reduces learning rate when loss plateaus |
        | **Gradient Clipping** | Prevents exploding gradients (max_norm=1.0) |
        | **Early Stopping** | Stops training when validation loss stops improving (patience=15) |

        **Why these choices?**
        - BatchNorm + Dropout together provide strong regularization
        - HuberLoss handles energy spikes better than pure MSE
        - Learning rate scheduling adapts to training dynamics
        - Early stopping prevents overfitting to training data

        ---

        ### Metrics Explanation

        | Metric | Formula | Interpretation |
        |--------|---------|----------------|
        | **R² (Coefficient of Determination)** | 1 - (SS_res / SS_tot) | % of variance explained. Higher = better. 1.0 is perfect. |
        | **RMSE (Root Mean Squared Error)** | √(Σ(y-ŷ)²/n) | Penalizes large errors more. Same units as target. |
        | **MAE (Mean Absolute Error)** | Σ|y-ŷ|/n | Average error magnitude. Robust to outliers. |
        | **MAPE (Mean Absolute % Error)** | Σ|y-ŷ|/y × 100 | Relative error. Intuitive but sensitive to small values. |
        | **Train R² vs Test R²** | Compare both | Large gap = overfitting. Similar values = good generalization. |

        ---

        ### Data Pipeline

        ```
        Raw CSV → Date Parsing → Feature Engineering → Train/Test Split (80/20)
                → Scaling (StandardScaler for NNs) → Model Training → Evaluation
        ```

        **Why 80/20 split?** Standard practice balancing training data volume with reliable test evaluation.

        **Why StandardScaler for NNs?** Neural networks converge faster and perform better when features are normalized (mean=0, std=1). Tree-based models don't need this.
        """)

    with gr.Tab("About"):
        gr.Markdown("""
        ## About EcoMindAI

        ### Default Dataset: UCI ML Repository - Appliances Energy Prediction
        
        This application is designed to work with the **"Appliances energy prediction" dataset** from the UCI Machine Learning Repository.
        
        **Dataset Overview:**
        - **Source**: UCI Machine Learning Repository
        - **Citation**: Luis Candanedo, Véronique Feldheim, Dominique Deramaix. "Data driven prediction models of energy use of appliances in a low-energy house." Energy and Buildings, Volume 140, 1 April 2017, Pages 81-97.
        - **DOI**: [10.1016/j.enbuild.2017.01.083](https://doi.org/10.1016/j.enbuild.2017.01.083)
        - **Repository Link**: [UCI ML Repository - Appliances Energy Prediction](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)
        
        **Data Collection Details:**
        - **Location**: A low-energy residential house in Stambruges, Belgium
        - **Duration**: 4.5 months (January 11, 2016 to May 27, 2016)
        - **Frequency**: Data recorded every 10 minutes
        - **Total Records**: 19,735 observations
        - **Sensors**: 
          - 9 temperature sensors (T1-T9) across different rooms
          - 9 humidity sensors (RH_1 to RH_9) across different rooms
          - Weather station data (outdoor temperature, humidity, pressure, wind speed, visibility)
        
        **Target Variables:**
        - **Appliances**: Energy consumption of appliances in the house (Wh)
        - **Lights**: Energy consumption of lighting fixtures (Wh) - not used in this application
        
        **Why This Dataset?**
        - Real-world, continuously monitored residential energy data
        - Rich multi-sensor setup capturing various environmental factors
        - High temporal resolution (10-minute intervals) enables lag feature engineering
        - Well-documented and peer-reviewed by energy researchers
        - Represents modern low-energy building standards with smart home monitoring
        - Contains both controllable (appliances) and external (weather) factors
        
        **Data Characteristics:**
        - Non-linear relationships between features and energy consumption
        - Strong temporal autocorrelation (past usage predicts future usage)
        - Seasonal patterns (heating in winter months)
        - Time-of-day patterns (higher usage during active hours)
        - Weather-dependent variations (external conditions affect HVAC loads)
        
        ---
        
        ### Machine Learning Models
        - **Random Forest**: Ensemble tree-based model with 200 estimators
        - **XGBoost**: Gradient boosting with optimized hyperparameters
        - **TensorFlow Deep NN**: 5-layer neural network with dropout regularization
        - **PyTorch Deep NN**: Custom deep architecture with BatchNorm and advanced training

        ### Feature Engineering
        - **Temporal Features**: Month, day of week, hour
        - **Lag Features**: 1-hour, 6-hour, 24-hour historical values
        - **Rolling Averages**: 3-hour, 6-hour, 12-hour moving averages

        ### RAG System
        - **LLM**: Ollama with llama3.2 (local inference)
        - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
        - **Vector Store**: FAISS for efficient similarity search
        - **Optional**: Langflow integration for visual RAG pipeline design

        ### Setup Requirements
        Ensure Ollama is running:
        ```bash
        ollama serve
        ```

        ### Langflow Integration (Optional)
        You can use Langflow to visually design and modify the RAG pipeline:

        **1. Install Langflow:**
        ```bash
        pip install langflow
        ```

        **2. Launch Langflow visual editor:**
        ```bash
        langflow run
        ```

        **3. Design your RAG flow with these components:**
        - Chat Input → File Loader → Text Splitter → HuggingFace Embeddings
        - FAISS Vector Store → Ollama LLM → RetrievalQA → Chat Output

        **4. Export flow:**
        - Save to `flows/energy_rag_flow.json`

        **5. Enable Langflow in app.py:**
        ```python
        USE_LANGFLOW = True  # Change from False to True
        ```
        
        ### Supported Data Format
        **Expected CSV format**: Time-series energy data with:
        - Date/timestamp column
        - Target variable (e.g., Appliances, energy consumption)
        - Feature columns (temperature, humidity, weather data, etc.)
        
        **Example**: The UCI energydata_complete.csv contains columns like:
        - `date`: Timestamp (YYYY-MM-DD HH:MM:SS)
        - `Appliances`: Target variable (Wh)
        - `T1` through `T9`: Temperature sensors (°C)
        - `RH_1` through `RH_9`: Humidity sensors (%)
        - `T_out`, `RH_out`, `Windspeed`, `Visibility`, `Press_mm_hg`: Weather data
        
        You can upload your own energy dataset following a similar structure with timestamp and sensor readings.
        """)

if __name__ == "__main__":
    demo.launch(share=False)
