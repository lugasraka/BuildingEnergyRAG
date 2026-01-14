# EcomindAI: Building Energy Prediction & AI Consultant

A production-ready machine learning platform for predicting residential energy consumption and providing AI-powered sustainability recommendations. Built for energy analysts, building managers, and sustainability professionals who need actionable insights from smart meter data.

## Problem Statement

Building energy consumption accounts for approximately 40% of global energy use. Yet most building operators lack the tools to:
- Predict energy demand accurately for operational planning
- Identify which factors drive consumption spikes
- Get expert guidance on energy efficiency without hiring consultants

EcomindAI addresses these gaps with automated ML model training and a RAG-powered AI consultant.

## Key Capabilities

### 1. Automated Multi-Model Training
Upload a CSV file and train four production-grade models simultaneously:
- **Random Forest** (200 estimators) - Robust baseline with feature importance
- **XGBoost** (gradient boosting) - Often best-in-class for tabular data
- **TensorFlow Neural Network** - Deep learning with dropout regularization
- **PyTorch Neural Network** - Advanced architecture with BatchNorm and early stopping

Each model reports: R² (train/test), RMSE, MAE, MAPE, training latency, and relative compute cost.

### 2. Intelligent Feature Engineering
The system automatically engineers 9 additional features from raw time-series data:
- **Lag features** (1h, 6h, 24h) - Captures temporal persistence
- **Rolling averages** (3h, 6h, 12h) - Smooths noise and captures trends
- **Temporal features** (hour, day of week, month) - Encodes cyclical patterns

This typically improves model R² by 15-20% compared to raw features alone.

### 3. RAG-Powered AI Consultant
Ask natural language questions about energy efficiency, HVAC optimization, and sustainability best practices. The system uses:
- **Ollama** (llama3.2) for local LLM inference - no API costs, full data privacy
- **FAISS** vector store for fast semantic search
- **Extensible knowledge base** that can be customized for your domain

### 4. Model Interpretability
Automatic feature importance visualization shows which variables drive predictions, enabling actionable operational decisions.

## Target Users

| Persona | Use Case |
|---------|----------|
| **Energy Analyst** | Compare ML models for forecasting accuracy; identify key consumption drivers |
| **Building Manager** | Predict peak demand for load scheduling; get efficiency recommendations |
| **Sustainability Consultant** | Rapid prototyping of energy models for client assessments |
| **Data Scientist** | Benchmark neural network architectures against tree-based models |
| **Researcher** | Experiment with feature engineering approaches on UCI energy dataset |

## Quick Start

### Prerequisites
- Python 3.9+
- Ollama installed and running (for AI consultant)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BuildingEnergyRAG.git
cd BuildingEnergyRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in a separate terminal)
ollama serve

# Pull the LLM model (first time only)
ollama pull llama3.2:1b

# Launch the application
python app.py
```

The web interface will be available at `http://localhost:7860`

## Data Format

The application is optimized for the [UCI Appliances Energy Prediction Dataset](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction), but works with any CSV containing:

- A `date` column (for temporal feature extraction)
- Numeric sensor readings (temperature, humidity, etc.)
- An `Appliances` column as the target variable (or any numeric target)

Example structure:
```
date,Appliances,T1,RH_1,T2,RH_2,...
2016-01-11 17:00:00,60,19.89,47.60,19.20,44.79,...
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio Web Interface                    │
├─────────────────────────────────────────────────────────────┤
│  ML Training Tab  │  AI Consultant Tab  │  Architecture Tab │
├───────────────────┼─────────────────────┼───────────────────┤
│   Feature         │   RAG Pipeline      │   Documentation   │
│   Engineering     │   ┌─────────────┐   │                   │
│        ↓          │   │   Ollama    │   │                   │
│   Model Training  │   │  (llama3.2) │   │                   │
│   ┌──────────┐    │   └──────┬──────┘   │                   │
│   │ RF │ XGB │    │          │          │                   │
│   ├────┼─────┤    │   ┌──────┴──────┐   │                   │
│   │ TF │ PT  │    │   │    FAISS    │   │                   │
│   └──────────┘    │   │ Vector Store│   │                   │
│        ↓          │   └─────────────┘   │                   │
│   Evaluation &    │          ↑          │                   │
│   Visualization   │   Knowledge Base    │                   │
└───────────────────┴─────────────────────┴───────────────────┘
```

## Model Performance

Typical results on UCI Energy Dataset (~19,000 samples after feature engineering):

| Model | Train R² | Test R² | RMSE (Wh) | Training Time |
|-------|----------|---------|-----------|---------------|
| Random Forest | 0.95-0.97 | 0.55-0.58 | 70-75 | ~3s |
| XGBoost | 0.92-0.95 | 0.56-0.60 | 68-73 | ~2s |
| TensorFlow NN | 0.65-0.70 | 0.50-0.55 | 75-80 | ~15s |
| PyTorch NN | 0.68-0.72 | 0.52-0.57 | 72-78 | ~10s |

Note: Tree-based models typically outperform neural networks on this dataset size. Neural networks may excel with larger datasets or when combined with embeddings.

## Configuration

### Customizing the Knowledge Base
Edit `knowledge_base/sustainability_manual.txt` to add domain-specific knowledge for the AI consultant:

```text
# HVAC Optimization
Maintain setpoint of 20°C in winter and 25°C in summer for optimal efficiency.
Variable frequency drives on HVAC fans can reduce energy use by 30%.

# Your custom knowledge
Add your organization's specific guidelines here.
```

### Adjusting Model Hyperparameters
Modify parameters directly in `app.py`:

```python
# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, ...)

# XGBoost
xgb_model = XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, ...)
```

## Project Structure

```
BuildingEnergyRAG/
├── app.py                 # Main application (Gradio interface + ML pipeline)
├── requirements.txt       # Python dependencies
├── knowledge_base/        # RAG knowledge documents
│   └── sustainability_manual.txt
├── dataset/               # Sample data
│   └── energydata_complete.csv
├── data-analysis.ipynb    # Exploratory analysis notebook
└── test_app.py           # Unit tests
```

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web Framework | Gradio | Interactive UI with minimal code |
| ML (Trees) | scikit-learn, XGBoost | Random Forest, Gradient Boosting |
| ML (Deep Learning) | TensorFlow, PyTorch | Neural network architectures |
| LLM Inference | Ollama | Local, private LLM hosting |
| RAG Framework | LangChain | Retrieval-augmented generation pipeline |
| Vector Store | FAISS | Efficient similarity search |
| Embeddings | sentence-transformers | Text-to-vector conversion |
| Visualization | Plotly | Interactive charts |

## Limitations

- Neural network performance is constrained by dataset size (~19k samples)
- RAG responses depend on knowledge base quality and coverage
- Real-time inference not implemented (batch training only)
- Single-target prediction (Appliances energy only)

## Roadmap

- [x] **MVP: Core ML Pipeline** - Multi-model training (RF, XGBoost, TensorFlow, PyTorch) with automated feature engineering
- [x] **MVP: RAG-Powered Consultant** - Local LLM integration via Ollama with FAISS vector store
- [x] **MVP: Web Interface** - Gradio-based UI for model training and AI consultation
- [ ] Hyperparameter tuning interface
- [ ] Expanded knowledge base with domain-specific documents
- [ ] Multi-target prediction (Appliances + HVAC + Lighting)
- [ ] Model export for edge deployment
- [ ] Integration with building management systems (BMS)

## License

MIT License - see LICENSE file for details.

## Author

**Raka Adrianto**  
[LinkedIn](https://www.linkedin.com/in/lugasraka/)

## Acknowledgments

- **Dataset**: Candanedo, L.M., Feldheim, V., and Deramaix, D. (2017). "Data driven prediction models of energy use of appliances in a low-energy house." *Energy and Buildings*, Volume 140, Pages 81-97. DOI: [10.1016/j.enbuild.2017.01.083](https://doi.org/10.1016/j.enbuild.2017.01.083). Data available from UCI Machine Learning Repository.
- LangChain and Ollama communities for RAG infrastructure
- Gradio team for the rapid prototyping framework
