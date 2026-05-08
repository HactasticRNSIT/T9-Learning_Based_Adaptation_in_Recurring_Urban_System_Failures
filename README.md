# UrbanMind AI 🚦🌧️

AI-powered adaptive urban resilience system focused on predicting and preventing recurring traffic congestion and flood-related disruptions in RR Nagar, Bengaluru.

---

# 📌 Problem Statement

Urban systems frequently experience recurring disruptions such as traffic congestion, urban flooding, infrastructure failures, and public-service interruptions. Despite years of monitoring and historical data collection, many cities continue to respond reactively instead of learning from previous failures.

RR Nagar, Bengaluru experiences recurring traffic congestion during peak office hours, rainfall conditions, metro connectivity rush periods, and road bottlenecks. Heavy rainfall further increases the risk of waterlogging and severe traffic buildup in critical junctions.

UrbanMind AI aims to solve this challenge by building an AI-powered adaptive intelligence system capable of:

- Detecting recurring urban failure patterns
- Predicting traffic congestion risks
- Identifying flood-prone zones
- Learning from historical disruptions
- Generating proactive adaptation recommendations

---

# 🎯 Project Objective

The objective of UrbanMind AI is to develop a smart urban resilience platform that:

✅ Predicts traffic congestion levels  
✅ Detects recurring disruption hotspots  
✅ Estimates flood-related traffic risk  
✅ Identifies hidden urban dependency patterns  
✅ Provides proactive adaptation recommendations  
✅ Supports intelligent urban traffic management  

---

# 💡 Proposed Solution

UrbanMind AI combines machine learning, traffic analytics, weather analysis, and interactive visualizations to monitor and predict urban disruptions in RR Nagar.

The system analyzes:

- Vehicle density
- Rainfall intensity
- Traffic signal behavior
- Historical incident records
- Public transport activity
- Congestion patterns
- Flood-prone regions

Using AI-driven prediction models, the platform helps authorities take preventive actions before major disruptions occur.

---

# 🚀 Key Features

- AI-based traffic congestion prediction
- Flood risk and rainfall correlation analysis
- RR Nagar hotspot detection
- Interactive smart dashboard
- Failure risk gauge visualization
- Feature importance analysis
- Adaptive traffic recommendations
- Historical congestion analytics
- Real-time risk alerts
- Interactive zone mapping using Mapbox

---

# 🛠️ Tech Stack

## 1. Core Language
- Python

---

## 2. Frontend & Web Framework

### Streamlit
Used to build the interactive web dashboard including:
- Sidebar controls
- User inputs
- Real-time prediction display
- Dynamic visual analytics

### Custom CSS
Used for:
- Dark mode UI
- Custom metric cards
- Animated alert badges
- Enhanced dashboard styling

---

## 3. Machine Learning & Data Processing

### Scikit-learn
- Random Forest Regressor
- Traffic congestion prediction model

### Pandas
Used for:
- Dataset loading
- Data preprocessing
- Historical traffic record management

### NumPy
Used for:
- Numerical operations
- Prediction optimization
- Feature importance calculations

---

## 4. Visualizations & Mapping

### Plotly Graph Objects (`go`)
Used for:
- Failure Risk Gauge
- Feature Importance Charts
- Advanced analytics visualizations

### Plotly Express (`px`)
Used for:
- Historical traffic analysis
- Data visualization graphs

### Mapbox
Used for:
- Interactive RR Nagar traffic zone mapping
- Hotspot visualization

---

## 5. Model Deployment & Utilities

### Joblib
Used to:
- Save trained ML models
- Load models instantly without retraining

Files:
- `traffic_model.pkl`
- `model_columns.pkl`

### OS Module
Used for:
- File system checks
- Model verification before deployment

---

# 🧠 Machine Learning Model

## Model Used
### Random Forest Regressor

The model predicts traffic congestion levels using:

- Rainfall intensity
- Vehicle count
- Traffic density
- Road conditions
- Historical congestion patterns

### Output:
- Congestion score
- Failure risk level
- Smart adaptation recommendations

---

# 🗂️ Project Structure

```text
UrbanMind-AI/
│
├── app.py
├── rr_nagar_traffic_dataset.csv
├── traffic_model.pkl
├── model_columns.pkl
├── requirements.txt
├── README.md
│
├── assets/
├── screenshots/
├── datasets/
└── docs/
```

---

# ⚙️ Installation & Setup

## Clone Repository

```bash
git clone https://github.com/your-username/UrbanMind-AI.git
```

---

## Navigate to Project Folder

```bash
cd UrbanMind-AI
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# 📊 Dashboard Features

The Streamlit dashboard includes:

✅ Congestion prediction panel  
✅ Flood-risk analysis  
✅ Interactive RR Nagar map  
✅ Historical traffic records  
✅ Failure risk gauge  
✅ Feature importance graph  
✅ Smart traffic alerts  
✅ Adaptation recommendations  

---

# 🌧️ Traffic + Flood Use Case

## Example Scenario

1. Heavy rainfall occurs in RR Nagar
2. Vehicle density increases rapidly
3. AI detects recurring congestion pattern
4. Flood-prone roads are identified
5. System predicts high-risk traffic zones
6. Dashboard generates adaptive recommendations:
   - Alternate route suggestions
   - Traffic signal optimization
   - Emergency response deployment

---

# 📈 Expected Outcomes

- Reduced recurring congestion
- Improved flood preparedness
- Faster emergency mobility
- Better urban traffic management
- Increased urban resilience
- Data-driven decision making

---

# 🔮 Future Scope

- IoT sensor integration
- Real-time traffic camera analytics
- Smart signal automation
- Reinforcement learning models
- Digital twin city simulation
- Multi-city deployment
- AI-powered emergency response systems

---

# 👨‍💻 Development Environment

## Recommended Tools

- Visual Studio Code (VS Code)
- Python 3.x
- Pip Package Manager

---

# 📦 Required Python Libraries

```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

---

# 👥 Team Members

- Member: Srajan Porwal
- Member: Sujal Parmar
- Member: Sparsh Jaiswal
- Member: Shreya Kumari

---

# 📜 License

This project is licensed under the MIT License.

---

# 🌟 Vision

UrbanMind AI transforms historical urban failures into adaptive intelligence, helping cities shift from reactive urban management to proactive resilience planning.