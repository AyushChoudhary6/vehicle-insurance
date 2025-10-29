# Vehicle Insurance Cross-Sell Prediction - MLOps Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-success)
![AWS](https://img.shields.io/badge/AWS-S3-orange)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![ML](https://img.shields.io/badge/ML-RandomForest-red)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Architecture](#project-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Data Pipeline Flow](#data-pipeline-flow)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)

---

## 🎯 Project Overview

This is an end-to-end **MLOps project** for predicting whether a customer would be interested in vehicle insurance cross-sell. The project implements a complete machine learning pipeline with best practices including:

- **Modular code structure** with logging and exception handling
- **MongoDB integration** for data storage
- **AWS S3** for model registry and artifact storage
- **FastAPI** web application for real-time predictions
- **Automated training pipeline** with data validation, transformation, and model evaluation
- **Docker containerization** for easy deployment
- **CI/CD ready** architecture

The system predicts customer response (Yes/No) based on various features like age, gender, vehicle details, driving license status, and previous insurance information.

---

## 🎯 Problem Statement

An insurance company wants to predict whether existing health insurance policyholders would also be interested in purchasing vehicle insurance. This prediction helps the company:

- **Optimize marketing campaigns** by targeting interested customers
- **Reduce operational costs** by avoiding unnecessary outreach
- **Improve customer experience** with personalized communication
- **Increase revenue** through effective cross-selling strategies

**Input Features:**
- Customer Demographics (Gender, Age)
- Driving License Status
- Region Code
- Previous Insurance Status
- Vehicle Details (Age, Damage History)
- Policy Information (Annual Premium, Sales Channel, Vintage)

**Target Variable:** Response (1 = Interested, 0 = Not Interested)

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                               │
│  MongoDB → Feature Store → Train/Test Split → Artifacts             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      DATA VALIDATION                                 │
│  Schema Validation → Column Check → Data Type Check → Report        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                   DATA TRANSFORMATION                                │
│  Encoding → Scaling → SMOTEENN (Imbalance) → Preprocessor Object    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      MODEL TRAINING                                  │
│  RandomForestClassifier → Hyperparameters → Metrics → Model.pkl     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                     MODEL EVALUATION                                 │
│  Compare with Production Model → Threshold Check → Accept/Reject    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      MODEL PUSHER                                    │
│  Save to AWS S3 → Model Registry → Version Control                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                   PREDICTION PIPELINE                                │
│  FastAPI → Load Model from S3 → Preprocess → Predict → Response     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Technology Stack

### **Core Technologies**
- **Python 3.10**: Primary programming language
- **FastAPI**: High-performance web framework for APIs
- **scikit-learn**: Machine learning library
- **Pandas & NumPy**: Data manipulation and analysis
- **imbalanced-learn**: Handling imbalanced datasets (SMOTEENN)

### **Database & Storage**
- **MongoDB**: NoSQL database for storing training data
- **AWS S3**: Cloud storage for model registry and artifacts

### **MLOps Tools**
- **Docker**: Containerization
- **Uvicorn**: ASGI server for FastAPI
- **PyYAML**: Configuration management
- **from_root**: Project root directory management

### **Frontend**
- **Jinja2**: Template engine
- **HTML/CSS**: User interface

### **Development Tools**
- **Git**: Version control
- **Logging**: Custom logging framework
- **Exception Handling**: Custom exception framework

---

## 📁 Project Structure

```
vehicle-insurance/
│
├── app.py                          # FastAPI application entry point
├── demo.py                         # Training pipeline execution script
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation configuration
├── Dockerfile                      # Docker configuration
├── pyproject.toml                  # Project metadata
│
├── config/
│   ├── model.yaml                  # Model configuration
│   └── schema.yaml                 # Data schema definition
│
├── notebook/
│   ├── data.csv                    # Sample dataset
│   ├── exp-notebook.ipynb          # Exploratory data analysis
│   └── mongoDB_demo.ipynb          # MongoDB integration demo
│
├── src/
│   ├── __init__.py
│   │
│   ├── components/                 # Pipeline components
│   │   ├── data_ingestion.py      # Data loading from MongoDB
│   │   ├── data_validation.py     # Schema and data validation
│   │   ├── data_transformation.py # Feature engineering & preprocessing
│   │   ├── model_trainer.py       # Model training
│   │   ├── model_evaluation.py    # Model performance evaluation
│   │   └── model_pusher.py        # Model deployment to S3
│   │
│   ├── pipline/                    # Pipeline orchestration
│   │   ├── training_pipeline.py   # End-to-end training workflow
│   │   └── prediction_pipeline.py # Real-time prediction workflow
│   │
│   ├── entity/                     # Data classes
│   │   ├── config_entity.py       # Configuration dataclasses
│   │   ├── artifact_entity.py     # Artifact dataclasses
│   │   ├── estimator.py           # Custom model wrapper
│   │   └── s3_estimator.py        # S3 model loader
│   │
│   ├── data_access/                # Data access layer
│   │   └── proj1_data.py          # MongoDB data operations
│   │
│   ├── configuration/              # External connections
│   │   ├── mongo_db_connection.py # MongoDB client
│   │   └── aws_connection.py      # AWS S3 client
│   │
│   ├── cloud_storage/              # Cloud operations
│   │   └── aws_storage.py         # S3 upload/download operations
│   │
│   ├── utils/                      # Utility functions
│   │   └── main_utils.py          # Common helper functions
│   │
│   ├── constants/                  # Project constants
│   │   └── __init__.py            # All constant definitions
│   │
│   ├── logger/                     # Logging module
│   │   └── __init__.py            # Custom logger configuration
│   │
│   └── exception/                  # Exception handling
│       └── __init__.py            # Custom exception class
│
├── static/                         # Static files
│   └── css/
│       └── style.css              # Web UI styling
│
└── templates/                      # HTML templates
    └── vehicledata.html           # Prediction form UI
```

---

## 🔄 Data Pipeline Flow

### **1. Data Ingestion**
```python
MongoDB → Export Collection → Feature Store (CSV) → Train/Test Split (75:25)
```
- Connects to MongoDB using credentials from environment variables
- Exports collection as DataFrame
- Saves to feature store
- Splits data into train and test sets
- Creates `DataIngestionArtifact` with file paths

### **2. Data Validation**
```python
Load Schema → Validate Columns → Check Data Types → Generate Report
```
- Reads schema configuration from `schema.yaml`
- Validates all required columns are present
- Checks data types match expected types
- Generates validation report
- Creates `DataValidationArtifact` with status

### **3. Data Transformation**
```python
Gender Mapping → Drop ID → Create Dummies → Rename → Scale → SMOTEENN
```
**Transformations Applied:**
- **Gender Encoding**: Female=0, Male=1
- **Drop Columns**: Remove `_id` column
- **One-Hot Encoding**: Create dummy variables for `Vehicle_Age` and `Vehicle_Damage`
- **Column Renaming**: Standardize column names
- **Scaling**: 
  - StandardScaler for: Age, Vintage
  - MinMaxScaler for: Annual_Premium
- **Imbalance Handling**: SMOTEENN technique for class balance
- Saves preprocessor object for future use

### **4. Model Training**
```python
Load Transformed Data → Train RandomForest → Evaluate → Save Model
```
**Model Configuration:**
- **Algorithm**: RandomForestClassifier
- **Hyperparameters**:
  - n_estimators: 200
  - min_samples_split: 7
  - min_samples_leaf: 6
  - max_depth: 10
  - criterion: entropy
  - random_state: 101

**Metrics Calculated:**
- Accuracy Score
- F1 Score
- Precision Score
- Recall Score

**Output**: Combined model object (preprocessor + trained model)

### **5. Model Evaluation**
```python
Load Production Model from S3 → Compare Metrics → Accept/Reject Decision
```
- Fetches current production model from S3 (if exists)
- Compares new model performance with production model
- Checks if improvement exceeds threshold (2%)
- Returns acceptance status

### **6. Model Pusher**
```python
Upload Model to S3 → Update Model Registry → Version Control
```
- Uploads accepted model to AWS S3
- Maintains model registry
- Enables model versioning and rollback

### **7. Prediction Pipeline**
```python
User Input → Load Model from S3 → Preprocess → Predict → Display Result
```
- Receives customer data via FastAPI endpoint
- Loads latest model from S3
- Applies same transformations as training
- Makes prediction (0 or 1)
- Returns "Response-Yes" or "Response-No"

---

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.10 or higher
- MongoDB instance (local or cloud)
- AWS account with S3 access
- Git

### **1. Clone the Repository**
```bash
git clone https://github.com/AyushChoudhary6/vehicle-insurance.git
cd vehicle-insurance
```

### **2. Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Environment Variables**
Create a `.env` file in the root directory:

```env
# MongoDB Configuration
MONGODB_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority

# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
```

### **5. Update Constants**
Edit `src/constants/__init__.py` if needed:
```python
DATABASE_NAME = "Proj1"
COLLECTION_NAME = "Proj1-Data"
MODEL_BUCKET_NAME = "my-model-mlopsproj"
REGION_NAME = "us-east-1"
```

---

## 📊 Usage

### **Training the Model**

**Option 1: Using demo.py**
```bash
python demo.py
```

**Option 2: Using FastAPI endpoint**
```bash
# Start the server
python app.py

# In browser or curl
curl http://localhost:5000/train
```

**Training Process:**
1. Data ingestion from MongoDB
2. Data validation against schema
3. Data transformation and preprocessing
4. Model training with RandomForest
5. Model evaluation against production model
6. Model pusher to S3 (if accepted)

### **Making Predictions**

**Option 1: Web Interface**
```bash
# Start the server
python app.py

# Open browser
http://localhost:5000
```

**Option 2: API Request**
```python
import requests

data = {
    "Gender": 1,
    "Age": 44,
    "Driving_License": 1,
    "Region_Code": 28.0,
    "Previously_Insured": 0,
    "Annual_Premium": 40454.0,
    "Policy_Sales_Channel": 26.0,
    "Vintage": 217,
    "Vehicle_Age_lt_1_Year": 0,
    "Vehicle_Age_gt_2_Years": 1,
    "Vehicle_Damage_Yes": 1
}

response = requests.post("http://localhost:5000/", data=data)
print(response.text)
```

---

## 🌐 API Endpoints

### **1. Home Page (GET)**
```
GET /
```
**Description**: Renders the prediction form interface

**Response**: HTML page with input form

---

### **2. Train Model (GET)**
```
GET /train
```
**Description**: Triggers the complete training pipeline

**Response**: 
```json
"Training successful!!!"
```

**Process Flow:**
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation
6. Model Pusher (if accepted)

---

### **3. Make Prediction (POST)**
```
POST /
Content-Type: application/x-www-form-urlencoded
```

**Request Parameters:**
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| Gender | int | 0=Female, 1=Male | 1 |
| Age | int | Customer age | 44 |
| Driving_License | int | 0=No, 1=Yes | 1 |
| Region_Code | float | Customer region code | 28.0 |
| Previously_Insured | int | 0=No, 1=Yes | 0 |
| Annual_Premium | float | Policy premium amount | 40454.0 |
| Policy_Sales_Channel | float | Sales channel code | 26.0 |
| Vintage | int | Days associated with company | 217 |
| Vehicle_Age_lt_1_Year | int | Vehicle < 1 year old | 0 |
| Vehicle_Age_gt_2_Years | int | Vehicle > 2 years old | 1 |
| Vehicle_Damage_Yes | int | Vehicle damaged before | 1 |

**Response**: HTML page with prediction result
- "Response-Yes" (interested in insurance)
- "Response-No" (not interested)

---

## 🤖 Model Details

### **Algorithm**
**RandomForestClassifier** - Ensemble learning method using multiple decision trees

### **Why RandomForest?**
- Handles both numerical and categorical features
- Robust to outliers and overfitting
- Provides feature importance
- Good performance on imbalanced datasets
- No need for feature scaling (but applied for better performance)

### **Hyperparameters**
```python
n_estimators = 200              # Number of trees
min_samples_split = 7          # Min samples to split node
min_samples_leaf = 6           # Min samples at leaf node
max_depth = 10                 # Maximum tree depth
criterion = 'entropy'          # Split quality measure
random_state = 101             # Reproducibility
```

### **Performance Metrics**
The model is evaluated using:
- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Correct positive predictions
- **Recall**: Coverage of actual positives

### **Minimum Expected Score**
- Baseline accuracy threshold: **60%**
- Model acceptance threshold: **2% improvement** over production model

### **Imbalanced Data Handling**
**SMOTEENN** (Synthetic Minority Over-sampling + Edited Nearest Neighbors)
- Combines over-sampling (SMOTE) and under-sampling (ENN)
- Balances the dataset while removing noisy samples
- Improves model performance on minority class

---

## ⚙️ Configuration

### **1. Schema Configuration** (`config/schema.yaml`)
```yaml
columns:
  - id, Gender, Age, Driving_License, etc.

numerical_columns:
  - Age, Driving_License, Region_Code, etc.

categorical_columns:
  - Gender, Vehicle_Age, Vehicle_Damage

drop_columns: _id

num_features:  # StandardScaler
  - Age
  - Vintage

mm_columns:    # MinMaxScaler
  - Annual_Premium
```

### **2. Model Configuration** (`config/model.yaml`)
Currently empty - reserved for future model configurations

### **3. Constants** (`src/constants/__init__.py`)
```python
# Database
DATABASE_NAME = "Proj1"
COLLECTION_NAME = "Proj1-Data"

# AWS
MODEL_BUCKET_NAME = "my-model-mlopsproj"
REGION_NAME = "us-east-1"

# Training
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.25
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02

# Server
APP_HOST = "0.0.0.0"
APP_PORT = 5000
```

---

## 🐳 Docker Deployment

### **Build Docker Image**
```bash
docker build -t vehicle-insurance:latest .
```

### **Run Container**
```bash
docker run -p 5000:5000 `
  -e MONGODB_URL="your_mongodb_url" `
  -e AWS_ACCESS_KEY_ID="your_access_key" `
  -e AWS_SECRET_ACCESS_KEY="your_secret_key" `
  vehicle-insurance:latest
```

### **Docker Compose** (Optional)
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MONGODB_URL=${MONGODB_URL}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
```

Run with:
```bash
docker-compose up
```

---

## 📝 Logging & Exception Handling

### **Custom Logging**
- All operations are logged with timestamps
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Logs stored with structured format
- Easy debugging and monitoring

### **Custom Exception Handling**
```python
from src.exception import MyException
import sys

try:
    # Your code
except Exception as e:
    raise MyException(e, sys) from e
```
- Captures detailed error information
- Provides file name, line number, and error message
- Facilitates quick debugging

---

## 🔒 Security Best Practices

1. **Environment Variables**: Store credentials in `.env` file (not in code)
2. **gitignore**: Exclude sensitive files from version control
3. **AWS IAM**: Use minimal required permissions for S3 access
4. **MongoDB**: Use strong passwords and IP whitelisting
5. **API Security**: Consider adding authentication for production

---

## 📈 Future Enhancements

- [ ] Add user authentication and authorization
- [ ] Implement A/B testing for model versions
- [ ] Add model monitoring and drift detection
- [ ] Create data quality checks dashboard
- [ ] Implement automated retraining pipeline
- [ ] Add more ML algorithms for comparison
- [ ] Deploy to AWS ECS/EKS or Azure
- [ ] Add unit tests and integration tests
- [ ] Implement CI/CD pipeline with GitHub Actions
- [ ] Add Grafana/Prometheus for monitoring
- [ ] Create REST API documentation with Swagger
- [ ] Add batch prediction capability

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- FastAPI documentation for excellent examples
- scikit-learn community for ML algorithms
- MongoDB and AWS for cloud infrastructure
- imbalanced-learn for SMOTEENN implementation



## ⭐ Show Your Support

Give a ⭐️ if this project helped you!
