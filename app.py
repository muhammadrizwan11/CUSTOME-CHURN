import streamlit as st

# Display initial loading message
st.markdown("### 🔄 Loading required packages...")

# List of required packages
required_packages = [
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'plotly',
    'scikit-learn',
    'joblib',
    'imbalanced-learn'
]

# Function to check and install missing packages
def install_packages():
    import subprocess
    import sys
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            st.warning(f"Installing {package}...")
            try:
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    "--user",
                    package
                ])
                st.success(f"✅ {package} installed successfully!")
            except Exception as e:
                st.error(f"❌ Failed to install {package}: {str(e)}")
                st.stop()

# Install any missing packages
install_packages()

# Now import all required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# Clear the loading message
st.empty()
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
import io
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🚀 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Analysis Section",
    ["📈 Data Overview", "🔍 Exploratory Analysis", "🤖 Machine Learning", "🎯 Make Predictions"]
)

@st.cache_data
def load_sample_data():
    """Generate sample data if no file is uploaded"""
    np.random.seed(42)
    n_samples = 1000
    
    customers = []
    for i in range(n_samples):
        age = np.random.normal(35, 11)
        age = max(18, min(75, age))  # Constrain age
        
        gender = np.random.choice(['Male', 'Female'])
        contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                       p=[0.6, 0.25, 0.15])
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                          p=[0.4, 0.4, 0.2])
        
        # Create logical relationships
        if contract_type == 'Month-to-month':
            monthly_charges = np.random.uniform(20, 120)
            churn_prob = 0.4
        elif contract_type == 'One year':
            monthly_charges = np.random.uniform(30, 100)
            churn_prob = 0.2
        else:  # Two year
            monthly_charges = np.random.uniform(25, 90)
            churn_prob = 0.1
            
        # Adjust churn probability based on charges
        if monthly_charges > 80:
            churn_prob *= 1.5
        
        total_charges = monthly_charges * np.random.uniform(1, 60)
        
        # Add some missing values
        if np.random.random() < 0.02:
            total_charges = np.nan
        if np.random.random() < 0.008:
            age = np.nan
        if np.random.random() < 0.007:
            gender = np.nan
            
        churn = 'Yes' if np.random.random() < churn_prob else 'No'
        
        customers.append({
            'customer_id': f'CUST_{i+1:04d}',
            'age': age,
            'gender': gender,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract_type': contract_type,
            'internet_service': internet_service,
            'churn': churn
        })
    
    return pd.DataFrame(customers)

@st.cache_data
def preprocess_data(df):
    """Clean and preprocess the data"""
    df_clean = df.copy()
    
    # Handle missing values
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
    
    cat_cols = df_clean.select_dtypes(include='object').columns
    cat_cols = [col for col in cat_cols if col not in ['customer_id', 'churn']]
    for col in cat_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # Convert data types
    df_clean['total_charges'] = pd.to_numeric(df_clean['total_charges'], errors='coerce')
    df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
    df_clean['monthly_charges'] = pd.to_numeric(df_clean['monthly_charges'], errors='coerce')
    
    return df_clean

# Data loading section
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")
else:
    df = load_sample_data()
    st.sidebar.info("📊 Using sample dataset")

# Clean the data
df_clean = preprocess_data(df)

# Main content based on page selection
if page == "📈 Data Overview":
    st.markdown('<h2 class="section-header">Data Overview & Quality Assessment</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Dataset Size</h3>
            <h2>{df_clean.shape[0]:,}</h2>
            <p>Total Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Features</h3>
            <h2>{df_clean.shape[1]-1}</h2>
            <p>Data Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        churn_rate = (df_clean['churn'] == 'Yes').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Churn Rate</h3>
            <h2>{churn_rate:.1f}%</h2>
            <p>Customer Churn</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data quality metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Data Quality Check")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig = px.bar(
                x=missing_data.index, 
                y=missing_data.values,
                title="Missing Values by Column"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    with col2:
        st.subheader("📊 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df_clean.columns,
            'Data Type': df_clean.dtypes.astype(str),
            'Non-Null Count': df_clean.count()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Basic statistics
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df_clean.describe(), use_container_width=True)
    
    # Show raw data
    with st.expander("👀 View Raw Data"):
        st.dataframe(df_clean.head(100), use_container_width=True)

elif page == "🔍 Exploratory Analysis":
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # Customer Age Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Age Distribution")
        fig = px.histogram(
            df_clean, 
            x='age', 
            nbins=30,
            title="Customer Age Distribution"
        )
        fig.add_vline(x=df_clean['age'].mean(), line_dash="dash", line_color="red")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Monthly Charges vs Churn")
        fig = px.box(
            df_clean, 
            x='churn', 
            y='monthly_charges',
            title="Monthly Charges by Churn Status"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Churn analysis
    st.subheader("📊 Churn Analysis by Categories")
    
    analysis_col = st.selectbox(
        "Select category for analysis:",
        ['contract_type', 'internet_service', 'gender']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Count plot
        churn_counts = df_clean.groupby([analysis_col, 'churn']).size().reset_index(name='count')
        fig = px.bar(
            churn_counts,
            x=analysis_col,
            y='count',
            color='churn',
            title=f"Churn Distribution by {analysis_col.replace('_', ' ').title()}"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn rate
        churn_rate = df_clean.groupby(analysis_col)['churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        churn_rate.columns = [analysis_col, 'churn_rate']
        
        fig = px.bar(
            churn_rate,
            x=analysis_col,
            y='churn_rate',
            title=f"Churn Rate by {analysis_col.replace('_', ' ').title()}"
        )
        fig.update_layout(height=400, yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Analysis")
    
    # Create a copy for correlation analysis
    df_corr = df_clean.copy()
    
    # Encode categorical variables for correlation
    le_dict = {}
    for col in ['gender', 'contract_type', 'internet_service', 'churn']:
        if col in df_corr.columns:
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))
            le_dict[col] = le
    
    # Calculate correlation matrix
    corr_matrix = df_corr.select_dtypes(include=[np.number]).corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Machine Learning":
    st.markdown('<h2 class="section-header">Machine Learning Models</h2>', unsafe_allow_html=True)
    
    # Prepare data for ML
    X = df_clean.drop(['customer_id', 'churn'], axis=1)
    y = df_clean['churn'].map({'No': 0, 'Yes': 1})
    
    # Encode categorical variables
    categorical_cols = ['gender', 'contract_type', 'internet_service']
    le_dict = {}
    
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model selection
    st.subheader("🎯 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        models_to_train = st.multiselect(
            "Select models to train:",
            ["Logistic Regression", "Decision Tree", "Random Forest", "Ensemble (Voting)"],
            default=["Logistic Regression", "Decision Tree"]
        )
    
    with col2:
        use_advanced_features = st.checkbox("Use Advanced Features (SMOTE, Scaling)", value=False)
    
    if st.button("🚀 Train Models", type="primary"):
        results = {}
        
        with st.spinner("Training models..."):
            # Apply advanced features if selected
            if use_advanced_features:
                # Apply SMOTE
                smote = SMOTE(random_state=42)
                X_train_processed, y_train_processed = smote.fit_resample(X_train, y_train)
                
                # Apply scaling
                scaler = RobustScaler()
                X_train_processed = scaler.fit_transform(X_train_processed)
                X_test_processed = scaler.transform(X_test)
            else:
                X_train_processed, y_train_processed = X_train, y_train
                X_test_processed = X_test
            
            # Train selected models
            for model_name in models_to_train:
                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000, class_weight='balanced' if use_advanced_features else None)
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42, class_weight='balanced' if use_advanced_features else None)
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=100, 
                        random_state=42, 
                        class_weight='balanced' if use_advanced_features else None
                    )
                elif model_name == "Ensemble (Voting)":
                    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
                    dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
                    rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
                    model = VotingClassifier(
                        estimators=[('lr', lr), ('dt', dt), ('rf', rf)],
                        voting='soft'
                    )
                
                # Train model
                model.fit(X_train_processed, y_train_processed)
                y_pred = model.predict(X_test_processed)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'confusion_matrix': cm
                }
        
        # Display results
        st.success("✅ Models trained successfully!")
        
        # Performance comparison
        st.subheader("📊 Model Performance Comparison")
        
        performance_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results.keys()]
        }).sort_values('Accuracy', ascending=False)
        
        fig = px.bar(
            performance_df,
            x='Model',
            y='Accuracy',
            title="Model Accuracy Comparison",
            text='Accuracy'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results for each model
        for model_name, result in results.items():
            with st.expander(f"📋 Detailed Results: {model_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Accuracy",
                        value=f"{result['accuracy']:.3f}",
                        delta=f"{(result['accuracy'] - 0.5):.3f} vs random"
                    )
                    
                    # Classification report
                    st.text("Classification Report:")
                    report = classification_report(y_test, result['predictions'])
                    st.text(report)
                
                with col2:
                    # Confusion matrix
                    cm = result['confusion_matrix']
                    fig = px.imshow(
                        cm,
                        title=f"Confusion Matrix - {model_name}",
                        labels=dict(x="Predicted", y="Actual"),
                        x=['No Churn', 'Churn'],
                        y=['No Churn', 'Churn'],
                        color_continuous_scale="Blues"
                    )
                    
                    # Add text annotations
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            fig.add_annotation(
                                x=j, y=i,
                                text=str(cm[i, j]),
                                showarrow=False,
                                font=dict(color="white" if cm[i, j] > cm.max()/2 else "black", size=16)
                            )
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (for tree-based models)
                if hasattr(result['model'], 'feature_importances_'):
                    st.subheader("🎯 Feature Importance")
                    
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': result['model'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f"Feature Importance - {model_name}"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Store best model in session state and save to disk
        best_model_name = performance_df.iloc[0]['Model']
        st.session_state.best_model = results[best_model_name]['model']
        st.session_state.le_dict = le_dict
        st.session_state.scaler = scaler if use_advanced_features else None
        st.session_state.feature_names = X.columns.tolist()
        
        # Save models and preprocessing objects
        try:
            if not os.path.exists('models'):
                os.makedirs('models')
            
            # Save the best model
            joblib.dump(results[best_model_name]['model'], 'models/best_model.joblib')
            joblib.dump(le_dict, 'models/label_encoders.joblib')
            if use_advanced_features and scaler is not None:
                joblib.dump(scaler, 'models/scaler.joblib')
            
            st.success("✅ Models saved successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not save models to disk: {str(e)}")
        
        st.success(f"🏆 Best performing model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.3f})")

elif page == "🎯 Make Predictions":
    st.markdown('<h2 class="section-header">Make Churn Predictions</h2>', unsafe_allow_html=True)
    
    # Try to load saved model if not in session state
    if 'best_model' not in st.session_state:
        try:
            if os.path.exists('models/best_model.joblib'):
                st.session_state.best_model = joblib.load('models/best_model.joblib')
                st.session_state.le_dict = joblib.load('models/label_encoders.joblib')
                if os.path.exists('models/scaler.joblib'):
                    st.session_state.scaler = joblib.load('models/scaler.joblib')
                else:
                    st.session_state.scaler = None
                st.success("✅ Loaded saved model successfully!")
            else:
                st.warning("⚠️ Please train models first in the Machine Learning section!")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error loading saved model: {str(e)}")
            st.warning("⚠️ Please train models first in the Machine Learning section!")
            st.stop()
    
    st.subheader("🔮 Predict Customer Churn")
    
    # Input form for prediction
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        with col2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
        
        with col3:
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        predict_button = st.form_submit_button("🎯 Predict Churn", type="primary")
    
    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'monthly_charges': [monthly_charges],
            'total_charges': [total_charges],
            'contract_type': [contract_type],
            'internet_service': [internet_service]
        })
        
        # Apply label encoding
        for col in ['gender', 'contract_type', 'internet_service']:
            if col in st.session_state.le_dict:
                input_data[col] = st.session_state.le_dict[col].transform(input_data[col])
        
        # Apply scaling if used during training
        if st.session_state.scaler is not None:
            input_data_scaled = st.session_state.scaler.transform(input_data)
        else:
            input_data_scaled = input_data
        
        # Make prediction
        prediction = st.session_state.best_model.predict(input_data_scaled)[0]
        prediction_proba = st.session_state.best_model.predict_proba(input_data_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("🔮 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ HIGH CHURN RISK")
                st.markdown("This customer is likely to churn.")
            else:
                st.success("✅ LOW CHURN RISK")
                st.markdown("This customer is likely to stay.")
        
        with col2:
            st.metric(
                label="Churn Probability",
                value=f"{prediction_proba[1]:.1%}"
            )
        
        with col3:
            st.metric(
                label="Retention Probability",
                value=f"{prediction_proba[0]:.1%}"
            )
        
        # Probability visualization
        fig = go.Figure(data=[
            go.Bar(
                x=['Retention', 'Churn'],
                y=[prediction_proba[0], prediction_proba[1]],
                marker_color=['green', 'red']
            )
        ])
        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if prediction == 1:  # High churn risk
            st.markdown("""
            <div class="info-box">
            <h4>🚨 Retention Strategies:</h4>
            <ul>
                <li><b>Immediate Action:</b> Contact customer within 24 hours</li>
                <li><b>Offer Incentives:</b> Discount on next 3 months or upgrade benefits</li>
                <li><b>Contract Upgrade:</b> Encourage longer-term contract with benefits</li>
                <li><b>Service Review:</b> Evaluate service quality and resolve issues</li>
                <li><b>Personalized Support:</b> Assign dedicated customer success manager</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:  # Low churn risk
            st.markdown("""
            <div class="info-box">
            <h4>🌟 Customer Success Strategies:</h4>
            <ul>
                <li><b>Upsell Opportunities:</b> Introduce premium services or add-ons</li>
                <li><b>Loyalty Program:</b> Enroll in rewards program for long-term benefits</li>
                <li><b>Regular Check-ins:</b> Quarterly satisfaction surveys</li>
                <li><b>Referral Program:</b> Incentivize customer referrals</li>
                <li><b>Service Enhancement:</b> Offer latest features or upgrades</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Batch prediction section
    st.markdown("---")
    st.subheader("📊 Batch Predictions")
    
    uploaded_batch = st.file_uploader(
        "Upload CSV file for batch predictions",
        type=['csv'],
        help="File should contain the same columns as the training data"
    )
    
    if uploaded_batch is not None:
        batch_df = pd.read_csv(uploaded_batch)
        
        if st.button("🚀 Process Batch Predictions"):
            try:
                # Prepare batch data (similar to single prediction)
                batch_processed = batch_df.copy()
                
                # Apply label encoding
                for col in ['gender', 'contract_type', 'internet_service']:
                    if col in batch_processed.columns and col in st.session_state.le_dict:
                        batch_processed[col] = st.session_state.le_dict[col].transform(batch_processed[col])
                
                # Select relevant features
                feature_cols = [col for col in st.session_state.feature_names if col in batch_processed.columns]
                batch_features = batch_processed[feature_cols]
                
                # Apply scaling if used
                if st.session_state.scaler is not None:
                    batch_features_scaled = st.session_state.scaler.transform(batch_features)
                else:
                    batch_features_scaled = batch_features
                
                # Make predictions
                predictions = st.session_state.best_model.predict(batch_features_scaled)
                probabilities = st.session_state.best_model.predict_proba(batch_features_scaled)
                
                # Add predictions to dataframe
                batch_df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                batch_df['Churn_Probability'] = probabilities[:, 1]
                batch_df['Risk_Level'] = pd.cut(
                    probabilities[:, 1], 
                    bins=[0, 0.3, 0.7, 1.0], 
                    labels=['Low', 'Medium', 'High']
                )
                
                st.success(f"✅ Processed {len(batch_df)} predictions!")
                
                # Display results
                st.dataframe(batch_df, use_container_width=True)
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_risk = (batch_df['Risk_Level'] == 'High').sum()
                    st.metric("High Risk Customers", high_risk)
                
                with col2:
                    medium_risk = (batch_df['Risk_Level'] == 'Medium').sum()
                    st.metric("Medium Risk Customers", medium_risk)
                
                with col3:
                    low_risk = (batch_df['Risk_Level'] == 'Low').sum()
                    st.metric("Low Risk Customers", low_risk)
                
            except Exception as e:
                st.error(f"❌ Error processing batch predictions: {str(e)}")
                st.info("Please ensure your file has the correct column names and data format.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🚀 Customer Churn Prediction Dashboard | Built with Streamlit</p>
    <p>📊 Upload your data or use the sample dataset to get started</p>
</div>
""", unsafe_allow_html=True)