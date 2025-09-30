# lightweight_efficient_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.express as px
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# Lightweight LLM - Using Google's Gemma model which is efficient
try:
    from transformers import pipeline
    from huggingface_hub import login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Efficient Data Analyzer", 
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EfficientAnalyzer:
    def __init__(self):
        self.pipeline = None
        self.model_loaded = False
        
    def setup_lightweight_model(self, hf_token):
        """Setup a lightweight but capable model"""
        try:
            if not HF_AVAILABLE:
                return False
                
            login(token=hf_token)
            
            # Use Google's Gemma model - efficient and capable
            self.pipeline = pipeline(
                "text-generation",
                model="google/gemma-2b-it",  # Very lightweight but capable
                torch_dtype="auto",
                device_map="auto", 
                trust_remote_code=True,
                max_length=1024
            )
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            # Fallback to even smaller model
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-small",  # Very small fallback
                    torch_dtype="auto",
                    max_length=512
                )
                self.model_loaded = True
                return True
            except:
                return False
    
    def analyze_dataset(self, df, filename):
        """Perform efficient but detailed analysis"""
        # First, extract key features from the dataset
        dataset_features = self._extract_dataset_features(df, filename)
        
        # Use LLM for intelligent analysis
        if self.model_loaded:
            analysis = self._get_llm_analysis(dataset_features, filename)
        else:
            analysis = self._get_smart_analysis(dataset_features, filename)
            
        return dataset_features, analysis
    
    def _extract_dataset_features(self, df, filename):
        """Extract comprehensive dataset features"""
        features = {}
        
        # Basic info
        features['basic'] = {
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'total_cells': len(df) * len(df.columns)
        }
        
        # Column analysis
        features['columns'] = {}
        for col in df.columns:
            features['columns'][col] = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_pct': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_pct': (df[col].nunique() / len(df)) * 100,
                'sample_values': df[col].dropna().head(3).tolist() if df[col].dtype == 'object' else None
            }
        
        # Data type distribution
        features['dtypes'] = {
            'numeric': len(df.select_dtypes(include=[np.number]).columns),
            'categorical': len(df.select_dtypes(include=['object']).columns),
            'boolean': len(df.select_dtypes(include=['bool']).columns),
            'datetime': len([col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()])
        }
        
        # Data quality issues
        features['quality_issues'] = self._find_quality_issues(df)
        
        # Patterns and characteristics
        features['patterns'] = self._detect_patterns(df)
        
        # Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            features['numeric_stats'] = {}
            for col in numeric_cols[:5]:  # Limit to first 5 for efficiency
                stats = df[col].describe()
                features['numeric_stats'][col] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': df[col].median()
                }
        
        return features
    
    def _find_quality_issues(self, df):
        """Find data quality issues"""
        issues = []
        
        # Null value issues
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > 50:
                issues.append(f"🚨 CRITICAL: {col} has {null_pct:.1f}% null values")
            elif null_pct > 20:
                issues.append(f"⚠️ HIGH: {col} has {null_pct:.1f}% null values")
            elif null_pct > 5:
                issues.append(f"📊 MEDIUM: {col} has {null_pct:.1f}% null values")
        
        # High cardinality issues
        for col in df.columns:
            unique_pct = (df[col].nunique() / len(df)) * 100
            if unique_pct > 95:
                issues.append(f"🔍 HIGH_CARDINALITY: {col} has {unique_pct:.1f}% unique values")
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                issues.append(f"📊 CONSTANT: {col} has only {df[col].nunique()} unique value")
        
        # Data type issues
        for col in df.select_dtypes(include=['object']).columns:
            # Check if numeric data stored as string
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if numeric_count > len(df) * 0.5:  # More than 50% numeric
                issues.append(f"🔧 TYPE_ISSUE: {col} appears to be numeric data stored as text")
        
        return issues
    
    def _detect_patterns(self, df):
        """Detect dataset patterns and characteristics"""
        patterns = {}
        
        # Domain detection
        column_text = ' '.join(df.columns).lower()
        domains = {
            'FINANCIAL': ['amount', 'price', 'cost', 'revenue', 'transaction', 'balance', 'payment'],
            'RETAIL': ['product', 'customer', 'order', 'sales', 'inventory', 'sku', 'purchase'],
            'HEALTHCARE': ['patient', 'diagnosis', 'treatment', 'medical', 'hospital', 'doctor'],
            'HR': ['employee', 'salary', 'department', 'performance', 'hire', 'manager'],
            'MARKETING': ['campaign', 'conversion', 'click', 'impression', 'lead', 'customer'],
            'TELECOM': ['call', 'minute', 'data', 'usage', 'subscription', 'plan']
        }
        
        detected_domains = []
        for domain, keywords in domains.items():
            if any(keyword in column_text for keyword in keywords):
                detected_domains.append(domain)
        
        patterns['domains'] = detected_domains if detected_domains else ['GENERAL']
        
        # Use case detection
        use_cases = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len([col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]) >= 1:
            use_cases.append("TIME_SERIES")
        
        if any(col.lower() in ['target', 'label', 'class'] for col in df.columns):
            use_cases.append("CLASSIFICATION")
        elif len(categorical_cols) >= 3:
            use_cases.append("SEGMENTATION")
        
        if len(numeric_cols) >= 5:
            use_cases.append("REGRESSION")
        
        if len(df.columns) > 20:
            use_cases.append("FEATURE_RICH_ANALYSIS")
        
        patterns['use_cases'] = use_cases if use_cases else ['EXPLORATORY_ANALYSIS']
        
        # Column purpose detection
        column_purposes = {}
        for col in df.columns:
            col_lower = col.lower()
            purposes = []
            
            if any(word in col_lower for word in ['id', 'code', 'key']):
                purposes.append('IDENTIFIER')
            if any(word in col_lower for word in ['date', 'time', 'year', 'month']):
                purposes.append('TEMPORAL')
            if any(word in col_lower for word in ['amount', 'price', 'cost', 'value']):
                purposes.append('MONETARY')
            if any(word in col_lower for word in ['count', 'total', 'sum', 'number']):
                purposes.append('QUANTITATIVE')
            if any(word in col_lower for word in ['name', 'description', 'title']):
                purposes.append('DESCRIPTIVE')
            if any(word in col_lower for word in ['type', 'category', 'status', 'flag']):
                purposes.append('CATEGORICAL')
            
            column_purposes[col] = purposes if purposes else ['UNCATEGORIZED']
        
        patterns['column_purposes'] = column_purposes
        
        return patterns
    
    def _get_llm_analysis(self, features, filename):
        """Get analysis from lightweight LLM"""
        try:
            prompt = self._create_efficient_prompt(features, filename)
            
            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False
            )
            
            return response[0]['generated_text']
            
        except Exception as e:
            return self._get_smart_analysis(features, filename)
    
    def _create_efficient_prompt(self, features, filename):
        """Create efficient prompt for lightweight model"""
        basic = features['basic']
        patterns = features['patterns']
        quality_issues = features['quality_issues']
        
        prompt = f"""
        Analyze this dataset and provide specific recommendations:

        DATASET: {filename}
        SIZE: {basic['rows']} rows, {basic['columns']} columns
        MEMORY: {basic['memory_mb']:.1f} MB
        DOMAIN: {', '.join(patterns['domains'])}
        USE CASES: {', '.join(patterns['use_cases'])}

        DATA TYPES:
        - Numeric: {features['dtypes']['numeric']} columns
        - Categorical: {features['dtypes']['categorical']} columns  
        - Boolean: {features['dtypes']['boolean']} columns
        - DateTime: {features['dtypes']['datetime']} columns

        QUALITY ISSUES ({len(quality_issues)} found):
        {chr(10).join(quality_issues[:5])}

        Provide specific recommendations for:
        1. Data quality improvements
        2. Storage optimization
        3. Suitable machine learning models
        4. Business applications
        5. Technical next steps

        Be very specific to this dataset's characteristics.
        """
        
        return prompt
    
    def _get_smart_analysis(self, features, filename):
        """Generate smart analysis without LLM"""
        basic = features['basic']
        patterns = features['patterns']
        quality_issues = features['quality_issues']
        
        analysis = f"""
## 🎯 COMPREHENSIVE ANALYSIS: {filename}

### 📊 DATASET CHARACTERISTICS
- **Size**: {basic['rows']:,} rows × {basic['columns']} columns
- **Memory**: {basic['memory_mb']:.1f} MB
- **Domain**: {', '.join(patterns['domains'])}
- **Primary Use Cases**: {', '.join(patterns['use_cases'])}

### 🚨 DATA QUALITY ASSESSMENT
**Issues Found**: {len(quality_issues)}
"""
        
        # Critical issues
        critical_issues = [issue for issue in quality_issues if '🚨 CRITICAL' in issue]
        if critical_issues:
            analysis += "\n**Critical Issues (Immediate Action Required):**\n"
            for issue in critical_issues[:3]:
                analysis += f"- {issue}\\n"
        
        # High priority issues
        high_issues = [issue for issue in quality_issues if '⚠️ HIGH' in issue]
        if high_issues:
            analysis += "\n**High Priority Issues:**\n"
            for issue in high_issues[:3]:
                analysis += f"- {issue}\\n"
        
        analysis += f"""
### 💡 SPECIFIC RECOMMENDATIONS

#### 1. DATA QUALITY IMPROVEMENT
"""
        
        # Data quality recommendations based on issues
        if any('null' in issue.lower() for issue in quality_issues):
            analysis += "- **Null Value Strategy**: Implement targeted imputation based on column importance\\n"
        
        if any('cardinality' in issue.lower() for issue in quality_issues):
            analysis += "- **High Cardinality**: Use target encoding or feature hashing for ML models\\n"
        
        if any('constant' in issue.lower() for issue in quality_issues):
            analysis += "- **Constant Columns**: Remove columns with no predictive value\\n"
        
        analysis += f"""
#### 2. STORAGE & PROCESSING OPTIMIZATION
"""
        
        # Storage recommendations
        if basic['memory_mb'] > 100:
            analysis += "- **Large Dataset**: Use chunked processing and consider distributed computing\\n"
            analysis += "- **Storage**: Convert to Parquet format for better compression\\n"
        else:
            analysis += "- **Moderate Size**: In-memory processing is optimal\\n"
            analysis += "- **Optimization**: Focus on data type efficiency\\n"
        
        analysis += f"""
#### 3. MACHINE LEARNING STRATEGY
"""
        
        # ML recommendations based on use cases
        use_cases = patterns['use_cases']
        if 'CLASSIFICATION' in use_cases:
            analysis += "- **Classification**: Use XGBoost, Random Forest, or Neural Networks\\n"
        if 'REGRESSION' in use_cases:
            analysis += "- **Regression**: Gradient Boosting or Linear Models with regularization\\n"
        if 'TIME_SERIES' in use_cases:
            analysis += "- **Time Series**: ARIMA, Prophet, or LSTM networks\\n"
        if 'SEGMENTATION' in use_cases:
            analysis += "- **Segmentation**: K-Means clustering or DBSCAN\\n"
        
        analysis += f"""
#### 4. BUSINESS APPLICATIONS
- **{patterns['domains'][0]} Focus**: Leverage domain-specific patterns for insights
- **Data Products**: Consider building predictive models or dashboards
- **Decision Support**: Use for strategic planning and optimization

#### 5. TECHNICAL NEXT STEPS
1. **Immediate**: Address critical data quality issues
2. **Short-term**: Implement storage optimizations
3. **Medium-term**: Develop ML models based on use cases
4. **Long-term**: Establish data governance and monitoring
"""
        
        return analysis

# Initialize analyzer
analyzer = EfficientAnalyzer()

def create_interactive_dashboard(features, analysis):
    """Create interactive dashboard"""
    
    st.subheader("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{features['basic']['rows']:,}")
    with col2:
        st.metric("Total Columns", features['basic']['columns'])
    with col3:
        st.metric("Memory Usage", f"{features['basic']['memory_mb']:.1f} MB")
    with col4:
        st.metric("Quality Issues", len(features['quality_issues']))
    
    # Data type distribution
    st.subheader("📈 Data Composition")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Numeric Columns", features['dtypes']['numeric'])
    with col2:
        st.metric("Categorical Columns", features['dtypes']['categorical'])
    with col3:
        st.metric("Boolean Columns", features['dtypes']['boolean'])
    with col4:
        st.metric("DateTime Columns", features['dtypes']['datetime'])
    
    # Quality issues
    with st.expander("🚨 Data Quality Issues", expanded=True):
        if features['quality_issues']:
            for issue in features['quality_issues'][:8]:
                st.write(issue)
        else:
            st.success("✅ No major quality issues detected")
    
    # Domain and use cases
    with st.expander("🎯 Domain & Use Cases", expanded=True):
        st.write(f"**Detected Domain**: {', '.join(features['patterns']['domains'])}")
        st.write(f"**Recommended Use Cases**: {', '.join(features['patterns']['use_cases'])}")
    
    # LLM Analysis
    st.subheader("🤖 AI-Powered Analysis")
    st.markdown(analysis)
    
    if analyzer.model_loaded:
        st.success("✅ Analysis powered by lightweight AI model")

def main():
    st.title("⚡ Efficient Data Analyzer")
    st.markdown("**Fast, intelligent analysis using lightweight AI models**")
    
    # Simple setup
    with st.sidebar:
        st.header("⚙️ Quick Setup")
        
        hf_token = st.text_input("HF Token (optional)", type="password",
                               help="For enhanced AI analysis. Get from huggingface.co")
        
        if st.button("🚀 Load AI Model") and hf_token:
            with st.spinner("Loading efficient model..."):
                if analyzer.setup_lightweight_model(hf_token):
                    st.success("AI model loaded!")
                else:
                    st.error("Using smart analysis without AI")
        
        st.markdown("---")
        st.markdown("""
        **⚡ Features:**
        - Fast analysis (seconds)
        - Lightweight AI models
        - Comprehensive insights
        - Actionable recommendations
        - No timeouts
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload your dataset",
        type=['csv', 'xlsx', 'parquet'],
        help="CSV, Excel, or Parquet files supported"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(tmp_path)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(tmp_path)
            else:
                df = pd.read_parquet(tmp_path)
            
            # Perform analysis
            with st.spinner("⚡ Analyzing dataset..."):
                features, analysis = analyzer.analyze_dataset(df, uploaded_file.name)
            
            # Display results
            create_interactive_dashboard(features, analysis)
            
            # Data preview
            with st.expander("🔍 Data Preview", expanded=False):
                st.dataframe(df.head(8), use_container_width=True)
                st.write(f"**Full dataset**: {len(df):,} rows × {len(df.columns)} columns")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    else:
        st.info("👆 Upload a dataset for fast, intelligent analysis!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Why This Works")
            st.markdown("""
            - **Lightweight Models**: Fast loading, no timeouts
            - **Smart Analysis**: Pattern-based insights
            - **Actionable**: Specific recommendations
            - **Reliable**: Works without external APIs
            """)
        
        with col2:
            st.subheader("🎯 What You Get")
            st.markdown("""
            - Data quality assessment
            - Storage optimization tips
            - ML model recommendations
            - Business applications
            - Technical next steps
            """)

if __name__ == "__main__":
    main()
