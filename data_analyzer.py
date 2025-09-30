# hf_powered_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Hugging Face imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

# Set page config
st.set_page_config(
    page_title="HF-Powered Data Analyzer",
    page_icon="🤗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class HFAnalyzer:
    def __init__(self):
        self.models = {
            "Zephyr-7B-Beta": "HuggingFaceH4/zephyr-7b-beta",
            "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.2", 
            "Phi-2": "microsoft/phi-2",
            "CodeLlama-7B": "codellama/CodeLlama-7b-Instruct-hf"
        }
        self.pipeline = None
        self.model_loaded = False
        self.current_model = None
    
    def setup_huggingface(self, hf_token, model_choice):
        """Setup Hugging Face with the chosen model"""
        try:
            # Login to Hugging Face
            login(token=hf_token)
            
            model_name = self.models[model_choice]
            st.info(f"🔄 Loading {model_choice}... This may take 2-3 minutes.")
            
            # Load with quantization to save memory
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True,  # Reduce memory usage
                max_length=2048
            )
            
            self.model_loaded = True
            self.current_model = model_choice
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    def analyze_with_hf(self, df, filename, analysis_type="comprehensive"):
        """Use Hugging Face model to analyze dataset"""
        
        # First create a detailed data summary
        data_summary = self._create_detailed_summary(df, filename)
        
        # Generate prompt based on analysis type
        if analysis_type == "comprehensive":
            prompt = self._create_comprehensive_prompt(data_summary, filename)
        elif analysis_type == "data_engineering":
            prompt = self._create_data_engineering_prompt(data_summary, filename)
        elif analysis_type == "business_insights":
            prompt = self._create_business_prompt(data_summary, filename)
        
        try:
            # Generate analysis using HF model
            response = self.pipeline(
                prompt,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False
            )
            
            return response[0]['generated_text']
            
        except Exception as e:
            return f"❌ HF Analysis failed: {str(e)}\n\n{self._fallback_analysis(data_summary)}"
    
    def _create_detailed_summary(self, df, filename):
        """Create comprehensive data summary for HF model"""
        summary = f"""
DATASET: {filename}
SHAPE: {df.shape[0]} rows, {df.shape[1]} columns
MEMORY_USAGE: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

COLUMN_ANALYSIS:
"""
        
        # Detailed column analysis
        for i, col in enumerate(df.columns):
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            unique_pct = (unique_count / len(df)) * 100
            
            summary += f"\nCOLUMN {i+1}: {col}"
            summary += f"\n  - Data Type: {dtype}"
            summary += f"\n  - Null Values: {null_count} ({null_pct:.1f}%)"
            summary += f"\n  - Unique Values: {unique_count} ({unique_pct:.1f}%)"
            
            # Add sample values for first few rows
            if i < 10:  # Limit to first 10 columns to avoid token limits
                sample_vals = df[col].dropna().head(3).tolist()
                summary += f"\n  - Sample: {sample_vals}"
        
        # Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary += f"\n\nNUMERIC_FEATURES: {len(numeric_cols)} columns"
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                stats = df[col].describe()
                summary += f"\n- {col}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}"
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary += f"\n\nCATEGORICAL_FEATURES: {len(categorical_cols)} columns"
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                top_values = df[col].value_counts().head(3)
                summary += f"\n- {col}: Top values: {dict(top_values)}"
        
        # Data quality issues
        quality_issues = []
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > 50:
                quality_issues.append(f"CRITICAL: {col} has {null_pct:.1f}% null values")
            elif null_pct > 20:
                quality_issues.append(f"WARNING: {col} has {null_pct:.1f}% null values")
        
        if quality_issues:
            summary += f"\n\nDATA_QUALITY_ISSUES:"
            for issue in quality_issues[:5]:
                summary += f"\n- {issue}"
        
        return summary
    
    def _create_comprehensive_prompt(self, data_summary, filename):
        """Create prompt for comprehensive analysis"""
        return f"""<|system|>
You are an expert data scientist and data engineer with deep experience in analyzing diverse datasets. 
Provide a comprehensive, technical analysis of the following dataset.

Focus on:
1. DATA CHARACTERISTICS - What type of dataset is this? What patterns do you see?
2. DATA QUALITY ASSESSMENT - Specific issues and their severity
3. BUSINESS CONTEXT - What domain does this data likely belong to?
4. TECHNICAL RECOMMENDATIONS - Specific, actionable advice for data engineering
5. MACHINE LEARNING OPPORTUNITIES - What modeling approaches would work well?

Be extremely specific to THIS dataset. Reference actual column names, data types, and patterns you observe.
Avoid generic advice - every recommendation should be tied to the actual data characteristics.
</|system|>
<|user|>
Please analyze this dataset and provide comprehensive recommendations:

{data_summary}

Provide a detailed analysis with specific recommendations for this particular dataset.
</|user|>
<|assistant|>
## 🎯 COMPREHENSIVE ANALYSIS OF {filename}

**Dataset Overview:**
"""
    
    def _create_data_engineering_prompt(self, data_summary, filename):
        """Create prompt for data engineering focused analysis"""
        return f"""<|system|>
You are a senior data engineer specializing in data quality, storage optimization, and pipeline design.
Analyze this dataset from a data engineering perspective and provide specific, technical recommendations.

Focus on:
1. STORAGE OPTIMIZATION - Data type conversions, compression strategies
2. DATA QUALITY - Null value handling, data validation rules
3. PROCESSING STRATEGY - Batch vs streaming, distributed computing needs
4. PIPELINE DESIGN - ETL/ELT recommendations, monitoring strategies
5. SCALABILITY - Performance considerations for current and future scale

Provide concrete, actionable recommendations with specific column-level suggestions.
</|system|>
<|user|>
Analyze this dataset for data engineering optimization:

{data_summary}

Provide specific data engineering recommendations for storage, processing, and quality.
</|user|>
<|assistant|>
## 🔧 DATA ENGINEERING ANALYSIS FOR {filename}

**Technical Assessment:**
"""
    
    def _create_business_prompt(self, data_summary, filename):
        """Create prompt for business insights analysis"""
        return f"""<|system|>
You are a business intelligence expert and data strategist. 
Analyze this dataset to extract business insights and identify opportunities.

Focus on:
1. BUSINESS DOMAIN - What industry/domain does this data represent?
2. KEY METRICS - What are the important business metrics in this data?
3. INSIGHTS - What patterns could drive business decisions?
4. OPPORTUNITIES - What business problems could this data solve?
5. RECOMMENDATIONS - Specific business actions based on data patterns

Connect data patterns to real business value and decisions.
</|system|>
<|user|>
Extract business insights from this dataset:

{data_summary}

What business value can be derived from this data? What decisions can it inform?
</|user|>
<|assistant|>
## 💼 BUSINESS INTELLIGENCE ANALYSIS FOR {filename}

**Business Context:**
"""
    
    def _fallback_analysis(self, data_summary):
        """Fallback analysis when HF model fails"""
        return f"""
## 🤖 Basic Analysis (HF Model Unavailable)

**Note**: This is a basic analysis. For intelligent, AI-powered insights, please load a Hugging Face model.

**Data Summary**:
{data_summary}

**General Recommendations**:
- Conduct exploratory data analysis to understand patterns
- Address data quality issues before modeling
- Consider both statistical and machine learning approaches
"""

# Initialize analyzer
hf_analyzer = HFAnalyzer()

def analyze_dataset_structure(df, filename):
    """Perform basic dataset structure analysis"""
    analysis = {
        'filename': filename,
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'total_columns': len(df.columns),
        'total_rows': len(df),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'datetime_columns': len([col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Basic data quality metrics
    null_analysis = {}
    for col in df.columns:
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        if null_pct > 0:
            null_analysis[col] = null_pct
    
    analysis['null_analysis'] = null_analysis
    analysis['total_null_columns'] = len(null_analysis)
    analysis['high_null_columns'] = len([pct for pct in null_analysis.values() if pct > 20])
    
    return analysis

def create_analysis_dashboard(analysis, hf_analysis):
    """Create interactive dashboard for analysis results"""
    
    st.subheader("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{analysis['shape'][0]:,}")
    with col2:
        st.metric("Total Columns", analysis['shape'][1])
    with col3:
        st.metric("Memory Usage", f"{analysis['memory_usage_mb']:.2f} MB")
    with col4:
        st.metric("Data Quality", f"{100 - analysis['high_null_columns']}/{analysis['total_columns']}")
    
    # Column type distribution
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Numeric Columns", analysis['numeric_columns'])
    with col2:
        st.metric("Categorical Columns", analysis['categorical_columns'])
    with col3:
        st.metric("DateTime Columns", analysis['datetime_columns'])
    
    # Data quality visualization
    if analysis['null_analysis']:
        with st.expander("📉 Null Value Analysis", expanded=True):
            null_df = pd.DataFrame({
                'Column': list(analysis['null_analysis'].keys()),
                'Null_Percentage': list(analysis['null_analysis'].values())
            }).sort_values('Null_Percentage', ascending=False)
            
            fig = px.bar(null_df.head(10), x='Column', y='Null_Percentage',
                        title='Top 10 Columns with Highest Null Percentages',
                        color='Null_Percentage')
            st.plotly_chart(fig, use_container_width=True)
    
    # Hugging Face Analysis Results
    st.subheader("🤗 AI-Powered Analysis")
    st.markdown(hf_analysis)
    
    # Model information
    if hf_analyzer.model_loaded:
        st.info(f"**Analysis generated using**: {hf_analyzer.current_model}")

def main():
    st.title("🤗 Hugging Face Powered Data Analyzer")
    st.markdown("Get **intelligent, AI-powered analysis** using state-of-the-art Hugging Face models")
    
    # Sidebar for Hugging Face configuration
    with st.sidebar:
        st.header("🤗 Hugging Face Setup")
        
        hf_token = st.text_input("Enter Hugging Face Token", type="password", 
                               help="Get your token from https://huggingface.co/settings/tokens")
        
        model_choice = st.selectbox(
            "Choose Model",
            list(hf_analyzer.models.keys()),
            help="Zephyr-7B: Good balance of speed and capability\nMistral-7B: Excellent for reasoning\nPhi-2: Fastest, good for basic analysis"
        )
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["comprehensive", "data_engineering", "business_insights"],
            help="Comprehensive: Full analysis\nData Engineering: Technical focus\nBusiness Insights: Business value focus"
        )
        
        if st.button("🚀 Load HF Model"):
            if hf_token:
                with st.spinner("Loading Hugging Face model..."):
                    if hf_analyzer.setup_huggingface(hf_token, model_choice):
                        st.success(f"✅ {model_choice} loaded successfully!")
            else:
                st.error("Please enter your Hugging Face token")
        
        st.markdown("---")
        st.markdown("""
        **🎯 Powered by:**
        - **Zephyr-7B-Beta**: Fine-tuned Mistral, excellent for instruction following
        - **Mistral-7B**: Strong reasoning capabilities
        - **Phi-2**: Microsoft's compact model, fast inference
        - **CodeLlama-7B**: Specialized in technical analysis
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
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(tmp_path)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(tmp_path)
            else:
                df = pd.read_csv(tmp_path)
            
            # Perform basic analysis
            with st.spinner("📊 Analyzing dataset structure..."):
                basic_analysis = analyze_dataset_structure(df, uploaded_file.name)
            
            # Hugging Face Analysis
            if hf_analyzer.model_loaded:
                with st.spinner("🤖 AI is analyzing your data with Hugging Face model..."):
                    hf_analysis = hf_analyzer.analyze_with_hf(df, uploaded_file.name, analysis_type)
            else:
                hf_analysis = hf_analyzer._fallback_analysis(hf_analyzer._create_detailed_summary(df, uploaded_file.name))
                st.warning("⚠️ Using basic analysis - Load a Hugging Face model for AI-powered insights")
            
            # Display results
            create_analysis_dashboard(basic_analysis, hf_analysis)
            
            # Data preview
            with st.expander("🔍 Data Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.write(f"**Full Dataset**: {basic_analysis['shape'][0]:,} rows × {basic_analysis['shape'][1]} columns")
                st.write(f"**Analysis Time**: {basic_analysis['timestamp']}")
        
        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    else:
        st.info("👆 Upload a dataset and configure Hugging Face to get AI-powered analysis!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 How It Works")
            st.markdown("""
            1. **Configure HF** - Enter token & choose model in sidebar
            2. **Upload Data** - Any CSV, Excel, or Parquet file
            3. **Get AI Analysis** - HF model provides intelligent insights
            4. **Implement** - Use specific, data-driven recommendations
            """)
        
        with col2:
            st.subheader("🎯 What You Get")
            st.markdown("""
            - **AI-Powered Insights** - Not rule-based templates
            - **Dataset-Specific** - Unique analysis for each file
            - **Technical Depth** - Data engineering expertise
            - **Business Context** - Domain-aware recommendations
            - **Actionable Advice** - Specific, implementable steps
            """)

if __name__ == "__main__":
    main()
