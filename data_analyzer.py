# universal_data_analyzer_hf.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Hugging Face imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

# Set page config
st.set_page_config(
    page_title="Data Engineer's Analyzer",
    page_icon="🔧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class HFDataEngineer:
    def __init__(self):
        self.models = {
            "Mixtral 8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "CodeLlama 34B": "codellama/CodeLlama-34b-Instruct-hf", 
            "Zephyr 7B": "HuggingFaceH4/zephyr-7b-beta",
            "Data Analysis Expert": "microsoft/DialoGPT-medium"  # Fallback
        }
        self.pipeline = None
        self.model_loaded = False
    
    def setup_huggingface(self, hf_token, model_choice):
        """Setup Hugging Face with the chosen model"""
        try:
            # Login to Hugging Face
            login(token=hf_token)
            
            model_name = self.models[model_choice]
            st.info(f"🔄 Loading {model_choice}... This may take a few minutes.")
            
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
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    def analyze_as_data_engineer(self, df, file_info):
        """Perform deep data engineering analysis"""
        
        # First, create comprehensive technical analysis
        technical_analysis = self._create_technical_analysis(df, file_info)
        
        # Then use LLM for intelligent insights
        prompt = self._create_data_engineer_prompt(technical_analysis, file_info['file_name'])
        
        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more factual responses
                top_p=0.9,
                return_full_text=False
            )
            
            return response[0]['generated_text']
            
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}\n\n{self._fallback_technical_analysis(technical_analysis)}"
    
    def _create_technical_analysis(self, df, file_info):
        """Create comprehensive technical data analysis"""
        analysis = {}
        
        # Basic metrics
        analysis['shape'] = df.shape
        analysis['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024
        analysis['file_size_mb'] = file_info['file_size_mb']
        
        # Data types analysis
        dtypes_analysis = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype not in dtypes_analysis:
                dtypes_analysis[dtype] = []
            dtypes_analysis[dtype].append(col)
        analysis['dtype_distribution'] = dtypes_analysis
        
        # Null analysis
        null_analysis = {}
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            if null_count > 0:
                null_analysis[col] = {
                    'count': null_count,
                    'percentage': round(null_pct, 2),
                    'type': 'CRITICAL' if null_pct > 50 else 'WARNING' if null_pct > 20 else 'INFO'
                }
        analysis['null_analysis'] = null_analysis
        
        # Cardinality analysis
        cardinality = {}
        for col in df.columns:
            unique_count = df[col].nunique()
            cardinality_pct = (unique_count / len(df)) * 100
            cardinality[col] = {
                'unique_count': unique_count,
                'cardinality_pct': round(cardinality_pct, 2),
                'type': 'HIGH' if cardinality_pct > 90 else 'MEDIUM' if cardinality_pct > 50 else 'LOW'
            }
        analysis['cardinality'] = cardinality
        
        # Statistical analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_analysis = {}
        for col in numeric_cols:
            stats = df[col].describe()
            skewness = df[col].skew()
            stats_analysis[col] = {
                'mean': round(stats['mean'], 4),
                'std': round(stats['std'], 4),
                'min': round(stats['min'], 4),
                'max': round(stats['max'], 4),
                'skewness': round(skewness, 4),
                'outlier_pct': self._calculate_outlier_percentage(df[col])
            }
        analysis['numeric_stats'] = stats_analysis
        
        # Data quality issues
        quality_issues = self._identify_data_quality_issues(df, analysis)
        analysis['quality_issues'] = quality_issues
        
        # Storage optimization suggestions
        storage_optimization = self._suggest_storage_optimization(df)
        analysis['storage_optimization'] = storage_optimization
        
        return analysis
    
    def _calculate_outlier_percentage(self, series):
        """Calculate percentage of outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            return 0
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return round((len(outliers) / len(series)) * 100, 2)
    
    def _identify_data_quality_issues(self, df, analysis):
        """Identify data quality issues from a data engineering perspective"""
        issues = []
        
        # High null percentage
        for col, null_info in analysis['null_analysis'].items():
            if null_info['type'] == 'CRITICAL':
                issues.append(f"🚨 CRITICAL: {col} has {null_info['percentage']}% missing values")
            elif null_info['type'] == 'WARNING':
                issues.append(f"⚠️ WARNING: {col} has {null_info['percentage']}% missing values")
        
        # High cardinality (potential data leakage)
        for col, card_info in analysis['cardinality'].items():
            if card_info['type'] == 'HIGH':
                issues.append(f"🔍 HIGH CARDINALITY: {col} has {card_info['unique_count']} unique values ({card_info['cardinality_pct']}%)")
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                issues.append(f"📊 CONSTANT COLUMN: {col} has only {df[col].nunique()} unique value")
        
        # High skewness in numeric columns
        for col, stats in analysis.get('numeric_stats', {}).items():
            if abs(stats['skewness']) > 2:
                issues.append(f"📈 HIGHLY SKEWED: {col} has skewness of {stats['skewness']}")
        
        # High outlier percentage
        for col, stats in analysis.get('numeric_stats', {}).items():
            if stats['outlier_pct'] > 10:
                issues.append(f"📊 OUTLIERS: {col} has {stats['outlier_pct']}% outliers")
        
        return issues
    
    def _suggest_storage_optimization(self, df):
        """Suggest storage optimization strategies"""
        optimizations = []
        
        current_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Check for downcasting opportunities
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 256:
                    optimizations.append(f"🔧 {col}: Convert to uint8 (saves ~75% memory)")
                elif col_max < 65536:
                    optimizations.append(f"🔧 {col}: Convert to uint16 (saves ~50% memory)")
            else:  # Signed integers
                if col_min > -128 and col_max < 127:
                    optimizations.append(f"🔧 {col}: Convert to int8 (saves ~75% memory)")
                elif col_min > -32768 and col_max < 32767:
                    optimizations.append(f"🔧 {col}: Convert to int16 (saves ~50% memory)")
        
        # Categorical optimization
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Good candidate for categorical
                optimizations.append(f"🔧 {col}: Convert to category (saves ~60% memory)")
        
        optimizations.append(f"💾 Current memory: {current_memory:.2f} MB")
        optimizations.append(f"💾 Estimated savings: {current_memory * 0.3:.2f} MB (30% reduction possible)")
        
        return optimizations
    
    def _create_data_engineer_prompt(self, technical_analysis, filename):
        """Create a detailed prompt for data engineering analysis"""
        
        prompt = f"""<|system|>
You are a senior data engineer with 10+ years of experience. Analyze this dataset from a data engineering perspective and provide detailed, technical recommendations.

Focus on:
1. DATA QUALITY ASSESSMENT
2. STORAGE OPTIMIZATION
3. PROCESSING CONSIDERATIONS
4. PIPELINE DESIGN
5. SCALABILITY CONCERNS

Be technical, specific, and provide actionable recommendations. Use data engineering terminology.
</|system|>
<|user|>
DATASET: {filename}

TECHNICAL ANALYSIS:
- Shape: {technical_analysis['shape']}
- Memory Usage: {technical_analysis['memory_usage_mb']:.2f} MB
- File Size: {technical_analysis['file_size_mb']:.2f} MB

DATA TYPE DISTRIBUTION:
{self._format_dtype_distribution(technical_analysis['dtype_distribution'])}

NULL VALUE ANALYSIS:
{self._format_null_analysis(technical_analysis['null_analysis'])}

CARDINALITY ANALYSIS:
{self._format_cardinality_analysis(technical_analysis['cardinality'])}

NUMERIC FEATURES STATISTICS:
{self._format_numeric_stats(technical_analysis.get('numeric_stats', {}))}

DATA QUALITY ISSUES:
{self._format_quality_issues(technical_analysis['quality_issues'])}

STORAGE OPTIMIZATION SUGGESTIONS:
{self._format_storage_optimization(technical_analysis['storage_optimization'])}

Provide a comprehensive data engineering analysis covering:
1. Data Quality Score and Issues
2. Storage Optimization Strategy
3. Processing Recommendations
4. Pipeline Design Considerations
5. Scalability Assessment
6. Specific Technical Actions

Be brutally honest about data quality problems and provide concrete solutions.
</|user|>
<|assistant|>
"""
        return prompt
    
    def _format_dtype_distribution(self, dtype_dist):
        text = ""
        for dtype, cols in dtype_dist.items():
            text += f"- {dtype}: {len(cols)} columns\n"
            if len(cols) <= 5:
                text += f"  Samples: {', '.join(cols)}\n"
        return text
    
    def _format_null_analysis(self, null_analysis):
        if not null_analysis:
            return "✅ No missing values detected"
        
        text = ""
        for col, info in null_analysis.items():
            text += f"- {col}: {info['count']} nulls ({info['percentage']}%) - {info['type']}\n"
        return text
    
    def _format_cardinality_analysis(self, cardinality):
        text = ""
        for col, info in cardinality.items():
            text += f"- {col}: {info['unique_count']} unique ({info['cardinality_pct']}%) - {info['type']} cardinality\n"
        return text
    
    def _format_numeric_stats(self, numeric_stats):
        if not numeric_stats:
            return "No numeric columns found"
        
        text = ""
        for col, stats in numeric_stats.items():
            text += f"- {col}: mean={stats['mean']}, std={stats['std']}, range=[{stats['min']}, {stats['max']}], skew={stats['skewness']}, outliers={stats['outlier_pct']}%\n"
        return text
    
    def _format_quality_issues(self, issues):
        if not issues:
            return "✅ No major data quality issues detected"
        
        text = ""
        for issue in issues[:10]:  # Limit to top 10 issues
            text += f"{issue}\n"
        return text
    
    def _format_storage_optimization(self, optimizations):
        text = ""
        for opt in optimizations:
            text += f"{opt}\n"
        return text
    
    def _fallback_technical_analysis(self, technical_analysis):
        """Fallback analysis when LLM fails"""
        return f"""
## 🔧 DATA ENGINEERING ANALYSIS (Fallback)

### 📊 DATASET OVERVIEW
- **Shape**: {technical_analysis['shape']}
- **Memory Usage**: {technical_analysis['memory_usage_mb']:.2f} MB
- **File Size**: {technical_analysis['file_size_mb']:.2f} MB

### 🚨 DATA QUALITY ISSUES
{chr(10).join(technical_analysis['quality_issues'][:10])}

### 💾 STORAGE OPTIMIZATION
{chr(10).join(technical_analysis['storage_optimization'])}

### 🎯 RECOMMENDATIONS
1. **Address data quality issues** before modeling
2. **Implement storage optimizations** to reduce memory usage
3. **Consider data type conversions** for better performance
4. **Monitor for data drift** in production
"""

# Initialize the data engineer analyzer
data_engineer = HFDataEngineer()

def analyze_file(file_path):
    """Analyze file and return basic info"""
    file_info = {
        'file_name': os.path.basename(file_path),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'file_extension': os.path.splitext(file_path)[1],
    }
    
    try:
        if file_info['file_extension'] == '.csv':
            df = pd.read_csv(file_path)
        elif file_info['file_extension'] == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        file_info['df'] = df
        file_info['shape'] = df.shape
        
    except Exception as e:
        file_info['error'] = str(e)
    
    return file_info

def create_technical_dashboard(technical_analysis):
    """Create a technical dashboard for data engineers"""
    
    st.subheader("🔧 Technical Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Shape", f"{technical_analysis['shape'][0]:,} × {technical_analysis['shape'][1]}")
    with col2:
        st.metric("Memory Usage", f"{technical_analysis['memory_usage_mb']:.2f} MB")
    with col3:
        st.metric("File Size", f"{technical_analysis['file_size_mb']:.2f} MB")
    with col4:
        quality_score = 10 - min(10, len(technical_analysis['quality_issues']) * 0.5)
        st.metric("Quality Score", f"{quality_score:.1f}/10")
    
    # Data Quality Issues
    with st.expander("🚨 Data Quality Issues", expanded=True):
        if technical_analysis['quality_issues']:
            for issue in technical_analysis['quality_issues'][:10]:
                st.write(issue)
        else:
            st.success("✅ No major data quality issues detected")
    
    # Storage Optimization
    with st.expander("💾 Storage Optimization", expanded=True):
        for opt in technical_analysis['storage_optimization']:
            st.write(opt)
    
    # Data Types Distribution
    with st.expander("📊 Data Types Analysis", expanded=False):
        dtype_data = []
        for dtype, cols in technical_analysis['dtype_distribution'].items():
            dtype_data.append({'Type': dtype, 'Count': len(cols)})
        
        if dtype_data:
            dtype_df = pd.DataFrame(dtype_data)
            fig = px.pie(dtype_df, values='Count', names='Type', title='Data Type Distribution')
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🔧 Data Engineer's Analyzer")
    st.markdown("Get **deep, technical data engineering analysis** with Hugging Face models")
    
    # Sidebar for Hugging Face configuration
    with st.sidebar:
        st.header("🤗 Hugging Face Setup")
        
        hf_token = st.text_input("Enter Hugging Face Token", type="password")
        
        model_choice = st.selectbox(
            "Choose Model",
            list(data_engineer.models.keys())
        )
        
        if st.button("Load Model"):
            if hf_token:
                with st.spinner("Loading model..."):
                    if data_engineer.setup_huggingface(hf_token, model_choice):
                        st.success(f"✅ {model_choice} loaded successfully!")
            else:
                st.error("Please enter your Hugging Face token")
        
        st.markdown("---")
        st.markdown("""
        **🔧 Data Engineering Focus:**
        - Data Quality Assessment
        - Storage Optimization  
        - Pipeline Design
        - Scalability Analysis
        - Technical Debt Identification
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload your dataset",
        type=['csv', 'xlsx', 'parquet'],
        help="Upload CSV, Excel, or Parquet files for analysis"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            with st.spinner("🔍 Performing technical analysis..."):
                file_info = analyze_file(tmp_path)
                
                if 'error' in file_info:
                    st.error(f"Error reading file: {file_info['error']}")
                    return
                
                # Create technical analysis
                technical_analysis = data_engineer._create_technical_analysis(file_info['df'], file_info)
                
                # Display technical dashboard
                create_technical_dashboard(technical_analysis)
                
                # LLM Analysis
                st.subheader("🧠 AI Data Engineering Analysis")
                
                if data_engineer.model_loaded:
                    with st.spinner("🤖 Data Engineer AI is analyzing your data..."):
                        llm_analysis = data_engineer.analyze_as_data_engineer(file_info['df'], file_info)
                    
                    st.markdown(llm_analysis)
                else:
                    st.warning("""
                    **⚠️ Hugging Face Model Not Loaded**
                    
                    Please configure your Hugging Face token and load a model in the sidebar to get AI-powered data engineering analysis.
                    """)
                    
                    # Show fallback analysis
                    st.markdown(data_engineer._fallback_technical_analysis(technical_analysis))
        
        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    else:
        st.info("👆 Upload a dataset to get data engineering analysis!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 What You'll Get")
            st.markdown("""
            - **Data Quality Score** and issues
            - **Storage Optimization** strategies  
            - **Memory Usage** analysis
            - **Data Type** optimization
            - **Pipeline Design** considerations
            - **Scalability** assessment
            """)
        
        with col2:
            st.subheader("🔧 Technical Focus")
            st.markdown("""
            - **Null Value Analysis**
            - **Cardinality Assessment** 
            - **Outlier Detection**
            - **Skewness Analysis**
            - **Memory Optimization**
            - **Data Type Recommendations**
            """)

if __name__ == "__main__":
    main()
