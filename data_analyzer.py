# lightweight_hf_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Lightweight Hugging Face imports
try:
    from transformers import pipeline
    from huggingface_hub import login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Lightweight HF Analyzer",
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class LightweightHFAnalyzer:
    def __init__(self):
        self.models = {
            "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Very small, fast loading
            "DistilGPT2": "distilgpt2",  # Smallest option
            "OPT-350M": "facebook/opt-350m"  # Good balance
        }
        self.pipeline = None
        self.model_loaded = False
        self.current_model = None
    
    def setup_huggingface(self, hf_token, model_choice):
        """Setup Hugging Face with lightweight model"""
        try:
            if not HF_AVAILABLE:
                st.error("🤗 Hugging Face not available. Please check dependencies.")
                return False
            
            # Login to Hugging Face
            login(token=hf_token)
            
            model_name = self.models[model_choice]
            st.info(f"🔄 Loading {model_choice}... This should be fast!")
            
            # Use a smaller, faster model with minimal settings
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                max_length=1024  # Shorter responses for speed
            )
            
            self.model_loaded = True
            self.current_model = model_choice
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.info("💡 Try using 'DistilGPT2' - it's the smallest and fastest option.")
            return False
    
    def quick_analyze(self, df, filename):
        """Quick analysis using HF model with optimized prompts"""
        
        # Create optimized data summary (shorter for token efficiency)
        data_summary = self._create_optimized_summary(df, filename)
        
        # Use shorter, more focused prompts
        prompt = self._create_quick_prompt(data_summary, filename)
        
        try:
            # Quick generation with lower token count
            response = self.pipeline(
                prompt,
                max_new_tokens=512,  # Shorter response
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False
            )
            
            return response[0]['generated_text']
            
        except Exception as e:
            return f"❌ Quick analysis failed: {str(e)}\n\n{self._smart_fallback_analysis(df, filename)}"
    
    def _create_optimized_summary(self, df, filename):
        """Create optimized data summary for faster processing"""
        summary = f"DATA: {filename}, Shape: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB\n"
        
        # Column summary (limited)
        summary += "COLUMNS:\n"
        for i, col in enumerate(df.columns[:8]):  # Limit to 8 columns
            dtype = str(df[col].dtype)
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            unique_pct = (df[col].nunique() / len(df)) * 100
            
            summary += f"- {col}: {dtype}, nulls: {null_pct:.1f}%, unique: {unique_pct:.1f}%\n"
        
        # Quick patterns
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(df.select_dtypes(include=['object']).columns)
        
        summary += f"\nPATTERNS: Numeric: {numeric_count}, Categorical: {categorical_count}\n"
        
        # Critical issues only
        critical_issues = []
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > 50:
                critical_issues.append(f"{col}({null_pct:.1f}% null)")
        
        if critical_issues:
            summary += f"ISSUES: {', '.join(critical_issues[:3])}\n"
        
        return summary
    
    def _create_quick_prompt(self, data_summary, filename):
        """Create quick, focused prompt"""
        return f"""Analyze this dataset briefly:

{data_summary}

Provide 3-4 specific recommendations for data quality, storage, and analysis. Be concise and focus on the most important issues.
"""
    
    def _smart_fallback_analysis(self, df, filename):
        """Smart fallback analysis without HF"""
        analysis = f"""
## ⚡ Quick Analysis of {filename}

**Dataset Overview:**
- **Size**: {df.shape[0]:,} rows × {df.shape[1]} columns
- **Memory**: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
- **Types**: {len(df.select_dtypes(include=[np.number]).columns)} numeric, {len(df.select_dtypes(include=['object']).columns)} categorical

**Key Findings:**
"""
        
        # Auto-detect patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > len(categorical_cols):
            analysis += "- **Numeric-rich dataset** - Good for regression and clustering\n"
        else:
            analysis += "- **Categorical-rich dataset** - Good for classification and segmentation\n"
        
        # Critical issues
        critical_issues = []
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > 50:
                critical_issues.append(col)
        
        if critical_issues:
            analysis += f"- **Critical**: {len(critical_issues)} columns with >50% null values\n"
        
        analysis += """
**Recommendations:**
1. Address data quality issues first
2. Optimize data types for memory efficiency  
3. Choose models based on data characteristics
4. Implement proper validation strategy
"""
        
        return analysis

# Initialize analyzer
analyzer = LightweightHFAnalyzer()

def quick_dataset_analysis(df, filename):
    """Quick dataset analysis"""
    return {
        'filename': filename,
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
        'total_nulls': df.isnull().sum().sum(),
        'null_columns': [col for col in df.columns if df[col].isnull().sum() > 0]
    }

def create_fast_dashboard(analysis, hf_analysis):
    """Create fast-loading dashboard"""
    
    st.subheader("📊 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{analysis['shape'][0]:,}")
    with col2:
        st.metric("Columns", analysis['shape'][1])
    with col3:
        st.metric("Memory", f"{analysis['memory_mb']:.1f} MB")
    with col4:
        st.metric("Null Columns", len(analysis['null_columns']))
    
    # Quick insights
    with st.expander("🔍 Quick Insights", expanded=True):
        st.write(f"**Numeric Columns**: {analysis['numeric_cols']}")
        st.write(f"**Categorical Columns**: {analysis['categorical_cols']}")
        
        if analysis['null_columns']:
            st.warning(f"**Columns with nulls**: {', '.join(analysis['null_columns'][:3])}")
        else:
            st.success("✅ No null values detected")
    
    # HF Analysis
    st.subheader("🤗 AI Analysis")
    st.markdown(hf_analysis)
    
    if analyzer.model_loaded:
        st.success(f"✅ Analysis powered by {analyzer.current_model}")

def main():
    st.title("⚡ Lightweight HF Data Analyzer")
    st.markdown("**Fast, AI-powered analysis with lightweight Hugging Face models**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Quick Setup")
        
        hf_token = st.text_input("HF Token (optional)", type="password",
                               help="Required for model loading. Get from huggingface.co/settings/tokens")
        
        if HF_AVAILABLE:
            model_choice = st.selectbox(
                "Choose Model",
                list(analyzer.models.keys()),
                index=1,  # Default to DistilGPT2
                help="DistilGPT2: Fastest, OPT-350M: Better quality, TinyLlama: Balanced"
            )
        else:
            st.warning("🤗 Hugging Face not available")
            model_choice = "DistilGPT2"
        
        if st.button("🚀 Load Model") and hf_token:
            with st.spinner("Loading lightweight model..."):
                if analyzer.setup_huggingface(hf_token, model_choice):
                    st.success("Model loaded!")
        
        st.markdown("---")
        st.markdown("""
        **⚡ Lightweight Models:**
        - **DistilGPT2**: 82M params, very fast
        - **TinyLlama**: 1.1B params, good balance  
        - **OPT-350M**: 350M params, better quality
        
        **🎯 Perfect for:**
        - Quick analysis
        - Data quality checks
        - Storage optimization
        - Model recommendations
        """)
    
    # File upload - simplified
    uploaded_file = st.file_uploader(
        "📤 Upload CSV/Excel",
        type=['csv', 'xlsx'],
        help="Upload your dataset for quick analysis"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Quick file reading
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(tmp_path)
            else:
                df = pd.read_excel(tmp_path)
            
            # Quick analysis
            with st.spinner("⚡ Quick analysis..."):
                basic_analysis = quick_dataset_analysis(df, uploaded_file.name)
                
                if analyzer.model_loaded:
                    hf_analysis = analyzer.quick_analyze(df, uploaded_file.name)
                else:
                    hf_analysis = analyzer._smart_fallback_analysis(df, uploaded_file.name)
                    if not analyzer.model_loaded:
                        st.info("💡 Load a model in sidebar for AI-powered analysis")
            
            # Display results
            create_fast_dashboard(basic_analysis, hf_analysis)
            
            # Quick preview
            with st.expander("👀 Data Preview", expanded=False):
                st.dataframe(df.head(5), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    else:
        # Simple landing page
        st.info("👆 Upload a CSV or Excel file for quick analysis!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Fast & Light")
            st.markdown("""
            - **Quick model loading** (seconds, not minutes)
            - **Lightweight analysis** 
            - **Fast responses**
            - **Streamlit Cloud compatible**
            """)
        
        with col2:
            st.subheader("🎯 Smart Analysis")
            st.markdown("""
            - **Data quality assessment**
            - **Storage optimization**
            - **Model recommendations**
            - **Pattern detection**
            - **Actionable insights**
            """)

if __name__ == "__main__":
    main()
