# universal_data_analyzer_llm.py
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

# LLM imports
from openai import OpenAI
import google.generativeai as genai
import anthropic

# Set page config
st.set_page_config(
    page_title="AI Data Analyzer",
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize LLM clients (user will configure their preferred API)
class LLMAnalyzer:
    def __init__(self):
        self.available_models = {
            "OpenAI GPT-4": "openai",
            "Google Gemini": "gemini", 
            "Anthropic Claude": "claude",
            "Local Fallback": "local"
        }
        self.client = None
        self.model_type = None
    
    def setup_openai(self, api_key):
        """Setup OpenAI client"""
        try:
            self.client = OpenAI(api_key=api_key)
            self.model_type = "openai"
            return True
        except Exception as e:
            st.error(f"OpenAI setup failed: {e}")
            return False
    
    def setup_gemini(self, api_key):
        """Setup Google Gemini client"""
        try:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel('gemini-pro')
            self.model_type = "gemini"
            return True
        except Exception as e:
            st.error(f"Gemini setup failed: {e}")
            return False
    
    def setup_claude(self, api_key):
        """Setup Anthropic Claude client"""
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_type = "claude"
            return True
        except Exception as e:
            st.error(f"Claude setup failed: {e}")
            return False
    
    def analyze_with_llm(self, data_summary, file_name):
        """Use LLM to analyze data and provide intelligent recommendations"""
        
        if self.model_type == "openai":
            return self._analyze_with_openai(data_summary, file_name)
        elif self.model_type == "gemini":
            return self._analyze_with_gemini(data_summary, file_name)
        elif self.model_type == "claude":
            return self._analyze_with_claude(data_summary, file_name)
        else:
            return self._local_fallback_analysis(data_summary, file_name)
    
    def _analyze_with_openai(self, data_summary, file_name):
        """Analyze using OpenAI GPT"""
        try:
            prompt = f"""
            You are an expert data scientist. Analyze this dataset and provide SPECIFIC, DETAILED recommendations.

            DATASET: {file_name}
            DATA SUMMARY:
            {data_summary}

            Provide a COMPREHENSIVE analysis with:

            1. **DATA CHARACTERISTICS**: Key insights about this specific dataset
            2. **PROBLEM TYPE DETECTION**: What type of ML problem is this most suited for?
            3. **MODEL RECOMMENDATIONS**: 3 specific models with reasoning for THIS dataset
            4. **PREPROCESSING STEPS**: Custom steps needed for THIS data
            5. **POTENTIAL CHALLENGES**: Dataset-specific issues to watch for
            6. **BUSINESS APPLICATIONS**: How this data could be used

            Be VERY specific to this dataset - don't give generic advice.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert data scientist specializing in practical ML applications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ OpenAI analysis failed: {str(e)}\n\n{self._local_fallback_analysis(data_summary, file_name)}"
    
    def _analyze_with_gemini(self, data_summary, file_name):
        """Analyze using Google Gemini"""
        try:
            prompt = f"""
            As an expert data scientist, provide detailed analysis and recommendations for this dataset:

            DATASET: {file_name}
            DATA SUMMARY:
            {data_summary}

            Provide specific recommendations including:
            - Data characteristics and patterns
            - Suitable machine learning problem types  
            - Recommended models with justifications
            - Necessary preprocessing steps
            - Potential challenges and solutions
            - Business use cases

            Focus on what makes THIS dataset unique.
            """
            
            response = self.client.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"❌ Gemini analysis failed: {str(e)}\n\n{self._local_fallback_analysis(data_summary, file_name)}"
    
    def _analyze_with_claude(self, data_summary, file_name):
        """Analyze using Anthropic Claude"""
        try:
            prompt = f"""
            Human: You are an expert data scientist. Please analyze this dataset and provide specific, detailed recommendations.

            DATASET: {file_name}
            DATA SUMMARY:
            {data_summary}

            Please provide a comprehensive analysis with:
            1. Key data characteristics and insights
            2. Recommended machine learning approaches  
            3. Specific model suggestions with reasoning
            4. Preprocessing requirements for this data
            5. Potential challenges and mitigation strategies
            6. Business applications

            Be very specific to this dataset - avoid generic advice.

            Assistant:
            """
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"❌ Claude analysis failed: {str(e)}\n\n{self._local_fallback_analysis(data_summary, file_name)}"
    
    def _local_fallback_analysis(self, data_summary, file_name):
        """Fallback analysis when no LLM is available"""
        return f"""
## 🤖 Basic Analysis (LLM Not Available)

**Dataset**: {file_name}

**Note**: This is a basic rule-based analysis. For intelligent, dataset-specific recommendations, please configure an LLM API key in the sidebar.

**General Recommendations**:
- Start with exploratory data analysis
- Consider both tree-based models and neural networks
- Focus on data quality and preprocessing

**Setup an LLM API** in the sidebar for intelligent, customized recommendations!
"""

# Initialize the LLM analyzer
llm_analyzer = LLMAnalyzer()

def detect_data_type(file_path):
    """Detect what kind of data we're dealing with"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        extension_map = {
            '.csv': 'tabular', '.xlsx': 'tabular', '.parquet': 'tabular',
            '.jpg': 'images', '.png': 'images', '.jpeg': 'images', '.tiff': 'images',
            '.txt': 'text', '.json': 'text', '.xml': 'text',
            '.wav': 'audio', '.mp3': 'audio'
        }
        
        if file_ext in extension_map:
            return extension_map[file_ext]
        else:
            return 'generic'
            
    except Exception as e:
        return 'generic'

def create_data_summary(df, file_info):
    """Create a comprehensive data summary for LLM analysis"""
    summary = f"""
DATASET OVERVIEW:
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- File Size: {file_info['file_size_mb']:.2f} MB

COLUMN ANALYSIS:
- Total Columns: {len(df.columns)}
- Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}
- Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}
- Boolean Columns: {len(df.select_dtypes(include=['bool']).columns)}

DATA QUALITY:
- Missing Values: {df.isnull().sum().sum()} total
- Duplicate Rows: {df.duplicated().sum()}
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

COLUMN DETAILS:
"""
    
    # Add details for first 10 columns (to avoid token limits)
    for i, col in enumerate(df.columns[:10]):
        col_details = f"- {col}: {df[col].dtype}, {df[col].nunique()} unique values"
        if df[col].isnull().sum() > 0:
            col_details += f", {df[col].isnull().sum()} missing"
        summary += col_details + "\n"
    
    # Add statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary += "\nNUMERIC FEATURES SUMMARY:\n"
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            stats = df[col].describe()
            summary += f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}\n"
    
    return summary

def analyze_file_with_llm(file_path):
    """Analyze file and prepare for LLM processing"""
    file_info = {
        'file_name': os.path.basename(file_path),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'file_extension': os.path.splitext(file_path)[1],
        'data_type': detect_data_type(file_path)
    }
    
    if file_info['data_type'] == 'tabular':
        try:
            if file_info['file_extension'] == '.csv':
                df = pd.read_csv(file_path)
            elif file_info['file_extension'] == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            file_info['shape'] = df.shape
            file_info['sample_data'] = df.head(10)
            file_info['data_summary'] = create_data_summary(df, file_info)
            
        except Exception as e:
            file_info['error'] = f"Error reading file: {str(e)}"
    
    return file_info

def main():
    st.title("🧠 AI-Powered Data Analyzer")
    st.markdown("Upload any dataset and get **intelligent, LLM-powered analysis** with unique recommendations!")
    
    # Sidebar for LLM configuration
    with st.sidebar:
        st.header("🔧 AI Configuration")
        st.markdown("Choose your preferred AI model for intelligent analysis:")
        
        llm_choice = st.selectbox(
            "Select AI Model",
            ["OpenAI GPT-4", "Google Gemini", "Anthropic Claude", "Local Fallback"]
        )
        
        api_key = None
        if llm_choice != "Local Fallback":
            api_key = st.text_input(f"Enter {llm_choice} API Key", type="password")
        
        if st.button("Connect AI Model"):
            if llm_choice == "OpenAI GPT-4" and api_key:
                if llm_analyzer.setup_openai(api_key):
                    st.success("✅ Connected to OpenAI GPT-4!")
            elif llm_choice == "Google Gemini" and api_key:
                if llm_analyzer.setup_gemini(api_key):
                    st.success("✅ Connected to Google Gemini!")
            elif llm_choice == "Anthropic Claude" and api_key:
                if llm_analyzer.setup_claude(api_key):
                    st.success("✅ Connected to Anthropic Claude!")
            elif llm_choice == "Local Fallback":
                st.info("Using local fallback analysis")
            else:
                st.error("Please enter a valid API key")
        
        st.markdown("---")
        st.markdown("""
        **Why use AI analysis?**
        - 🧠 **Intelligent insights** specific to your data
        - 🎯 **Custom recommendations** based on patterns
        - 🔍 **Deep understanding** of data characteristics
        - 💡 **Creative suggestions** for model selection
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload your dataset",
        type=['csv', 'xlsx', 'txt', 'json', 'jpg', 'png', 'jpeg', 'parquet'],
        help="Supported formats: CSV, Excel, Images, Text, JSON, Parquet"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            with st.spinner("📊 Analyzing dataset structure..."):
                analysis = analyze_file_with_llm(tmp_path)
            
            # Display basic file info
            st.subheader("📄 File Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("File Type", analysis['data_type'].upper())
            with col2:
                st.metric("File Size", f"{analysis['file_size_mb']:.2f} MB")
            with col3:
                st.metric("Rows", f"{analysis['shape'][0]:,}")
            with col4:
                st.metric("Columns", analysis['shape'][1])
            
            # Data preview
            st.subheader("🔍 Data Preview")
            st.dataframe(analysis.get('sample_data', pd.DataFrame()), use_container_width=True)
            
            # LLM Analysis Section
            st.subheader("🧠 AI-Powered Analysis")
            
            if llm_analyzer.model_type or llm_choice == "Local Fallback":
                with st.spinner("🤖 AI is analyzing your data and generating intelligent recommendations..."):
                    llm_analysis = llm_analyzer.analyze_with_llm(
                        analysis.get('data_summary', 'No data summary available'),
                        analysis['file_name']
                    )
                
                st.markdown(llm_analysis)
                
                # Add user feedback
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful Analysis"):
                        st.success("Thanks for your feedback! 🎉")
                with col2:
                    if st.button("👎 Needs Improvement"):
                        st.info("We'll work on improving the analysis! 📈")
            
            else:
                st.warning("""
                **⚠️ No AI Model Configured**
                
                Please configure an AI model in the sidebar to get intelligent, dataset-specific recommendations.
                
                Currently showing basic analysis only.
                """)
                
                # Show basic data summary
                st.subheader("📊 Basic Data Summary")
                st.text(analysis.get('data_summary', 'No data available for analysis'))
        
        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    else:
        # Demo and instructions
        st.info("👆 Upload a dataset to get started!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 How It Works")
            st.markdown("""
            1. **Configure AI** - Choose your preferred LLM in sidebar
            2. **Upload Data** - Any CSV, Excel, or other format
            3. **Get Analysis** - AI provides intelligent recommendations
            4. **Implement** - Use the insights for your projects
            """)
        
        with col2:
            st.subheader("🎯 What You Get")
            st.markdown("""
            - **Problem Type Detection** - Classification, regression, etc.
            - **Model Recommendations** - Specific to your data
            - **Preprocessing Steps** - Customized for your dataset
            - **Business Applications** - Practical use cases
            - **Challenge Identification** - Potential issues to watch for
            """)

if __name__ == "__main__":
    main()
