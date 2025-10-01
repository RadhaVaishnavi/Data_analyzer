import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import ast
import io
import os
from typing import Dict, Any
import streamlit as st

# Configuration
class Config:
    HF_TOKEN = os.getenv("HF_TOKEN", "your_hf_token_here")
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    MAX_ROWS = 10000
    MAX_LENGTH = 1024

class HFCSVAnalyzerAgent:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self._load_model()
        
        self.safe_imports = {
            'pd': pd,
            'np': __import__('numpy'),
            'plt': __import__('matplotlib.pyplot')
        }
    
    def _load_model(self):
        """Load Hugging Face model"""
        try:
            st.info("🔄 Loading AI model... This may take a minute.")
            
            # Use smaller model for faster loading
            self.model_name = "microsoft/DialoGPT-large"  # Faster to load
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=Config.HF_TOKEN
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=Config.HF_TOKEN,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            st.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            # Fallback - will use simple analysis without LLM
            self.pipe = None
    
    def analyze_csv(self, csv_file, question: str) -> Dict[str, Any]:
        """Main method to analyze CSV and answer questions"""
        try:
            # Read CSV
            df = self._read_csv(csv_file)
            
            # Create data profile
            data_profile = self._create_data_profile(df)
            
            if self.pipe is None:
                # Fallback analysis without LLM
                return self._simple_analysis(df, question, data_profile)
            
            # Generate analysis code
            analysis_code = self._generate_analysis_code(data_profile, question)
            
            # Execute analysis safely
            result = self._execute_analysis_safely(df, analysis_code)
            
            # Generate explanation
            explanation = self._generate_explanation(question, analysis_code, result)
            
            return {
                "success": True,
                "answer": result,
                "explanation": explanation,
                "code_used": analysis_code,
                "data_profile": data_profile
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": None,
                "explanation": f"Error analyzing data: {str(e)}"
            }
    
    def _read_csv(self, csv_file) -> pd.DataFrame:
        """Read CSV file with error handling"""
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        else:
            df = pd.read_csv(io.BytesIO(csv_file.read()))
        
        if len(df) > Config.MAX_ROWS:
            df = df.head(Config.MAX_ROWS)
            st.warning(f"Dataset limited to first {Config.MAX_ROWS} rows for demo")
            
        return df
    
    def _create_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive data profile"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict('records'),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        }
    
    def _generate_analysis_code(self, data_profile: Dict, question: str) -> str:
        """Use HF model to generate analysis code"""
        prompt = f"""
        You are a data analyst. Given this data profile:
        
        COLUMNS: {data_profile['columns']}
        DATA TYPES: {data_profile['data_types']}
        
        Answer this question: "{question}"
        
        Generate Python pandas code that:
        1. Uses the dataframe 'df' 
        2. Answers the question
        3. Returns result as 'result'
        
        Return ONLY the Python code.
        """
        
        try:
            response = self.pipe(
                prompt,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True
            )[0]['generated_text']
            
            # Extract code from response
            code = self._extract_code_from_response(response)
            return code if code else "result = 'Could not generate analysis code'"
            
        except Exception as e:
            return f"result = 'Error generating code: {str(e)}'"
    
    def _extract_code_from_response(self, text: str) -> str:
        """Extract Python code from model response"""
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        else:
            return text.strip()
    
    def _execute_analysis_safely(self, df: pd.DataFrame, code: str) -> Any:
        """Safely execute the generated analysis code"""
        try:
            # Security check
            self._validate_code_safety(code)
            
            # Create execution environment
            exec_globals = {**self.safe_imports, 'df': df}
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Get the result
            result = exec_locals.get('result', 'Analysis completed')
            
            return self._format_result(result)
            
        except Exception as e:
            return f"Error executing analysis: {str(e)}"
    
    def _validate_code_safety(self, code: str):
        """Validate that code doesn't contain dangerous operations"""
        forbidden_keywords = [
            'os.', 'sys.', 'subprocess', 'eval', 'exec', 'open', 
            '__import__', 'shutil', 'requests.', 'urllib'
        ]
        
        for keyword in forbidden_keywords:
            if keyword in code:
                raise Exception(f"Forbidden operation: {keyword}")
    
    def _format_result(self, result):
        """Format the result for display"""
        if isinstance(result, pd.DataFrame):
            return {
                "type": "dataframe",
                "data": result.head(10).to_dict('records'),
                "shape": result.shape,
                "columns": list(result.columns)
            }
        elif isinstance(result, (int, float, str, bool)):
            return {"type": "scalar", "value": result}
        elif isinstance(result, (list, tuple)):
            return {"type": "list", "data": result[:10]}
        else:
            return {"type": "unknown", "data": str(result)}
    
    def _generate_explanation(self, question: str, code: str, result: Any) -> str:
        """Generate explanation of the analysis"""
        prompt = f"""
        The user asked: "{question}"
        
        We generated this code:
        {code}
        
        The result was: {result}
        
        Explain what was done in simple terms.
        """
        
        try:
            response = self.pipe(prompt, max_new_tokens=200)[0]['generated_text']
            return response.strip()
        except:
            return "Analysis completed. Check the results above."
    
    def _simple_analysis(self, df: pd.DataFrame, question: str, data_profile: Dict) -> Dict[str, Any]:
        """Fallback analysis without LLM"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['average', 'mean']):
            if data_profile['numeric_columns']:
                result = df[data_profile['numeric_columns']].mean().to_dict()
                explanation = f"Calculated average for numeric columns: {list(result.keys())}"
            else:
                result = "No numeric columns found for average calculation"
                explanation = "The dataset doesn't contain numeric columns for average calculation"
                
        elif any(word in question_lower for word in ['count', 'how many']):
            result = f"Total rows: {len(df)}"
            explanation = "Counted total number of rows in the dataset"
            
        elif any(word in question_lower for word in ['column', 'columns']):
            result = f"Columns: {data_profile['columns']}"
            explanation = "Listed all columns in the dataset"
            
        else:
            result = df.head(5).to_dict('records')
            explanation = "Showing sample data from the dataset"
        
        return {
            "success": True,
            "answer": result,
            "explanation": explanation,
            "code_used": "# Simple analysis without LLM",
            "data_profile": data_profile
        }


import streamlit as st
import pandas as pd
from csv_agent import HFCSVAnalyzerAgent
import time

# Page configuration
st.set_page_config(
    page_title="CSV Analyzer Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0px;
    }
    .code-box {
        background-color: #262730;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 CSV Data Analyzer Agent</h1>', unsafe_allow_html=True)
st.markdown("Upload a CSV file and ask questions in natural language! Powered by Hugging Face 🤗")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("This tool uses AI to analyze your CSV data and answer questions in plain English.")
    
    st.subheader("📁 Sample Queries")
    st.write("""
    - "What are the average values?"
    - "Show me the distribution of data"
    - "Which columns have missing values?"
    - "What is the correlation between numeric columns?"
    - "Show summary statistics"
    """)
    
    st.subheader("ℹ️ How to Use")
    st.write("""
    1. Upload a CSV file
    2. Ask a question about your data
    3. Get instant analysis with code
    """)

# Initialize agent (cached)
@st.cache_resource(show_spinner=False)
def load_agent():
    return HFCSVAnalyzerAgent()

def main():
    # File upload section
    st.header("📤 Upload Your CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display data
            df = pd.read_csv(uploaded_file)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"✅ File loaded successfully! Shape: {df.shape}")
                
            with col2:
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
            
            # Data preview
            with st.expander("🔍 Data Preview", expanded=True):
                tab1, tab2, tab3 = st.tabs(["First 10 rows", "Data Types", "Missing Values"])
                
                with tab1:
                    st.dataframe(df.head(10), use_container_width=True)
                
                with tab2:
                    dtype_df = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.values,
                        'Non-Null Count': df.count().values
                    })
                    st.dataframe(dtype_df, use_container_width=True)
                
                with tab3:
                    missing_df = pd.DataFrame({
                        'Column': df.columns,
                        'Missing Values': df.isnull().sum().values,
                        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
                    })
                    st.dataframe(missing_df, use_container_width=True)
            
            # Question input
            st.header("❓ Ask Questions About Your Data")
            
            # Quick question templates
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📈 Summary Statistics"):
                    st.session_state.question = "Show summary statistics for numeric columns"
            with col2:
                if st.button("🔍 Data Overview"):
                    st.session_state.question = "Give me an overview of this dataset"
            with col3:
                if st.button("🎯 Find Insights"):
                    st.session_state.question = "What are the key insights from this data?"
            
            question = st.text_input(
                "Or type your own question:",
                placeholder="e.g., 'What is the average sales by region?'",
                key="question"
            )
            
            if question:
                # Initialize agent
                agent = load_agent()
                
                # Analyze
                with st.spinner("🤖 Analyzing your data with AI..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    result = agent.analyze_csv(uploaded_file, question)
                
                # Display results
                if result["success"]:
                    st.balloons()
                    st.success("✅ Analysis Complete!")
                    
                    # Results section
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("📝 Explanation")
                    st.write(result["explanation"])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display results
                    st.subheader("📊 Results")
                    answer_data = result["answer"]
                    
                    if isinstance(answer_data, dict):
                        if answer_data.get("type") == "dataframe":
                            st.dataframe(answer_data["data"], use_container_width=True)
                            st.write(f"**Shape:** {answer_data['shape']}")
                        elif answer_data.get("type") == "scalar":
                            st.metric("Result", answer_data["value"])
                        elif answer_data.get("type") == "list":
                            st.write(answer_data["data"])
                        else:
                            st.write(answer_data)
                    else:
                        st.write(answer_data)
                    
                    # Code used
                    with st.expander("🔧 View Generated Code"):
                        st.markdown('<div class="code-box">', unsafe_allow_html=True)
                        st.code(result["code_used"], language="python")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                else:
                    st.error(f"❌ Error: {result['error']}")
                    
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
    
    else:
        # Welcome message when no file uploaded
        st.info("👆 Please upload a CSV file to get started")
        
        # Sample data section
        with st.expander("🎯 Don't have a CSV? Try with sample data"):
            sample_option = st.selectbox(
                "Choose sample dataset:",
                ["Sales Data", "Customer Data", "Product Inventory"]
            )
            
            if st.button("Load Sample Data"):
                # Create sample data
                if sample_option == "Sales Data":
                    sample_df = pd.DataFrame({
                        'Region': ['North', 'South', 'East', 'West'] * 25,
                        'Product': ['A', 'B', 'C'] * 33 + ['A'],
                        'Sales': range(100, 500, 4),
                        'Profit': range(20, 120, 1),
                        'Month': ['Jan', 'Feb', 'Mar', 'Apr'] * 25
                    })
                elif sample_option == "Customer Data":
                    sample_df = pd.DataFrame({
                        'CustomerID': range(1, 101),
                        'Age': range(18, 118),
                        'City': ['New York', 'London', 'Tokyo'] * 33 + ['New York'],
                        'Spending': range(100, 1100, 10),
                        'Membership': ['Gold', 'Silver', 'Bronze'] * 33 + ['Gold']
                    })
                else:
                    sample_df = pd.DataFrame({
                        'ProductID': range(1, 101),
                        'Category': ['Electronics', 'Clothing', 'Books'] * 33 + ['Electronics'],
                        'Price': range(10, 1010, 10),
                        'Stock': range(100, 0, -1),
                        'Supplier': ['Supplier A', 'Supplier B'] * 50
                    })
                
                # Convert to CSV for upload
                csv_data = sample_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=csv_data,
                    file_name=f"sample_{sample_option.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                st.dataframe(sample_df.head(10))

if __name__ == "__main__":
    main()
