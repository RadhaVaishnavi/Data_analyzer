import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import re

# Configure the page
st.set_page_config(
    page_title="CSV Analyzer AI",
    page_icon="📊",
    layout="wide"
)

# Security check function
def is_code_safe(code: str) -> bool:
    """Check if code contains dangerous operations"""
    forbidden_patterns = [
        'os.', 'sys.', 'eval', 'exec', 'open', '__import__', 
        'subprocess', 'shutil', 'rmdir', 'remove', 'unlink',
        'write(', 'read(', 'delete', 'rm ', 'format(', 'compile(',
        'input(', 'getpass', 'pickle', 'yaml', 'json.loads'
    ]
    return not any(pattern in code for pattern in forbidden_patterns)

def clean_generated_code(code: str) -> str:
    """Clean and validate generated code"""
    # Remove any markdown formatting
    code = re.sub(r'```python|```', '', code).strip()
    
    # Remove any explanatory text before/after code
    lines = code.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.strip() and not line.strip().startswith(('#', '"', "'")) and 'print(' not in line:
            if any(keyword in line for keyword in ['import', 'def ', 'class ', '=', 'df.', 'result =']):
                in_code = True
            if in_code:
                code_lines.append(line)
    
    cleaned_code = '\n'.join(code_lines)
    
    # Ensure result variable is set
    if 'result =' not in cleaned_code:
        cleaned_code += '\nresult = "Analysis completed"'
    
    return cleaned_code

# Cache the model loading
@st.cache_resource(show_spinner=False)
def load_analyzer_model():
    """Load the model with secure token handling"""
    try:
        # Get token from Streamlit secrets
        HF_TOKEN = st.secrets.get("HUGGINGFACE_KEY")
        
        if not HF_TOKEN:
            st.error("🔐 HUGGINGFACE_KEY not found in Streamlit secrets. Please add it to your secrets.")
            return None
            
        model_name = "microsoft/DialoGPT-medium"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.info("💡 Make sure your HUGGINGFACE_KEY is valid and has access to the model.")
        return None

class CSVAnalyzerAI:
    def __init__(self):
        model_components = load_analyzer_model()
        if model_components:
            self.tokenizer, self.model = model_components
            self.safe_imports = {'pd': pd, 'np': __import__('numpy')}
        else:
            raise Exception("Model failed to load")
    
    def _generate_text_fast(self, prompt: str, max_length: int = 200) -> str:
        """Generate text using the model with optimized settings"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=0.3,  # Lower temperature for more focused output
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.replace(prompt, "").strip()
        except Exception as e:
            return f"# Error in code generation\nresult = 'Failed to generate analysis code'"
    
    def analyze_data_fast(self, df, question: str):
        """Fast analysis with predefined patterns for common questions"""
        try:
            # Quick analysis for common patterns
            quick_result = self._quick_analysis(df, question)
            if quick_result:
                return quick_result
            
            # For complex questions, use AI with timeout
            profile = {
                "columns": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict(),
                "shape": df.shape,
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
            }
            
            # Simplified prompt for faster generation
            code_prompt = f"Data: {profile['columns']}. Question: {question}. Code:"
            
            generated_text = self._generate_text_fast(code_prompt, max_length=150)
            code = clean_generated_code(generated_text)
            
            # Security check
            if not is_code_safe(code):
                return {
                    "success": False,
                    "error": "Security violation detected"
                }
            
            # Execute code safely
            exec_globals = {**self.safe_imports, 'df': df}
            exec_locals = {}
            exec(code, exec_globals, exec_locals)
            result = exec_locals.get('result', 'Analysis completed')
            
            return {
                "success": True,
                "answer": result,
                "explanation": f"Analysis for: {question}",
                "code": code
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}"
            }
    
    def _quick_analysis(self, df, question: str):
        """Handle common questions with predefined code patterns"""
        question_lower = question.lower()
        
        # Basic statistics
        if any(phrase in question_lower for phrase in ['missing', 'null', 'nan']):
            missing_data = df.isnull().sum()
            result = f"Missing values per column:\n{missing_data}"
            return {
                "success": True,
                "answer": result,
                "explanation": "Shows count of missing/null values for each column in the dataset",
                "code": "result = df.isnull().sum()"
            }
        
        # Dataset size
        elif any(phrase in question_lower for phrase in ['size', 'shape', 'dimension']):
            result = f"Dataset shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})"
            return {
                "success": True,
                "answer": result,
                "explanation": "Shows the dimensions of your dataset",
                "code": "result = f'Dataset shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})'"
            }
        
        # Data types
        elif any(phrase in question_lower for phrase in ['data type', 'dtype', 'data types']):
            result = f"Data types:\n{df.dtypes}"
            return {
                "success": True,
                "answer": result,
                "explanation": "Shows the data types of each column",
                "code": "result = df.dtypes"
            }
        
        # Duplicates
        elif any(phrase in question_lower for phrase in ['duplicate', 'duplicates']):
            duplicate_count = df.duplicated().sum()
            result = f"Number of duplicate rows: {duplicate_count}"
            return {
                "success": True,
                "answer": result,
                "explanation": "Counts duplicate rows in the dataset",
                "code": "result = f'Number of duplicate rows: {df.duplicated().sum()}'"
            }
        
        # Basic info
        elif any(phrase in question_lower for phrase in ['info', 'information', 'overview']):
            result = f"Dataset Info:\n- Shape: {df.shape}\n- Columns: {list(df.columns)}\n- Data types: {df.dtypes.to_dict()}"
            return {
                "success": True,
                "answer": result,
                "explanation": "Provides basic overview of the dataset",
                "code": "result = f'Dataset Info: Shape {df.shape}, Columns: {list(df.columns)}'"
            }
        
        return None

# Main Streamlit app
def main():
    st.title("📊 CSV Analyzer AI")
    st.markdown("Upload your CSV file and get instant data analysis! 🤖")
    
    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Fast AI-powered data analysis:**
        - ⚡ Quick responses
        - 🔍 Smart analysis
        - 💡 Clear explanations
        - 🔒 Secure execution
        """)
    
    # Initialize analyzer
    try:
        analyzer = CSVAnalyzerAI()
        st.success("✅ AI Model loaded successfully!")
    except Exception as e:
        st.error("❌ Failed to initialize AI analyzer. Please check your HUGGINGFACE_KEY in secrets.")
        st.stop()
    
    # File upload section
    st.header("1. Upload Your CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your CSV data file for analysis"
    )
    
    df = None
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ CSV file loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.stop()
    
    # Display data preview
    if df is not None:
        st.header("2. Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Total Cells", df.shape[0] * df.shape[1])
        
        # Show data preview
        with st.expander("📋 View Data Preview", expanded=False):
            tab1, tab2 = st.tabs(["First 10 Rows", "Data Info"])
            with tab1:
                st.dataframe(df.head(10), use_container_width=True)
            with tab2:
                st.write("**Column Information:**")
                for col in df.columns:
                    missing = df[col].isna().sum()
                    st.write(f"- **{col}**: {df[col].dtype} ({missing} missing)")
        
        # Question section
        st.header("3. Ask Questions")
        
        # Comprehensive question categories
        data_quality_questions = [
            "How many missing/null values per column?",
            "Are there duplicates?",
            "Does it reflect reality (accuracy check)?",
            "Are formats uniform (consistency)?",
            "Conforms to rules (validity)?",
            "Is the data fresh enough (timeliness)?",
            "Any sampling biases?",
            "Distributions and outliers?"
        ]
        
        business_questions = [
            "What's the dataset's purpose or intended use case?",
            "What's the size and growth projection?",
            "Includes needed features (relevance)?",
            "Relationships preserved (integrity)?",
            "Correlations or patterns?",
            "What if it's wrong (impact assessment)?"
        ]
        
        technical_questions = [
            "Can it integrate into pipelines?",
            "Performance acceptable?",
            "What compute resources are available?",
            "How to handle small vs big datasets?"
        ]
        
        basic_questions = [
            "What is the average of numeric columns?",
            "Show column distributions",
            "What are the most frequent categories?",
            "Show data types overview",
            "Basic statistics summary"
        ]
        
        # Question selection
        question_category = st.selectbox(
            "Choose question category:",
            ["Basic Analysis", "Data Quality", "Business Context", "Technical Aspects"]
        )
        
        if question_category == "Basic Analysis":
            quick_questions = basic_questions
        elif question_category == "Data Quality":
            quick_questions = data_quality_questions
        elif question_category == "Business Context":
            quick_questions = business_questions
        else:
            quick_questions = technical_questions
        
        selected_question = st.selectbox(
            "Choose a question:",
            [""] + quick_questions,
            help="Select from comprehensive analysis questions"
        )
        
        custom_question = st.text_input(
            "Or type your own question:",
            placeholder="e.g., Show correlation between numeric columns",
            help="Ask any specific question about your data"
        )
        
        # Use selected question or custom question
        question = custom_question if custom_question else selected_question
        
        if st.button("🚀 Analyze Data", type="primary", use_container_width=True) and question:
            with st.spinner("⚡ Analyzing your data..."):
                result = analyzer.analyze_data_fast(df, question)
            
            if result["success"]:
                st.success("✅ Analysis Complete!")
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["📊 Results", "💡 Explanation", "🔧 Generated Code"])
                
                with tab1:
                    st.subheader("Analysis Results")
                    if isinstance(result["answer"], (pd.DataFrame, pd.Series)):
                        st.dataframe(result["answer"], use_container_width=True)
                    else:
                        st.write(result["answer"])
                
                with tab2:
                    st.subheader("Explanation")
                    st.write(result["explanation"])
                
                with tab3:
                    st.subheader("Generated Code")
                    st.code(result["code"], language="python")
            
            else:
                st.error(f"❌ Analysis failed: {result['error']}")
                st.info("💡 Try a different question or rephrase your query.")
        
        # Quick analysis buttons
        st.header("4. Quick Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Basic Info", use_container_width=True):
                st.session_state.quick_question = "Show basic dataset information"
        
        with col2:
            if st.button("🔍 Missing Values", use_container_width=True):
                st.session_state.quick_question = "How many missing/null values per column?"
        
        with col3:
            if st.button("📈 Statistics", use_container_width=True):
                st.session_state.quick_question = "Show basic statistics for numeric columns"
        
        with col4:
            if st.button("🔄 Duplicates", use_container_width=True):
                st.session_state.quick_question = "Are there duplicates?"
        
        # Handle quick question from buttons
        if hasattr(st.session_state, 'quick_question'):
            question = st.session_state.quick_question
            with st.spinner("⚡ Quick analysis..."):
                result = analyzer.analyze_data_fast(df, question)
            
            if result["success"]:
                st.success("✅ Quick Analysis Complete!")
                st.write("**Results:**", result["answer"])
            else:
                st.error(f"Quick analysis failed: {result['error']}")
            
            # Clear the quick question
            del st.session_state.quick_question
    
    else:
        # Welcome message when no file is uploaded
        st.info("👆 Please upload a CSV file to get started with data analysis!")
        
        # Sample data option
        if st.button("🎯 Try with Sample Data"):
            sample_data = {
                'Employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace'],
                'Age': [25, 30, 35, 28, 32, 29, 31],
                'Salary': [50000, 60000, 70000, 55000, 65000, 58000, 62000],
                'Department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance'],
                'Experience_Years': [2, 5, 8, 3, 6, 4, 5],
                'Performance_Score': [85, 90, 78, 92, 88, 85, 91]
            }
            df = pd.DataFrame(sample_data)
            st.session_state.sample_data = df
            st.rerun()

if __name__ == "__main__":
    main()
