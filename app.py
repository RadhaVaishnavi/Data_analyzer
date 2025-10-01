import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

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
        'write(', 'read(', 'delete', 'rm ', 'format(', 'compile('
    ]
    return not any(pattern in code for pattern in forbidden_patterns)

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
            torch_dtype=torch.float16,  # Fixed: using dtype instead of torch_dtype
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
            st.success("✅ AI Model loaded successfully!")
        else:
            raise Exception("Model failed to load")
    
    def _generate_text(self, prompt: str, max_length: int = 400) -> str:
        """Generate text using the model"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from response
            return response.replace(prompt, "").strip()
        except Exception as e:
            return f"Generation error: {str(e)}"
    
    def _extract_code(self, text: str) -> str:
        """Extract code from generated text"""
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
    
    def analyze_data(self, df, question: str):
        """Analyze dataframe with the given question"""
        try:
            # Create data profile
            profile = {
                "columns": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict(),
                "shape": f"{df.shape[0]} rows, {df.shape[1]} columns",
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "sample_data": df.head(2).to_dict('records')
            }
            
            # Generate analysis code
            code_prompt = f"""
            You are a data analyst. Given this data:
            - Columns: {profile['columns']}
            - Data Types: {profile['data_types']}
            - Sample Data: {profile['sample_data']}
            
            Question: {question}
            
            Generate Python pandas code to answer this question. Use dataframe 'df'.
            Store the final result in variable 'result'.
            Return ONLY the code without any explanations or markdown formatting.
            
            Code:
            """
            
            generated_text = self._generate_text(code_prompt)
            code = self._extract_code(generated_text)
            
            # Security check
            if not is_code_safe(code):
                return {
                    "success": False,
                    "error": "Security violation detected in generated code"
                }
            
            # Execute code safely
            exec_globals = {**self.safe_imports, 'df': df}
            exec_locals = {}
            exec(code, exec_globals, exec_locals)
            result = exec_locals.get('result', 'Analysis completed successfully')
            
            # Generate explanation
            explanation_prompt = f"""
            Question: {question}
            Result: {result}
            
            Explain this analysis in simple, clear terms:
            """
            explanation = self._generate_text(explanation_prompt, max_length=300)
            
            return {
                "success": True,
                "answer": result,
                "explanation": explanation,
                "code": code,
                "data_profile": profile
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Main Streamlit app
def main():
    st.title("📊 CSV Analyzer AI")
    st.markdown("Upload your CSV file and ask questions about your data! 🤖")
    
    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI-powered app can:
        - 📈 Analyze your CSV data automatically
        - 🔍 Generate insights and answers
        - 💡 Explain results in simple terms
        - 🔒 Run securely in your browser
        
        **How to use:**
        1. Upload CSV file
        2. Ask questions about your data
        3. Get AI-powered analysis
        """)
    
    # Initialize analyzer
    try:
        analyzer = CSVAnalyzerAI()
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
        with st.expander("📋 View Data Preview", expanded=True):
            tab1, tab2 = st.tabs(["First 10 Rows", "Data Info"])
            with tab1:
                st.dataframe(df.head(10), use_container_width=True)
            with tab2:
                st.write("**Column Information:**")
                for col in df.columns:
                    st.write(f"- **{col}**: {df[col].dtype} ({(df[col].isna().sum())} missing)")
        
        # Question section
        st.header("3. Ask Questions")
        
        # Quick question suggestions
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        quick_questions = []
        
        if numeric_cols:
            quick_questions.extend([
                f"What is the average {numeric_cols[0]}?",
                f"What is the maximum and minimum {numeric_cols[0]}?",
                f"Show the distribution of {numeric_cols[0]}"
            ])
        
        if categorical_cols:
            quick_questions.extend([
                f"What is the most common {categorical_cols[0]}?",
                f"Show the count of each {categorical_cols[0]}"
            ])
        
        if len(numeric_cols) >= 2:
            quick_questions.append(f"Is there correlation between {numeric_cols[0]} and {numeric_cols[1]}?")
        
        quick_questions.extend([
            "What are the main insights from this data?",
            "Are there any missing values in the data?",
            "What patterns can you find in this data?"
        ])
        
        # Question input
        selected_question = st.selectbox(
            "Choose a quick question:",
            [""] + quick_questions,
            help="Select a pre-defined question or type your own below"
        )
        
        custom_question = st.text_input(
            "Or type your own question:",
            placeholder="e.g., What is the average salary by department? Who has the highest sales?",
            help="Ask any question about your data"
        )
        
        # Use selected question or custom question
        question = custom_question if custom_question else selected_question
        
        if st.button("🚀 Analyze Data", type="primary", use_container_width=True) and question:
            with st.spinner("🤖 AI is analyzing your data. This may take a few seconds..."):
                result = analyzer.analyze_data(df, question)
            
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
                    st.subheader("AI Explanation")
                    st.write(result["explanation"])
                
                with tab3:
                    st.subheader("Generated Analysis Code")
                    st.code(result["code"], language="python")
                    st.info("💡 This code was automatically generated by the AI to answer your question.")
            
            else:
                st.error(f"❌ Analysis failed: {result['error']}")
                st.info("💡 Try rephrasing your question or check if your data supports this analysis.")
    
    else:
        # Welcome message when no file is uploaded
        st.info("👆 Please upload a CSV file to get started with data analysis!")
        
        # Sample data option
        if st.button("🎯 Try with Sample Data"):
            sample_data = {
                'Employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
                'Age': [25, 30, 35, 28, 32],
                'Salary': [50000, 60000, 70000, 55000, 65000],
                'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
                'Experience_Years': [2, 5, 8, 3, 6]
            }
            df = pd.DataFrame(sample_data)
            st.session_state.sample_data = df
            st.rerun()

if __name__ == "__main__":
    main()
