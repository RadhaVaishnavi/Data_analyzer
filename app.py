
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

load_dotenv()

class StreamlitCSVAnalyzer:
    def __init__(self):
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.model_name = "microsoft/DialoGPT-medium"
        self.tokenizer, self.model = self._load_model()
        self.safe_imports = {'pd': pd, 'np': __import__('numpy')}
    
    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model
    
    def _generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=300, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
    
    def analyze(self, df, question: str):
        try:
            # Simple data profile
            profile = {
                "columns": list(df.columns),
                "sample": df.head(2).to_dict('records')
            }
            
            # Generate code
            prompt = f"Data: {profile}. Question: {question}. Generate pandas code:"
            code = self._generate_text(prompt)
            
            # Execute
            exec_globals = {**self.safe_imports, 'df': df}
            exec_locals = {}
            exec(code, exec_globals, exec_locals)
            result = exec_locals.get('result', 'Done')
            
            # Explain
            explanation = self._generate_text(f"Explain this analysis: {question}. Result: {result}")
            
            return {
                "success": True,
                "result": result,
                "explanation": explanation,
                "code": code
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# Streamlit UI
st.set_page_config(page_title="CSV Analyzer", page_icon="📊")

st.title("📊 CSV Data Analyzer")
st.write("Upload a CSV file and ask questions!")

# Initialize agent
@st.cache_resource
def load_agent():
    return StreamlitCSVAnalyzer()

agent = load_agent()

uploaded_file = st.file_uploader("Choose CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(df.head())
    
    question = st.text_input("Ask a question about your data:")
    
    if question:
        with st.spinner("Analyzing..."):
            result = agent.analyze(df, question)
        
        if result["success"]:
            st.success("Analysis complete!")
            st.write("**Result:**", result["result"])
            st.write("**Explanation:**", result["explanation"])
            with st.expander("View generated code"):
                st.code(result["code"])
        else:
            st.error(f"Error: {result['error']}")
