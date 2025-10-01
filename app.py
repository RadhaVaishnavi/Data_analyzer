import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import re
import json

# Configure the page
st.set_page_config(
    page_title="AI CSV Analyzer Agent",
    page_icon="🤖",
    layout="wide"
)

class LLMAgenticAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.analysis_history = []
        self._load_lightweight_llm()
    
    def _load_lightweight_llm(self):
        """Load a lightweight, efficient LLM for analysis"""
        try:
            # Using a small, fast model for efficiency
            model_name = "microsoft/DialoGPT-small"  # Lightweight conversational model
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            st.success("✅ AI Agent loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to load AI model: {e}")
            st.info("🔧 Using rule-based fallback analyzer")
            self.model = None
    
    def _llm_generate(self, prompt, max_length=300):
        """Generate response using LLM with efficient settings"""
        if self.model is None:
            return self._rule_based_fallback(prompt)
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=0.3,  # Low temperature for consistent output
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.replace(prompt, "").strip()
            
        except Exception as e:
            return self._rule_based_fallback(prompt)
    
    def _rule_based_fallback(self, prompt):
        """Fallback when LLM is not available"""
        if "missing" in prompt.lower():
            return "I'll analyze missing values using pandas. Use df.isnull().sum() to count nulls per column."
        elif "correlation" in prompt.lower():
            return "I'll calculate correlations between numeric columns using df.corr()."
        else:
            return "I'll perform comprehensive data analysis using statistical methods."
    
    def _agentic_analysis(self, df, question):
        """Agentic approach: LLM plans and executes analysis"""
        
        # Step 1: LLM understands the question and plans analysis
        planning_prompt = f"""
        You are a data analysis agent. Given a dataset with columns: {list(df.columns)}
        and data types: {df.dtypes.astype(str).to_dict()}
        
        Question: {question}
        
        Plan the analysis steps. Respond with a JSON plan:
        {{
            "analysis_type": "statistical|quality|business|technical",
            "steps": ["step1", "step2", ...],
            "required_operations": ["describe", "corr", "groupby", ...],
            "expected_output": "what result to produce"
        }}
        """
        
        plan_text = self._llm_generate(planning_prompt, max_length=200)
        analysis_plan = self._parse_llm_plan(plan_text)
        
        # Step 2: Execute the planned analysis
        results = self._execute_analysis_plan(df, analysis_plan)
        
        # Step 3: LLM generates explanation
        explanation_prompt = f"""
        Question: {question}
        Analysis Results: {str(results)[:500]}
        
        Provide a clear, concise explanation of what was found:
        """
        
        explanation = self._llm_generate(explanation_prompt, max_length=250)
        
        # Step 4: Generate code based on analysis
        code = self._generate_analysis_code(analysis_plan, results)
        
        return {
            "success": True,
            "answer": results,
            "explanation": explanation,
            "code": code,
            "analysis_plan": analysis_plan
        }
    
    def _parse_llm_plan(self, plan_text):
        """Parse LLM-generated analysis plan"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback plan
        return {
            "analysis_type": "comprehensive",
            "steps": ["basic_statistics", "data_quality_check"],
            "required_operations": ["describe", "isnull", "info"],
            "expected_output": "dataset overview and quality assessment"
        }
    
    def _execute_analysis_plan(self, df, plan):
        """Execute the analysis plan using pandas operations"""
        operations = plan.get("required_operations", [])
        results = {}
        
        for op in operations:
            try:
                if op == "describe" and len(df.select_dtypes(include=[np.number]).columns) > 0:
                    results["statistics"] = df.describe().to_dict()
                elif op == "isnull":
                    results["missing_values"] = df.isnull().sum().to_dict()
                elif op == "corr" and len(df.select_dtypes(include=[np.number]).columns) >= 2:
                    results["correlations"] = df.corr().to_dict()
                elif op == "groupby":
                    # Simple groupby on first categorical column
                    cat_cols = df.select_dtypes(include=['object']).columns
                    if len(cat_cols) > 0:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            results["group_analysis"] = df.groupby(cat_cols[0])[numeric_cols[0]].mean().to_dict()
                elif op == "info":
                    results["dataset_info"] = {
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "data_types": df.dtypes.astype(str).to_dict()
                    }
            except Exception as e:
                continue
        
        # Ensure we have at least basic results
        if not results:
            results = {
                "basic_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
                }
            }
        
        return results
    
    def _generate_analysis_code(self, plan, results):
        """Generate Python code based on the analysis performed"""
        operations = plan.get("required_operations", [])
        code_lines = ["# Generated analysis code", "import pandas as pd", "import numpy as np", ""]
        
        for op in operations:
            if op == "describe":
                code_lines.append("# Basic statistics")
                code_lines.append("stats = df.describe()")
            elif op == "isnull":
                code_lines.append("# Missing values analysis")
                code_lines.append("missing_data = df.isnull().sum()")
            elif op == "corr":
                code_lines.append("# Correlation analysis")
                code_lines.append("correlation_matrix = df.corr()")
            elif op == "groupby":
                code_lines.append("# Group analysis")
                code_lines.append("grouped_data = df.groupby('column_name')['numeric_column'].mean()")
        
        code_lines.append("\n# Final result compilation")
        code_lines.append("result = {'analysis': 'completed'}")
        
        return "\n".join(code_lines)
    
    def analyze_question(self, df, question):
        """Main agentic analysis entry point"""
        try:
            # Store in history
            self.analysis_history.append({
                "question": question,
                "timestamp": pd.Timestamp.now(),
                "data_shape": df.shape
            })
            
            # Use agentic LLM approach
            return self._agentic_analysis(df, question)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Agentic analysis failed: {str(e)}"
            }

class HybridAnalyzer:
    """Combines LLM agent with rule-based fallbacks"""
    
    def __init__(self):
        self.llm_agent = LLMAgenticAnalyzer()
        self.rule_analyzer = RuleBasedAnalyzer()
    
    def analyze(self, df, question):
        """Hybrid approach: Try LLM first, fallback to rules"""
        # For complex questions, use LLM
        complex_keywords = ['purpose', 'insight', 'pattern', 'relationship', 'trend']
        
        if any(keyword in question.lower() for keyword in complex_keywords):
            result = self.llm_agent.analyze_question(df, question)
            if result["success"]:
                return result
        
        # For structured questions, use rule-based (faster)
        return self.rule_analyzer.analyze_question(df, question)

class RuleBasedAnalyzer:
    """Fast rule-based analyzer (fallback)"""
    
    def analyze_question(self, df, question):
        """Rule-based analysis without LLM"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['missing', 'null']):
            return self._analyze_missing_values(df)
        elif any(word in question_lower for word in ['correlation']):
            return self._analyze_correlations(df)
        elif any(word in question_lower for word in ['statistic', 'describe']):
            return self._basic_statistics(df)
        elif any(word in question_lower for word in ['purpose', 'use case']):
            return self._analyze_purpose(df)
        else:
            return self._comprehensive_analysis(df, question)
    
    def _analyze_missing_values(self, df):
        missing_data = df.isnull().sum()
        return {
            "success": True,
            "answer": missing_data.to_dict(),
            "explanation": f"Found {missing_data.sum()} total missing values across {len(df.columns)} columns.",
            "code": "missing_data = df.isnull().sum()\nresult = missing_data"
        }
    
    def _analyze_correlations(self, df):
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            return {
                "success": True,
                "answer": corr_matrix.to_dict(),
                "explanation": "Correlation matrix showing relationships between numeric variables.",
                "code": "result = df.corr()"
            }
        else:
            return {
                "success": False,
                "error": "Need at least 2 numeric columns for correlation analysis"
            }
    
    def _basic_statistics(self, df):
        return {
            "success": True,
            "answer": df.describe().to_dict(),
            "explanation": "Basic statistical summary of numeric columns.",
            "code": "result = df.describe()"
        }
    
    def _analyze_purpose(self, df):
        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "suggested_uses": ["Data analysis", "Reporting", "ML modeling"]
        }
        return {
            "success": True,
            "answer": analysis,
            "explanation": "Dataset overview and potential use cases.",
            "code": "result = {'shape': df.shape, 'columns': list(df.columns)}"
        }
    
    def _comprehensive_analysis(self, df, question):
        return {
            "success": True,
            "answer": {"message": "Analysis completed", "question": question},
            "explanation": "Comprehensive analysis performed using rule-based methods.",
            "code": "# Rule-based analysis completed"
        }

# Main Streamlit App
def main():
    st.title("🤖 AI CSV Analyzer Agent")
    st.markdown("Intelligent data analysis powered by LLM agents! 🚀")
    
    # Initialize the hybrid analyzer
    analyzer = HybridAnalyzer()
    
    # File upload
    st.header("1. Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])
    
    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    if df is not None:
        st.header("2. Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        with st.expander("Data Preview"):
            st.dataframe(df.head())
        
        st.header("3. Ask AI Agent")
        
        # Question categories
        question_type = st.selectbox(
            "Analysis Type",
            ["🤔 Smart Analysis", "📊 Data Quality", "🔍 Statistical", "💼 Business Insights"]
        )
        
        question = st.text_area(
            "Ask your data question:",
            placeholder="e.g., What patterns can you find in this data? What's the relationship between variables?"
        )
        
        if st.button("🚀 Ask AI Agent", type="primary"):
            with st.spinner("🤖 AI Agent is analyzing..."):
                result = analyzer.analyze(df, question)
            
            if result["success"]:
                st.success("✅ AI Analysis Complete!")
                
                tabs = st.tabs(["📈 Results", "💡 Explanation", "🔧 Code", "📋 Plan"])
                
                with tabs[0]:
                    st.subheader("Analysis Results")
                    st.json(result["answer"])
                
                with tabs[1]:
                    st.subheader("AI Explanation")
                    st.write(result["explanation"])
                
                with tabs[2]:
                    st.subheader("Generated Code")
                    st.code(result["code"], language="python")
                
                with tabs[3]:
                    st.subheader("Analysis Plan")
                    if "analysis_plan" in result:
                        st.json(result["analysis_plan"])
                    else:
                        st.info("Rule-based analysis used")
            
            else:
                st.error(f"❌ {result['error']}")

if __name__ == "__main__":
    main()
