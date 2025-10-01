import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.express as px
import re
import json

# Configure the page
st.set_page_config(
    page_title="AI CSV Analyzer Agent",
    page_icon="🤖",
    layout="wide"
)

class WorkingCSVAnalyzer:
    def __init__(self):
        self.analysis_history = []
    
    def analyze_question(self, df, question):
        """Main analysis function that actually works"""
        try:
            question_lower = question.lower()
            
            # Handle duplicates question specifically
            if any(word in question_lower for word in ['duplicate', 'duplicates']):
                return self._analyze_duplicates(df)
            
            # Handle missing values
            elif any(word in question_lower for word in ['missing', 'null', 'nan']):
                return self._analyze_missing_values(df)
            
            # Handle data types
            elif any(word in question_lower for word in ['data type', 'dtype', 'type']):
                return self._analyze_data_types(df)
            
            # Handle correlations
            elif any(word in question_lower for word in ['correlation', 'relationship']):
                return self._analyze_correlations(df)
            
            # Handle basic statistics
            elif any(word in question_lower for word in ['statistic', 'summary', 'describe']):
                return self._basic_statistics(df)
            
            # Handle purpose analysis
            elif any(word in question_lower for word in ['purpose', 'use case']):
                return self._analyze_purpose(df)
            
            # Handle validity checks
            elif any(word in question_lower for word in ['valid', 'validity', 'rule']):
                return self._analyze_validity(df)
            
            # Handle outliers
            elif any(word in question_lower for word in ['outlier', 'anomaly']):
                return self._analyze_outliers(df)
            
            # Handle distributions
            elif any(word in question_lower for word in ['distribution', 'histogram']):
                return self._analyze_distributions(df)
            
            # Handle patterns
            elif any(word in question_lower for word in ['pattern', 'insight']):
                return self._analyze_patterns(df)
            
            # Default comprehensive analysis
            else:
                return self._comprehensive_analysis(df, question)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis error: {str(e)}"
            }
    
    def _analyze_duplicates(self, df):
        """Analyze duplicate records - FIXED VERSION"""
        try:
            # Count duplicates
            duplicate_count = df.duplicated().sum()
            duplicate_percentage = (duplicate_count / len(df)) * 100
            
            # Get duplicate rows for inspection
            duplicate_rows = df[df.duplicated(keep=False)]
            
            result = {
                'total_records': len(df),
                'duplicate_count': duplicate_count,
                'duplicate_percentage': round(duplicate_percentage, 2),
                'unique_records': len(df) - duplicate_count,
                'duplicate_rows_sample': duplicate_rows.head(5).to_dict('records') if duplicate_count > 0 else []
            }
            
            # Create visualization if duplicates exist
            if duplicate_count > 0:
                fig = px.bar(
                    x=['Unique Records', 'Duplicate Records'],
                    y=[len(df) - duplicate_count, duplicate_count],
                    title="Duplicate Records Analysis",
                    color=['Unique', 'Duplicate'],
                    labels={'x': 'Record Type', 'y': 'Count'}
                )
                visualization = fig
            else:
                visualization = None
            
            explanation = f"Duplicate analysis: Found {duplicate_count} duplicate records ({duplicate_percentage:.1f}% of total). "
            if duplicate_count > 0:
                explanation += "Consider removing duplicates for accurate analysis."
            else:
                explanation += "No duplicate records found - data is clean!"
            
            code = """
# Check for duplicate rows
duplicate_count = df.duplicated().sum()
duplicate_rows = df[df.duplicated(keep=False)]

result = {
    'total_records': len(df),
    'duplicate_count': duplicate_count,
    'duplicate_percentage': round((duplicate_count / len(df)) * 100, 2),
    'unique_records': len(df) - duplicate_count
}
"""
            
            return {
                "success": True,
                "answer": result,
                "explanation": explanation,
                "code": code,
                "visualization": visualization
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Duplicate analysis failed: {str(e)}"
            }
    
    def _analyze_missing_values(self, df):
        """Analyze missing values"""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        result_df = pd.DataFrame({
            'missing_count': missing_data,
            'missing_percentage': missing_percentage.round(2),
            'data_type': df.dtypes.astype(str)
        })
        
        # Create visualization
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Values by Column",
            labels={'x': 'Columns', 'y': 'Missing Count'},
            color=missing_data.values
        )
        
        explanation = f"Missing values analysis: {missing_data.sum()} total missing values across {len(df.columns)} columns."
        
        code = """
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
result = pd.DataFrame({
    'missing_count': missing_data,
    'missing_percentage': missing_percentage.round(2),
    'data_type': df.dtypes.astype(str)
})
"""
        
        return {
            "success": True,
            "answer": result_df.to_dict(),
            "explanation": explanation,
            "code": code,
            "visualization": fig
        }
    
    def _analyze_data_types(self, df):
        """Analyze data types"""
        type_analysis = {}
        for col in df.columns:
            type_analysis[col] = {
                'data_type': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            }
        
        explanation = f"Data types analysis: Found {len(df.columns)} columns with various data types."
        
        code = """
type_analysis = {}
for col in df.columns:
    type_analysis[col] = {
        'data_type': str(df[col].dtype),
        'non_null_count': df[col].count(),
        'null_count': df[col].isnull().sum(),
        'unique_count': df[col].nunique()
    }
result = type_analysis
"""
        
        return {
            "success": True,
            "answer": type_analysis,
            "explanation": explanation,
            "code": code
        }
    
    def _analyze_correlations(self, df):
        """Analyze correlations"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {
                "success": False,
                "error": "Need at least 2 numeric columns for correlation analysis"
            }
        
        correlation_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Matrix",
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        
        explanation = "Correlation analysis shows relationships between numeric variables."
        
        code = "result = df.corr()"
        
        return {
            "success": True,
            "answer": correlation_matrix.to_dict(),
            "explanation": explanation,
            "code": code,
            "visualization": fig
        }
    
    def _basic_statistics(self, df):
        """Basic statistics"""
        stats = df.describe(include='all').to_dict()
        
        explanation = "Basic statistical summary of the dataset."
        
        code = "result = df.describe(include='all')"
        
        return {
            "success": True,
            "answer": stats,
            "explanation": explanation,
            "code": code
        }
    
    def _analyze_purpose(self, df):
        """Analyze dataset purpose"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        purpose_analysis = {
            "dataset_shape": df.shape,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "suggested_use_cases": [
                "Data analysis and visualization",
                "Statistical modeling",
                "Business intelligence"
            ]
        }
        
        explanation = f"Dataset with {len(df.columns)} features suitable for various analytical purposes."
        
        code = """
purpose_analysis = {
    "dataset_shape": df.shape,
    "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
    "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
}
result = purpose_analysis
"""
        
        return {
            "success": True,
            "answer": purpose_analysis,
            "explanation": explanation,
            "code": code
        }
    
    def _analyze_validity(self, df):
        """Analyze data validity"""
        validity_checks = {}
        for col in df.columns:
            validity_checks[col] = {
                'total_count': len(df),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2),
                'unique_count': df[col].nunique(),
                'data_type': str(df[col].dtype)
            }
        
        explanation = "Data validity assessment across all columns."
        
        code = """
validity_checks = {}
for col in df.columns:
    validity_checks[col] = {
        'non_null_count': df[col].count(),
        'null_count': df[col].isnull().sum(),
        'unique_count': df[col].nunique(),
        'data_type': str(df[col].dtype)
    }
result = validity_checks
"""
        
        return {
            "success": True,
            "answer": validity_checks,
            "explanation": explanation,
            "code": code
        }
    
    def _analyze_outliers(self, df):
        """Analyze outliers"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            outlier_analysis[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': round((len(outliers) / len(df)) * 100, 2)
            }
        
        explanation = "Outlier detection using IQR method."
        
        code = """
outlier_analysis = {}
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_analysis[col] = len(outliers)
result = outlier_analysis
"""
        
        return {
            "success": True,
            "answer": outlier_analysis,
            "explanation": explanation,
            "code": code
        }
    
    def _analyze_distributions(self, df):
        """Analyze distributions"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {
                "success": False,
                "error": "No numeric columns for distribution analysis"
            }
        
        # Create histograms for first 3 numeric columns
        figs = []
        for col in numeric_cols[:3]:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            figs.append(fig)
        
        explanation = "Distribution analysis of numeric columns."
        
        return {
            "success": True,
            "answer": {"columns_analyzed": numeric_cols.tolist()},
            "explanation": explanation,
            "code": "# Use px.histogram(df, x='column_name') for distribution analysis",
            "visualization": figs[0] if figs else None
        }
    
    def _analyze_patterns(self, df):
        """Analyze patterns"""
        patterns = {
            'dataset_size': df.shape,
            'missing_values_total': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns_count': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns_count': len(df.select_dtypes(include=['object']).columns)
        }
        
        explanation = "Pattern analysis revealing dataset characteristics."
        
        code = """
patterns = {
    'dataset_size': df.shape,
    'missing_values_total': df.isnull().sum().sum(),
    'duplicate_rows': df.duplicated().sum()
}
result = patterns
"""
        
        return {
            "success": True,
            "answer": patterns,
            "explanation": explanation,
            "code": code
        }
    
    def _comprehensive_analysis(self, df, question):
        """Comprehensive analysis fallback"""
        analysis = {
            'question': question,
            'dataset_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'data_types': df.dtypes.astype(str).to_dict()
            },
            'data_quality': {
                'total_missing': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum()
            }
        }
        
        explanation = f"Comprehensive analysis for: {question}"
        
        code = "# Comprehensive dataset analysis"
        
        return {
            "success": True,
            "answer": analysis,
            "explanation": explanation,
            "code": code
        }

# Main Streamlit App
def main():
    st.title("📊 Smart CSV Analyzer")
    st.markdown("Upload your CSV and get instant, reliable analysis! ⚡")
    
    # Initialize analyzer
    analyzer = WorkingCSVAnalyzer()
    
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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(10))
        
        st.header("3. Ask Analysis Questions")
        
        # Quick action buttons
        st.subheader("Quick Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Check Duplicates", use_container_width=True):
                st.session_state.quick_question = "r there any duplicates?"
        
        with col2:
            if st.button("📊 Missing Values", use_container_width=True):
                st.session_state.quick_question = "how many missing values?"
        
        with col3:
            if st.button("📈 Basic Stats", use_container_width=True):
                st.session_state.quick_question = "show basic statistics"
        
        with col4:
            if st.button("🎯 Data Types", use_container_width=True):
                st.session_state.quick_question = "what are the data types?"
        
        # Question input
        question = st.text_input(
            "Or ask your own question:",
            placeholder="e.g., r there any duplicates? how many missing values? show correlations...",
            key="question_input"
        )
        
        # Use quick question if set
        if hasattr(st.session_state, 'quick_question'):
            question = st.session_state.quick_question
            # Clear it after use
            del st.session_state.quick_question
        
        if question and st.button("🚀 Analyze", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing your data..."):
                result = analyzer.analyze_question(df, question)
            
            if result["success"]:
                st.success("✅ Analysis Complete!")
                
                tabs = st.tabs(["📊 Results", "💡 Explanation", "🔧 Code"])
                
                with tabs[0]:
                    st.subheader("Analysis Results")
                    answer = result["answer"]
                    
                    if isinstance(answer, dict):
                        st.json(answer)
                    else:
                        st.write(answer)
                    
                    # Show visualization if available
                    if "visualization" in result and result["visualization"]:
                        st.plotly_chart(result["visualization"], use_container_width=True)
                
                with tabs[1]:
                    st.subheader("Explanation")
                    st.write(result["explanation"])
                
                with tabs[2]:
                    st.subheader("Analysis Code")
                    st.code(result["code"], language="python")
            
            else:
                st.error(f"❌ {result['error']}")
    
    else:
        st.info("👆 Upload a CSV file to start analyzing!")
        
        # Sample data
        if st.button("🎯 Try Sample Data"):
            sample_data = {
                'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Eve', 'Frank', 'Grace', 'Bob', 'Ivy', 'Jack'],
                'Age': [25, 30, 35, 25, 28, 32, 29, 30, 27, 31],
                'Salary': [50000, 60000, 70000, 50000, 55000, 65000, 58000, 60000, 52000, 62000],
                'Department': ['IT', 'HR', 'IT', 'IT', 'Finance', 'HR', 'IT', 'HR', 'Finance', 'IT']
            }
            df = pd.DataFrame(sample_data)
            st.session_state.sample_data = df
            st.rerun()

if __name__ == "__main__":
    main()
