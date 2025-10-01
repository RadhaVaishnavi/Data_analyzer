import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import io

# Configure the page
st.set_page_config(
    page_title="Smart CSV Analyzer",
    page_icon="📊",
    layout="wide"
)

class SmartCSVAnalyzer:
    def __init__(self):
        self.analysis_templates = self._load_analysis_templates()
    
    def _load_analysis_templates(self):
        """Pre-defined analysis templates for reliable execution"""
        return {
            'data_quality': {
                'missing_values': self._analyze_missing_values,
                'data_types': self._analyze_data_types,
                'duplicates': self._analyze_duplicates,
                'validity': self._analyze_validity,
                'outliers': self._analyze_outliers
            },
            'statistical': {
                'basic_stats': self._basic_statistics,
                'correlations': self._analyze_correlations,
                'distributions': self._analyze_distributions
            },
            'business': {
                'purpose': self._analyze_purpose,
                'patterns': self._analyze_patterns,
                'insights': self._generate_insights
            }
        }
    
    def analyze_question(self, df, question):
        """Main analysis function with reliable template matching"""
        try:
            question_lower = question.lower()
            
            # Map questions to analysis functions
            if any(word in question_lower for word in ['missing', 'null', 'nan']):
                return self._analyze_missing_values(df)
            
            elif any(word in question_lower for word in ['validity', 'rules', 'conform', 'valid']):
                return self._analyze_validity(df)
            
            elif any(word in question_lower for word in ['purpose', 'use case', 'intended']):
                return self._analyze_purpose(df)
            
            elif any(word in question_lower for word in ['duplicate', 'duplicates']):
                return self._analyze_duplicates(df)
            
            elif any(word in question_lower for word in ['correlation', 'relationship']):
                return self._analyze_correlations(df)
            
            elif any(word in question_lower for word in ['outlier', 'anomaly']):
                return self._analyze_outliers(df)
            
            elif any(word in question_lower for word in ['distribution', 'histogram']):
                return self._analyze_distributions(df)
            
            elif any(word in question_lower for word in ['statistic', 'summary', 'describe']):
                return self._basic_statistics(df)
            
            elif any(word in question_lower for word in ['pattern', 'insight', 'trend']):
                return self._generate_insights(df)
            
            elif any(word in question_lower for word in ['size', 'shape', 'dimension']):
                return self._analyze_dataset_size(df)
            
            elif any(word in question_lower for word in ['type', 'dtype', 'data type']):
                return self._analyze_data_types(df)
            
            else:
                # Default comprehensive analysis
                return self._comprehensive_analysis(df, question)
                
        except Exception as e:
            return self._error_response(f"Analysis error: {str(e)}")
    
    def _analyze_missing_values(self, df):
        """Analyze missing values comprehensively"""
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
            labels={'x': 'Columns', 'y': 'Missing Count'}
        )
        fig.update_layout(showlegend=False)
        
        return {
            "success": True,
            "answer": result_df,
            "explanation": f"Missing values analysis: {missing_data.sum()} total missing values across {len(df.columns)} columns. Columns with >20% missing data may need special attention.",
            "visualization": fig,
            "code": """
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
result = pd.DataFrame({
    'missing_count': missing_data,
    'missing_percentage': missing_percentage.round(2),
    'data_type': df.dtypes.astype(str)
})
"""
        }
    
    def _analyze_validity(self, df):
        """Analyze data validity and rule compliance"""
        validity_checks = {}
        
        for col in df.columns:
            col_checks = {
                'total_count': len(df),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2),
                'unique_count': df[col].nunique(),
                'data_type': str(df[col].dtype),
                'sample_values': df[col].dropna().head(3).tolist() if df[col].dtype == 'object' else 'N/A'
            }
            
            # Additional checks based on data type
            if pd.api.types.is_numeric_dtype(df[col]):
                col_checks.update({
                    'min_value': df[col].min(),
                    'max_value': df[col].max(),
                    'mean_value': df[col].mean(),
                    'has_negative': (df[col] < 0).any()
                })
            
            validity_checks[col] = col_checks
        
        result_df = pd.DataFrame(validity_checks).T
        
        return {
            "success": True,
            "answer": result_df,
            "explanation": "Data validity assessment: Checks completeness, uniqueness, and basic data integrity rules for each column. Look for high null percentages or unexpected data types.",
            "code": """
validity_checks = {}
for col in df.columns:
    col_checks = {
        'total_count': len(df),
        'non_null_count': df[col].count(),
        'null_count': df[col].isnull().sum(),
        'null_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2),
        'unique_count': df[col].nunique(),
        'data_type': str(df[col].dtype)
    }
    if pd.api.types.is_numeric_dtype(df[col]):
        col_checks.update({
            'min_value': df[col].min(),
            'max_value': df[col].max(),
            'mean_value': df[col].mean()
        })
    validity_checks[col] = col_checks
result = pd.DataFrame(validity_checks).T
"""
        }
    
    def _analyze_purpose(self, df):
        """Analyze dataset purpose and potential use cases"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Determine potential use cases
        use_cases = []
        
        if len(numeric_cols) >= 2:
            use_cases.append("Predictive modeling and regression analysis")
        if len(categorical_cols) >= 1:
            use_cases.append("Classification and segmentation analysis")
        if len(date_cols) >= 1:
            use_cases.append("Time series analysis and forecasting")
        if len(numeric_cols) >= 3:
            use_cases.append("Multivariate analysis and dimensionality reduction")
        
        if not use_cases:
            use_cases = ["General data analysis and exploration"]
        
        purpose_analysis = {
            "dataset_shape": df.shape,
            "total_records": len(df),
            "total_features": len(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": date_cols,
            "suggested_use_cases": use_cases,
            "data_complexity": "High" if len(numeric_cols) > 5 else "Medium" if len(numeric_cols) > 2 else "Low"
        }
        
        return {
            "success": True,
            "answer": purpose_analysis,
            "explanation": f"Dataset purpose analysis: This {purpose_analysis['data_complexity'].lower()} complexity dataset with {len(df.columns)} features suggests use cases like {', '.join(use_cases[:2])}.",
            "code": """
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
purpose_analysis = {
    "dataset_shape": df.shape,
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "suggested_use_cases": ["Data analysis", "Machine learning", "Business intelligence"]
}
result = purpose_analysis
"""
        }
    
    def _analyze_correlations(self, df):
        """Analyze correlations between numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return self._error_response("Need at least 2 numeric columns for correlation analysis")
        
        correlation_matrix = numeric_df.corr()
        
        # Create heatmap visualization
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Matrix Heatmap",
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'feature_1': correlation_matrix.columns[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': round(corr_value, 3)
                    })
        
        analysis_summary = {
            'correlation_matrix': correlation_matrix,
            'strong_correlations': strong_correlations,
            'matrix_shape': correlation_matrix.shape
        }
        
        return {
            "success": True,
            "answer": analysis_summary,
            "explanation": f"Correlation analysis: Found {len(strong_correlations)} strong correlations (|r| > 0.7). Values close to 1/-1 indicate strong positive/negative relationships.",
            "visualization": fig,
            "code": """
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
result = {
    'correlation_matrix': correlation_matrix,
    'strong_correlations': []  # Would be calculated separately
}
"""
        }
    
    def _analyze_duplicates(self, df):
        """Analyze duplicate records"""
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        analysis_result = {
            'total_records': len(df),
            'duplicate_count': duplicate_count,
            'duplicate_percentage': round(duplicate_percentage, 2),
            'unique_records': len(df) - duplicate_count
        }
        
        return {
            "success": True,
            "answer": analysis_result,
            "explanation": f"Duplicate analysis: {duplicate_count} duplicate records ({duplicate_percentage:.1f}% of total). High duplication rates may indicate data collection issues.",
            "code": "result = {'duplicate_count': df.duplicated().sum(), 'total_records': len(df)}"
        }
    
    def _basic_statistics(self, df):
        """Generate comprehensive basic statistics"""
        stats_summary = {
            'dataset_shape': df.shape,
            'column_stats': {},
            'overall_metrics': {
                'total_missing': df.isnull().sum().sum(),
                'total_duplicates': df.duplicated().sum(),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            }
        }
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats_summary['column_stats'][col] = {
                    'type': 'numeric',
                    'stats': df[col].describe().to_dict(),
                    'missing': df[col].isnull().sum()
                }
            else:
                stats_summary['column_stats'][col] = {
                    'type': 'categorical',
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].empty else 'N/A',
                    'missing': df[col].isnull().sum()
                }
        
        return {
            "success": True,
            "answer": stats_summary,
            "explanation": f"Basic statistics: Dataset with {df.shape[0]} rows, {df.shape[1]} columns. Total memory usage: {stats_summary['overall_metrics']['memory_usage_mb']} MB.",
            "code": """
stats_summary = {
    'dataset_shape': df.shape,
    'numeric_stats': df.describe().to_dict(),
    'missing_values': df.isnull().sum().to_dict()
}
result = stats_summary
"""
        }
    
    def _comprehensive_analysis(self, df, question):
        """Fallback comprehensive analysis"""
        analysis_result = {
            'question_asked': question,
            'dataset_overview': {
                'shape': df.shape,
                'columns': list(df.columns),
                'data_types': df.dtypes.astype(str).to_dict()
            },
            'data_quality': {
                'total_missing': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'recommendations': [
                "Use specific questions for detailed analysis",
                "Try asking about correlations, distributions, or data quality",
                "Upload larger datasets for more comprehensive insights"
            ]
        }
        
        return {
            "success": True,
            "answer": analysis_result,
            "explanation": f"Comprehensive analysis for: '{question}'. For more specific insights, try asking about particular columns or data aspects.",
            "code": "# General dataset analysis performed"
        }
    
    def _error_response(self, error_message):
        """Standard error response"""
        return {
            "success": False,
            "error": error_message
        }

# Main Streamlit App
def main():
    st.title("📊 Smart CSV Analyzer")
    st.markdown("Upload your CSV and get instant, reliable data analysis! ⚡")
    
    # Initialize analyzer
    analyzer = SmartCSVAnalyzer()
    
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
        st.header("2. Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Quick data preview
        with st.expander("📋 Data Preview", expanded=False):
            tab1, tab2 = st.tabs(["First 10 Rows", "Column Info"])
            with tab1:
                st.dataframe(df.head(10), use_container_width=True)
            with tab2:
                for col in df.columns:
                    missing = df[col].isna().sum()
                    dtype = df[col].dtype
                    unique = df[col].nunique()
                    st.write(f"**{col}**: {dtype} | {missing} missing | {unique} unique values")
        
        # Question section
        st.header("3. Ask Analysis Questions")
        
        # Categorized questions
        analysis_categories = {
            "📈 Data Quality": [
                "How many missing/null values per column?",
                "Conforms to rules (validity)?",
                "Are there duplicates?",
                "Data types overview"
            ],
            "🔍 Statistical Analysis": [
                "Basic statistics summary",
                "Correlations between numeric columns",
                "Data distributions",
                "Outliers detection"
            ],
            "💼 Business Insights": [
                "What's the dataset's purpose or intended use case?",
                "Key patterns and insights",
                "Data freshness and relevance",
                "Integration potential"
            ]
        }
        
        # Category selection
        selected_category = st.selectbox(
            "Choose analysis category:",
            list(analysis_categories.keys())
        )
        
        # Question selection
        selected_question = st.selectbox(
            "Choose a question:",
            [""] + analysis_categories[selected_category]
        )
        
        # Custom question
        custom_question = st.text_input(
            "Or type your own question:",
            placeholder="e.g., Show relationship between age and salary"
        )
        
        # Use selected or custom question
        question = custom_question if custom_question else selected_question
        
        # Quick analysis buttons
        st.header("4. Quick Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Data Quality", use_container_width=True):
                question = "Conforms to rules (validity)?"
        
        with col2:
            if st.button("📈 Correlations", use_container_width=True):
                question = "Correlations between numeric columns"
        
        with col3:
            if st.button("📊 Statistics", use_container_width=True):
                question = "Basic statistics summary"
        
        with col4:
            if st.button("🎯 Purpose", use_container_width=True):
                question = "What's the dataset's purpose or intended use case?"
        
        # Perform analysis
        if question and st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("⚡ Analyzing your data..."):
                result = analyzer.analyze_question(df, question)
            
            if result["success"]:
                st.success("✅ Analysis Complete!")
                
                # Display results
                tab1, tab2, tab3 = st.tabs(["📊 Results", "💡 Explanation", "🔧 Code"])
                
                with tab1:
                    st.subheader("Analysis Results")
                    answer = result["answer"]
                    
                    if isinstance(answer, pd.DataFrame):
                        st.dataframe(answer, use_container_width=True)
                    elif isinstance(answer, dict):
                        # Pretty print dictionaries
                        for key, value in answer.items():
                            with st.expander(f"**{key}**", expanded=False):
                                if isinstance(value, (dict, list)):
                                    st.json(value)
                                else:
                                    st.write(value)
                    else:
                        st.write(answer)
                    
                    # Show visualization if available
                    if "visualization" in result:
                        st.plotly_chart(result["visualization"], use_container_width=True)
                
                with tab2:
                    st.subheader("Explanation")
                    st.write(result["explanation"])
                
                with tab3:
                    st.subheader("Analysis Code")
                    st.code(result["code"], language="python")
            
            else:
                st.error(f"❌ Analysis failed: {result['error']}")
    
    else:
        # Welcome message
        st.info("👆 Please upload a CSV file to start analyzing your data!")
        
        # Sample data option
        if st.button("🎯 Try with Sample Data"):
            sample_data = {
                'Employee_ID': [101, 102, 103, 104, 105, 106, 107],
                'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace'],
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
