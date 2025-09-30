# comprehensive_data_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# Hugging Face imports
try:
    from transformers import pipeline
    from huggingface_hub import login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Comprehensive Data Analyzer",
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class ComprehensiveAnalyzer:
    def __init__(self):
        self.pipeline = None
        self.model_loaded = False
    
    def setup_huggingface(self, hf_token):
        """Setup Hugging Face with a reliable model"""
        try:
            if not HF_AVAILABLE:
                return False
            
            login(token=hf_token)
            
            # Use a model that's good at reasoning and analysis
            self.pipeline = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.2",
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                max_length=2048
            )
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    def comprehensive_analysis(self, df, filename):
        """Perform comprehensive analysis using the 21-question framework"""
        
        # Generate detailed analysis using all 21 questions
        analysis_results = self._analyze_all_questions(df, filename)
        
        # Use HF model to synthesize insights
        if self.model_loaded:
            hf_insights = self._get_hf_insights(analysis_results, filename)
            return analysis_results, hf_insights
        else:
            return analysis_results, self._synthesize_insights(analysis_results)
    
    def _analyze_all_questions(self, df, filename):
        """Analyze all 21 questions comprehensively"""
        analysis = {}
        
        # Q1: Purpose and Use Case
        analysis['purpose'] = self._analyze_purpose(df, filename)
        
        # Q2: Source and Provenance
        analysis['provenance'] = self._analyze_provenance(df, filename)
        
        # Q3: Documentation
        analysis['documentation'] = self._analyze_documentation(df)
        
        # Q4: Sensitive Information
        analysis['sensitivity'] = self._analyze_sensitivity(df)
        
        # Q5: Format and Structure
        analysis['structure'] = self._analyze_structure(df, filename)
        
        # Q6: Size and Growth
        analysis['size'] = self._analyze_size(df, filename)
        
        # Q7: Missing Values
        analysis['completeness'] = self._analyze_completeness(df)
        
        # Q8: Accuracy
        analysis['accuracy'] = self._analyze_accuracy(df)
        
        # Q9: Consistency
        analysis['consistency'] = self._analyze_consistency(df)
        
        # Q10: Validity
        analysis['validity'] = self._analyze_validity(df)
        
        # Q11: Uniqueness
        analysis['uniqueness'] = self._analyze_uniqueness(df)
        
        # Q12: Timeliness
        analysis['timeliness'] = self._analyze_timeliness(df)
        
        # Q13: Relevance
        analysis['relevance'] = self._analyze_relevance(df)
        
        # Q14: Integrity
        analysis['integrity'] = self._analyze_integrity(df)
        
        # Q15: Sampling Biases
        analysis['biases'] = self._analyze_biases(df)
        
        # Q16: Distributions and Outliers
        analysis['distributions'] = self._analyze_distributions(df)
        
        # Q17: Correlations and Patterns
        analysis['correlations'] = self._analyze_correlations(df)
        
        # Q18: Pipeline Integration
        analysis['pipeline'] = self._analyze_pipeline_integration(df)
        
        # Q19: Performance
        analysis['performance'] = self._analyze_performance(df)
        
        # Q20: Impact Assessment
        analysis['impact'] = self._analyze_impact(df)
        
        # Q21: Compute Resources
        analysis['resources'] = self._analyze_resources(df)
        
        return analysis
    
    def _analyze_purpose(self, df, filename):
        """Q1: Analyze dataset purpose and use cases"""
        analysis = {}
        
        # Detect potential use cases based on column patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        datetime_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'year', 'month'])]
        
        # Determine primary use case
        use_cases = []
        
        if len(datetime_cols) >= 1 and len(numeric_cols) >= 2:
            use_cases.append("TIME_SERIES_ANALYSIS")
            if len(numeric_cols) >= 5:
                use_cases.append("FORECASTING")
        
        if any('target' in col.lower() or 'label' in col.lower() for col in df.columns):
            use_cases.append("SUPERVISED_LEARNING")
        
        if len(categorical_cols) >= 3:
            use_cases.append("SEGMENTATION")
        
        if len(numeric_cols) >= 8:
            use_cases.append("FEATURE_ANALYSIS")
        
        # Detect domain
        domain_keywords = {
            'FINANCIAL': ['amount', 'price', 'cost', 'revenue', 'balance', 'transaction'],
            'RETAIL': ['product', 'customer', 'order', 'sales', 'inventory'],
            'HEALTHCARE': ['patient', 'diagnosis', 'treatment', 'medical'],
            'HR': ['employee', 'salary', 'department', 'performance'],
            'MARKETING': ['campaign', 'conversion', 'click', 'lead']
        }
        
        domain = 'GENERAL'
        column_text = ' '.join(df.columns).lower()
        for dom, keywords in domain_keywords.items():
            if any(keyword in column_text for keyword in keywords):
                domain = dom
                break
        
        analysis['primary_use_cases'] = use_cases if use_cases else ['EXPLORATORY_ANALYSIS']
        analysis['domain'] = domain
        analysis['suitability_score'] = self._calculate_suitability_score(df, use_cases)
        
        return analysis
    
    def _analyze_provenance(self, df, filename):
        """Q2: Analyze data source and provenance"""
        analysis = {}
        
        # Check for common data quality issues that indicate source problems
        issues = []
        
        # Check for mixed data types (common in scraped data)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column has mixed numeric and string values
                numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                if 0.1 < (numeric_count / len(df)) < 0.9:  # Mixed types
                    issues.append(f"Mixed data types in {col}")
        
        # Check for inconsistent formatting
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].str.contains(r'\s{2,}', na=False).any():  # Multiple spaces
                issues.append(f"Inconsistent spacing in {col}")
        
        analysis['quality_issues'] = issues
        analysis['recommended_actions'] = [
            "Validate against source system if available",
            "Check data collection methodology",
            "Implement data quality monitoring"
        ]
        
        return analysis
    
    def _analyze_documentation(self, df):
        """Q3: Analyze documentation needs"""
        analysis = {}
        
        # Infer schema and data dictionary
        data_dict = {}
        for col in df.columns:
            data_dict[col] = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'sample_values': df[col].dropna().head(3).tolist() if df[col].dtype == 'object' else None
            }
        
        analysis['inferred_schema'] = data_dict
        analysis['documentation_gaps'] = [
            "No formal data dictionary found",
            "Column descriptions missing",
            "Business definitions needed"
        ]
        
        return analysis
    
    def _analyze_sensitivity(self, df):
        """Q4: Analyze sensitive information"""
        analysis = {}
        
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b',
            'ssn': r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b'
        }
        
        sensitive_columns = []
        for col in df.select_dtypes(include=['object']).columns:
            col_text = ' '.join(df[col].dropna().astype(str))
            for pii_type, pattern in pii_patterns.items():
                if re.search(pattern, col_text):
                    sensitive_columns.append((col, pii_type))
                    break
        
        analysis['sensitive_columns'] = sensitive_columns
        analysis['risk_level'] = 'HIGH' if sensitive_columns else 'LOW'
        analysis['recommendations'] = [
            "Implement data masking for sensitive columns",
            "Review compliance requirements (GDPR, HIPAA)",
            "Consider anonymization techniques"
        ]
        
        return analysis
    
    def _analyze_structure(self, df, filename):
        """Q5: Analyze format and structure"""
        analysis = {}
        
        structure_issues = []
        
        # Check for nested structures
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].str.contains(r'\{.*\}', na=False).any():  # JSON-like
                structure_issues.append(f"Potential nested data in {col}")
            if df[col].str.contains(r'\[.*\]', na=False).any():  # Array-like
                structure_issues.append(f"Potential array data in {col}")
        
        analysis['format_issues'] = structure_issues
        analysis['recommended_format'] = 'Parquet' if len(df) > 10000 else 'CSV'
        analysis['normalization_needed'] = len(structure_issues) > 0
        
        return analysis
    
    def _analyze_size(self, df, filename):
        """Q6: Analyze size and growth projections"""
        analysis = {}
        
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        row_count = len(df)
        col_count = len(df.columns)
        
        analysis['current_size'] = {
            'rows': row_count,
            'columns': col_count,
            'memory_mb': memory_mb,
            'cells': row_count * col_count
        }
        
        # Growth projections (simple heuristic)
        if memory_mb > 100:
            growth_category = 'LARGE_SCALE'
            processing = 'DISTRIBUTED'
        elif memory_mb > 10:
            growth_category = 'MEDIUM_SCALE'
            processing = 'SINGLE_MACHINE_OPTIMIZED'
        else:
            growth_category = 'SMALL_SCALE'
            processing = 'SINGLE_MACHINE'
        
        analysis['growth_category'] = growth_category
        analysis['recommended_processing'] = processing
        
        return analysis
    
    def _analyze_completeness(self, df):
        """Q7: Analyze missing values"""
        analysis = {}
        
        missing_analysis = {}
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            if null_count > 0:
                missing_analysis[col] = {
                    'count': null_count,
                    'percentage': null_pct,
                    'severity': 'CRITICAL' if null_pct > 50 else 'HIGH' if null_pct > 20 else 'MEDIUM'
                }
        
        analysis['missing_values'] = missing_analysis
        analysis['completeness_score'] = 100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        
        # Imputation recommendations
        imputation_strategies = []
        for col, info in missing_analysis.items():
            if info['severity'] == 'CRITICAL':
                imputation_strategies.append(f"Consider dropping {col} (>50% missing)")
            elif df[col].dtype in ['float64', 'int64']:
                imputation_strategies.append(f"Use median imputation for {col}")
            else:
                imputation_strategies.append(f"Use mode imputation for {col}")
        
        analysis['imputation_recommendations'] = imputation_strategies
        
        return analysis
    
    # Continue with other analysis methods for remaining questions...
    # [The rest of the analysis methods would follow the same pattern]
    
    def _analyze_accuracy(self, df):
        """Q8: Analyze data accuracy"""
        return {"analysis": "Accuracy check based on value ranges and patterns"}
    
    def _analyze_consistency(self, df):
        """Q9: Analyze consistency"""
        return {"analysis": "Format and value consistency analysis"}
    
    # ... [Implement all remaining analysis methods]
    
    def _calculate_suitability_score(self, df, use_cases):
        """Calculate how suitable the data is for detected use cases"""
        score = 50  # Base score
        
        # Adjust based on data quality
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        score -= null_percentage * 0.5  # Penalize for nulls
        
        # Bonus for having clear use cases
        if use_cases:
            score += len(use_cases) * 10
        
        return max(0, min(100, score))
    
    def _get_hf_insights(self, analysis_results, filename):
        """Get insights from Hugging Face model"""
        try:
            # Create comprehensive prompt
            prompt = self._create_analysis_prompt(analysis_results, filename)
            
            response = self.pipeline(
                prompt,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False
            )
            
            return response[0]['generated_text']
            
        except Exception as e:
            return f"AI insights unavailable: {str(e)}"
    
    def _create_analysis_prompt(self, analysis_results, filename):
        """Create detailed prompt for HF model"""
        return f"""
        As a senior data engineer, provide a comprehensive analysis of this dataset:

        DATASET: {filename}
        
        KEY FINDINGS:
        - Purpose: {analysis_results['purpose']['primary_use_cases']}
        - Domain: {analysis_results['purpose']['domain']}
        - Suitability Score: {analysis_results['purpose']['suitability_score']}/100
        - Data Quality: {analysis_results['completeness']['completeness_score']:.1f}% complete
        - Size: {analysis_results['size']['current_size']['rows']} rows, {analysis_results['size']['current_size']['columns']} columns
        - Memory: {analysis_results['size']['current_size']['memory_mb']:.1f} MB
        
        Provide specific recommendations for:
        1. Data quality improvement
        2. Storage optimization
        3. Processing strategy
        4. Potential use cases
        5. Risk mitigation
        
        Be very specific and reference the actual data characteristics.
        """
    
    def _synthesize_insights(self, analysis_results):
        """Synthesize insights when HF model is not available"""
        insights = f"""
## 🎯 COMPREHENSIVE ANALYSIS RESULTS

### 📊 EXECUTIVE SUMMARY
- **Primary Use Cases**: {', '.join(analysis_results['purpose']['primary_use_cases'])}
- **Domain**: {analysis_results['purpose']['domain']}
- **Suitability Score**: {analysis_results['purpose']['suitability_score']}/100
- **Data Quality**: {analysis_results['completeness']['completeness_score']:.1f}% complete

### 🚨 CRITICAL FINDINGS
"""
        
        # Add critical issues
        if analysis_results['completeness']['missing_values']:
            critical_missing = [col for col, info in analysis_results['completeness']['missing_values'].items() 
                              if info['severity'] == 'CRITICAL']
            if critical_missing:
                insights += f"- **Critical Data Gaps**: {len(critical_missing)} columns with >50% missing values\\n"
        
        if analysis_results['sensitivity']['sensitive_columns']:
            insights += f"- **Sensitive Data**: {len(analysis_results['sensitivity']['sensitive_columns'])} columns contain PII\\n"
        
        insights += """
### 💡 RECOMMENDATIONS

#### 1. IMMEDIATE ACTIONS
- Address critical data quality issues
- Implement proper data governance
- Set up monitoring for data quality

#### 2. STRATEGIC PLANNING
- Develop data quality framework
- Implement proper documentation
- Establish data lineage tracking

#### 3. TECHNICAL OPTIMIZATION
- Optimize storage format
- Implement proper data types
- Set up automated quality checks
"""
        
        return insights

# Initialize analyzer
analyzer = ComprehensiveAnalyzer()

def main():
    st.title("🔍 Comprehensive Data Analyzer")
    st.markdown("**Deep analysis using 21-question data engineering framework**")
    
    # Hugging Face setup
    with st.sidebar:
        st.header("🤗 Hugging Face Setup")
        hf_token = st.text_input("HF Token (for enhanced analysis)", type="password")
        
        if st.button("Load AI Model") and hf_token:
            with st.spinner("Loading AI model..."):
                if analyzer.setup_huggingface(hf_token):
                    st.success("AI model loaded!")
                else:
                    st.error("Failed to load model")
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload Dataset", type=['csv', 'xlsx', 'parquet'])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(tmp_path)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(tmp_path)
            else:
                df = pd.read_parquet(tmp_path)
            
            # Perform comprehensive analysis
            with st.spinner("🔍 Performing comprehensive analysis..."):
                analysis_results, insights = analyzer.comprehensive_analysis(df, uploaded_file.name)
            
            # Display results
            st.subheader("📋 Analysis Results")
            st.markdown(insights)
            
            # Show detailed findings
            with st.expander("🔍 Detailed Findings", expanded=True):
                st.json(analysis_results)
            
            # Data preview
            with st.expander("👀 Data Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.write(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
                st.write(f"**Memory**: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    else:
        st.info("👆 Upload a dataset for comprehensive analysis!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 21-Question Framework")
            st.markdown("""
            **Data Assessment:**
            1. Purpose & Use Case
            2. Source & Provenance  
            3. Documentation
            4. Sensitivity
            5. Structure
            6. Size & Growth
            7. Completeness
            8. Accuracy
            9. Consistency
            10. Validity
            """)
        
        with col2:
            st.subheader("🎯 Continued...")
            st.markdown("""
            **Quality & Technical:**
            11. Uniqueness
            12. Timeliness
            13. Relevance
            14. Integrity
            15. Biases
            16. Distributions
            17. Correlations
            18. Pipeline Integration
            19. Performance
            20. Impact
            21. Resources
            """)

if __name__ == "__main__":
    main()
