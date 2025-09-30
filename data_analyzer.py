# universal_data_analyzer_fast.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Data Engineer's Analyzer",
    page_icon="🔧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class FastDataEngineer:
    def __init__(self):
        self.analysis_ready = False
    
    def perform_deep_analysis(self, df, file_info):
        """Perform comprehensive data engineering analysis without LLM dependencies"""
        
        analysis = self._create_comprehensive_analysis(df, file_info)
        recommendations = self._generate_data_engineer_recommendations(analysis)
        
        return analysis, recommendations
    
    def _create_comprehensive_analysis(self, df, file_info):
        """Create detailed technical analysis"""
        analysis = {}
        
        # Basic metrics
        analysis['shape'] = df.shape
        analysis['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024
        analysis['file_size_mb'] = file_info['file_size_mb']
        analysis['total_cells'] = df.shape[0] * df.shape[1]
        
        # Data types analysis
        dtypes_analysis = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype not in dtypes_analysis:
                dtypes_analysis[dtype] = []
            dtypes_analysis[dtype].append({
                'name': col,
                'memory_usage': df[col].memory_usage(deep=True) / 1024,  # KB
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            })
        analysis['dtype_analysis'] = dtypes_analysis
        
        # Comprehensive null analysis
        null_analysis = {}
        total_nulls = 0
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            total_nulls += null_count
            
            if null_count > 0:
                null_analysis[col] = {
                    'count': null_count,
                    'percentage': round(null_pct, 2),
                    'severity': 'CRITICAL' if null_pct > 50 else 'HIGH' if null_pct > 20 else 'MEDIUM' if null_pct > 5 else 'LOW'
                }
        
        analysis['null_analysis'] = null_analysis
        analysis['total_nulls'] = total_nulls
        analysis['null_percentage'] = (total_nulls / analysis['total_cells']) * 100
        
        # Advanced cardinality analysis
        cardinality = {}
        high_cardinality_cols = []
        for col in df.columns:
            unique_count = df[col].nunique()
            cardinality_pct = (unique_count / len(df)) * 100
            
            cardinality[col] = {
                'unique_count': unique_count,
                'cardinality_pct': round(cardinality_pct, 2),
                'type': 'VERY_HIGH' if cardinality_pct > 95 else 'HIGH' if cardinality_pct > 70 else 'MEDIUM' if cardinality_pct > 30 else 'LOW'
            }
            
            if cardinality_pct > 95:
                high_cardinality_cols.append(col)
        
        analysis['cardinality'] = cardinality
        analysis['high_cardinality_cols'] = high_cardinality_cols
        
        # Statistical analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_analysis = {}
        skewed_cols = []
        high_variance_cols = []
        
        for col in numeric_cols:
            stats = df[col].describe()
            skewness = df[col].skew()
            cv = (stats['std'] / abs(stats['mean'])) * 100 if stats['mean'] != 0 else float('inf')
            
            stats_analysis[col] = {
                'mean': round(stats['mean'], 4),
                'std': round(stats['std'], 4),
                'min': round(stats['min'], 4),
                '25%': round(stats['25%'], 4),
                '50%': round(stats['50%'], 4),
                '75%': round(stats['75%'], 4),
                'max': round(stats['max'], 4),
                'skewness': round(skewness, 4),
                'cv': round(cv, 2),
                'outlier_pct': self._calculate_outlier_percentage(df[col]),
                'zeros_pct': ((df[col] == 0).sum() / len(df)) * 100
            }
            
            if abs(skewness) > 2:
                skewed_cols.append((col, skewness))
            if cv > 200:  # Coefficient of variation > 200%
                high_variance_cols.append((col, cv))
        
        analysis['numeric_stats'] = stats_analysis
        analysis['skewed_cols'] = skewed_cols
        analysis['high_variance_cols'] = high_variance_cols
        
        # Data quality issues
        quality_issues = self._identify_data_quality_issues(df, analysis)
        analysis['quality_issues'] = quality_issues
        
        # Storage optimization
        storage_analysis = self._analyze_storage_optimization(df, analysis)
        analysis['storage_analysis'] = storage_analysis
        
        # Data profiling
        analysis['data_profile'] = self._create_data_profile(df)
        
        return analysis
    
    def _calculate_outlier_percentage(self, series):
        """Calculate percentage of outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            return 0
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return round((len(outliers) / len(series)) * 100, 2)
    
    def _identify_data_quality_issues(self, df, analysis):
        """Identify comprehensive data quality issues"""
        issues = []
        
        # Null value issues
        for col, null_info in analysis['null_analysis'].items():
            if null_info['severity'] == 'CRITICAL':
                issues.append(f"🚨 CRITICAL_NULL: {col} has {null_info['percentage']}% missing values - Consider dropping or advanced imputation")
            elif null_info['severity'] == 'HIGH':
                issues.append(f"⚠️ HIGH_NULL: {col} has {null_info['percentage']}% missing values - Needs imputation strategy")
            elif null_info['severity'] == 'MEDIUM':
                issues.append(f"📊 MEDIUM_NULL: {col} has {null_info['percentage']}% missing values - Monitor and impute")
        
        # High cardinality issues
        for col in analysis['high_cardinality_cols']:
            card_info = analysis['cardinality'][col]
            issues.append(f"🔍 HIGH_CARDINALITY: {col} has {card_info['unique_count']} unique values ({card_info['cardinality_pct']}%) - Potential ID column or data leakage")
        
        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        for col in constant_cols:
            issues.append(f"📊 CONSTANT_COLUMN: {col} has only {df[col].nunique()} unique value - Consider dropping")
        
        # Skewed distributions
        for col, skewness in analysis['skewed_cols']:
            issues.append(f"📈 HIGHLY_SKEWED: {col} has skewness {skewness:.2f} - Consider transformation (log, box-cox)")
        
        # High variance
        for col, cv in analysis['high_variance_cols']:
            issues.append(f"📊 HIGH_VARIANCE: {col} has coefficient of variation {cv:.0f}% - May need scaling")
        
        # High outlier percentage
        for col, stats in analysis.get('numeric_stats', {}).items():
            if stats['outlier_pct'] > 15:
                issues.append(f"📊 HIGH_OUTLIERS: {col} has {stats['outlier_pct']}% outliers - Investigate data quality")
        
        # High zero percentage
        for col, stats in analysis.get('numeric_stats', {}).items():
            if stats['zeros_pct'] > 80:
                issues.append(f"🔲 SPARSE_COLUMN: {col} has {stats['zeros_pct']:.1f}% zeros - Consider sparse matrix representation")
        
        # Memory inefficiency
        current_memory = analysis['memory_usage_mb']
        if current_memory > 100:  # 100MB threshold
            issues.append(f"💾 LARGE_MEMORY: Dataset uses {current_memory:.1f} MB - Implement storage optimizations")
        
        return issues
    
    def _analyze_storage_optimization(self, df, analysis):
        """Analyze storage optimization opportunities"""
        optimizations = []
        current_memory = analysis['memory_usage_mb']
        potential_savings = 0
        
        # Numeric column optimizations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            current_dtype = str(df[col].dtype)
            col_min = df[col].min()
            col_max = df[col].max()
            
            suggested_dtype = None
            savings_pct = 0
            
            if 'float' in current_dtype:
                if col_min >= -3.4e38 and col_max <= 3.4e38:
                    suggested_dtype = 'float32'
                    savings_pct = 50
                elif col_min > -65500 and col_max < 65500:
                    suggested_dtype = 'float16'
                    savings_pct = 75
            elif 'int' in current_dtype:
                if col_min >= 0:  # Unsigned
                    if col_max < 256:
                        suggested_dtype = 'uint8'
                        savings_pct = 75
                    elif col_max < 65536:
                        suggested_dtype = 'uint16'
                        savings_pct = 50
                else:  # Signed
                    if col_min > -128 and col_max < 127:
                        suggested_dtype = 'int8'
                        savings_pct = 75
                    elif col_min > -32768 and col_max < 32767:
                        suggested_dtype = 'int16'
                        savings_pct = 50
            
            if suggested_dtype:
                optimizations.append({
                    'column': col,
                    'current_dtype': current_dtype,
                    'suggested_dtype': suggested_dtype,
                    'savings_pct': savings_pct,
                    'reason': f"Range [{col_min}, {col_max}] fits {suggested_dtype}"
                })
                potential_savings += savings_pct * (df[col].memory_usage(deep=True) / df.memory_usage(deep=True).sum())
        
        # Categorical optimizations
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                current_mem = df[col].memory_usage(deep=True)
                optimizations.append({
                    'column': col,
                    'current_dtype': 'object',
                    'suggested_dtype': 'category',
                    'savings_pct': 60,
                    'reason': f"Low cardinality ({df[col].nunique()} unique values, {unique_ratio:.1%} uniqueness)"
                })
                potential_savings += 60 * (current_mem / df.memory_usage(deep=True).sum())
        
        total_savings_mb = current_memory * (potential_savings / 100)
        
        return {
            'current_memory_mb': current_memory,
            'potential_savings_mb': total_savings_mb,
            'potential_savings_pct': potential_savings,
            'optimizations': optimizations
        }
    
    def _create_data_profile(self, df):
        """Create comprehensive data profile"""
        profile = {
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'total_memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'complete_cases': df.notnull().all(axis=1).sum(),
            'complete_cases_pct': (df.notnull().all(axis=1).sum() / len(df)) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_rows_pct': (df.duplicated().sum() / len(df)) * 100
        }
        
        # Column type counts
        dtypes_count = df.dtypes.value_counts()
        profile['dtype_counts'] = {str(k): v for k, v in dtypes_count.items()}
        
        return profile
    
    def _generate_data_engineer_recommendations(self, analysis):
        """Generate detailed data engineering recommendations"""
        
        rec = f"""
## 🔧 DATA ENGINEERING ANALYSIS REPORT

### 📊 EXECUTIVE SUMMARY
- **Dataset**: {analysis['shape'][0]:,} rows × {analysis['shape'][1]} columns
- **Memory Usage**: {analysis['memory_usage_mb']:.2f} MB
- **Data Quality Score**: {self._calculate_quality_score(analysis):.1f}/10
- **Total Issues**: {len(analysis['quality_issues'])} identified

### 🚨 CRITICAL ISSUES (Immediate Action Required)
"""
        
        # Critical issues
        critical_issues = [issue for issue in analysis['quality_issues'] if '🚨 CRITICAL' in issue]
        if critical_issues:
            for issue in critical_issues[:5]:
                rec += f"- {issue}\\n"
        else:
            rec += "- ✅ No critical issues detected\\n"
        
        rec += """
### ⚠️ HIGH PRIORITY ISSUES
"""
        # High priority issues
        high_issues = [issue for issue in analysis['quality_issues'] if '⚠️ HIGH' in issue]
        if high_issues:
            for issue in high_issues[:5]:
                rec += f"- {issue}\\n"
        else:
            rec += "- ✅ No high priority issues\\n"
        
        rec += f"""
### 💾 STORAGE OPTIMIZATION
- **Current Memory**: {analysis['storage_analysis']['current_memory_mb']:.2f} MB
- **Potential Savings**: {analysis['storage_analysis']['potential_savings_mb']:.2f} MB ({analysis['storage_analysis']['potential_savings_pct']:.1f}%)
- **Optimization Opportunities**: {len(analysis['storage_analysis']['optimizations'])} columns

**Top Storage Optimizations:**
"""
        
        # Top storage optimizations
        for opt in analysis['storage_analysis']['optimizations'][:5]:
            rec += f"- **{opt['column']}**: {opt['current_dtype']} → {opt['suggested_dtype']} ({opt['savings_pct']}% savings)\\n"
        
        rec += """
### 📈 DATA CHARACTERISTICS
"""
        # Data characteristics
        rec += f"- **Null Values**: {analysis['total_nulls']:,} total ({analysis['null_percentage']:.2f}% of data)\\n"
        rec += f"- **High Cardinality Columns**: {len(analysis['high_cardinality_cols'])} identified\\n"
        rec += f"- **Skewed Distributions**: {len(analysis['skewed_cols'])} numeric columns\\n"
        rec += f"- **High Variance Columns**: {len(analysis['high_variance_cols'])} identified\\n"
        
        rec += """
### 🛠️ TECHNICAL RECOMMENDATIONS

#### 1. DATA QUALITY IMPROVEMENT
"""
        # Data quality recommendations
        if analysis['null_analysis']:
            rec += "- Implement strategic null value imputation based on column importance\\n"
            rec += "- Consider multiple imputation techniques for critical columns\\n"
        
        if analysis['skewed_cols']:
            rec += "- Apply transformations (log, box-cox) to highly skewed numeric columns\\n"
        
        rec += """
#### 2. STORAGE OPTIMIZATION
- Implement suggested data type conversions for memory reduction
- Use categorical encoding for low-cardinality string columns
- Consider Parquet format for better compression

#### 3. PROCESSING OPTIMIZATION
- Use chunk processing for large datasets
- Implement lazy evaluation where possible
- Consider distributed computing for scaling

#### 4. MONITORING & GOVERNANCE
- Set up data quality monitoring
- Implement data validation rules
- Create data lineage documentation
"""
        
        return rec
    
    def _calculate_quality_score(self, analysis):
        """Calculate data quality score (0-10)"""
        score = 10.0
        
        # Penalize for null values
        null_penalty = min(3.0, analysis['null_percentage'] / 10)
        score -= null_penalty
        
        # Penalize for quality issues
        issue_penalty = min(3.0, len(analysis['quality_issues']) * 0.1)
        score -= issue_penalty
        
        # Penalize for memory inefficiency
        if analysis['memory_usage_mb'] > 100:
            mem_penalty = min(2.0, analysis['memory_usage_mb'] / 1000)
            score -= mem_penalty
        
        return max(0, score)

# Initialize the analyzer
data_engineer = FastDataEngineer()

def analyze_file(file_path):
    """Analyze file and return basic info"""
    file_info = {
        'file_name': os.path.basename(file_path),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'file_extension': os.path.splitext(file_path)[1],
    }
    
    try:
        if file_info['file_extension'] == '.csv':
            df = pd.read_csv(file_path)
        elif file_info['file_extension'] == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        file_info['df'] = df
        file_info['shape'] = df.shape
        
    except Exception as e:
        file_info['error'] = str(e)
    
    return file_info

def create_technical_dashboard(analysis):
    """Create interactive technical dashboard"""
    
    st.subheader("🔧 Technical Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Shape", f"{analysis['shape'][0]:,} × {analysis['shape'][1]}")
    with col2:
        st.metric("Memory Usage", f"{analysis['memory_usage_mb']:.2f} MB")
    with col3:
        st.metric("Data Quality", f"{data_engineer._calculate_quality_score(analysis):.1f}/10")
    with col4:
        st.metric("Total Issues", len(analysis['quality_issues']))
    
    # Data Quality Issues
    with st.expander("🚨 Data Quality Issues", expanded=True):
        if analysis['quality_issues']:
            for issue in analysis['quality_issues'][:10]:
                st.write(issue)
        else:
            st.success("✅ No data quality issues detected")
    
    # Storage Optimization
    with st.expander("💾 Storage Optimization", expanded=True):
        st.metric("Potential Savings", 
                 f"{analysis['storage_analysis']['potential_savings_mb']:.2f} MB",
                 f"{analysis['storage_analysis']['potential_savings_pct']:.1f}%")
        
        if analysis['storage_analysis']['optimizations']:
            st.write("**Top Optimization Opportunities:**")
            for opt in analysis['storage_analysis']['optimizations'][:5]:
                st.write(f"- **{opt['column']}**: {opt['current_dtype']} → {opt['suggested_dtype']} ({opt['savings_pct']}% savings)")
    
    # Data Types Distribution
    with st.expander("📊 Data Types Analysis", expanded=False):
        dtype_data = []
        for dtype, cols in analysis['dtype_analysis'].items():
            dtype_data.append({'Type': dtype, 'Count': len(cols)})
        
        if dtype_data:
            dtype_df = pd.DataFrame(dtype_data)
            fig = px.pie(dtype_df, values='Count', names='Type', title='Data Type Distribution')
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🔧 Data Engineer's Analyzer")
    st.markdown("**Fast, comprehensive data engineering analysis without model dependencies**")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload your dataset",
        type=['csv', 'xlsx', 'parquet'],
        help="Upload CSV, Excel, or Parquet files for analysis"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            with st.spinner("🔍 Performing deep data engineering analysis..."):
                file_info = analyze_file(tmp_path)
                
                if 'error' in file_info:
                    st.error(f"Error reading file: {file_info['error']}")
                    return
                
                # Perform comprehensive analysis
                analysis, recommendations = data_engineer.perform_deep_analysis(file_info['df'], file_info)
                
                # Display technical dashboard
                create_technical_dashboard(analysis)
                
                # Display detailed recommendations
                st.subheader("📋 Data Engineering Recommendations")
                st.markdown(recommendations)
                
                # Show sample data
                with st.expander("🔍 Data Preview", expanded=False):
                    st.dataframe(file_info['df'].head(10), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    else:
        st.info("👆 Upload a dataset to get comprehensive data engineering analysis!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 What You'll Get")
            st.markdown("""
            - **Data Quality Assessment** with scoring
            - **Storage Optimization** recommendations
            - **Memory Usage** analysis
            - **Data Type** optimization
            - **Null Value** analysis
            - **Cardinality** assessment
            """)
        
        with col2:
            st.subheader("⚡ Fast & Reliable")
            st.markdown("""
            - **No external dependencies**
            - **Instant analysis**
            - **Technical focus**
            - **Actionable insights**
            - **Production-ready recommendations**
            """)

if __name__ == "__main__":
    main()
