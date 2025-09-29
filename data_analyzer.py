# universal_data_analyzer.py
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

# Set page config
st.set_page_config(
    page_title="Universal Data Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def smart_detect_column_types(df):
    """Smart detection of column types that actually works"""
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    boolean_cols = []
    text_cols = []
    
    for col in df.columns:
        # Skip if all values are null
        if df[col].isnull().all():
            categorical_cols.append(col)
            continue
            
        # Sample of non-null values
        sample_size = min(100, len(df[col].dropna()))
        if sample_size == 0:
            categorical_cols.append(col)
            continue
            
        sample = df[col].dropna().sample(sample_size) if len(df[col].dropna()) > sample_size else df[col].dropna()
        
        # Try to detect numeric
        numeric_count = 0
        try:
            # Try converting to numeric
            pd.to_numeric(df[col].dropna().head(100), errors='coerce')
            # Check if most values are numeric
            numeric_pct = (pd.to_numeric(df[col], errors='coerce').notna().sum() / len(df)) * 100
            if numeric_pct > 80:  # If 80%+ values are numeric
                numeric_cols.append(col)
                continue
        except:
            pass
        
        # Check for boolean
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 3:
            if all(str(val).lower() in ['true', 'false', '0', '1', 'yes', 'no', 'y', 'n'] for val in unique_vals[:3]):
                boolean_cols.append(col)
                continue
        
        # Check for datetime
        datetime_count = 0
        try:
            pd.to_datetime(df[col].dropna().head(10), errors='coerce')
            datetime_pct = (pd.to_datetime(df[col], errors='coerce').notna().sum() / len(df)) * 100
            if datetime_pct > 50:
                datetime_cols.append(col)
                continue
        except:
            pass
        
        # Check if it's text (long strings)
        avg_length = df[col].astype(str).str.len().mean()
        if avg_length > 50 and len(unique_vals) > len(df) * 0.8:
            text_cols.append(col)
        else:
            # Default to categorical
            categorical_cols.append(col)
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
        'boolean': boolean_cols,
        'text': text_cols
    }

def analyze_tabular_content(df):
    """Deep analysis of tabular data content"""
    analysis = {}
    
    # Basic stats
    analysis['shape'] = df.shape
    analysis['columns'] = list(df.columns)
    analysis['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # Smart column type detection
    column_types = smart_detect_column_types(df)
    analysis.update(column_types)
    
    # Convert boolean to categorical for analysis
    analysis['categorical_columns'].extend(analysis['boolean'])
    
    # Missing values
    missing_data = df.isnull().sum()
    analysis['missing_values'] = missing_data.to_dict()
    analysis['missing_percentage'] = (missing_data / len(df) * 100).round(2).to_dict()
    analysis['total_missing'] = missing_data.sum()
    
    # Numeric analysis
    if analysis['numeric']:
        numeric_stats = df[analysis['numeric']].describe()
        analysis['numeric_stats'] = numeric_stats.to_dict()
        
        # Detect outliers using IQR
        outlier_info = {}
        for col in analysis['numeric']:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_info[col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(df) * 100).round(2)
                    }
            except:
                outlier_info[col] = {'count': 0, 'percentage': 0}
        analysis['outliers'] = outlier_info
    
    # Categorical analysis
    if analysis['categorical']:
        categorical_stats = {}
        for col in analysis['categorical']:
            try:
                value_counts = df[col].value_counts()
                categorical_stats[col] = {
                    'unique_values': len(value_counts),
                    'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
                    'top_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'value_distribution': value_counts.head(10).to_dict()
                }
            except:
                categorical_stats[col] = {'unique_values': 0, 'top_value': None, 'top_count': 0}
        analysis['categorical_stats'] = categorical_stats
    
    # Correlation analysis for numeric columns
    if len(analysis['numeric']) > 1:
        try:
            correlation_matrix = df[analysis['numeric']].corr()
            analysis['correlation'] = correlation_matrix.to_dict()
            
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = abs(correlation_matrix.iloc[i, j])
                    if corr_val > 0.8 and not pd.isna(corr_val):
                        high_corr_pairs.append({
                            'col1': correlation_matrix.columns[i],
                            'col2': correlation_matrix.columns[j],
                            'correlation': corr_val.round(3)
                        })
            analysis['high_correlations'] = high_corr_pairs
        except:
            analysis['correlation'] = {}
            analysis['high_correlations'] = []
    
    # Data quality assessment
    quality_issues = []
    
    # Check for constant columns
    for col in df.columns:
        try:
            if df[col].nunique() <= 1:
                quality_issues.append(f"Constant column: {col} (only one unique value)")
        except:
            pass
    
    # Check for high cardinality categorical
    for col in analysis['categorical']:
        try:
            unique_count = df[col].nunique()
            if unique_count > 50:
                quality_issues.append(f"High cardinality: {col} ({unique_count} unique values)")
            elif unique_count == len(df):
                quality_issues.append(f"Potential ID column: {col} (all values unique)")
        except:
            pass
    
    # Check for high missing percentage
    for col, missing_pct in analysis['missing_percentage'].items():
        if missing_pct > 50:
            quality_issues.append(f"High missing values: {col} ({missing_pct}% missing)")
        elif missing_pct > 20:
            quality_issues.append(f"Moderate missing values: {col} ({missing_pct}% missing)")
    
    analysis['quality_issues'] = quality_issues
    
    # Sample data for display
    analysis['sample_data'] = df.head(10)
    
    return analysis

def analyze_file_basics(file_path):
    """Enhanced analysis that deeply analyzes content"""
    file_stats = {
        'file_name': os.path.basename(file_path),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'file_extension': os.path.splitext(file_path)[1],
        'data_type': detect_data_type(file_path)
    }
    
    # Add type-specific deep analysis
    if file_stats['data_type'] == 'tabular':
        try:
            # Try different encodings and separators for CSV
            if file_stats['file_extension'] == '.csv':
                try:
                    df = pd.read_csv(file_path)
                except:
                    # Try with different encoding
                    try:
                        df = pd.read_csv(file_path, encoding='latin-1')
                    except:
                        df = pd.read_csv(file_path, sep=None, engine='python')
            elif file_stats['file_extension'] == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Basic info
            file_stats['columns'] = list(df.columns)
            file_stats['total_rows'] = len(df)
            
            # Deep content analysis
            content_analysis = analyze_tabular_content(df)
            file_stats.update(content_analysis)
            
        except Exception as e:
            file_stats['error'] = f"Could not read tabular data: {e}"
            # Create basic analysis even if deep analysis fails
            file_stats['numeric'] = []
            file_stats['categorical'] = file_stats['columns']
            file_stats['quality_issues'] = [f"Analysis error: {str(e)}"]
    
    return file_stats

def get_content_based_recommendations(file_analysis):
    """Recommendations based on actual data content analysis"""
    if file_analysis['data_type'] != 'tabular':
        return "## Analysis for non-tabular data\n\nPlease upload a CSV or Excel file for detailed analysis."
    
    recommendations = "## 📊 CONTENT-BASED ANALYSIS\n\n"
    
    # Show actual column types detected
    numeric_count = len(file_analysis.get('numeric', []))
    categorical_count = len(file_analysis.get('categorical', []))
    datetime_count = len(file_analysis.get('datetime', []))
    boolean_count = len(file_analysis.get('boolean', []))
    text_count = len(file_analysis.get('text', []))
    
    # Data Overview
    recommendations += f"### 📈 DATASET OVERVIEW\n"
    recommendations += f"- **File**: {file_analysis['file_name']}\n"
    recommendations += f"- **Size**: {file_analysis['total_rows']} rows × {len(file_analysis['columns'])} columns\n"
    recommendations += f"- **Memory**: {file_analysis['file_size_mb']:.2f} MB\n"
    recommendations += f"- **Numeric columns**: {numeric_count}\n"
    recommendations += f"- **Categorical columns**: {categorical_count}\n"
    if datetime_count > 0:
        recommendations += f"- **Datetime columns**: {datetime_count}\n"
    if boolean_count > 0:
        recommendations += f"- **Boolean columns**: {boolean_count}\n"
    if text_count > 0:
        recommendations += f"- **Text columns**: {text_count}\n"
    recommendations += "\n"
    
    # Show sample of detected column types
    if numeric_count > 0:
        recommendations += f"- **Sample numeric columns**: {', '.join(file_analysis['numeric'][:3])}\n"
    if categorical_count > 0:
        recommendations += f"- **Sample categorical columns**: {', '.join(file_analysis['categorical'][:3])}\n"
    recommendations += "\n"
    
    # Data quality issues
    quality_issues = file_analysis.get('quality_issues', [])
    if quality_issues:
        recommendations += "### 🚨 DATA QUALITY ISSUES\n"
        for issue in quality_issues[:5]:
            recommendations += f"- {issue}\n"
        recommendations += "\n"
    else:
        recommendations += "### ✅ DATA QUALITY\n- No major quality issues detected\n\n"
    
    # Missing values analysis
    missing_data = file_analysis.get('missing_values', {})
    if any(missing_data.values()):
        total_missing = sum(missing_data.values())
        total_cells = file_analysis['total_rows'] * len(file_analysis['columns'])
        missing_percentage = (total_missing / total_cells * 100)
        
        recommendations += "### 📉 MISSING VALUES ANALYSIS\n"
        recommendations += f"- **Total missing values**: {total_missing} ({missing_percentage:.1f}% of data)\n"
        
        # Top 3 columns with most missing values
        high_missing = sorted([(col, cnt) for col, cnt in missing_data.items() if cnt > 0], 
                            key=lambda x: x[1], reverse=True)[:3]
        if high_missing:
            recommendations += "- **Columns needing attention**:\n"
            for col, cnt in high_missing:
                pct = (cnt / file_analysis['total_rows'] * 100)
                recommendations += f"  - {col}: {cnt} missing ({pct:.1f}%)\n"
        recommendations += "\n"
    else:
        recommendations += "### ✅ MISSING VALUES\n- No missing values found\n\n"
    
    # Model recommendations based on ACTUAL data characteristics
    recommendations += "\n### 🤖 INTELLIGENT MODEL RECOMMENDATIONS\n"
    
    # Smart model selection based on actual column types
    if numeric_count >= 5 and categorical_count >= 2:
        recommendations += "- **Primary**: XGBoost (excellent for mixed data types)\n"
        recommendations += "- **Alternative**: LightGBM (fast, great for categorical features)\n"
        recommendations += "- **Advanced**: CatBoost (handles categorical natively)\n"
        recommendations += "- **Use case**: Complex pattern recognition with mixed data types\n"
        
    elif numeric_count >= 8:
        recommendations += "- **Primary**: Neural Networks (capture complex numeric patterns)\n"
        recommendations += "- **Alternative**: XGBoost (robust, fast training)\n" 
        recommendations += "- **Baseline**: Random Forest (interpretable)\n"
        recommendations += "- **Use case**: Complex numerical pattern recognition\n"
        
    elif categorical_count >= 5:
        recommendations += "- **Primary**: CatBoost (best for categorical data)\n"
        recommendations += "- **Alternative**: LightGBM (fast with categorical)\n"
        recommendations += "- **Classic**: Random Forest (handles mixed types)\n"
        recommendations += "- **Use case**: Categorical feature analysis\n"
        
    elif numeric_count == 0 and categorical_count > 0:
        recommendations += "- **Primary**: CatBoost or LightGBM (optimized for categorical)\n"
        recommendations += "- **Alternative**: Random Forest (handles categorical well)\n"
        recommendations += "- **Baseline**: Logistic Regression with encoding\n"
        recommendations += "- **Use case**: Pure categorical data analysis\n"
        
    elif categorical_count == 0 and numeric_count > 0:
        recommendations += "- **Primary**: XGBoost (excellent for numeric data)\n"
        recommendations += "- **Alternative**: Neural Networks (complex patterns)\n"
        recommendations += "- **Baseline**: Linear Models (fast, interpretable)\n"
        recommendations += "- **Use case**: Pure numerical data analysis\n"
        
    else:
        recommendations += "- **Primary**: XGBoost (versatile for most datasets)\n"
        recommendations += "- **Alternative**: Random Forest (stable, interpretable)\n"
        recommendations += "- **Baseline**: Logistic/Linear Regression (fast)\n"
        recommendations += "- **Use case**: General purpose analysis\n"
    
    # Preprocessing recommendations based on ACTUAL data
    recommendations += "\n### 🛠️ SPECIFIC PREPROCESSING STEPS\n"
    
    if any(missing_data.values()):
        missing_percentage = (sum(missing_data.values()) / (file_analysis['total_rows'] * len(file_analysis['columns'])) * 100
        if missing_percentage < 5:
            recommendations += "- **Missing values**: Simple imputation (median for numeric, mode for categorical)\n"
        elif missing_percentage < 20:
            recommendations += "- **Missing values**: Advanced imputation (KNN) or indicator variables\n"
        else:
            recommendations += "- **Missing values**: Consider removal or advanced methods\n"
    
    if categorical_count > 0:
        # Check for high cardinality
        high_cardinality_cols = []
        for col in file_analysis.get('categorical', []):
            unique_vals = file_analysis.get('categorical_stats', {}).get(col, {}).get('unique_values', 0)
            if unique_vals > 20:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            recommendations += f"- **Categorical encoding**: Target encoding for {len(high_cardinality_cols)} high-cardinality columns\n"
            recommendations += f"- **High-cardinality columns**: {', '.join(high_cardinality_cols[:3])}\n"
        else:
            recommendations += "- **Categorical encoding**: One-hot encoding recommended\n"
    
    if numeric_count > 0:
        recommendations += "- **Numeric scaling**: StandardScaler for normal distributions\n"
        
        # Check for outliers
        outlier_cols = [col for col, info in file_analysis.get('outliers', {}).items() 
                       if info.get('percentage', 0) > 5]
        if outlier_cols:
            recommendations += f"- **Outlier handling**: RobustScaler for {len(outlier_cols)} columns with outliers\n"
    
    high_corrs = file_analysis.get('high_correlations', [])
    if high_corrs:
        recommendations += f"- **Multicollinearity**: Remove one from {len(high_corrs)} highly correlated pairs\n"
        for pair in high_corrs[:2]:
            recommendations += f"  - {pair['col1']} ↔ {pair['col2']} (r={pair['correlation']})\n"
    
    # Hardware recommendations
    recommendations += "\n### 💻 HARDWARE OPTIMIZATION\n"
    file_size = file_analysis['file_size_mb']
    total_rows = file_analysis['total_rows']
    
    if file_size < 10 or total_rows < 10000:
        recommendations += "- **Memory**: Dataset fits easily in RAM\n"
        recommendations += "- **Processing**: Single machine sufficient\n"
        recommendations += "- **Speed**: Fast training expected (< 1 minute)\n"
    elif file_size < 100 or total_rows < 100000:
        recommendations += "- **Memory**: Moderate size, monitor usage\n"
        recommendations += "- **Processing**: Single machine still suitable\n"
        recommendations += "- **Speed**: Reasonable training times (1-10 minutes)\n"
    else:
        recommendations += "- **Memory**: Large dataset, consider sampling for exploration\n"
        recommendations += "- **Processing**: May need distributed computing for full dataset\n"
        recommendations += "- **Speed**: Plan for longer training times (10+ minutes)\n"
    
    return recommendations

def create_advanced_visualizations(analysis):
    """Create advanced visualizations based on content analysis"""
    if analysis['data_type'] == 'tabular':
        
        tab1, tab2, tab3 = st.tabs([
            "📋 Data Preview", "📊 Column Types", "🔍 Data Quality"
        ])
        
        with tab1:
            st.subheader("Data Preview")
            if 'sample_data' in analysis:
                st.dataframe(analysis['sample_data'], use_container_width=True)
            else:
                st.warning("Could not load sample data")
            
            # Basic stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", analysis['total_rows'])
            with col2:
                st.metric("Total Columns", len(analysis['columns']))
            with col3:
                st.metric("Numeric Columns", len(analysis.get('numeric', [])))
            with col4:
                st.metric("Categorical Columns", len(analysis.get('categorical', [])))
        
        with tab2:
            st.subheader("Column Type Distribution")
            
            # Create column type distribution chart
            type_counts = {
                'Numeric': len(analysis.get('numeric', [])),
                'Categorical': len(analysis.get('categorical', [])),
                'Datetime': len(analysis.get('datetime', [])),
                'Boolean': len(analysis.get('boolean', [])),
                'Text': len(analysis.get('text', []))
            }
            
            # Remove zero counts
            type_counts = {k: v for k, v in type_counts.items() if v > 0}
            
            if type_counts:
                fig = px.pie(values=list(type_counts.values()), 
                           names=list(type_counts.keys()),
                           title='Column Type Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show sample columns of each type
            if analysis.get('numeric'):
                st.write("**Sample Numeric Columns:**", ", ".join(analysis['numeric'][:5]))
            if analysis.get('categorical'):
                st.write("**Sample Categorical Columns:**", ", ".join(analysis['categorical'][:5]))
        
        with tab3:
            st.subheader("Data Quality Report")
            
            # Quality issues
            issues = analysis.get('quality_issues', [])
            if issues:
                st.error(f"🚨 {len(issues)} Quality Issues Found:")
                for issue in issues[:5]:
                    st.write(f"- {issue}")
            else:
                st.success("✅ No major quality issues detected!")
            
            # Missing values summary
            missing_total = analysis.get('total_missing', 0)
            if missing_total > 0:
                total_cells = analysis['total_rows'] * len(analysis['columns'])
                missing_pct = (missing_total / total_cells * 100)
                st.warning(f"⚠️ {missing_total} missing values ({missing_pct:.1f}% of data)")

def main():
    st.title("🤖 Advanced Universal Data Analyzer")
    st.markdown("Upload **ANY** dataset and get **deep content analysis** with intelligent recommendations!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload your dataset",
        type=['csv', 'xlsx', 'txt', 'json', 'jpg', 'png', 'jpeg', 'parquet'],
        help="Supported formats: CSV, Excel, Images, Text, JSON, Parquet"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Analyze file with deep content analysis
            with st.spinner("🔍 Performing deep content analysis..."):
                analysis = analyze_file_basics(tmp_path)
            
            # Display basic info
            st.subheader("📄 File Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("File Type", analysis['data_type'].upper())
            with col2:
                st.metric("File Size", f"{analysis['file_size_mb']:.2f} MB")
            with col3:
                st.metric("Columns", len(analysis['columns']))
            with col4:
                st.metric("Total Rows", analysis.get('total_rows', 'N/A'))
            
            # Advanced Visualizations
            st.subheader("📊 Deep Data Exploration")
            create_advanced_visualizations(analysis)
            
            # Content-Based Recommendations
            st.subheader("🎯 Intelligent Content-Based Recommendations")
            with st.expander("View Detailed Analysis", expanded=True):
                recommendations = get_content_based_recommendations(analysis)
                st.markdown(recommendations)
            
        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")
        
        finally:
            # Clean up
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    else:
        # Demo section
        st.info("👆 Upload a CSV, Excel, or other data file to see deep content analysis!")

if __name__ == "__main__":
    main()
