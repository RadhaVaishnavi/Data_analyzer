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

def analyze_tabular_content(df):
    """Deep analysis of tabular data content without scipy"""
    analysis = {}
    
    # Basic stats
    analysis['shape'] = df.shape
    analysis['columns'] = list(df.columns)
    analysis['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # Missing values
    missing_data = df.isnull().sum()
    analysis['missing_values'] = missing_data.to_dict()
    analysis['missing_percentage'] = (missing_data / len(df) * 100).round(2).to_dict()
    analysis['total_missing'] = missing_data.sum()
    
    # Numeric analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    analysis['numeric_columns'] = list(numeric_cols)
    
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()
        analysis['numeric_stats'] = numeric_stats.to_dict()
        
        # Detect outliers using IQR (manual calculation without scipy)
        outlier_info = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df) * 100).round(2)
            }
        analysis['outliers'] = outlier_info
    
    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    analysis['categorical_columns'] = list(categorical_cols)
    
    if len(categorical_cols) > 0:
        categorical_stats = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_values': len(value_counts),
                'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
                'top_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'value_distribution': value_counts.head(10).to_dict()
            }
        analysis['categorical_stats'] = categorical_stats
    
    # Correlation analysis for numeric columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        analysis['correlation'] = correlation_matrix.to_dict()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > 0.8:
                    high_corr_pairs.append({
                        'col1': correlation_matrix.columns[i],
                        'col2': correlation_matrix.columns[j],
                        'correlation': corr_val.round(3)
                    })
        analysis['high_correlations'] = high_corr_pairs
    
    # Data quality assessment
    quality_issues = []
    
    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            quality_issues.append(f"Constant column: {col} (only one unique value)")
    
    # Check for high cardinality categorical
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count > 50:
            quality_issues.append(f"High cardinality: {col} ({unique_count} unique values)")
        elif unique_count == len(df):
            quality_issues.append(f"Potential ID column: {col} (all values unique)")
    
    # Check for high missing percentage
    for col, missing_pct in analysis['missing_percentage'].items():
        if missing_pct > 50:
            quality_issues.append(f"High missing values: {col} ({missing_pct}% missing)")
        elif missing_pct > 20:
            quality_issues.append(f"Moderate missing values: {col} ({missing_pct}% missing)")
    
    # Check for skewed distributions
    for col in numeric_cols:
        if col in analysis.get('numeric_stats', {}):
            mean_val = analysis['numeric_stats'][col].get('mean', 0)
            std_val = analysis['numeric_stats'][col].get('std', 1)
            if std_val > 3 * abs(mean_val) and mean_val != 0:
                quality_issues.append(f"High variance: {col} (std > 3*mean)")
    
    analysis['quality_issues'] = quality_issues
    
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
            if file_stats['file_extension'] == '.csv':
                df = pd.read_csv(file_path)
            elif file_stats['file_extension'] == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Basic info
            file_stats['columns'] = list(df.columns)
            file_stats['total_rows'] = len(df)
            file_stats['sample_data'] = df.head(10)
            
            # Deep content analysis
            content_analysis = analyze_tabular_content(df)
            file_stats.update(content_analysis)
            
        except Exception as e:
            file_stats['error'] = f"Could not read tabular data: {e}"
            
    elif file_stats['data_type'] == 'text':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(10000)
            file_stats['content_preview'] = content[:1000] + "..." if len(content) > 1000 else content
            file_stats['line_count'] = len(content.split('\n'))
            file_stats['word_count'] = len(content.split())
            file_stats['char_count'] = len(content)
        except Exception as e:
            file_stats['error'] = f"Could not read text file: {e}"
    
    elif file_stats['data_type'] == 'images':
        try:
            with Image.open(file_path) as img:
                file_stats['image_size'] = img.size
                file_stats['image_mode'] = img.mode
                file_stats['image_format'] = img.format
                file_stats['aspect_ratio'] = img.size[0] / img.size[1] if img.size[1] > 0 else 0
        except Exception as e:
            file_stats['error'] = f"Could not read image: {e}"
    
    return file_stats

def get_content_based_recommendations(file_analysis):
    """Recommendations based on actual data content analysis"""
    if file_analysis['data_type'] != 'tabular':
        return get_universal_recommendations(file_analysis)
    
    recommendations = "## 📊 CONTENT-BASED ANALYSIS\n\n"
    
    # Data Overview
    recommendations += f"### 📈 DATASET OVERVIEW\n"
    recommendations += f"- **File**: {file_analysis['file_name']}\n"
    recommendations += f"- **Size**: {file_analysis['total_rows']} rows × {len(file_analysis['columns'])} columns\n"
    recommendations += f"- **Memory**: {file_analysis['file_size_mb']:.2f} MB\n\n"
    
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
    
    # Column type analysis
    numeric_cols = file_analysis.get('numeric_columns', [])
    categorical_cols = file_analysis.get('categorical_columns', [])
    
    recommendations += "### 🔍 FEATURE ANALYSIS\n"
    recommendations += f"- **Numeric columns**: {len(numeric_cols)}\n"
    recommendations += f"- **Categorical columns**: {len(categorical_cols)}\n"
    
    if categorical_cols:
        high_cardinality = [col for col in categorical_cols 
                          if file_analysis.get('categorical_stats', {}).get(col, {}).get('unique_values', 0) > 20]
        if high_cardinality:
            recommendations += f"- **High cardinality features**: {len(high_cardinality)} columns\n"
    
    # Model recommendations
    recommendations += "\n### 🤖 INTELLIGENT MODEL RECOMMENDATIONS\n"
    
    # Determine the best approach based on data characteristics
    if len(numeric_cols) >= 5 and len(categorical_cols) >= 2:
        recommendations += "- **Primary**: XGBoost (excellent for mixed data types)\n"
        recommendations += "- **Alternative**: LightGBM (fast, great for categorical features)\n"
        recommendations += "- **Advanced**: CatBoost (handles categorical natively)\n"
    elif len(numeric_cols) >= 8:
        recommendations += "- **Primary**: Neural Networks (capture complex patterns)\n"
        recommendations += "- **Alternative**: XGBoost (robust, fast training)\n"
        recommendations += "- **Baseline**: Random Forest (interpretable)\n"
    elif len(categorical_cols) >= 5:
        recommendations += "- **Primary**: CatBoost (best for categorical data)\n"
        recommendations += "- **Alternative**: LightGBM (fast with categorical)\n"
        recommendations += "- **Classic**: Random Forest (handles mixed types)\n"
    else:
        recommendations += "- **Primary**: XGBoost (versatile for most datasets)\n"
        recommendations += "- **Alternative**: Random Forest (stable, interpretable)\n"
        recommendations += "- **Baseline**: Logistic/Linear Regression (fast)\n"
    
    # Preprocessing recommendations
    recommendations += "\n### 🛠️ SPECIFIC PREPROCESSING STEPS\n"
    
    if any(missing_data.values()):
        if missing_percentage < 5:
            recommendations += "- **Missing values**: Simple imputation (median/mode)\n"
        elif missing_percentage < 20:
            recommendations += "- **Missing values**: Advanced imputation (KNN) or indicator variables\n"
        else:
            recommendations += "- **Missing values**: Consider removal or advanced methods\n"
    
    if categorical_cols:
        if any(file_analysis.get('categorical_stats', {}).get(col, {}).get('unique_values', 0) > 20 for col in categorical_cols):
            recommendations += "- **Categorical encoding**: Target encoding for high-cardinality, one-hot for low\n"
        else:
            recommendations += "- **Categorical encoding**: One-hot encoding recommended\n"
    
    if numeric_cols:
        recommendations += "- **Numeric scaling**: StandardScaler for normal, RobustScaler for outliers\n"
    
    high_corrs = file_analysis.get('high_correlations', [])
    if high_corrs:
        recommendations += f"- **Multicollinearity**: Remove one from {len(high_corrs)} highly correlated pairs\n"
    
    # Outlier handling
    outliers = file_analysis.get('outliers', {})
    outlier_cols = [col for col, info in outliers.items() if info.get('percentage', 0) > 5]
    if outlier_cols:
        recommendations += f"- **Outliers**: {len(outlier_cols)} columns have significant outliers\n"
    
    # Hardware recommendations
    recommendations += "\n### 💻 HARDWARE OPTIMIZATION\n"
    file_size = file_analysis['file_size_mb']
    if file_size < 10:
        recommendations += "- **Memory**: Dataset fits easily in RAM\n"
        recommendations += "- **Processing**: Single machine sufficient\n"
        recommendations += "- **Speed**: Fast training expected\n"
    elif file_size < 100:
        recommendations += "- **Memory**: Moderate size, monitor usage\n"
        recommendations += "- **Processing**: Single machine still suitable\n"
        recommendations += "- **Speed**: Reasonable training times\n"
    else:
        recommendations += "- **Memory**: Large dataset, consider sampling\n"
        recommendations += "- **Processing**: May need distributed computing\n"
        recommendations += "- **Speed**: Plan for longer training times\n"
    
    return recommendations

def get_universal_recommendations(file_analysis):
    """Fallback recommendations"""
    data_type = file_analysis['data_type']
    return f"## Analysis for {data_type.upper()} data\n\nBasic recommendations will be shown here."

def calculate_quality_score(file_analysis):
    """Calculate data quality score"""
    score = 10.0
    
    # Penalize for missing values
    total_missing = file_analysis.get('total_missing', 0)
    total_cells = file_analysis.get('total_rows', 1) * len(file_analysis.get('columns', 1))
    missing_pct = total_missing / total_cells
    score -= missing_pct * 20
    
    # Penalize for quality issues
    score -= min(len(file_analysis.get('quality_issues', [])), 5) * 0.3
    
    return max(0, min(10, score))

def create_advanced_visualizations(analysis):
    """Create advanced visualizations based on content analysis"""
    if analysis['data_type'] == 'tabular':
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Data Preview", "📊 Missing Values", "📈 Distributions", "🔗 Correlations"
        ])
        
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(analysis['sample_data'], use_container_width=True)
            
            # Basic stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", analysis['total_rows'])
            with col2:
                st.metric("Total Columns", len(analysis['columns']))
            with col3:
                st.metric("Numeric Columns", len(analysis.get('numeric_columns', [])))
            with col4:
                st.metric("Categorical Columns", len(analysis.get('categorical_columns', [])))
        
        with tab2:
            st.subheader("Missing Values Analysis")
            if analysis.get('missing_values'):
                missing_data = pd.DataFrame({
                    'Column': list(analysis['missing_values'].keys()),
                    'Missing_Count': list(analysis['missing_values'].values()),
                    'Missing_Percentage': list(analysis['missing_percentage'].values())
                }).sort_values('Missing_Count', ascending=False)
                
                # Plot top 10 columns with missing values
                fig = px.bar(missing_data.head(10), 
                           x='Column', y='Missing_Percentage',
                           title='Top 10 Columns with Missing Values (%)',
                           color='Missing_Percentage')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values detected!")
        
        with tab3:
            st.subheader("Feature Distributions")
            numeric_cols = analysis.get('numeric_columns', [])
            if numeric_cols:
                selected_col = st.selectbox("Select numeric column:", numeric_cols)
                if selected_col:
                    fig = px.histogram(analysis['sample_data'], x=selected_col, 
                                     title=f'Distribution of {selected_col}',
                                     nbins=20)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Feature Correlations")
            numeric_cols = analysis.get('numeric_columns', [])
            if len(numeric_cols) > 1:
                # Create correlation matrix
                corr_matrix = analysis['sample_data'][numeric_cols].corr()
                
                # Plot correlation heatmap
                fig = px.imshow(corr_matrix, 
                              aspect='auto', 
                              color_continuous_scale='RdBu_r',
                              title='Correlation Matrix',
                              labels=dict(color="Correlation"))
                st.plotly_chart(fig, use_container_width=True)
                
                # Show high correlation pairs
                high_corrs = analysis.get('high_correlations', [])
                if high_corrs:
                    st.subheader("Highly Correlated Features (|r| > 0.8)")
                    for pair in high_corrs[:5]:
                        st.write(f"- **{pair['col1']}** ↔ **{pair['col2']}**: {pair['correlation']}")

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
