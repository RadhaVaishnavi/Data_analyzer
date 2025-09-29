# universal_data_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
from transformers import pipeline
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
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
    """Deep analysis of tabular data content"""
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
        
        # Detect outliers using IQR
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
                        'correlation': corr_val
                    })
        analysis['high_correlations'] = high_corr_pairs
    
    # Data quality assessment
    quality_issues = []
    
    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            quality_issues.append(f"Constant column: {col}")
    
    # Check for high cardinality categorical
    for col in categorical_cols:
        if df[col].nunique() > 50:
            quality_issues.append(f"High cardinality: {col} ({df[col].nunique()} unique values)")
    
    # Check for high missing percentage
    for col, missing_pct in analysis['missing_percentage'].items():
        if missing_pct > 50:
            quality_issues.append(f"High missing values: {col} ({missing_pct}% missing)")
    
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
    
    # Content-based recommendations
    recommendations = "## 📊 CONTENT-BASED ANALYSIS\n\n"
    
    # Data quality issues
    if file_analysis.get('quality_issues'):
        recommendations += "### 🚨 DATA QUALITY ISSUES\n"
        for issue in file_analysis['quality_issues'][:5]:  # Show top 5 issues
            recommendations += f"- {issue}\n"
        recommendations += "\n"
    
    # Missing values summary
    total_missing = file_analysis.get('total_missing', 0)
    if total_missing > 0:
        missing_pct = (total_missing / (file_analysis['total_rows'] * len(file_analysis['columns'])) * 100)
        recommendations += f"### 📉 MISSING DATA\n"
        recommendations += f"- Total missing values: {total_missing} ({missing_pct:.1f}% of all data)\n"
        
        # Show columns with highest missing percentage
        high_missing_cols = sorted(
            [(col, pct) for col, pct in file_analysis.get('missing_percentage', {}).items() if pct > 10],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if high_missing_cols:
            recommendations += "- Columns needing attention:\n"
            for col, pct in high_missing_cols:
                recommendations += f"  - {col}: {pct}% missing\n"
        recommendations += "\n"
    
    # Model recommendations based on data characteristics
    numeric_cols = file_analysis.get('numeric_columns', [])
    categorical_cols = file_analysis.get('categorical_columns', [])
    total_rows = file_analysis.get('total_rows', 0)
    
    recommendations += "### 🤖 INTELLIGENT MODEL SUGGESTIONS\n"
    
    # Determine problem type based on data
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        # Mixed data - classification or regression
        if any('target' in col.lower() or 'label' in col.lower() or 'class' in col.lower() for col in file_analysis['columns']):
            recommendations += "- **Primary**: XGBoost Classifier (excellent for mixed data types)\n"
            recommendations += "- **Alternative**: Random Forest Classifier (robust, interpretable)\n"
            recommendations += "- **Advanced**: LightGBM (fast, great for categorical features)\n"
        else:
            recommendations += "- **Primary**: XGBoost Regressor (best for structured data)\n"
            recommendations += "- **Alternative**: Random Forest Regressor (handles outliers well)\n"
            recommendations += "- **Advanced**: CatBoost (excellent for categorical data)\n"
    elif len(numeric_cols) > 5:
        recommendations += "- **Primary**: Neural Networks (capture complex numeric patterns)\n"
        recommendations += "- **Alternative**: Gradient Boosting (XGBoost/LightGBM)\n"
        recommendations += "- **Baseline**: Linear Models (fast, interpretable)\n"
    else:
        recommendations += "- **Primary**: XGBoost (versatile for most tabular data)\n"
        recommendations += "- **Alternative**: Random Forest (stable, good default)\n"
        recommendations += "- **Baseline**: Logistic Regression/Linear Regression\n"
    
    recommendations += "\n### 🛠️ SPECIFIC PREPROCESSING STEPS\n"
    
    # Preprocessing based on actual data
    if file_analysis.get('missing_values', {}):
        recommendations += "- **Handle missing values**:\n"
        if total_missing < (0.05 * total_rows * len(file_analysis['columns'])):
            recommendations += "  - Use imputation (median for numeric, mode for categorical)\n"
        else:
            recommendations += "  - Consider advanced imputation (KNN, MICE) or removal\n"
    
    if categorical_cols:
        recommendations += "- **Encode categorical variables**:\n"
        high_card_cols = [col for col in categorical_cols 
                         if file_analysis.get('categorical_stats', {}).get(col, {}).get('unique_values', 0) > 10]
        if high_card_cols:
            recommendations += "  - High-cardinality features: Use target encoding or embedding\n"
        recommendations += "  - Low-cardinality: One-hot encoding\n"
    
    if numeric_cols:
        recommendations += "- **Scale numeric features**: StandardScaler or MinMaxScaler\n"
    
    if file_analysis.get('high_correlations'):
        recommendations += "- **Address multicollinearity**: Remove highly correlated features\n"
    
    # Outlier handling
    outlier_cols = [col for col, info in file_analysis.get('outliers', {}).items() 
                   if info.get('percentage', 0) > 5]
    if outlier_cols:
        recommendations += "- **Handle outliers**: Use robust models or outlier treatment\n"
    
    recommendations += "\n### 📈 DATA CHARACTERISTICS SUMMARY\n"
    recommendations += f"- **Dataset size**: {total_rows} rows × {len(file_analysis['columns'])} columns\n"
    recommendations += f"- **Numeric features**: {len(numeric_cols)}\n"
    recommendations += f"- **Categorical features**: {len(categorical_cols)}\n"
    recommendations += f"- **Data quality score**: {calculate_quality_score(file_analysis):.1f}/10\n"
    
    return recommendations

def calculate_quality_score(file_analysis):
    """Calculate data quality score based on multiple factors"""
    score = 10.0
    
    # Penalize for missing values
    missing_pct = file_analysis.get('total_missing', 0) / (file_analysis.get('total_rows', 1) * len(file_analysis.get('columns', 1)))
    score -= missing_pct * 20  # Up to 2 points penalty
    
    # Penalize for quality issues
    score -= min(len(file_analysis.get('quality_issues', [])), 5) * 0.5
    
    # Penalize for high cardinality
    high_card_cols = [col for col in file_analysis.get('categorical_columns', [])
                     if file_analysis.get('categorical_stats', {}).get(col, {}).get('unique_values', 0) > 50]
    score -= len(high_card_cols) * 0.3
    
    return max(0, min(10, score))

def get_universal_recommendations(file_analysis):
    """Fallback to universal recommendations"""
    data_type = file_analysis['data_type']
    file_size = file_analysis['file_size_mb']
    num_columns = len(file_analysis.get('columns', []))
    
    recommendations = {
        'tabular': f"""
**SMART ANALYSIS FOR TABULAR DATA**

📊 **Dataset Overview:**
- **Columns:** {num_columns} features
- **Size:** {file_size:.2f} MB
- **Type:** Structured tabular data

🤖 **MODEL SUGGESTIONS:**
- **Primary:** XGBoost (best for structured data)
- **Alternative:** Random Forest (interpretable, robust)
- **Advanced:** Neural Networks

🛠️ **PREPROCESSING NEEDED:**
- Handle missing values
- Encode categorical variables
- Scale numerical features
- Feature selection
""",
        'images': "Image analysis recommendations...",
        'text': "Text analysis recommendations...",
        'generic': "Generic data recommendations..."
    }
    
    return recommendations.get(data_type, recommendations['generic'])

def create_advanced_visualizations(analysis):
    """Create advanced visualizations based on content analysis"""
    if analysis['data_type'] == 'tabular':
        
        # Create multiple tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Data Preview", "📊 Missing Values", "📈 Distributions", 
            "🔗 Correlations", "📉 Quality Report"
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
                })
                
                fig = px.bar(missing_data.nlargest(10, 'Missing_Count'), 
                           x='Column', y='Missing_Percentage',
                           title='Top 10 Columns with Missing Values',
                           color='Missing_Percentage')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show missing values table
                st.subheader("Missing Values Details")
                st.dataframe(missing_data[missing_data['Missing_Count'] > 0], use_container_width=True)
            else:
                st.success("✅ No missing values detected!")
        
        with tab3:
            st.subheader("Data Distributions")
            numeric_cols = analysis.get('numeric_columns', [])
            if numeric_cols:
                selected_col = st.selectbox("Select numeric column:", numeric_cols)
                if selected_col:
                    fig = px.histogram(analysis['sample_data'], x=selected_col, 
                                     title=f'Distribution of {selected_col}')
                    st.plotly_chart(fig, use_container_width=True)
            
            categorical_cols = analysis.get('categorical_columns', [])
            if categorical_cols:
                selected_cat = st.selectbox("Select categorical column:", categorical_cols)
                if selected_cat:
                    value_counts = analysis['sample_data'][selected_cat].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f'Top 10 Values in {selected_cat}')
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Feature Correlations")
            numeric_cols = analysis.get('numeric_columns', [])
            if len(numeric_cols) > 1:
                corr_matrix = analysis['sample_data'][numeric_cols].corr()
                fig = px.imshow(corr_matrix, aspect='auto', color_continuous_scale='RdBu_r',
                              title='Correlation Matrix')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show high correlation pairs
                high_corrs = analysis.get('high_correlations', [])
                if high_corrs:
                    st.subheader("Highly Correlated Feature Pairs (>0.8)")
                    for pair in high_corrs[:5]:
                        st.write(f"- {pair['col1']} ↔ {pair['col2']}: {pair['correlation']:.3f}")
        
        with tab5:
            st.subheader("Data Quality Report")
            
            # Quality score
            quality_score = calculate_quality_score(analysis)
            st.metric("Overall Quality Score", f"{quality_score:.1f}/10")
            
            # Quality issues
            issues = analysis.get('quality_issues', [])
            if issues:
                st.error("🚨 Quality Issues Found:")
                for issue in issues[:5]:
                    st.write(f"- {issue}")
            else:
                st.success("✅ No major quality issues detected!")
                
            # Outlier summary
            outliers = analysis.get('outliers', {})
            if outliers:
                outlier_cols = [col for col, info in outliers.items() if info.get('percentage', 0) > 5]
                if outlier_cols:
                    st.warning(f"⚠️ Columns with significant outliers: {', '.join(outlier_cols[:3])}")

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
        
        # Show what the tool analyzes
        st.subheader("🔍 What This Tool Analyzes:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📈 For Tabular Data:**
            - Data quality and missing values
            - Feature distributions and outliers
            - Correlation patterns
            - Data type analysis
            - Quality scoring
            """)
        
        with col2:
            st.markdown("""
            **🤖 Intelligent Recommendations:**
            - Model selection based on data characteristics
            - Specific preprocessing steps needed
            - Data quality improvements
            - Hardware optimization tips
            """)

if __name__ == "__main__":
    main()
