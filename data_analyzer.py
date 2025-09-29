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

# Set page config
st.set_page_config(
    page_title="Universal Data Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'llm_loaded' not in st.session_state:
    st.session_state.llm_loaded = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

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
            detected_type = extension_map[file_ext]
            
            # Verify the file can be read
            if detected_type == 'tabular':
                try:
                    if file_ext == '.csv':
                        pd.read_csv(file_path, nrows=5)
                    elif file_ext == '.xlsx':
                        pd.read_excel(file_path, nrows=5)
                    return 'tabular'
                except:
                    return 'generic'
                    
            elif detected_type == 'images':
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    return 'images'
                except:
                    return 'generic'
                    
            elif detected_type == 'text':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read(1024)
                    return 'text'
                except:
                    return 'generic'
            
            return detected_type
        else:
            return 'generic'
            
    except Exception as e:
        st.error(f"Error detecting data type: {e}")
        return 'generic'

def analyze_file_basics(file_path):
    """Basic analysis that works for any file type"""
    file_stats = {
        'file_name': os.path.basename(file_path),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'file_extension': os.path.splitext(file_path)[1],
        'data_type': detect_data_type(file_path)
    }
    
    # Add type-specific basic info
    if file_stats['data_type'] == 'tabular':
        try:
            if file_stats['file_extension'] == '.csv':
                df = pd.read_csv(file_path, nrows=1000)
            elif file_stats['file_extension'] == '.xlsx':
                df = pd.read_excel(file_path, nrows=1000)
            else:
                df = pd.read_csv(file_path, nrows=1000)
                
            file_stats['columns'] = list(df.columns)
            file_stats['sample_rows'] = len(df)
            file_stats['data_types'] = df.dtypes.astype(str).to_dict()
            file_stats['missing_values'] = df.isnull().sum().to_dict()
            file_stats['sample_data'] = df.head(10)
            
        except Exception as e:
            file_stats['error'] = f"Could not read tabular data: {e}"
            
    elif file_stats['data_type'] == 'text':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(5000)
            file_stats['content_preview'] = content[:500] + "..." if len(content) > 500 else content
            file_stats['line_count'] = len(content.split('\n'))
        except Exception as e:
            file_stats['error'] = f"Could not read text file: {e}"
    
    elif file_stats['data_type'] == 'images':
        try:
            with Image.open(file_path) as img:
                file_stats['image_size'] = img.size
                file_stats['image_mode'] = img.mode
                file_stats['image_format'] = img.format
        except Exception as e:
            file_stats['error'] = f"Could not read image: {e}"
    
    return file_stats

def get_universal_recommendations(file_analysis):
    """Smart recommendations that work for ANY data type automatically"""
    data_type = file_analysis['data_type']
    file_size = file_analysis['file_size_mb']
    num_columns = len(file_analysis.get('columns', []))
    
    # Universal base template
    recommendations = {
        'tabular': f"""
**SMART ANALYSIS FOR TABULAR DATA**

📊 **Dataset Overview:**
- **Columns:** {num_columns} features
- **Size:** {file_size:.2f} MB
- **Type:** Structured tabular data

🤖 **MODEL SUGGESTIONS:**
- **Primary:** XGBoost (best for structured data, handles missing values well)
- **Alternative:** Random Forest (interpretable, robust to outliers)
- **Advanced:** Neural Networks (if patterns are complex and non-linear)

🛠️ **PREPROCESSING NEEDED:**
- Handle missing values (impute with median/mode or remove)
- Encode categorical variables (one-hot encoding for few categories, label encoding for many)
- Scale numerical features (StandardScaler for normal distribution, MinMaxScaler for bounded ranges)
- Feature selection (remove highly correlated or low-variance features)

💻 **HARDWARE OPTIMIZATION:**
- Your system can handle this dataset easily
- Use GPU-accelerated XGBoost for faster training
- Process in memory - no sampling needed
- Consider distributed computing if dataset grows significantly
""",
        'images': f"""
**SMART ANALYSIS FOR IMAGE DATA**

🖼️ **Dataset Overview:**
- **Size:** {file_size:.2f} MB
- **Type:** Image data
- **Format:** {file_analysis.get('image_format', 'Unknown')}

🤖 **MODEL SUGGESTIONS:**
- **Primary:** ResNet-50 (excellent for most vision tasks, good balance of accuracy/speed)
- **Alternative:** EfficientNet (better speed/accuracy balance for mobile/edge devices)
- **Object Detection:** YOLOv8 (fast and accurate for real-time detection)
- **Segmentation:** U-Net (medical images, precise boundaries)

🛠️ **PREPROCESSING NEEDED:**
- Resize images to consistent dimensions (224x224 for ResNet, 320x320 for EfficientNet)
- Normalize pixel values (divide by 255 for 0-1 range or use ImageNet stats)
- Data augmentation (random rotation, flip, brightness/contrast adjustments)
- Split into train/validation/test sets (70/15/15 recommended)

💻 **HARDWARE OPTIMIZATION:**
- Use mixed precision training (FP16) to save GPU memory
- Batch size 16-32 depending on image size and GPU memory
- Enable CUDA acceleration for all operations
- Use data loaders with multiple workers for faster loading
""",
        'text': f"""
**SMART ANALYSIS FOR TEXT DATA**

📝 **Dataset Overview:**
- **Size:** {file_size:.2f} MB
- **Type:** Text data
- **Lines:** {file_analysis.get('line_count', 'Unknown')}

🤖 **MODEL SUGGESTIONS:**
- **Primary:** BERT (state-of-the-art for most NLP tasks, excellent understanding)
- **Alternative:** DistilBERT (40% smaller, 60% faster, 95% of BERT's performance)
- **Classic:** TF-IDF + SVM (fast baseline, good for simple classification)
- **Sequence:** LSTM/GRU (for time-series text data)

🛠️ **PREPROCESSING NEEDED:**
- Tokenization (split text into words/subwords)
- Padding/truncation to consistent length
- Remove stop words and special characters (domain-dependent)
- Text cleaning (lowercase, remove URLs/emails, handle contractions)
- Train/validation split with stratification if classification

💻 **HARDWARE OPTIMIZATION:**
- Fine-tune BERT on GPU for best results
- Use smaller batch sizes for long texts to avoid OOM
- Consider gradient accumulation for effective larger batches
- Use attention optimization (flash attention) if available
""",
        'generic': f"""
**SMART ANALYSIS FOR UNKNOWN DATA**

🔍 **Dataset Overview:**
- **Size:** {file_size:.2f} MB
- **Type:** Unknown/Generic data format

🤖 **MODEL SUGGESTIONS:**
- Start with exploratory data analysis to understand patterns
- Try multiple approaches: simple models first, then complex
- Consider both supervised and unsupervised learning
- Use cross-validation to evaluate model performance

🛠️ **PREPROCESSING NEEDED:**
- Explore data structure and format first
- Convert to appropriate data format if needed
- Handle missing/incorrect values appropriately
- Feature engineering based on domain knowledge
- Normalize/standardize based on data distribution

💻 **HARDWARE OPTIMIZATION:**
- Start with small samples to test approaches
- Scale up as understanding improves
- Monitor resource usage during processing
- Use appropriate data structures for efficiency
"""
    }
    
    # Select the appropriate template
    base_recommendation = recommendations.get(data_type, recommendations['generic'])
    
    # Add data-specific insights
    additional_insights = "\n**📈 DATA-SPECIFIC INSIGHTS:**\n"
    
    if data_type == 'tabular':
        if num_columns > 50:
            additional_insights += "• **High-dimensional data:** Consider PCA or feature selection to reduce complexity\n"
        if num_columns < 10:
            additional_insights += "• **Low-dimensional data:** Focus on feature engineering and interaction terms\n"
        if file_size > 100:
            additional_insights += "• **Large dataset:** Use incremental learning or sampling for initial experiments\n"
        else:
            additional_insights += "• **Dataset size:** Easily manageable on standard hardware\n"
        
        # Add column type insights
        if 'data_types' in file_analysis:
            numeric_cols = sum(1 for dtype in file_analysis['data_types'].values() 
                             if 'int' in dtype or 'float' in dtype)
            categorical_cols = num_columns - numeric_cols
            additional_insights += f"• **Column types:** {numeric_cols} numeric, {categorical_cols} categorical\n"
    
    elif data_type == 'images':
        if file_size > 500:
            additional_insights += "• **Large image dataset:** Consider transfer learning to save training time\n"
        else:
            additional_insights += "• **Image dataset:** Suitable for training from scratch or fine-tuning\n"
        
        if 'image_size' in file_analysis:
            additional_insights += f"• **Image dimensions:** {file_analysis['image_size']}\n"
    
    elif data_type == 'text':
        if file_size > 50:
            additional_insights += "• **Large text corpus:** Consider distributed processing for efficiency\n"
        else:
            additional_insights += "• **Text dataset:** Easily processable on single machine\n"
    
    return base_recommendation + additional_insights

def create_visualizations(analysis):
    """Create visualizations based on data type"""
    if analysis['data_type'] == 'tabular' and 'sample_data' in analysis:
        df = analysis['sample_data']
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Missing Values", "Data Types"])
        
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df, use_container_width=True)
        
        with tab2:
            st.subheader("Missing Values Analysis")
            missing_data = pd.DataFrame({
                'Column': list(analysis['missing_values'].keys()),
                'Missing_Count': list(analysis['missing_values'].values())
            })
            missing_data['Missing_Percentage'] = (missing_data['Missing_Count'] / len(df) * 100).round(2)
            
            fig = px.bar(missing_data, x='Column', y='Missing_Percentage', 
                        title='Missing Values by Column (%)',
                        color='Missing_Percentage')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Data Types Distribution")
            type_counts = pd.Series(analysis['data_types']).value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        title='Data Types Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis['data_type'] == 'text':
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Text Preview")
            st.text_area("Content", analysis.get('content_preview', 'No content'), height=200)
        
        with col2:
            st.subheader("Text Statistics")
            stats_data = {
                'Metric': ['File Size', 'Lines', 'Characters'],
                'Value': [
                    f"{analysis['file_size_mb']:.2f} MB",
                    analysis.get('line_count', 'N/A'),
                    len(analysis.get('content_preview', ''))
                ]
            }
            st.table(pd.DataFrame(stats_data))
    
    elif analysis['data_type'] == 'images':
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image Preview")
            try:
                image = Image.open(analysis['file_name'])
                st.image(image, caption=analysis['file_name'], use_column_width=True)
            except:
                st.error("Could not display image")
        
        with col2:
            st.subheader("Image Properties")
            props_data = {
                'Property': ['Size', 'Dimensions', 'Mode', 'Format'],
                'Value': [
                    f"{analysis['file_size_mb']:.2f} MB",
                    str(analysis.get('image_size', 'N/A')),
                    analysis.get('image_mode', 'N/A'),
                    analysis.get('image_format', 'N/A')
                ]
            }
            st.table(pd.DataFrame(props_data))

# Streamlit UI
def main():
    st.title("🤖 Universal Data Analyzer")
    st.markdown("Upload **ANY** dataset (CSV, Images, Text, Excel) and get intelligent analysis and model recommendations!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("This tool analyzes any dataset and provides tailored machine learning recommendations.")
        
        st.subheader("Hardware Info")
        if torch.cuda.is_available():
            st.success(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ No GPU detected - using CPU")
    
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
            # Analyze file
            with st.spinner("🔍 Analyzing your dataset..."):
                analysis = analyze_file_basics(tmp_path)
                st.session_state.analysis_results = analysis
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("File Type", analysis['data_type'].upper())
            with col2:
                st.metric("File Size", f"{analysis['file_size_mb']:.2f} MB")
            with col3:
                if 'columns' in analysis:
                    st.metric("Columns", len(analysis['columns']))
                else:
                    st.metric("Properties", "N/A")
            with col4:
                if 'sample_rows' in analysis:
                    st.metric("Sample Rows", analysis['sample_rows'])
                else:
                    st.metric("Status", "Analyzed")
            
            # Visualizations
            st.subheader("📊 Data Exploration")
            create_visualizations(analysis)
            
            # Recommendations
            st.subheader("🎯 Intelligent Recommendations")
            with st.expander("View Detailed Recommendations", expanded=True):
                recommendations = get_universal_recommendations(analysis)
                st.markdown(recommendations)
            
            # Next steps
            st.subheader("🚀 Implementation Guide")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**Immediate Actions:**")
                st.markdown("""
                1. **Review data quality** - Check for missing values and inconsistencies
                2. **Implement preprocessing** - Follow the recommended steps
                3. **Start with primary model** - Quick prototype with suggested approach
                4. **Validate results** - Use cross-validation and appropriate metrics
                """)
            
            with col2:
                st.info("**Advanced Steps:**")
                st.markdown("""
                1. **Feature engineering** - Create domain-specific features
                2. **Model tuning** - Hyperparameter optimization
                3. **Ensemble methods** - Combine multiple models
                4. **Deployment** - Model serving and monitoring
                """)
        
        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    else:
        # Show demo when no file uploaded
        st.info("👆 Upload a file to get started!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Tabular Data")
            st.markdown("""
            - CSV, Excel, Parquet files
            - Automatic column analysis
            - Missing value detection
            - Model recommendations
            """)
        
        with col2:
            st.subheader("🖼️ Image Data")
            st.markdown("""
            - JPG, PNG, JPEG files
            - Image property analysis
            - Computer vision models
            - Augmentation strategies
            """)
        
        with col3:
            st.subheader("📝 Text Data")
            st.markdown("""
            - TXT, JSON, XML files
            - Text statistics
            - NLP model suggestions
            - Preprocessing guidance
            """)

if __name__ == "__main__":
    main()
