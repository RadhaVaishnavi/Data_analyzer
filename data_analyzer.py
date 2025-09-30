import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class LightweightDatasetAnalyzer:
    def __init__(self):
        self.analysis_results = {}
        
    def analyze(self, df, analysis_type="quick"):
        """Main analysis method"""
        self.df = df.copy()
        self.analysis_type = analysis_type
        
        analyses = [
            self._basic_info,
            self._data_quality_check,
            self._statistical_analysis,
            self._pattern_detection
        ]
        
        if analysis_type == "comprehensive":
            analyses.extend([
                self._bias_detection,
                self._compliance_check,
                self._resource_analysis
            ])
        
        for analysis in analyses:
            try:
                analysis()
            except Exception as e:
                print(f"Analysis failed: {analysis.__name__}: {e}")
        
        return self.analysis_results
    
    def _basic_info(self):
        """Question 1, 3, 5, 6: Basic dataset info"""
        info = {
            'shape': self.df.shape,
            'size_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'sample_data': self.df.head(3).to_dict('records')
        }
        self.analysis_results['basic_info'] = info
    
    def _data_quality_check(self):
        """Questions 7, 9, 10, 11: Data quality assessment"""
        quality = {}
        
        # Missing values (Q7)
        missing = self.df.isnull().sum()
        quality['missing_values'] = {
            'total': int(missing.sum()),
            'by_column': missing[missing > 0].to_dict(),
            'percentage': (missing / len(self.df) * 100).round(2).to_dict()
        }
        
        # Duplicates (Q11)
        quality['duplicates'] = {
            'exact_duplicates': self.df.duplicated().sum(),
            'percentage_duplicates': (self.df.duplicated().sum() / len(self.df) * 100).round(2)
        }
        
        # Data consistency (Q9)
        consistency_issues = {}
        for col in self.df.select_dtypes(include=['object']).columns:
            unique_vals = self.df[col].dropna().unique()
            if len(unique_vals) < min(50, len(self.df) * 0.1):  # Avoid high cardinality
                consistency_issues[col] = {
                    'unique_count': len(unique_vals),
                    'values': unique_vals.tolist()[:10]  # First 10 values
                }
        quality['consistency_check'] = consistency_issues
        
        # Validity (Q10) - Basic checks
        validity_issues = {}
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check for common validity issues
                sample_vals = self.df[col].dropna().head(100)
                if sample_vals.str.contains(r'[^\w\s@.-]', na=False).any():
                    validity_issues[col] = 'Potential special character issues'
        
        quality['validity_issues'] = validity_issues
        self.analysis_results['data_quality'] = quality
    
    def _statistical_analysis(self):
        """Questions 8, 16, 17: Statistical analysis"""
        stats_info = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Basic stats
            stats_info['descriptive_stats'] = self.df[numeric_cols].describe().to_dict()
            
            # Outliers (Q16)
            outliers = {}
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                               (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                outliers[col] = {
                    'outlier_count': int(outlier_count),
                    'percentage': (outlier_count / len(self.df) * 100).round(2)
                }
            stats_info['outliers'] = outliers
            
            # Correlations (Q17)
            if len(numeric_cols) > 1:
                corr_matrix = self.df[numeric_cols].corr()
                # Find high correlations
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr.append({
                                'features': [corr_matrix.columns[i], corr_matrix.columns[j]],
                                'correlation': round(corr_matrix.iloc[i, j], 3)
                            })
                stats_info['high_correlations'] = high_corr
        
        self.analysis_results['statistical_analysis'] = stats_info
    
    def _pattern_detection(self):
        """Questions 13, 15: Pattern and bias detection"""
        patterns = {}
        
        # Feature relevance (Q13)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Variance analysis - low variance features might be less useful
            variances = self.df[numeric_cols].var()
            low_variance = variances[variances < 0.01]  # Threshold for low variance
            patterns['low_variance_features'] = low_variance.index.tolist()
        
        # Sampling bias (Q15)
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        bias_analysis = {}
        for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            value_counts = self.df[col].value_counts(normalize=True)
            if len(value_counts) > 1:  # Multiple classes
                imbalance_ratio = value_counts.max() / value_counts.min()
                bias_analysis[col] = {
                    'imbalance_ratio': round(imbalance_ratio, 2),
                    'majority_class_percentage': round(value_counts.max() * 100, 2)
                }
        
        patterns['bias_analysis'] = bias_analysis
        self.analysis_results['pattern_analysis'] = patterns
    
    def _bias_detection(self):
        """Comprehensive bias and fairness analysis"""
        bias_results = {}
        
        # Numerical bias detection
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            skewness = self.df[col].skew()
            if abs(skewness) > 1:  # Highly skewed
                bias_results[col] = f"High skewness: {skewness:.2f}"
        
        self.analysis_results['bias_detection'] = bias_results
    
    def _compliance_check(self):
        """Question 4: Basic PII/sensitive data detection"""
        pii_keywords = ['email', 'phone', 'ssn', 'address', 'name', 'credit', 'password']
        sensitive_cols = []
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in pii_keywords):
                sensitive_cols.append({
                    'column': col,
                    'sensitive_type': 'potential_pii',
                    'unique_values': self.df[col].nunique() if self.df[col].dtype == 'object' else 'numeric'
                })
        
        self.analysis_results['compliance_check'] = {
            'sensitive_columns': sensitive_cols,
            'recommendation': 'Review identified columns for PII compliance'
        }
    
    def _resource_analysis(self):
        """Question 21: Resource requirements analysis"""
        size_mb = self.df.memory_usage(deep=True).sum() / 1024**2
        n_rows, n_cols = self.df.shape
        
        resource_guide = {}
        if size_mb < 100:  # Small dataset
            resource_guide['size_category'] = 'Small (<100MB)'
            resource_guide['recommended_approach'] = 'Single machine processing'
            resource_guide['tools'] = ['Pandas', 'scikit-learn']
        elif size_mb < 1000:  # Medium dataset
            resource_guide['size_category'] = 'Medium (100MB-1GB)'
            resource_guide['recommended_approach'] = 'Optimized single machine or basic distributed'
            resource_guide['tools'] = ['Pandas with chunking', 'Dask', 'Vaex']
        else:  # Large dataset
            resource_guide['size_category'] = 'Large (>1GB)'
            resource_guide['recommended_approach'] = 'Distributed processing'
            resource_guide['tools'] = ['Spark', 'Dask', 'Ray']
        
        resource_guide['estimated_memory'] = f"{size_mb:.2f} MB"
        resource_guide['shape'] = f"{n_rows} rows × {n_cols} columns"
        
        self.analysis_results['resource_analysis'] = resource_guide
    
    def generate_report(self):
        """Generate a text report from analysis results"""
        report = []
        report.append("=" * 60)
        report.append("DATASET ANALYSIS REPORT")
        report.append("=" * 60)
        
        for section, content in self.analysis_results.items():
            report.append(f"\n{section.upper().replace('_', ' ')}:")
            report.append("-" * 40)
            
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, (dict, list)):
                        report.append(f"{key}:")
                        for item in (value if isinstance(value, list) else value.items()):
                            report.append(f"  - {item}")
                    else:
                        report.append(f"{key}: {value}")
            else:
                report.append(str(content))
        
        return "\n".join(report)
    
    def suggest_solutions(self, issues):
        """Generate solution suggestions based on detected issues"""
        solutions = []
        
        # Missing values solutions
        if 'missing_values' in str(issues):
            solutions.extend([
                "Imputation: Use mean/median/mode for numerical, mode for categorical",
                "Dropping: Remove rows/columns with high missing percentage",
                "Advanced: Use KNN imputation or ML-based imputation"
            ])
        
        # Outlier solutions
        if 'outliers' in str(issues):
            solutions.extend([
                "Capping: Use IQR method to cap extreme values",
                "Transformation: Apply log/Box-Cox transformation",
                "Modeling: Use robust statistical methods"
            ])
        
        # Bias solutions
        if 'bias' in str(issues).lower():
            solutions.extend([
                "Resampling: Over/under sampling for class imbalance",
                "Augmentation: Synthetic data generation",
                "Weighting: Use class weights in models"
            ])
        
        return solutions

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    sample_data = {
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'email': [f'user{i}@test.com' for i in range(1000)],
        'category': np.random.choice(['A', 'B', 'C'], 1000, p=[0.7, 0.2, 0.1]),
        'missing_col': np.random.choice([1, 2, np.nan], 1000, p=[0.4, 0.4, 0.2])
    }
    
    df = pd.DataFrame(sample_data)
    
    # Analyze dataset
    analyzer = LightweightDatasetAnalyzer()
    results = analyzer.analyze(df, analysis_type="comprehensive")
    
    # Generate report
    report = analyzer.generate_report()
    print(report)
    
    # Save report
    with open('dataset_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("\nAnalysis complete! Report saved to 'dataset_analysis_report.txt'")
