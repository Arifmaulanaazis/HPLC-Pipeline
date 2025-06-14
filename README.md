# HPLC-Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![GUI](https://img.shields.io/badge/GUI-PySide6-green)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange)

A modern Python-based GUI application for HPLC chromatography data analysis, machine learning model training (regression & classification), prediction, condition optimization, and data visualization. This application is designed for researchers, laboratory analysts, and method developers in analytical chemistry, pharmaceuticals, and related sciences.

## 🚀 Key Features

### 🎨 Modern User Interface
- **Modern GUI** built with PySide6 featuring dark/light theme support
- Intuitive workflow with tabbed interface
- Real-time progress tracking with progress bars
- Comprehensive logging system

### 📊 Data Processing & Analysis
- **Multi-format support**: CSV & Excel file compatibility
- **Automatic molecular descriptor extraction** from SMILES using RDKit
- **QSRR Analysis** (Quantitative Structure-Retention Relationship)
- Advanced data preprocessing and feature engineering

### 🤖 Machine Learning Capabilities
- **Multiple algorithms**: Random Forest, SVM, MLP Neural Networks, Linear/Ridge Regression
- **Dual modeling**: Both regression (retention time) and classification (separation quality)
- **Hyperparameter optimization** with customizable parameters
- **Model evaluation**: Cross-validation, R², RMSE, accuracy metrics
- **Model persistence**: Save/load trained models

### 🔬 HPLC-Specific Features
- **Condition optimization**: Automated pH, buffer, and organic phase optimization
- **Retention time prediction** for new compounds
- **Separation quality assessment** with quality scoring
- **Method development support** with optimal condition suggestions

### 📈 Advanced Visualizations
- **Interactive plots**: Model performance, residuals, feature importance
- **Dimensionality reduction**: PCA, t-SNE, UMAP analysis
- **Heatmaps**: Correlation matrices and feature relationships
- **Dendrograms**: Hierarchical clustering visualization
- **Export capabilities**: High-quality PNG, PDF, and SVG formats

### 💾 Export & Documentation
- **Comprehensive exports**: Results to Excel, models to pickle
- **Batch processing**: Handle multiple compounds simultaneously
- **Built-in help system** with detailed documentation
- **Report generation**: Automated analysis reports

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Arifmaulanaazis/HPLC-Pipeline.git
   cd HPLC-Pipeline
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Install UMAP for advanced dimensionality reduction:**
   ```bash
   pip install umap-learn
   ```

## 📦 Dependencies

### Core Dependencies
- **PySide6**: Modern Qt-based GUI framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing
- **RDKit**: Cheminformatics and molecular descriptors
- **joblib**: Model serialization and parallel processing

### Optional Dependencies
- **umap-learn**: UMAP dimensionality reduction
- **seaborn**: Enhanced statistical visualizations

## 🚀 Quick Start

### Running the Application
```bash
python main.py
```

### Basic Workflow

1. **Launch the application**
   ```bash
   python main.py
   ```

2. **Load your data**
   - Click "Browse" to select your CSV or Excel file
   - Ensure your data contains required columns (see Data Format below)
   - Click "Load Data"

3. **Configure models**
   - Go to **Model > Select Models...** 
   - Choose desired algorithms (Random Forest, SVM, MLP, etc.)
   - Adjust hyperparameters if needed
   - Click "Apply"

4. **Train models**
   - Click "Train Models" button
   - Monitor progress in the status bar
   - View training logs in the log panel

5. **Analyze results**
   - Navigate to **Model Results** tab for performance metrics
   - Check **Visualizations** tab for plots and charts
   - Review feature importance and model insights

6. **Make predictions**
   - Enter SMILES notation for new compounds
   - Select trained model from dropdown
   - Click "Predict" for retention time and separation quality

7. **Optimize conditions**
   - Click "Suggest Optimal Condition" 
   - Review recommended pH, buffer, and organic phase settings
   - Export optimization results

8. **Export results**
   - Save models for future use
   - Export results to Excel
   - Generate visualization reports

## 📋 Data Format Requirements

Your input data should contain the following columns:

| Column Name | Description | Example |
|-------------|-------------|---------|
| `Compound` | Compound identifier | "Caffeine" |
| `SMILES` | SMILES molecular notation | "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" |
| `Retention_Time_min` | Retention time in minutes | 2.35 |
| `Separation_Quality` | Quality category | "Completely separated" |
| `Buffer_%` | Buffer percentage | 60 |
| `Organic_%` | Organic phase percentage | 40 |
| `pH` | pH value | 3.0 |

### Sample Data Format
```csv
Compound,SMILES,Retention_Time_min,Separation_Quality,Buffer_%,Organic_%,pH
Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,2.35,Completely separated,60,40,3.0
Theophylline,CN1C(=O)N(C)C2=C1N=CN2,1.85,Partially separated,65,35,3.5
Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O,3.20,Completely separated,55,45,2.8
```

## 🎯 Advanced Features

### Model Selection & Optimization
- **Regression Models**: Linear, Ridge, Random Forest, SVM, MLP
- **Classification Models**: Random Forest, SVM, MLP classifiers
- **Hyperparameter tuning**: Grid search and random search options
- **Cross-validation**: K-fold validation for robust evaluation

### Molecular Descriptors
Automatically calculated from SMILES:
- **Basic descriptors**: Molecular weight, LogP, TPSA
- **Structural features**: Ring count, aromatic atoms, H-bond donors/acceptors
- **Complexity measures**: Bertz complexity index
- **Pharmacophore features**: Custom pharmacophore fingerprints

### Visualization Suite
- **Performance plots**: R² vs RMSE, confusion matrices
- **Feature analysis**: Importance plots, correlation heatmaps
- **Dimensionality reduction**: PCA, t-SNE, UMAP projections
- **Clustering analysis**: Hierarchical clustering dendrograms

## 🔧 Configuration

### Model Parameters
Access model configuration through **Model > Select Models...**:

- **Random Forest**: n_estimators, max_depth, min_samples_split
- **SVM**: C, gamma, kernel type
- **MLP**: hidden_layer_sizes, activation, solver
- **Linear Models**: alpha (for Ridge regression)

### Application Settings
- **Theme**: Toggle between dark and light modes
- **Export format**: Choose PNG, PDF, or SVG for visualizations
- **Logging level**: Adjust verbosity of log messages

## 🐛 Troubleshooting

### Common Issues

1. **RDKit import error**
   ```bash
   pip install rdkit-pypi
   ```

2. **Missing UMAP functionality**
   ```bash
   pip install umap-learn
   ```

3. **GUI not displaying properly**
   - Ensure PySide6 is properly installed
   - Check system Qt compatibility

4. **Memory issues with large datasets**
   - Reduce dataset size for initial testing
   - Close unused applications
   - Consider using a machine with more RAM

## 📈 Performance Tips

- **Data size**: Optimal performance with 100-10,000 compounds
- **Feature selection**: Use correlation analysis to remove redundant features
- **Model selection**: Start with Random Forest for baseline performance
- **Cross-validation**: Use 5-fold CV for balanced accuracy/speed

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support & Contact

- **Author**: Arif Maulana Azis
- **Email**: titandigitalsoft@gmail.com
- **GitHub**: [@Arifmaulanaazis](https://github.com/Arifmaulanaazis)
- **Issues**: [GitHub Issues](https://github.com/Arifmaulanaazis/HPLC-Pipeline/issues)

## 🙏 Acknowledgments

- **RDKit** community for excellent cheminformatics tools
- **Scikit-learn** team for comprehensive ML algorithms
- **PySide6** developers for the modern GUI framework
- **Open source community** for continuous inspiration and support

## 📚 Documentation

For detailed documentation, tutorials, and examples:
- Launch the application and navigate to **Help > Documentation**
- Check the `docs/` folder in the repository
- Visit our [Wiki](https://github.com/Arifmaulanaazis/HPLC-Pipeline/wiki) for advanced tutorials

## 🔮 Roadmap

- [ ] **Deep learning models** integration (TensorFlow/PyTorch)
- [ ] **Real-time data streaming** from HPLC instruments
- [ ] **Cloud deployment** options
- [ ] **Multi-language support** (Indonesian, Chinese, Spanish)
- [ ] **Advanced optimization algorithms** (Bayesian optimization)
- [ ] **Integration with chemical databases** (ChEMBL, PubChem)

---

**© 2025 Titan Digitalsoft. For more information, see the built-in help documentation (Help menu).**

*Made with ❤️ for the analytical chemistry community*