import os
os.environ['MPLBACKEND'] = 'QtAgg'
import sys
import traceback
import math
import pickle
import base64

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from image_data import HPLC, ICON_APP

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QProgressBar,
    QTabWidget, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QGridLayout, QScrollArea, QFrame, QDialog, QCheckBox, QComboBox,
    QSizePolicy, QTextBrowser
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QIcon, QPixmap


class WorkerThread(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished_with_data = Signal(object)
    error_occurred = Signal(str)
    
    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
        
    def run(self):
        try:
            if self.task_type == "load_data":
                self.load_data_task()
            elif self.task_type == "train_models":
                self.train_models_task()
            elif self.task_type == "predict":
                self.predict_task()
        except Exception as e:
            self.error_occurred.emit(f"Error in {self.task_type}: {str(e)}\n{traceback.format_exc()}")
    
    def load_data_task(self):
        file_path = self.kwargs['file_path']
        self.status.emit("Loading data...")
        self.progress.emit(10)
        
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
            chrom_df = None
        elif file_path.lower().endswith(('.xls', '.xlsx')):
            xls = pd.read_excel(file_path, sheet_name=None)
            df = xls[list(xls.keys())[0]].copy()
            chrom_df = xls.get('chromatogram')
        else:
            raise ValueError("Input must be .csv or .xlsx/.xls")
        
        self.progress.emit(30)
        self.status.emit("Computing molecular descriptors...")
        
        descriptors_list = []
        for i, smiles in enumerate(df['SMILES']):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    desc_dict = {
                        'MolWt': Descriptors.MolWt(mol),
                        'LogP': Descriptors.MolLogP(mol),
                        'TPSA': Descriptors.TPSA(mol),
                        'HBD': Descriptors.NumHDonors(mol),
                        'HBA': Descriptors.NumHAcceptors(mol),
                    }
                else:
                    desc_dict = {k: 0 for k in ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA']}
                descriptors_list.append(desc_dict)
                
                if i % max(1, len(df) // 10) == 0:
                    progress = 30 + int((i / len(df)) * 30)
                    self.progress.emit(progress)
                    
            except Exception as e:
                descriptors_list.append({k: 0 for k in ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA']})
        
        descs_df = pd.DataFrame(descriptors_list)
        df = pd.concat([df, descs_df], axis=1)
        
        self.progress.emit(60)
        self.status.emit("Processing features...")
        
        solvents = [c for c in df.columns if c.endswith('_%')]
        
        if 'pH' in df.columns:
            df['pH'] = pd.to_numeric(df['pH'], errors='coerce')
        
        if 'Separation_Quality' in df.columns:
            df['Separation_Quality'] = df['Separation_Quality'].fillna('No data')
            le = LabelEncoder()
            df['SepQuality_Code'] = le.fit_transform(df['Separation_Quality'])
        else:
            le = None
        
        self.progress.emit(100)
        self.status.emit("Data loaded successfully!")
        
        result = {
            'dataframe': df,
            'chromatogram': chrom_df,
            'solvents': solvents,
            'label_encoder': le
        }
        self.finished_with_data.emit(result)
    
    def train_models_task(self):
        df = self.kwargs['dataframe']
        solvents = self.kwargs['solvents']
        label_enc = self.kwargs['label_encoder']
        output_dir = self.kwargs['output_dir']
        selected_reg_models = self.kwargs.get('selected_reg_models', ['LR', 'RF', 'GB', 'SVR', 'MLP'])
        selected_clf_models = self.kwargs.get('selected_clf_models', ['RFC', 'GBC', 'SVC', 'MLPC'])
        custom_hyperparams = self.kwargs.get('custom_hyperparams', {})
        os.makedirs(output_dir, exist_ok=True)
        feat_cols = solvents + ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'pH']
        reg_models = {
            'LR': (
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('mdl', LinearRegression())
                ]),
                {'mdl__fit_intercept': [True, False]}
            ),
            'Ridge': (
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('mdl', Ridge())
                ]),
                {'mdl__alpha': [1.0, 10.0]}
            ),
            'Lasso': (
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('mdl', Lasso(max_iter=1000))
                ]),
                {'mdl__alpha': [0.1, 1.0]}
            ),
            'RF': (
                RandomForestRegressor(random_state=42),
                {'n_estimators': [50, 100], 'max_depth': [None, 10]}
            ),
            'GB': (
                GradientBoostingRegressor(random_state=42),
                {'n_estimators': [50, 100], 'learning_rate': [0.1], 'max_depth': [3, 5]}
            ),
            'SVR': (
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('mdl', SVR())
                ]),
                {'mdl__C': [1, 10], 'mdl__gamma': ['scale']}
            ),
            'MLP': (
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('mdl', MLPRegressor(
                        random_state=42, max_iter=500,
                        early_stopping=True, tol=1e-4
                    ))
                ]),
                {'mdl__hidden_layer_sizes': [(50,)], 'mdl__alpha': [1e-3]}
            ),
            'KNN': (
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('mdl', KNeighborsRegressor())
                ]),
                {'mdl__n_neighbors': [3, 5, 7]}
            ),
            'DT': (
                DecisionTreeRegressor(random_state=42),
                {'max_depth': [None, 5, 10]}
            ),
        }
        clf_models = {
            'RFC': (RandomForestClassifier(random_state=42),
                    {'n_estimators': [50, 100], 'max_depth': [None, 10]}),
            'GBC': (GradientBoostingClassifier(random_state=42),
                    {'n_estimators': [50, 100], 'learning_rate': [0.1], 'max_depth': [3]}),
            'SVC': (Pipeline([
                        ('scaler', StandardScaler()), 
                        ('mdl', SVC(probability=True))
                    ]),
                    {'mdl__C': [1, 10], 'mdl__gamma': ['scale']}),
            'MLPC': (Pipeline([
                          ('scaler', StandardScaler()),
                          ('mdl', MLPClassifier(
                              random_state=42, max_iter=500,
                              early_stopping=True, tol=1e-4
                          ))
                      ]),
                      {'mdl__hidden_layer_sizes': [(50,)], 'mdl__alpha': [1e-3]}),
            'KNN': (Pipeline([
                        ('scaler', StandardScaler()),
                        ('mdl', KNeighborsClassifier())
                    ]),
                    {'mdl__n_neighbors': [3, 5, 7]}),
            'DT': (DecisionTreeClassifier(random_state=42),
                   {'max_depth': [None, 5, 10]}),
            'LR': (Pipeline([
                        ('scaler', StandardScaler()),
                        ('mdl', LogisticRegression(max_iter=500))
                    ]),
                    {'mdl__C': [1.0, 10.0]})
        }
        total_models = len(selected_reg_models) + len(selected_clf_models)
        current_model = 0
        results = {'regression': {}, 'classification': {}, 'all_reg_models': {}, 'all_clf_models': {}}
        if 'Retention_Time_min' in df.columns:
            self.status.emit("Training regression models...")
            df_reg = df.dropna(subset=['Retention_Time_min'])
            if not df_reg.empty:
                X_reg = df_reg[feat_cols].fillna(0)
                y_reg = df_reg['Retention_Time_min']
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_reg, y_reg, test_size=0.2, random_state=42
                )
                best_reg, best_rmse = None, np.inf
                reg_results = []
                for name in selected_reg_models:
                    if name not in reg_models:
                        continue
                    model, params = reg_models[name]
                    if name in custom_hyperparams and custom_hyperparams[name]:
                        params = custom_hyperparams[name]
                    current_model += 1
                    progress = int((current_model / max(1, total_models)) * 100)
                    self.progress.emit(progress)
                    self.status.emit(f"Training {name} regression...")
                    gs = GridSearchCV(
                        model, params, cv=3,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=1
                    )
                    gs.fit(X_tr, y_tr)
                    y_pred = gs.predict(X_te)
                    rmse = mean_squared_error(y_te, y_pred, squared=False)
                    r2 = r2_score(y_te, y_pred)
                    reg_results.append({
                        'Model': name, 'RMSE': rmse, 'R2': r2,
                        'Params': str(gs.best_params_)
                    })
                    model_path = os.path.join(output_dir, f"model_reg_{name}.joblib")
                    joblib.dump(gs.best_estimator_, model_path)
                    results['all_reg_models'][name] = gs.best_estimator_
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_reg = gs.best_estimator_
                results['regression'] = {
                    'results_df': pd.DataFrame(reg_results),
                    'best_model': best_reg,
                    'test_data': (X_te, y_te),
                    'feature_cols': feat_cols,
                    'all_models': results['all_reg_models']
                }
        if 'SepQuality_Code' in df.columns and label_enc is not None:
            self.status.emit("Training classification models...")
            X_clf = df[feat_cols].fillna(0)
            y_clf = df['SepQuality_Code']
            counts = y_clf.value_counts()
            min_count = counts.min()
            if min_count >= 2:
                if min_count < 2:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_clf, y_clf, test_size=0.2, random_state=42, shuffle=True
                    )
                else:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
                    )
                n_splits = min(3, max(2, y_tr.value_counts().min()))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                best_clf, best_acc = None, 0.0
                clf_results = []
                for name in selected_clf_models:
                    if name not in clf_models:
                        continue
                    model, params = clf_models[name]
                    if name in custom_hyperparams and custom_hyperparams[name]:
                        params = custom_hyperparams[name]
                    current_model += 1
                    progress = int((current_model / max(1, total_models)) * 100)
                    self.progress.emit(progress)
                    self.status.emit(f"Training {name} classification...")
                    gs = GridSearchCV(
                        model, params, cv=cv,
                        scoring='accuracy', n_jobs=1
                    )
                    gs.fit(X_tr, y_tr)
                    y_pred = gs.predict(X_te)
                    acc = accuracy_score(y_te, y_pred)
                    clf_results.append({
                        'Model': name, 'Accuracy': acc,
                        'Params': str(gs.best_params_)
                    })
                    model_path = os.path.join(output_dir, f"model_clf_{name}.joblib")
                    joblib.dump(gs.best_estimator_, model_path)
                    results['all_clf_models'][name] = gs.best_estimator_
                    if acc > best_acc:
                        best_acc = acc
                        best_clf = gs.best_estimator_
                results['classification'] = {
                    'results_df': pd.DataFrame(clf_results),
                    'best_model': best_clf,
                    'test_data': (X_te, y_te),
                    'feature_cols': feat_cols,
                    'label_encoder': label_enc,
                    'all_models': results['all_clf_models']
                }
        if results['regression'].get('best_model'):
            joblib.dump(results['regression']['best_model'],
                       os.path.join(output_dir, "best_model_regression.joblib"))
        if results['classification'].get('best_model'):
            joblib.dump(results['classification']['best_model'],
                       os.path.join(output_dir, "best_model_classification.joblib"))
        self.progress.emit(100)
        self.status.emit("Training completed!")
        self.finished_with_data.emit(results)
    
    def predict_task(self):
        try:
            model = self.kwargs['model']
            data = self.kwargs['data']
            self.status.emit("Making predictions...")
            self.progress.emit(50)
            
            predictions = model.predict(data)
            
            self.progress.emit(100)
            self.status.emit("Prediction completed!")
            self.finished_with_data.emit(predictions)
        except Exception as e:
            self.error_occurred.emit(f"Error in prediction: {str(e)}\n{traceback.format_exc()}")


class PlotWidget(QWidget):
    """Widget for displaying matplotlib plots"""
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        # Set icon for this widget (if shown as window)
        self.setWindowIcon(self._get_icon())
        # Multi-select plot menu in grid
        self.plot_menu_container = QWidget()
        self.plot_menu_grid = QGridLayout(self.plot_menu_container)
        self.plot_menu_grid.setContentsMargins(0, 0, 0, 0)
        self.plot_menu_grid.setSpacing(2)
        self.plot_menu_items = []
        self.layout.insertWidget(0, self.plot_menu_container)
        # Save button
        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_current_plot)
        self.layout.addWidget(self.save_btn)
        # Fullscreen button
        self.full_btn = QPushButton("View Fullscreen")
        self.full_btn.clicked.connect(self.show_fullscreen_plot)
        self.layout.addWidget(self.full_btn)
        # Customize button
        self.custom_btn = QPushButton("Customize Plot")
        self.custom_btn.clicked.connect(self.show_customize_dialog)
        self.layout.addWidget(self.custom_btn)
        # Data for plotting
        self.plot_data = {}
        self.plot_items = []
        # Customization options
        self.show_legend = True
        self.show_grid = False
        self.plot_colors = {}  # {plot_name: color}
    
    def _get_icon(self):
        from PySide6.QtGui import QPixmap, QIcon
        import base64
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(ICON_APP))
        return QIcon(pixmap)
    
    def set_plot_data(self, plot_data):
        self.plot_data = plot_data
        self.update_plot_menu()
    
    def update_plot_menu(self):
        # Remove old checkboxes
        for cb in self.plot_menu_items:
            cb.deleteLater()
        self.plot_menu_items = []
        # Default important plots
        items = ["Model Comparison"]
        if self.plot_data.get('regression'):
            items += ["Regression: Predicted vs Actual", "Regression: Residuals", "Regression: Learning Curve", "Regression: Feature Importance"]
        if self.plot_data.get('classification'):
            items += ["Classification: Confusion Matrix", "Classification: ROC Curve", "Classification: PR Curve", "Classification: Feature Importance", "Classification: Classification Report"]
        # --- Analysis plots ---
        if self.plot_data.get('PCA'):
            items += ["PCA 2D", "PCA 3D"]
        if self.plot_data.get('Clustering'):
            items += ["Hierarchical Clustering Dendrogram"]
        if self.plot_data.get('TSNE'):
            items += ["t-SNE 2D", "t-SNE 3D"]
        if self.plot_data.get('UMAP'):
            items += ["UMAP 2D", "UMAP 3D"]
        if self.plot_data.get('QSRR'):
            items += ["QSRR: Predicted vs Actual"]
        # Grid layout: 2 columns
        ncols = 2
        nrows = math.ceil(len(items) / ncols)
        for i, item in enumerate(items):
            cb = QCheckBox(item)
            cb.setChecked(item == "Model Comparison")
            cb.stateChanged.connect(self.on_plot_menu_changed)
            self.plot_menu_items.append(cb)
            row = i // ncols
            col = i % ncols
            self.plot_menu_grid.addWidget(cb, row, col)
        self.show_selected_plots()
    
    def on_plot_menu_changed(self, state):
        self.show_selected_plots()
    
    def get_selected_plots(self):
        return [cb.text() for cb in self.plot_menu_items if cb.isChecked()]
    
    def show_selected_plots(self):
        selected = self.get_selected_plots()
        self.figure.clear()
        # Special case: Model Comparison
        if "Model Comparison" in selected:
            n = len(selected) - 1 + 3  # 3 for model comparison subplots
        else:
            n = len(selected)
        if n == 0:
            self.canvas.draw()
            return
        # Grid layout
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        plot_idx = 1
        for plot_name in selected:
            # --- Perbaikan: gunakan projection='3d' untuk plot 3D ---
            is_3d = plot_name in ["PCA 3D", "t-SNE 3D", "UMAP 3D"]
            if plot_name == "Model Comparison":
                reg_results = self.plot_data.get('regression', {}).get('results_df')
                clf_results = self.plot_data.get('classification', {}).get('results_df')
                ax1 = self.figure.add_subplot(nrows, ncols, plot_idx)
                self.plot_rmse_bar(reg_results, ax1)
                plot_idx += 1
                ax2 = self.figure.add_subplot(nrows, ncols, plot_idx)
                self.plot_r2_bar(reg_results, ax2)
                plot_idx += 1
                ax3 = self.figure.add_subplot(nrows, ncols, plot_idx)
                self.plot_acc_bar(clf_results, ax3)
                plot_idx += 1
            else:
                if is_3d:
                    ax = self.figure.add_subplot(nrows, ncols, plot_idx, projection='3d')
                    self.show_plot(plot_name, ax, fig=self.figure)
                else:
                    ax = self.figure.add_subplot(nrows, ncols, plot_idx)
                    self.show_plot(plot_name, ax)
                plot_idx += 1
        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_plot(self, plot_name, ax, fig=None):
        # Model Comparison (should not be called, handled in show_selected_plots)
        # Regression
        if plot_name == "Regression: Predicted vs Actual":
            d = self.plot_data.get('regression', {})
            if d.get('best_model') and d.get('test_data'):
                X_test, y_test = d['test_data']
                self.plot_predicted_vs_actual(d['best_model'], X_test, y_test, 'Best Regression', ax)
        elif plot_name == "Regression: Residuals":
            d = self.plot_data.get('regression', {})
            if d.get('best_model') and d.get('test_data'):
                X_test, y_test = d['test_data']
                self.plot_residuals(d['best_model'], X_test, y_test, 'Best Regression', ax)
        elif plot_name == "Regression: Learning Curve":
            d = self.plot_data.get('regression', {})
            if d.get('best_model') and d.get('test_data'):
                X_test, y_test = d['test_data']
                self.plot_learning_curve(d['best_model'], X_test, y_test, 'Best Regression', ax)
        elif plot_name == "Regression: Feature Importance":
            d = self.plot_data.get('regression', {})
            if d.get('best_model') and d.get('feature_cols'):
                self.plot_feature_importance(d['best_model'], d['feature_cols'], 'Best Regression', ax)
        # Classification
        elif plot_name == "Classification: Confusion Matrix":
            d = self.plot_data.get('classification', {})
            if d.get('best_model') and d.get('test_data'):
                X_test, y_test = d['test_data']
                self.plot_confusion_matrix(d['best_model'], X_test, y_test, d.get('label_encoder'), ax)
        elif plot_name == "Classification: ROC Curve":
            d = self.plot_data.get('classification', {})
            if d.get('best_model') and d.get('test_data'):
                X_test, y_test = d['test_data']
                self.plot_roc_curve(d['best_model'], X_test, y_test, d.get('label_encoder'), ax)
        elif plot_name == "Classification: PR Curve":
            d = self.plot_data.get('classification', {})
            if d.get('best_model') and d.get('test_data'):
                X_test, y_test = d['test_data']
                self.plot_pr_curve(d['best_model'], X_test, y_test, d.get('label_encoder'), ax)
        elif plot_name == "Classification: Feature Importance":
            d = self.plot_data.get('classification', {})
            if d.get('best_model') and d.get('feature_cols'):
                self.plot_feature_importance(d['best_model'], d['feature_cols'], 'Best Classifier', ax)
        elif plot_name == "Classification: Classification Report":
            d = self.plot_data.get('classification', {})
            if d.get('best_model') and d.get('test_data'):
                X_test, y_test = d['test_data']
                self.plot_classification_report(d['best_model'], X_test, y_test, d.get('label_encoder'), ax)
        # --- Analysis plots ---
        elif plot_name == "PCA 2D":
            d = self.plot_data.get('PCA', {})
            X_pca = d.get('X_pca')
            labels = d.get('labels')
            if X_pca is not None:
                if labels is not None:
                    labels = np.array(labels)
                    unique_labels = pd.unique(labels)
                    colors = plt.cm.tab10.colors
                    for i, ulab in enumerate(unique_labels):
                        idx = (labels == ulab)
                        ax.scatter(X_pca[idx,0], X_pca[idx,1], label=str(ulab), color=colors[i % len(colors)], alpha=0.7)
                        # Tampilkan label pada titik jika data sedikit
                        if len(X_pca) <= 30:
                            for x, y in zip(X_pca[idx,0], X_pca[idx,1]):
                                ax.text(x, y, str(ulab), fontsize=8)
                    ax.legend(title="Separation_Quality")
                else:
                    ax.scatter(X_pca[:,0], X_pca[:,1], color='b', alpha=0.7)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_title('PCA 2D')
                if self.show_grid:
                    ax.grid(True)
        elif plot_name == "PCA 3D":
            from mpl_toolkits.mplot3d import Axes3D
            d = self.plot_data.get('PCA', {})
            X_pca = d.get('X_pca')
            labels = d.get('labels')
            if X_pca is not None and X_pca.shape[1] >= 3:
                if labels is not None:
                    labels = np.array(labels)
                    unique_labels = pd.unique(labels)
                    colors = plt.cm.tab10.colors
                    for i, ulab in enumerate(unique_labels):
                        idx = (labels == ulab)
                        ax.scatter(X_pca[idx,0], X_pca[idx,1], X_pca[idx,2], label=str(ulab), color=colors[i % len(colors)], alpha=0.7)
                    ax.legend(title="Separation_Quality")
                else:
                    ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], color='b', alpha=0.7)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                if hasattr(ax, 'set_zlabel'):
                    ax.set_zlabel('PC3')
                ax.set_title('PCA 3D')
        elif plot_name == "Hierarchical Clustering Dendrogram":
            d = self.plot_data.get('Clustering', {})
            Z = d.get('Z')
            # Ambil label custom: nama senyawa B%:O% pH
            labels = None
            if hasattr(self, 'current_data') and self.current_data is not None:
                df = self.current_data.get('dataframe')
                if df is not None and all(col in df.columns for col in ['Compound', 'Buffer_%', 'Organic_%', 'pH']):
                    labels = [f"{row['Compound']} {int(row['Buffer_%'])}:{int(row['Organic_%'])} {row['pH']}" for _, row in df.iterrows()]
            # fallback ke label PCA jika tidak ada
            if labels is None:
                labels = self.plot_data.get('PCA', {}).get('labels')
                if labels is not None:
                    labels = list(labels)
            if Z is not None:
                n_samples = Z.shape[0] + 1
                if labels is not None and len(labels) == n_samples:
                    dendrogram(Z, ax=ax, labels=labels)
                else:
                    dendrogram(Z, ax=ax)
                ax.set_title('Hierarchical Clustering Dendrogram')
        elif plot_name == "t-SNE 2D":
            d = self.plot_data.get('TSNE', {})
            X_tsne2 = d.get('X_tsne2')
            labels = d.get('labels')
            if X_tsne2 is not None:
                if labels is not None:
                    labels = np.array(labels)
                    unique_labels = pd.unique(labels)
                    colors = plt.cm.tab10.colors
                    for i, ulab in enumerate(unique_labels):
                        idx = (labels == ulab)
                        ax.scatter(X_tsne2[idx,0], X_tsne2[idx,1], label=str(ulab), color=colors[i % len(colors)], alpha=0.7)
                        if len(X_tsne2) <= 30:
                            for x, y in zip(X_tsne2[idx,0], X_tsne2[idx,1]):
                                ax.text(x, y, str(ulab), fontsize=8)
                    ax.legend(title="Separation_Quality")
                else:
                    ax.scatter(X_tsne2[:,0], X_tsne2[:,1], color='b', alpha=0.7)
                ax.set_xlabel('t-SNE1')
                ax.set_ylabel('t-SNE2')
                ax.set_title('t-SNE 2D')
        elif plot_name == "t-SNE 3D":
            from mpl_toolkits.mplot3d import Axes3D
            d = self.plot_data.get('TSNE', {})
            X_tsne3 = d.get('X_tsne3')
            labels = d.get('labels')
            if X_tsne3 is not None and X_tsne3.shape[1] >= 3:
                if labels is not None:
                    labels = np.array(labels)
                    unique_labels = pd.unique(labels)
                    colors = plt.cm.tab10.colors
                    for i, ulab in enumerate(unique_labels):
                        idx = (labels == ulab)
                        ax.scatter(X_tsne3[idx,0], X_tsne3[idx,1], X_tsne3[idx,2], label=str(ulab), color=colors[i % len(colors)], alpha=0.7)
                    ax.legend(title="Separation_Quality")
                else:
                    ax.scatter(X_tsne3[:,0], X_tsne3[:,1], X_tsne3[:,2], color='b', alpha=0.7)
                ax.set_xlabel('t-SNE1')
                ax.set_ylabel('t-SNE2')
                if hasattr(ax, 'set_zlabel'):
                    ax.set_zlabel('t-SNE3')
                ax.set_title('t-SNE 3D')
        elif plot_name == "UMAP 2D":
            d = self.plot_data.get('UMAP', {})
            X_umap2 = d.get('X_umap2')
            labels = d.get('labels')
            if X_umap2 is not None:
                if labels is not None:
                    labels = np.array(labels)
                    unique_labels = pd.unique(labels)
                    colors = plt.cm.tab10.colors
                    for i, ulab in enumerate(unique_labels):
                        idx = (labels == ulab)
                        ax.scatter(X_umap2[idx,0], X_umap2[idx,1], label=str(ulab), color=colors[i % len(colors)], alpha=0.7)
                        if len(X_umap2) <= 30:
                            for x, y in zip(X_umap2[idx,0], X_umap2[idx,1]):
                                ax.text(x, y, str(ulab), fontsize=8)
                    ax.legend(title="Separation_Quality")
                else:
                    ax.scatter(X_umap2[:,0], X_umap2[:,1], color='b', alpha=0.7)
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.set_title('UMAP 2D')
        elif plot_name == "UMAP 3D":
            from mpl_toolkits.mplot3d import Axes3D
            d = self.plot_data.get('UMAP', {})
            X_umap3 = d.get('X_umap3')
            labels = d.get('labels')
            if X_umap3 is not None and X_umap3.shape[1] >= 3:
                if labels is not None:
                    labels = np.array(labels)
                    unique_labels = pd.unique(labels)
                    colors = plt.cm.tab10.colors
                    for i, ulab in enumerate(unique_labels):
                        idx = (labels == ulab)
                        ax.scatter(X_umap3[idx,0], X_umap3[idx,1], X_umap3[idx,2], label=str(ulab), color=colors[i % len(colors)], alpha=0.7)
                    ax.legend(title="Separation_Quality")
                else:
                    ax.scatter(X_umap3[:,0], X_umap3[:,1], X_umap3[:,2], color='b', alpha=0.7)
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                if hasattr(ax, 'set_zlabel'):
                    ax.set_zlabel('UMAP3')
                ax.set_title('UMAP 3D')
        elif plot_name == "QSRR: Predicted vs Actual":
            d = self.plot_data.get('QSRR', {})
            y_true = d.get('y_true')
            y_pred = d.get('y_pred')
            if y_true is not None and y_pred is not None:
                ax.scatter(y_true, y_pred, alpha=0.7)
                ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
                ax.set_xlabel('Actual Retention Time')
                ax.set_ylabel('Predicted Retention Time')
                ax.set_title('QSRR: Predicted vs Actual')
    
    def plot_rmse_bar(self, reg_results, ax):
        if reg_results is not None and not reg_results.empty:
            reg_results.set_index('Model')['RMSE'].plot(kind='bar', ax=ax, color='tab:blue', alpha=0.7)
            ax.set_title('Regression RMSE (lower better)')
            ax.set_ylabel('RMSE')
            if self.show_legend:
                ax.legend()
            if self.show_grid:
                ax.grid(True)
    
    def plot_r2_bar(self, reg_results, ax):
        if reg_results is not None and not reg_results.empty:
            reg_results.set_index('Model')['R2'].plot(kind='bar', ax=ax, color='tab:green', alpha=0.7)
            ax.set_title('Regression R² (higher better)')
            ax.set_ylabel('R²')
            if self.show_legend:
                ax.legend()
            if self.show_grid:
                ax.grid(True)
    
    def plot_acc_bar(self, clf_results, ax):
        if clf_results is not None and not clf_results.empty:
            clf_results.set_index('Model')['Accuracy'].plot(kind='bar', ax=ax, color='tab:orange', alpha=0.7)
            ax.set_title('Classification Accuracy (higher better)')
            ax.set_ylabel('Accuracy')
            if self.show_legend:
                ax.legend()
            if self.show_grid:
                ax.grid(True)
    
    def plot_predicted_vs_actual(self, model, X_test, y_test, model_name, ax, fig=None):
        y_pred = model.predict(X_test)
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{model_name} Predicted vs Actual')
        if self.show_legend:
            ax.legend(["Prediction", "Perfect"], loc='best')
        if self.show_grid:
            ax.grid(True)
    
    def plot_residuals(self, model, X_test, y_test, model_name, ax, fig=None):
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel('Predicted RT')
        ax.set_ylabel('Residual (Actual - Pred)')
        ax.set_title(f'{model_name} Residual Plot')
        if self.show_legend:
            ax.legend(["Residuals"], loc='best')
        if self.show_grid:
            ax.grid(True)
    
    def plot_learning_curve(self, model, X, y, model_name, ax, fig=None):
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=3, scoring='neg_root_mean_squared_error', n_jobs=1
        )
        ax.plot(train_sizes, -train_scores.mean(axis=1), label='Train RMSE')
        ax.plot(train_sizes, -test_scores.mean(axis=1), label='Test RMSE')
        ax.set_xlabel('Train Size')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{model_name} Learning Curve')
        if self.show_legend:
            ax.legend()
        if self.show_grid:
            ax.grid(True)
    
    def plot_feature_importance(self, model, feat_cols, model_name, ax, fig=None):
        importance_attr = None
        if hasattr(model, 'feature_importances_'):
            importance_attr = model.feature_importances_
        elif hasattr(model, 'named_steps'):
            for step in model.named_steps.values():
                if hasattr(step, 'feature_importances_'):
                    importance_attr = step.feature_importances_
                    break
        if importance_attr is not None:
            indices = np.argsort(importance_attr)[::-1]
            ax.bar(range(len(importance_attr)), importance_attr[indices])
            ax.set_xticks(range(len(importance_attr)))
            ax.set_xticklabels([feat_cols[i] for i in indices], rotation=45)
            ax.set_title(f'Feature Importance - {model_name}')
            ax.set_ylabel('Importance')
        else:
            ax.text(0.5, 0.5, 'No feature importance available', ha='center')
        if self.show_legend:
            ax.legend(["Importance"], loc='best')
        if self.show_grid:
            ax.grid(True)
    
    def plot_confusion_matrix(self, model, X_test, y_test, label_encoder, ax, fig=None):
        from sklearn.metrics import confusion_matrix
        import itertools
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        classes = label_encoder.classes_ if label_encoder is not None else np.unique(y_test)
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion Matrix')
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticklabels(classes)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        if self.show_legend:
            ax.legend(["Confusion Matrix"], loc='best')
        if self.show_grid:
            ax.grid(False)
    
    def plot_roc_curve(self, model, X_test, y_test, label_encoder, ax, fig=None):
        from sklearn.metrics import roc_curve, auc
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            if y_score.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                if self.show_legend:
                    ax.legend(loc="lower right")
                if self.show_grid:
                    ax.grid(True)
            else:
                ax.text(0.5, 0.5, 'ROC: multiclass', ha='center')
        else:
            ax.text(0.5, 0.5, 'No predict_proba', ha='center')
    
    def plot_pr_curve(self, model, X_test, y_test, label_encoder, ax, fig=None):
        from sklearn.metrics import precision_recall_curve
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            if y_score.shape[1] == 2:
                precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1])
                ax.plot(recall, precision, label='PR curve')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                if self.show_legend:
                    ax.legend(loc="lower left")
                if self.show_grid:
                    ax.grid(True)
            else:
                ax.text(0.5, 0.5, 'PR: multiclass', ha='center')
        else:
            ax.text(0.5, 0.5, 'No predict_proba', ha='center')
    
    def plot_classification_report(self, model, X_test, y_test, label_encoder, ax, fig=None):
        from sklearn.metrics import classification_report
        y_pred = model.predict(X_test)
        if label_encoder is not None:
            labels = label_encoder.transform(label_encoder.classes_)
            report = classification_report(y_test, y_pred, labels=labels, target_names=label_encoder.classes_, zero_division=0)
        else:
            report = classification_report(y_test, y_pred, zero_division=0)
        ax.text(0.01, 0.01, report, {'fontsize': 10}, fontproperties = 'monospace')
        ax.axis('off')
        ax.set_title('Classification Report')
    
    def save_current_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot Image", "plot.png", "PNG Files (*.png)")
        if file_path:
            self.figure.savefig(file_path, dpi=150, bbox_inches='tight')
    
    def show_fullscreen_plot(self):
        selected = self.get_selected_plots()
        if not selected:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Fullscreen Plot")
        dlg.setWindowIcon(self._get_icon())
        layout = QVBoxLayout(dlg)
        fig = Figure(figsize=(16, 9))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        # Copy plotting logic
        if "Model Comparison" in selected:
            n = len(selected) - 1 + 3
        else:
            n = len(selected)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        plot_idx = 1
        for plot_name in selected:
            if plot_name == "Model Comparison":
                reg_results = self.plot_data.get('regression', {}).get('results_df')
                clf_results = self.plot_data.get('classification', {}).get('results_df')
                ax1 = fig.add_subplot(nrows, ncols, plot_idx)
                self.plot_rmse_bar(reg_results, ax1)
                plot_idx += 1
                ax2 = fig.add_subplot(nrows, ncols, plot_idx)
                self.plot_r2_bar(reg_results, ax2)
                plot_idx += 1
                ax3 = fig.add_subplot(nrows, ncols, plot_idx)
                self.plot_acc_bar(clf_results, ax3)
                plot_idx += 1
            else:
                ax = fig.add_subplot(nrows, ncols, plot_idx)
                self.show_plot(plot_name, ax, fig=fig)
                plot_idx += 1
        fig.tight_layout()
        canvas.draw()
        dlg.resize(1200, 800)
        dlg.exec()
    
    def show_customize_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Customize Plot")
        dlg.setWindowIcon(self._get_icon())
        layout = QVBoxLayout(dlg)
        # Legend
        legend_cb = QCheckBox("Show Legend")
        legend_cb.setChecked(self.show_legend)
        layout.addWidget(legend_cb)
        # Grid
        grid_cb = QCheckBox("Show Grid")
        grid_cb.setChecked(self.show_grid)
        layout.addWidget(grid_cb)
        # (Color picker UI can be added here)
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)
        def accept():
            self.show_legend = legend_cb.isChecked()
            self.show_grid = grid_cb.isChecked()
            dlg.accept()
            self.show_selected_plots()
        ok_btn.clicked.connect(accept)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec()


class HPLCPipelineGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HPLC Pipeline GUI")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(self._get_icon())
        
        # Data storage
        self.current_data = None
        self.current_results = None
        self.output_directory = "output"
        
        # Model selection storage
        self.available_reg_models = {
            'LR': 'Linear Regression',
            'Ridge': 'Ridge Regression',
            'Lasso': 'Lasso Regression',
            'RF': 'Random Forest Regressor',
            'GB': 'Gradient Boosting Regressor',
            'SVR': 'Support Vector Regressor',
            'MLP': 'MLP Regressor',
            'KNN': 'K-Nearest Neighbors Regressor',
            'DT': 'Decision Tree Regressor',
        }
        self.available_clf_models = {
            'RFC': 'Random Forest Classifier',
            'GBC': 'Gradient Boosting Classifier',
            'SVC': 'Support Vector Classifier',
            'MLPC': 'MLP Classifier',
            'KNN': 'K-Nearest Neighbors Classifier',
            'DT': 'Decision Tree Classifier',
            'LR': 'Logistic Regression',
        }
        self.selected_reg_models = set(['LR', 'RF', 'GB', 'SVR', 'MLP'])
        self.selected_clf_models = set(['RFC', 'GBC', 'SVC', 'MLPC'])
        # Store custom hyperparameters per model
        self.model_hyperparams = {}
        
        # Setup UI
        self.setup_ui()
        
        # Setup menu bar and status bar
        self.setup_menubar_statusbar()
        
        # Setup worker thread
        self.worker = None
    
    def _get_icon(self):
        from PySide6.QtGui import QPixmap, QIcon
        import base64
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(ICON_APP))
        return QIcon(pixmap)
    
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        # Left panel - Controls
        left_panel = self.create_left_panel()
        from PySide6.QtWidgets import QScrollArea
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        main_layout.addWidget(left_scroll, 1)
        # Right panel - Results and plots
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        # Tambahkan tab optimasi
        self.optimasi_tab = QWidget()
        self.optimasi_layout = QVBoxLayout(self.optimasi_tab)
        # Heatmap
        self.optimasi_heatmap_label = QLabel("Separation Quality Prediction Heatmap (pH vs Organic_%)")
        self.optimasi_layout.addWidget(self.optimasi_heatmap_label)
        self.optimasi_heatmap_canvas = None  # Will be filled after training
        # Saran
        self.optimasi_saran_label = QLabel("Optimization Suggestion:")
        self.optimasi_layout.addWidget(self.optimasi_saran_label)
        self.optimasi_saran_text = QTextEdit()
        self.optimasi_saran_text.setReadOnly(True)
        self.optimasi_layout.addWidget(self.optimasi_saran_text)
        # Tabel komposisi terbaik
        self.optimasi_best_table = QTableWidget()
        self.optimasi_layout.addWidget(QLabel("Top pH & Composition Combinations:"))
        self.optimasi_layout.addWidget(self.optimasi_best_table)
        # Tabel feature importance
        self.optimasi_featimp_table = QTableWidget()
        self.optimasi_layout.addWidget(QLabel("Feature Importance Ranking (Classification):"))
        self.optimasi_layout.addWidget(self.optimasi_featimp_table)
        # Penjelasan
        self.optimasi_explain_label = QLabel("Data Explanation:")
        self.optimasi_layout.addWidget(self.optimasi_explain_label)
        self.optimasi_explain_text = QTextEdit()
        self.optimasi_explain_text.setReadOnly(True)
        self.optimasi_layout.addWidget(self.optimasi_explain_text)
        # Bungkus optimasi_tab dengan QScrollArea
        optimasi_scroll = QScrollArea()
        optimasi_scroll.setWidget(self.optimasi_tab)
        optimasi_scroll.setWidgetResizable(True)
        self.right_panel = right_panel
        right_panel.addTab(optimasi_scroll, "Optimization")
        
    def create_left_panel(self):
        """Create left control panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(16777215)  # biarkan lebar mengikuti isi
        panel.setMinimumWidth(220)  # minimal agar tidak terlalu kecil
        # Tambahkan size policy agar QScrollArea bisa scroll
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout(panel)
        
        # File loading section
        file_group = QGroupBox("Data Loading")
        file_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        file_layout = QVBoxLayout(file_group)
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select Excel or CSV file...")
        
        browse_btn = QPushButton("Browse File")
        browse_btn.clicked.connect(self.browse_file)
        
        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(self.load_data)
        
        file_layout.addWidget(QLabel("File Path:"))
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(browse_btn)
        file_layout.addWidget(load_btn)
        
        layout.addWidget(file_group)
        
        # Model loading section
        model_group = QGroupBox("Model Management")
        model_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        model_layout = QVBoxLayout(model_group)
        
        load_model_btn = QPushButton("Load Saved Models")
        load_model_btn.clicked.connect(self.load_saved_models)
        
        save_results_btn = QPushButton("Save Results")
        save_results_btn.clicked.connect(self.save_results)
        
        model_layout.addWidget(load_model_btn)
        model_layout.addWidget(save_results_btn)
        
        layout.addWidget(model_group)
        
        # Prediction section
        pred_group = QGroupBox("Prediction")
        pred_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        pred_layout = QVBoxLayout(pred_group)
        self.smiles_input = QLineEdit()
        self.smiles_input.setPlaceholderText("Enter SMILES string...")
        # Model selection dropdowns
        self.reg_model_combo = QComboBox()
        self.reg_model_combo.setPlaceholderText("Select regression model...")
        self.clf_model_combo = QComboBox()
        self.clf_model_combo.setPlaceholderText("Select classification model...")
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.make_prediction)
        suggest_btn = QPushButton("Suggest Optimal Condition")
        suggest_btn.clicked.connect(self.suggest_optimal_condition)
        self.prediction_result = QTextEdit()
        self.prediction_result.setMaximumHeight(100)
        self.prediction_result.setReadOnly(True)
        pred_layout.addWidget(QLabel("SMILES:"))
        pred_layout.addWidget(self.smiles_input)
        pred_layout.addWidget(QLabel("Regression Model:"))
        pred_layout.addWidget(self.reg_model_combo)
        pred_layout.addWidget(QLabel("Classification Model:"))
        pred_layout.addWidget(self.clf_model_combo)
        pred_layout.addWidget(predict_btn)
        pred_layout.addWidget(suggest_btn)
        pred_layout.addWidget(QLabel("Result:"))
        pred_layout.addWidget(self.prediction_result)
        layout.addWidget(pred_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Log section
        log_group = QGroupBox("Log")
        log_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(clear_log_btn)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create right results panel"""
        panel = QTabWidget()
        # Data view tab
        self.data_table = QTableWidget()
        data_scroll = QScrollArea()
        data_scroll.setWidget(self.data_table)
        data_scroll.setWidgetResizable(True)
        panel.addTab(data_scroll, "Data View")
        # Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        self.results_table = QTableWidget()
        results_layout.addWidget(self.results_table)
        results_scroll = QScrollArea()
        results_scroll.setWidget(results_widget)
        results_scroll.setWidgetResizable(True)
        panel.addTab(results_scroll, "Model Results")
        # Plots tab (sudah pakai QScrollArea)
        self.plot_widget = PlotWidget()
        plot_scroll = QScrollArea()
        plot_scroll.setWidget(self.plot_widget)
        plot_scroll.setWidgetResizable(True)
        panel.addTab(plot_scroll, "Visualizations")
        # Model details tab
        self.model_details = QTextEdit()
        self.model_details.setReadOnly(True)
        details_scroll = QScrollArea()
        details_scroll.setWidget(self.model_details)
        details_scroll.setWidgetResizable(True)
        panel.addTab(details_scroll, "Model Details")
        # Analysis tab
        self.analysis_tab = QWidget()
        self.analysis_layout = QVBoxLayout(self.analysis_tab)
        # --- Analysis Controls ---
        btn_layout = QHBoxLayout()
        self.pca_btn = QPushButton("Run PCA")
        self.pca_btn.clicked.connect(self.run_pca_analysis)
        btn_layout.addWidget(self.pca_btn)
        self.clust_btn = QPushButton("Run Hierarchical Clustering")
        self.clust_btn.clicked.connect(self.run_clustering_analysis)
        btn_layout.addWidget(self.clust_btn)
        self.tsne_btn = QPushButton("Run t-SNE")
        self.tsne_btn.clicked.connect(self.run_tsne_analysis)
        btn_layout.addWidget(self.tsne_btn)
        if HAS_UMAP:
            self.umap_btn = QPushButton("Run UMAP")
            self.umap_btn.clicked.connect(self.run_umap_analysis)
            btn_layout.addWidget(self.umap_btn)
        self.qsrr_btn = QPushButton("Run QSRR Analysis")
        self.qsrr_btn.clicked.connect(self.run_qsrr_analysis)
        btn_layout.addWidget(self.qsrr_btn)
        self.analysis_layout.addLayout(btn_layout)
        # --- Results Area ---
        self.analysis_result_label = QLabel("Analysis Results:")
        self.analysis_layout.addWidget(self.analysis_result_label)
        self.analysis_result_table = QTableWidget()
        self.analysis_layout.addWidget(self.analysis_result_table)
        self.analysis_result_text = QTextEdit()
        self.analysis_result_text.setReadOnly(True)
        self.analysis_layout.addWidget(self.analysis_result_text)
        analysis_scroll = QScrollArea()
        analysis_scroll.setWidget(self.analysis_tab)
        analysis_scroll.setWidgetResizable(True)
        panel.addTab(analysis_scroll, "Analysis")
        return panel
    
    def browse_file(self):
        """Browse for input file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select HPLC Data File",
            "", "Excel Files (*.xlsx *.xls);;CSV Files (*.csv)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)
    
    def load_data(self):
        """Load data in worker thread"""
        file_path = self.file_path_edit.text().strip()
        if not file_path or not os.path.exists(file_path):
            msg = QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please select a valid file!")
            msg.setWindowIcon(self._get_icon())
            msg.exec()
            return
        
        self.log("Starting data loading...")
        self.progress_bar.setValue(0)
        
        # Start worker thread
        self.worker = WorkerThread("load_data", file_path=file_path)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished_with_data.connect(self.on_data_loaded)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
    
    def on_data_loaded(self, data):
        """Handle loaded data"""
        self.current_data = data
        df = data['dataframe']
        
        # Update data table
        self.update_data_table(df)
        
        # Log success
        self.log(f"Data loaded successfully! Shape: {df.shape}")
        self.log(f"Columns: {', '.join(df.columns)}")
        
        # Update status
        self.status_label.setText("Data loaded - ready for training")
    
    def update_data_table(self, df):
        """Update data table widget"""
        self.data_table.setRowCount(df.shape[0])
        self.data_table.setColumnCount(df.shape[1])
        self.data_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        # Populate table (show first 1000 rows to avoid UI lag)
        display_rows = min(1000, df.shape[0])
        for i in range(display_rows):
            for j, col in enumerate(df.columns):
                value = str(df.iloc[i, j])
                if len(value) > 50:  # Truncate long values
                    value = value[:50] + "..."
                item = QTableWidgetItem(value)
                self.data_table.setItem(i, j, item)
        
        if df.shape[0] > 1000:
            self.log(f"Note: Showing first 1000 rows out of {df.shape[0]} total rows")
    
    def train_models(self):
        """Train models in worker thread"""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
        output_dir = self.output_directory
        self.log("Starting model training...")
        self.progress_bar.setValue(0)
        # Pass selected models to worker
        selected_reg = list(self.selected_reg_models)
        selected_clf = list(self.selected_clf_models)
        # Pass custom hyperparameters
        self.worker = WorkerThread(
            "train_models",
            dataframe=self.current_data['dataframe'],
            solvents=self.current_data['solvents'],
            label_encoder=self.current_data['label_encoder'],
            output_dir=output_dir,
            selected_reg_models=selected_reg,
            selected_clf_models=selected_clf,
            custom_hyperparams=self.model_hyperparams
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished_with_data.connect(self.on_training_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
    
    def on_training_finished(self, results):
        """Handle training completion"""
        self.current_results = results
        # Save training data for later reload
        if self.current_data:
            try:
                output_dir = self.output_directory
                os.makedirs(output_dir, exist_ok=True)
                # Save dataframe as CSV
                self.current_data['dataframe'].to_csv(os.path.join(output_dir, "training_data.csv"), index=False)
                # Save all training context (data, solvents, label_encoder, results, optimization_heatmap) as pickle
                ctx = {
                        'data': self.current_data,
                        'results': results
                }
                if hasattr(self, 'optimization_heatmap'):
                    ctx['optimization_heatmap'] = self.optimization_heatmap
                with open(os.path.join(output_dir, "training_context.pkl"), "wb") as f:
                    pickle.dump(ctx, f)
                self.log("Training data and context saved for reload.")
            except Exception as e:
                self.log(f"Error saving training data: {str(e)}")
        # Update results table
        self.update_results_table(results)
        # Update plots
        self.plot_widget.set_plot_data(results)
        # Update model details (now includes feature importance & saran)
        self.update_model_details(results)
        # Log success
        self.log("Training completed successfully!")
        # Generate additional plots
        self.generate_additional_plots(results)
        self.status_label.setText("Training completed")
        # Update model selection dropdowns
        self.update_model_selection_dropdowns(results)
        # Update tab optimasi
        self.update_optimasi_tab(results)
    
    def update_results_table(self, results):
        """Update results table"""
        # Combine regression and classification results
        all_results = []
        # Regression metrics
        if 'regression' in results and 'results_df' in results['regression']:
            reg_df = results['regression']['results_df'].copy()
            reg_df['Type'] = 'Regression'
            # Tambahkan metrik MAE
            if results['regression'].get('test_data') and results['regression'].get('all_models'):
                X_test, y_test = results['regression']['test_data']
                for i, row in reg_df.iterrows():
                    model_key = row['Model']
                    model = results['regression']['all_models'].get(model_key)
                    if model is not None:
                        y_pred = model.predict(X_test)
                        mae = np.mean(np.abs(y_test - y_pred))
                        reg_df.at[i, 'MAE'] = mae
            all_results.append(reg_df)
        # Classification metrics
        if 'classification' in results and 'results_df' in results['classification']:
            clf_df = results['classification']['results_df'].copy()
            clf_df['Type'] = 'Classification'
            # Tambahkan metrik precision, recall, f1, support
            if results['classification'].get('test_data') and results['classification'].get('all_models'):
                X_test, y_test = results['classification']['test_data']
                label_enc = results['classification'].get('label_encoder')
                for i, row in clf_df.iterrows():
                    model_key = row['Model']
                    model = results['classification']['all_models'].get(model_key)
                    if model is not None:
                        y_pred = model.predict(X_test)
                        if label_enc is not None:
                            labels = label_enc.transform(label_enc.classes_)
                            report = classification_report(
                                y_test, y_pred, output_dict=True,
                                labels=labels,
                                target_names=label_enc.classes_,
                                zero_division=0
                            )
                        else:
                            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        # Ambil average (macro)
                        clf_df.at[i, 'Precision'] = report['macro avg']['precision']
                        clf_df.at[i, 'Recall'] = report['macro avg']['recall']
                        clf_df.at[i, 'F1'] = report['macro avg']['f1-score']
                        clf_df.at[i, 'Support'] = report['macro avg']['support']
            all_results.append(clf_df)
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            self.results_table.setRowCount(combined_df.shape[0])
            self.results_table.setColumnCount(combined_df.shape[1])
            self.results_table.setHorizontalHeaderLabels(combined_df.columns.tolist())
            for i in range(combined_df.shape[0]):
                for j, col in enumerate(combined_df.columns):
                    value = str(combined_df.iloc[i, j])
                    if len(value) > 100:
                        value = value[:100] + "..."
                    item = QTableWidgetItem(value)
                    self.results_table.setItem(i, j, item)
    
    def update_model_details(self, results):
        """Update model details text"""
        details = []
        # --- REGRESSION ---
        if 'regression' in results:
            details.append("=== REGRESSION MODELS ===")
            details.append("Lower RMSE and MAE are better. Higher R² is better.")
            reg_results = results['regression'].get('results_df')
            if reg_results is not None:
                details.append(reg_results.to_string())
                details.append("")
        # --- CLASSIFICATION ---
        if 'classification' in results:
            details.append("=== CLASSIFICATION MODELS ===")
            details.append("Higher Accuracy, Precision, Recall, and F1-score are better.")
            clf_results = results['classification'].get('results_df')
            if clf_results is not None:
                details.append(clf_results.to_string())
                details.append("")
            # --- FEATURE IMPORTANCE TABLE & SUGGESTION ---
            best_clf = results['classification'].get('best_model')
            feat_cols = results['classification'].get('feature_cols')
            if best_clf is not None and feat_cols is not None:
                importance_attr = None
                if hasattr(best_clf, 'feature_importances_'):
                    importance_attr = best_clf.feature_importances_
                elif hasattr(best_clf, 'named_steps'):
                    for step in best_clf.named_steps.values():
                        if hasattr(step, 'feature_importances_'):
                            importance_attr = step.feature_importances_
                            break
                if importance_attr is not None:
                    indices = np.argsort(importance_attr)[::-1]
                    details.append("=== FEATURE IMPORTANCE (Classification) ===")
                    details.append("Rank | Feature | Importance\n-----|---------|----------")
                    details.append("Features at the top have the most influence on separation quality prediction.")
                    for rank, idx in enumerate(indices, 1):
                        details.append(f"{rank} | {feat_cols[idx]} | {importance_attr[idx]:.4f}")
                    # Automatic suggestion
                    top_feat = feat_cols[indices[0]]
                    details.append("")
                    details.append(f"Most influential factor for separation: {top_feat}")
        # --- FEATURE INFO ---
        if self.current_data:
            details.append("=== FEATURE INFORMATION ===")
            details.append(f"Solvents: {', '.join(self.current_data['solvents'])}")
            details.append("Molecular Descriptors: MolWt, LogP, TPSA, HBD, HBA")
            details.append("Additional Features: pH")
            details.append("\nExplanation:\n- Higher LogP: more hydrophobic\n- Higher TPSA: more polar\n- Higher HBD/HBA: more hydrogen bonding\n- pH: mobile phase pH\n")
        # --- OPTIMIZATION SUGGESTION ---
        if 'classification' in results and results['classification'].get('best_model') is not None:
            try:
                best_clf = results['classification']['best_model']
                feat_cols = results['classification']['feature_cols']
                label_enc = results['classification'].get('label_encoder')
                solvents = [col for col in feat_cols if col.endswith('_%')]
                if solvents and 'pH' in feat_cols:
                    organic_col = solvents[0]
                    buffer_col = solvents[1] if len(solvents) > 1 else None
                    df = self.current_data['dataframe']
                    pH_range = np.sort(df['pH'].dropna().unique())
                    org_range = np.sort(df[organic_col].dropna().unique())
                    # Make list of all combinations in data
                    combos = []
                    for pH in pH_range:
                        for org in org_range:
                            buf = 100 - org if buffer_col else 0
                            if buffer_col:
                                exists = ((df['pH'] == pH) & (df[organic_col] == org) & (df[buffer_col] == buf)).any()
                            else:
                                exists = ((df['pH'] == pH) & (df[organic_col] == org)).any()
                            if not exists:
                                continue
                            feat_dict = {col: 0 for col in feat_cols}
                            feat_dict[organic_col] = org
                            if buffer_col:
                                feat_dict[buffer_col] = buf
                            feat_dict['pH'] = pH
                            for desc in ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'Peak_Area']:
                                if desc in feat_cols:
                                    feat_dict[desc] = 0
                            X_df = pd.DataFrame([feat_dict], columns=feat_cols)
                            pred = best_clf.predict(X_df)[0]
                            combos.append((pH, org, buf, pred))
                    # Sort best class by priority
                    if label_enc is not None:
                        class_priority = [
                            'Completely separated',
                            'Switching-fully separated',
                            'Switching-partialy separated',
                            'Partialy separated',
                            'Overlaping',
                            'No data'
                        ]
                        class_to_int = {c: label_enc.transform([c])[0] for c in class_priority if c in label_enc.classes_}
                        # Find prediction with best class
                        best_pred = None
                        best_combo = None
                        for class_name in class_priority:
                            class_val = class_to_int.get(class_name)
                            if class_val is None:
                                continue
                            filtered = [c for c in combos if c[3] == class_val]
                            if filtered:
                                best_combo = filtered[0]
                                best_pred = class_val
                                break
                        if best_combo is not None:
                            best_label = class_name
                            details.append("")
                            details.append(f"SUGGESTION: Combination pH={best_combo[0]}, {organic_col}={best_combo[1]}%{f', {buffer_col}={best_combo[2]}%' if buffer_col else ''} is predicted to yield the best Separation Quality: {best_label}")
                            details.append("")
                            details.append("Explanation: 'Completely separated' is the best possible separation. The higher the class in the list, the better the separation quality.")
                            # Save for heatmap plot
                            pred_matrix = np.full((len(pH_range), len(org_range)), np.nan)
                            for i, pH in enumerate(pH_range):
                                for j, org in enumerate(org_range):
                                    buf = 100 - org if buffer_col else 0
                                    if buffer_col:
                                        exists = ((df['pH'] == pH) & (df[organic_col] == org) & (df[buffer_col] == buf)).any()
                                    else:
                                        exists = ((df['pH'] == pH) & (df[organic_col] == org)).any()
                                    if not exists:
                                        continue
                                    feat_dict = {col: 0 for col in feat_cols}
                                    feat_dict[organic_col] = org
                                    if buffer_col:
                                        feat_dict[buffer_col] = buf
                                    feat_dict['pH'] = pH
                                    for desc in ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'Peak_Area']:
                                        if desc in feat_cols:
                                            feat_dict[desc] = 0
                                    X_df = pd.DataFrame([feat_dict], columns=feat_cols)
                                    pred = best_clf.predict(X_df)[0]
                                    pred_matrix[i, j] = pred
                            self.optimization_heatmap = {
                                'pH_range': pH_range,
                                'org_range': org_range,
                                'pred_matrix': pred_matrix,
                                'label_enc': label_enc,
                                'organic_col': organic_col,
                                'buffer_col': buffer_col
                            }
                        else:
                            details.append("")
                            details.append("SUGGESTION: No combination with the best class found in the data.")
                            details.append("")
                    else:
                        details.append("")
                        details.append("SUGGESTION: Label encoder not found.")
                        details.append("")
            except Exception as e:
                details.append(f"[Optimization error: {str(e)}]")
        self.model_details.setText("\n".join(details))
    
    def plot_optimization_heatmap(self):
        """Plot heatmap prediksi Separation Quality (pH vs Organic_%)"""
        if not hasattr(self, 'optimization_heatmap'):
            return
        data = self.optimization_heatmap
        pH_range = data['pH_range']
        org_range = data['org_range']
        pred_matrix = data['pred_matrix']
        label_enc = data['label_enc']
        organic_col = data['organic_col']
        buffer_col = data['buffer_col']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pred_matrix, aspect='auto', origin='lower',
                      extent=[org_range[0], org_range[-1], pH_range[0], pH_range[-1]],
                      cmap='viridis')
        cbar = plt.colorbar(im, ax=ax)
        if label_enc is not None:
            ticks = np.arange(len(label_enc.classes_))
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(label_enc.classes_)
        ax.set_xlabel(f"{organic_col} (%)")
        ax.set_ylabel("pH")
        ax.set_title("Prediksi Separation Quality (Heatmap)")
        plt.tight_layout()
        plt.show()
    
    def generate_additional_plots(self, results):
        """Generate additional visualization plots"""
        if not self.current_data:
            return
        
        df = self.current_data['dataframe']
        feat_cols = (self.current_data['solvents'] + 
                    ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'pH'])
        
        # Save correlation matrix plot
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_cols = [col for col in feat_cols if col in df.columns]
            if 'Retention_Time_min' in df.columns:
                corr_cols.append('Retention_Time_min')
            
            corr_data = df[corr_cols].corr()
            im = ax.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            
            ax.set_xticks(range(len(corr_data.columns)))
            ax.set_xticklabels(corr_data.columns, rotation=90)
            ax.set_yticks(range(len(corr_data.index)))
            ax.set_yticklabels(corr_data.index)
            ax.set_title('Feature Correlation Matrix')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_directory, "correlation_matrix.png"), dpi=150)
            plt.close()
            
            self.log("Correlation matrix plot saved")
        except Exception as e:
            self.log(f"Error generating correlation plot: {str(e)}")
        
        # Save residual plots for regression models
        if 'regression' in results and 'test_data' in results['regression']:
            try:
                X_test, y_test = results['regression']['test_data']
                best_model = results['regression']['best_model']
                
                if best_model is not None:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    y_pred = best_model.predict(X_test)
                    residuals = y_test - y_pred
                    
                    ax.scatter(y_pred, residuals, alpha=0.6)
                    ax.axhline(0, color='gray', linestyle='--')
                    ax.set_xlabel('Predicted RT')
                    ax.set_ylabel('Residual (Actual - Pred)')
                    ax.set_title('Best Model Residual Plot')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_directory, "best_model_residuals.png"), dpi=150)
                    plt.close()
                    
                    self.log("Residual plot saved")
            except Exception as e:
                self.log(f"Error generating residual plot: {str(e)}")
        
        # Save feature importance plot if available
        self.save_feature_importance_plots(results)
    
    def save_feature_importance_plots(self, results):
        """Save feature importance plots for tree-based models"""
        try:
            if 'regression' in results and 'best_model' in results['regression']:
                model = results['regression']['best_model']
                feat_cols = results['regression']['feature_cols']
                
                # Check if model has feature importance
                importance_attr = None
                if hasattr(model, 'feature_importances_'):
                    importance_attr = model.feature_importances_
                elif hasattr(model, 'named_steps'):
                    for step in model.named_steps.values():
                        if hasattr(step, 'feature_importances_'):
                            importance_attr = step.feature_importances_
                            break
                
                if importance_attr is not None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    indices = np.argsort(importance_attr)[::-1]
                    
                    ax.bar(range(len(importance_attr)), importance_attr[indices])
                    ax.set_xticks(range(len(importance_attr)))
                    ax.set_xticklabels([feat_cols[i] for i in indices], rotation=45)
                    ax.set_title('Feature Importance - Best Regression Model')
                    ax.set_ylabel('Importance')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_directory, "feature_importance_regression.png"), dpi=150)
                    plt.close()
                    
                    self.log("Feature importance plot saved")
        except Exception as e:
            self.log(f"Error generating feature importance plot: {str(e)}")
    
    def load_saved_models(self):
        """Load previously saved models"""
        model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if not model_dir:
            return
        
        try:
            # Look for saved models
            model_files = []
            for file in os.listdir(model_dir):
                if file.endswith('.joblib') and 'model' in file:
                    model_files.append(file)
            
            if not model_files:
                msg = QMessageBox(self)
                msg.setWindowTitle("Info")
                msg.setIcon(QMessageBox.Information)
                msg.setText("No model files found in selected directory")
                msg.setWindowIcon(self._get_icon())
                msg.exec()
                return
            
            self.log(f"Found {len(model_files)} model files:")
            for file in model_files:
                self.log(f"  - {file}")
            
            # Load best models if available
            best_reg_path = os.path.join(model_dir, "best_model_regression.joblib")
            best_clf_path = os.path.join(model_dir, "best_model_classification.joblib")
            
            loaded_models = {}
            
            if os.path.exists(best_reg_path):
                loaded_models['regression'] = joblib.load(best_reg_path)
                self.log("Best regression model loaded")
            
            if os.path.exists(best_clf_path):
                loaded_models['classification'] = joblib.load(best_clf_path)
                self.log("Best classification model loaded")
            
            if loaded_models:
                self.current_results = {'loaded_models': loaded_models}
                msg = QMessageBox(self)
                msg.setWindowTitle("Success")
                msg.setIcon(QMessageBox.Information)
                msg.setText(f"Loaded {len(loaded_models)} models successfully!")
                msg.setWindowIcon(self._get_icon())
                msg.exec()
            
        except Exception as e:
            self.log(f"Error loading models: {str(e)}")
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"Failed to load models: {str(e)}")
            msg.setWindowIcon(self._get_icon())
            msg.exec()
    
    def save_results(self):
        """Save current results to files"""
        if not self.current_results:
            msg = QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No results to save!")
            msg.setWindowIcon(self._get_icon())
            msg.exec()
            return
        
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return
        
        try:
            # Save result tables
            if 'regression' in self.current_results:
                reg_df = self.current_results['regression'].get('results_df')
                if reg_df is not None:
                    reg_df.to_excel(os.path.join(save_dir, "regression_results.xlsx"), index=False)
                    self.log("Regression results saved to Excel")
            
            if 'classification' in self.current_results:
                clf_df = self.current_results['classification'].get('results_df')
                if clf_df is not None:
                    clf_df.to_excel(os.path.join(save_dir, "classification_results.xlsx"), index=False)
                    self.log("Classification results saved to Excel")
            
            # Save model details
            details_text = self.model_details.toPlainText()
            if details_text:
                with open(os.path.join(save_dir, "model_details.txt"), 'w') as f:
                    f.write(details_text)
                self.log("Model details saved to text file")
            
            # Save current plot
            plot_path = os.path.join(save_dir, "current_plot.png")
            self.plot_widget.figure.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.log("Current plot saved")
            
            msg = QMessageBox(self)
            msg.setWindowTitle("Success")
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"Results saved to {save_dir}")
            msg.setWindowIcon(self._get_icon())
            msg.exec()
            
        except Exception as e:
            self.log(f"Error saving results: {str(e)}")
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"Failed to save results: {str(e)}")
            msg.setWindowIcon(self._get_icon())
            msg.exec()
    
    def make_prediction(self):
        """Make prediction for entered SMILES"""
        smiles = self.smiles_input.text().strip()
        if not smiles:
            msg = QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please enter a SMILES string!")
            msg.setWindowIcon(self._get_icon())
            msg.exec()
            return
        
        if not self.current_results:
            msg = QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No trained models available!")
            msg.setWindowIcon(self._get_icon())
            msg.exec()
            return
        
        try:
            # Compute descriptors for input SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                msg = QMessageBox(self)
                msg.setWindowTitle("Warning")
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Invalid SMILES string!")
                msg.setWindowIcon(self._get_icon())
                msg.exec()
                return
            
            descriptors = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
            }
            
            # Prepare feature vector (assuming default values for missing features)
            if self.current_data:
                solvents = self.current_data['solvents']
                solvent_values = {solv: 100.0/len(solvents) for solv in solvents}
            else:
                solvent_values = {}
            
            # Combine all features
            feature_dict = {**solvent_values, **descriptors, 'pH': 7.0}
            
            result_text = ""
            # Regression prediction
            reg_model_fullname = self.reg_model_combo.currentText()
            reg_model_key = self.reg_model_key_map.get(reg_model_fullname, reg_model_fullname)
            reg_models = self.current_results.get('regression', {}).get('all_models', {})
            if reg_model_key and reg_model_key in reg_models:
                feat_cols = self.current_results['regression']['feature_cols']
                feature_vector = np.array([[feature_dict.get(col, 0) for col in feat_cols]])
                reg_model = reg_models[reg_model_key]
                pred_rt = reg_model.predict(feature_vector)[0]
                result_text += f"Predicted Retention Time ({reg_model_fullname}): {pred_rt:.2f} minutes\n"
            else:
                result_text += "No regression model selected or available\n"
            # Classification prediction
            clf_model_fullname = self.clf_model_combo.currentText()
            clf_model_key = self.clf_model_key_map.get(clf_model_fullname, clf_model_fullname)
            clf_models = self.current_results.get('classification', {}).get('all_models', {})
            label_enc = self.current_results.get('classification', {}).get('label_encoder')
            if clf_model_key and clf_model_key in clf_models and label_enc is not None:
                feat_cols = self.current_results['classification']['feature_cols']
                feature_vector = np.array([[feature_dict.get(col, 0) for col in feat_cols]])
                clf_model = clf_models[clf_model_key]
                pred_class = clf_model.predict(feature_vector)[0]
                pred_label = label_enc.inverse_transform([pred_class])[0]
                result_text += f"Predicted Separation Quality ({clf_model_fullname}): {pred_label}"
                # Show accuracy for selected model
                clf_results_df = self.current_results['classification'].get('results_df')
                if clf_results_df is not None and 'Model' in clf_results_df.columns and 'Accuracy' in clf_results_df.columns:
                    acc_row = clf_results_df[clf_results_df['Model'] == clf_model_key]
                    if acc_row.shape[0] > 0:
                        acc = acc_row['Accuracy'].values[0]
                        result_text += f"\nModel accuracy on test set: {acc*100:.2f}%"
            else:
                result_text += "No classification model selected or available"
            self.prediction_result.setText(result_text)
            self.log(f"Prediction made for SMILES: {smiles}")
            
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            self.log(error_msg)
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText(error_msg)
            msg.setWindowIcon(self._get_icon())
            msg.exec()
            self.status_label.setText("Error occurred")
            self.progress_bar.setValue(0)
    
    def on_error(self, error_msg):
        """Handle errors from worker threads"""
        self.log(f"ERROR: {error_msg}")
        msg = QMessageBox(self)
        msg.setWindowTitle("Error")
        msg.setIcon(QMessageBox.Critical)
        msg.setText(error_msg)
        msg.setWindowIcon(self._get_icon())
        msg.exec()
        self.status_label.setText("Error occurred")
        self.progress_bar.setValue(0)
    
    def log(self, message):
        """Add message to log"""
        self.log_text.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, 'Quit', 'Training is in progress. Are you sure you want to quit?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.terminate()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
    
    def update_model_selection_dropdowns(self, results):
        """Update regression and classification model dropdowns after training"""
        self.reg_model_combo.clear()
        self.clf_model_combo.clear()
        reg_models = results.get('regression', {}).get('all_models', {})
        # Use full names in combobox, but store mapping for key lookup
        self.reg_model_key_map = {}
        for key in reg_models:
            full_name = self.available_reg_models.get(key, key)
            self.reg_model_combo.addItem(full_name)
            self.reg_model_key_map[full_name] = key
        clf_models = results.get('classification', {}).get('all_models', {})
        self.clf_model_key_map = {}
        for key in clf_models:
            full_name = self.available_clf_models.get(key, key)
            self.clf_model_combo.addItem(full_name)
            self.clf_model_key_map[full_name] = key
    
    def setup_menubar_statusbar(self):
        # Menu Bar
        menubar = self.menuBar()
        # File menu
        file_menu = menubar.addMenu("File")
        load_action = file_menu.addAction("Load Data File")
        load_action.triggered.connect(self.browse_file)
        save_action = file_menu.addAction("Save Results")
        save_action.triggered.connect(self.save_results)
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        # Model menu
        model_menu = menubar.addMenu("Model")
        # Model selection dialog
        model_select_action = model_menu.addAction("Select Models...")
        model_select_action.triggered.connect(self.show_model_selection_dialog)
        # Output dir
        output_dir_action = model_menu.addAction("Set Output Directory...")
        output_dir_action.triggered.connect(self.set_output_directory)
        # Train
        train_action = model_menu.addAction("Train Models")
        train_action.triggered.connect(self.train_models)
        # Load Training Data
        load_train_data_action = model_menu.addAction("Load Training Data...")
        load_train_data_action.triggered.connect(self.load_training_data)
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_app_action = help_menu.addAction("Help App")
        help_app_action.triggered.connect(self.show_help_app_dialog)
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about_dialog)
        # Status Bar
        statusbar = self.statusBar()
        statusbar.showMessage("Ready | Version 1.0.0")
        # --- THEME SWITCHER ---
        from PySide6.QtWidgets import QComboBox, QLabel
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentText("Light")
        self.theme_combo.setMaximumWidth(100)
        self.theme_combo.setToolTip("Switch theme")
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        statusbar.addPermanentWidget(QLabel("Theme:"))
        statusbar.addPermanentWidget(self.theme_combo)
        self.apply_theme("Light")


    def apply_theme(self, theme_name):
        """Apply modern dark or light theme to the whole app"""
        
        # Modern Dark Theme - Inspired by VS Code Dark+ and modern design systems
        dark_stylesheet = """
        QWidget {
            background-color: #1e1e1e;
            color: #cccccc;
            font-family: 'Segoe UI', 'SF Pro Display', 'Inter', 'Roboto', sans-serif;
            font-size: 9pt;
            selection-background-color: #264f78;
            selection-color: #ffffff;
        }
        
        QMainWindow, QDialog {
            background-color: #1e1e1e;
            border: none;
        }
        
        QFrame {
            background-color: #252526;
            border: none;
            border-radius: 8px;
        }
        
        QGroupBox {
            background-color: #252526;
            border: 2px solid #3c3c3c;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: 600;
            color: #cccccc;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px 0 8px;
            color: #ffffff;
        }
        
        /* Tabs */
        QTabWidget::pane {
            background-color: #252526;
            border: 1px solid #3c3c3c;
            border-radius: 8px;
            margin-top: -1px;
        }
        
        QTabBar::tab {
            background: #2d2d30;
            color: #cccccc;
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            min-width: 80px;
            font-weight: 500;
        }
        
        QTabBar::tab:selected {
            background: #007acc;
            color: #ffffff;
            font-weight: 600;
        }
        
        QTabBar::tab:hover:!selected {
            background: #3c3c3c;
            color: #ffffff;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #0e639c;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 10px 16px;
            font-weight: 600;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background-color: #1177bb;
        }
        
        QPushButton:pressed {
            background-color: #005a9e;
        }
        
        QPushButton:disabled {
            background-color: #3c3c3c;
            color: #808080;
        }
        
        /* Secondary Button Style */
        QPushButton[class="secondary"] {
            background-color: #3c3c3c;
            color: #cccccc;
            border: 1px solid #5a5a5a;
        }
        
        QPushButton[class="secondary"]:hover {
            background-color: #4d4d4d;
            border-color: #6a6a6a;
        }
        
        /* Input Fields */
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #3c3c3c;
            color: #cccccc;
            border: 2px solid #5a5a5a;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 9pt;
        }
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border-color: #007acc;
            background-color: #404040;
        }
        
        /* ComboBox */
        QComboBox {
            background-color: #3c3c3c;
            color: #cccccc;
            border: 2px solid #5a5a5a;
            border-radius: 6px;
            padding: 8px 12px;
            min-width: 100px;
        }
        
        QComboBox:hover {
            border-color: #007acc;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #cccccc;
            margin-right: 5px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #3c3c3c;
            color: #cccccc;
            border: 1px solid #5a5a5a;
            border-radius: 6px;
            selection-background-color: #007acc;
        }
        
        /* Tables */
        QTableWidget, QTableView {
            background-color: #252526;
            alternate-background-color: #2d2d30;
            color: #cccccc;
            border: 1px solid #3c3c3c;
            border-radius: 6px;
            gridline-color: #3c3c3c;
        }
        
        QHeaderView::section {
            background-color: #37373d;
            color: #ffffff;
            border: none;
            padding: 10px;
            font-weight: 600;
            border-bottom: 2px solid #007acc;
        }
        
        QTableWidget::item:selected, QTableView::item:selected {
            background-color: #264f78;
            color: #ffffff;
        }
        
        /* Progress Bar */
        QProgressBar {
            background-color: #3c3c3c;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            text-align: center;
            padding: 2px;
            font-weight: 600;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #007acc, stop:1 #1177bb);
            border-radius: 6px;
        }
        
        /* Scrollbars */
        QScrollBar:vertical {
            background: #2d2d30;
            border: none;
            width: 14px;
            border-radius: 7px;
        }
        
        QScrollBar::handle:vertical {
            background: #5a5a5a;
            border-radius: 7px;
            min-height: 30px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #6a6a6a;
        }
        
        QScrollBar:horizontal {
            background: #2d2d30;
            border: none;
            height: 14px;
            border-radius: 7px;
        }
        
        QScrollBar::handle:horizontal {
            background: #5a5a5a;
            border-radius: 7px;
            min-width: 30px;
            margin: 2px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: #6a6a6a;
        }
        
        QScrollBar::add-line, QScrollBar::sub-line {
            background: none;
            border: none;
        }
        
        /* Checkboxes and Radio Buttons */
        QCheckBox {
            color: #cccccc;
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #5a5a5a;
            border-radius: 4px;
            background-color: #3c3c3c;
        }
        
        QCheckBox::indicator:checked {
            background-color: #007acc;
            border-color: #007acc;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
        }
        
        /* Labels and Text */
        QLabel {
            color: #cccccc;
        }
        
        /* Menus */
        QMenuBar {
            background-color: #2d2d30;
            color: #cccccc;
            border-bottom: 1px solid #3c3c3c;
            padding: 4px;
        }
        
        QMenuBar::item {
            background: transparent;
            padding: 6px 12px;
            border-radius: 4px;
        }
        
        QMenuBar::item:selected {
            background-color: #37373d;
            color: #ffffff;
        }
        
        QMenu {
            background-color: #3c3c3c;
            color: #cccccc;
            border: 1px solid #5a5a5a;
            border-radius: 6px;
            padding: 4px;
        }
        
        QMenu::item {
            padding: 8px 16px;
            border-radius: 4px;
        }
        
        QMenu::item:selected {
            background-color: #007acc;
            color: #ffffff;
        }
        
        /* Tooltips */
        QToolTip {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #007acc;
            border-radius: 6px;
            padding: 8px;
            font-size: 9pt;
        }
        """
        
        # Modern Light Theme - Inspired by macOS Big Sur and Material Design
        light_stylesheet = """
        QWidget {
            background-color: #ffffff;
            color: #1d1d1f;
            font-family: 'Segoe UI', 'SF Pro Display', 'Inter', 'Roboto', sans-serif;
            font-size: 9pt;
            selection-background-color: #007acc;
            selection-color: #ffffff;
        }
        
        QMainWindow, QDialog {
            background-color: #f8f9fa;
            border: none;
        }
        
        QFrame {
            background-color: #ffffff;
            border: 1px solid #e5e5e7;
            border-radius: 12px;
        }
        
        QGroupBox {
            background-color: #ffffff;
            border: 2px solid #d2d2d7;
            border-radius: 12px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: 600;
            color: #1d1d1f;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px 0 8px;
            color: #1d1d1f;
        }
        
        /* Tabs */
        QTabWidget::pane {
            background-color: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 12px;
            margin-top: -1px;
        }
        
        QTabBar::tab {
            background: #f2f2f7;
            color: #8e8e93;
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
            min-width: 80px;
            font-weight: 500;
        }
        
        QTabBar::tab:selected {
            background: #007acc;
            color: #ffffff;
            font-weight: 600;
        }
        
        QTabBar::tab:hover:!selected {
            background: #e5e5ea;
            color: #1d1d1f;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #007acc;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            font-weight: 600;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background-color: #0051a2;
        }
        
        QPushButton:pressed {
            background-color: #004085;
        }
        
        QPushButton:disabled {
            background-color: #d2d2d7;
            color: #8e8e93;
        }
        
        /* Secondary Button Style */
        QPushButton[class="secondary"] {
            background-color: #f2f2f7;
            color: #1d1d1f;
            border: 2px solid #d2d2d7;
        }
        
        QPushButton[class="secondary"]:hover {
            background-color: #e5e5ea;
            border-color: #a1a1a6;
        }
        
        /* Input Fields */
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #ffffff;
            color: #1d1d1f;
            border: 2px solid #d2d2d7;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 9pt;
        }
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border-color: #007acc;
            background-color: #ffffff;
        }
        
        /* ComboBox */
        QComboBox {
            background-color: #ffffff;
            color: #1d1d1f;
            border: 2px solid #d2d2d7;
            border-radius: 8px;
            padding: 10px 14px;
            min-width: 100px;
        }
        
        QComboBox:hover {
            border-color: #007acc;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #8e8e93;
            margin-right: 5px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #1d1d1f;
            border: 1px solid #d2d2d7;
            border-radius: 8px;
            selection-background-color: #007acc;
            selection-color: #ffffff;
        }
        
        /* Tables */
        QTableWidget, QTableView {
            background-color: #ffffff;
            alternate-background-color: #f8f9fa;
            color: #1d1d1f;
            border: 1px solid #d2d2d7;
            border-radius: 8px;
            gridline-color: #e5e5e7;
        }
        
        QHeaderView::section {
            background-color: #f2f2f7;
            color: #1d1d1f;
            border: none;
            padding: 12px;
            font-weight: 600;
            border-bottom: 2px solid #007acc;
        }
        
        QTableWidget::item:selected, QTableView::item:selected {
            background-color: #007acc;
            color: #ffffff;
        }
        
        /* Progress Bar */
        QProgressBar {
            background-color: #f2f2f7;
            color: #1d1d1f;
            border: none;
            border-radius: 8px;
            text-align: center;
            padding: 2px;
            font-weight: 600;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #007acc, stop:1 #0051a2);
            border-radius: 8px;
        }
        
        /* Scrollbars */
        QScrollBar:vertical {
            background: #f2f2f7;
            border: none;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background: #c7c7cc;
            border-radius: 6px;
            min-height: 30px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #a1a1a6;
        }
        
        QScrollBar:horizontal {
            background: #f2f2f7;
            border: none;
            height: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:horizontal {
            background: #c7c7cc;
            border-radius: 6px;
            min-width: 30px;
            margin: 2px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: #a1a1a6;
        }
        
        QScrollBar::add-line, QScrollBar::sub-line {
            background: none;
            border: none;
        }
        
        /* Checkboxes and Radio Buttons */
        QCheckBox {
            color: #1d1d1f;
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border: 2px solid #d2d2d7;
            border-radius: 6px;
            background-color: #ffffff;
        }
        
        QCheckBox::indicator:checked {
            background-color: #007acc;
            border-color: #007acc;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
        }
        
        /* Labels and Text */
        QLabel {
            color: #1d1d1f;
        }
        
        /* Menus */
        QMenuBar {
            background-color: #f8f9fa;
            color: #1d1d1f;
            border-bottom: 1px solid #d2d2d7;
            padding: 4px;
        }
        
        QMenuBar::item {
            background: transparent;
            padding: 8px 16px;
            border-radius: 6px;
        }
        
        QMenuBar::item:selected {
            background-color: #e5e5ea;
            color: #1d1d1f;
        }
        
        QMenu {
            background-color: #ffffff;
            color: #1d1d1f;
            border: 1px solid #d2d2d7;
            border-radius: 12px;
            padding: 6px;
        }
        
        QMenu::item {
            padding: 10px 16px;
            border-radius: 8px;
        }
        
        QMenu::item:selected {
            background-color: #007acc;
            color: #ffffff;
        }
        
        /* Tooltips */
        QToolTip {
            background-color: #1d1d1f;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 10px;
            font-size: 9pt;
        }
        """
        
        # Apply the selected theme
        if theme_name.lower() == "dark":
            self.setStyleSheet(dark_stylesheet)
        elif theme_name.lower() == "light":
            self.setStyleSheet(light_stylesheet)
        else:
            QMessageBox.warning(self, "Invalid Theme", "Invalid theme name. Using default light theme.")
            self.setStyleSheet(light_stylesheet)

    def show_about_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("About HPLC Pipeline GUI")
        dlg.setWindowIcon(self._get_icon())
        dlg.setMinimumWidth(400)
        layout = QVBoxLayout(dlg)
        from PySide6.QtWidgets import QTextBrowser, QPushButton, QHBoxLayout
        text = QTextBrowser()
        text.setOpenExternalLinks(True)
        text.setHtml(
            """
            <h2>HPLC Pipeline GUI</h2>
            <p><b>Created by:</b> Arif Maulana Azis<br>
            <b>Company:</b> Titan Digitalsoft</p>
            <p>
            <b>Github:</b> <a href='https://github.com/Arifmaulanaazis'>Arifmaulanaazis</a><br>
            <b>Instagram:</b> <a href='https://instagram.com/arif_maulana_19'>arif_maulana_19</a><br>
            <b>Facebook:</b> <a href='https://facebook.com/ArifMaulanaAzis19'>ArifMaulanaAzis19</a>
            </p>
            <hr>
            <p style='font-size:10pt;color:gray;'>HPLC Pipeline GUI is a tool for chromatographic data analysis, model training, and prediction. For more info, contact the author via social media above.</p>
            """
        )
        layout.addWidget(text)
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dlg.accept)
        btn_box.addStretch()
        btn_box.addWidget(ok_btn)
        layout.addLayout(btn_box)
        dlg.exec()
    
    def show_model_selection_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Models to Train")
        dlg.setWindowIcon(self._get_icon())
        layout = QVBoxLayout(dlg)
        reg_label = QLabel("Regression Models:")
        layout.addWidget(reg_label)
        reg_checks = {}
        reg_hyper_btns = {}
        for key, name in self.available_reg_models.items():
            row = QHBoxLayout()
            cb = QCheckBox(name)
            cb.setChecked(key in self.selected_reg_models)
            reg_checks[key] = cb
            row.addWidget(cb)
            # Add hyperparameter button
            btn = QPushButton("Set Hyperparameters")
            btn.clicked.connect(lambda _, k=key: self.show_hyperparam_dialog(k, is_reg=True))
            reg_hyper_btns[key] = btn
            row.addWidget(btn)
            layout.addLayout(row)
        clf_label = QLabel("Classification Models:")
        layout.addWidget(clf_label)
        clf_checks = {}
        clf_hyper_btns = {}
        for key, name in self.available_clf_models.items():
            row = QHBoxLayout()
            cb = QCheckBox(name)
            cb.setChecked(key in self.selected_clf_models)
            clf_checks[key] = cb
            row.addWidget(cb)
            # Add hyperparameter button
            btn = QPushButton("Set Hyperparameters")
            btn.clicked.connect(lambda _, k=key: self.show_hyperparam_dialog(k, is_reg=False))
            clf_hyper_btns[key] = btn
            row.addWidget(btn)
            layout.addLayout(row)
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)
        def accept():
            self.selected_reg_models = set(k for k, cb in reg_checks.items() if cb.isChecked())
            self.selected_clf_models = set(k for k, cb in clf_checks.items() if cb.isChecked())
            dlg.accept()
        ok_btn.clicked.connect(accept)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec()
    
    def show_hyperparam_dialog(self, model_key, is_reg):
        # Define default hyperparameter grids for each model
        default_grids = {
            # Regression
            'LR': {'mdl__fit_intercept': [True, False]},
            'Ridge': {'mdl__alpha': [1.0, 10.0]},
            'Lasso': {'mdl__alpha': [0.1, 1.0]},
            'RF': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
            'GB': {'n_estimators': [50, 100], 'learning_rate': [0.1], 'max_depth': [3, 5]},
            'SVR': {'mdl__C': [1, 10], 'mdl__gamma': ['scale']},
            'MLP': {'mdl__hidden_layer_sizes': [(50,)], 'mdl__alpha': [1e-3]},
            'KNN': {'mdl__n_neighbors': [3, 5, 7]},
            'DT': {'max_depth': [None, 5, 10]},
            # Classification
            'RFC': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
            'GBC': {'n_estimators': [50, 100], 'learning_rate': [0.1], 'max_depth': [3]},
            'SVC': {'mdl__C': [1, 10], 'mdl__gamma': ['scale']},
            'MLPC': {'mdl__hidden_layer_sizes': [(50,)], 'mdl__alpha': [1e-3]},
            'KNN': {'mdl__n_neighbors': [3, 5, 7]},
            'DT': {'max_depth': [None, 5, 10]},
            'LR': {'mdl__C': [1.0, 10.0]},
        }
        grid = self.model_hyperparams.get(model_key, default_grids.get(model_key, {})).copy()
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Set Hyperparameters for {model_key}")
        dlg.setWindowIcon(self._get_icon())
        layout = QVBoxLayout(dlg)
        edits = {}
        for param, val in grid.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(param))
            edit = QLineEdit(str(val))
            edits[param] = edit
            row.addWidget(edit)
            layout.addLayout(row)
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)
        def accept():
            new_grid = {}
            for param, edit in edits.items():
                # Try to eval as python literal, fallback to string
                try:
                    v = eval(edit.text())
                except Exception:
                    v = edit.text()
                new_grid[param] = v
            self.model_hyperparams[model_key] = new_grid
            dlg.accept()
        ok_btn.clicked.connect(accept)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec()
    
    def set_output_directory(self):
        dir_ = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_:
            self.output_directory = dir_
            self.statusBar().showMessage(f"Output directory set: {dir_}")
    
    def load_training_data(self):
        # User selects a .pkl file (training_context.pkl)
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Training Data Context", "", "Pickle Files (*.pkl)")
        if not file_path:
            return
        try:
            with open(file_path, "rb") as f:
                ctx = pickle.load(f)
            self.current_data = ctx['data']
            self.current_results = ctx['results']
            # Restore optimization_heatmap if available
            if 'optimization_heatmap' in ctx:
                self.optimization_heatmap = ctx['optimization_heatmap']
            # Update all UI: data table, results, plots, model details
            self.update_data_table(self.current_data['dataframe'])
            self.update_results_table(self.current_results)
            self.plot_widget.set_plot_data(self.current_results)
            self.update_model_details(self.current_results)
            self.update_model_selection_dropdowns(self.current_results)
            self.update_optimasi_tab(self.current_results)
            self.log("Training data loaded successfully!")
            self.status_label.setText("Training data loaded - ready for prediction/visualization")
        except Exception as e:
            self.log(f"Error loading training data: {str(e)}")
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"Failed to load training data: {str(e)}")
            msg.setWindowIcon(self._get_icon())
            msg.exec()
    
    def update_optimasi_tab(self, results):
        """Update isi tab Optimasi setelah training selesai"""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt
        # Hapus canvas lama jika ada
        if self.optimasi_heatmap_canvas:
            self.optimasi_layout.removeWidget(self.optimasi_heatmap_canvas)
            self.optimasi_heatmap_canvas.setParent(None)
            self.optimasi_heatmap_canvas = None
        # Ambil data heatmap dan grid dari data asli
        df = self.current_data['dataframe'] if self.current_data else None
        pH_range = None
        org_range = None
        buf_range = None
        if df is not None:
            # Ambil unique values dari data
            feat_cols = results.get('classification', {}).get('feature_cols', [])
            solvents = [col for col in feat_cols if col.endswith('_%')]
            if solvents and 'pH' in feat_cols:
                organic_col = solvents[0]
                buffer_col = solvents[1] if len(solvents) > 1 else None
                pH_range = np.sort(df['pH'].dropna().unique())
                org_range = np.sort(df[organic_col].dropna().unique())
                if buffer_col:
                    buf_range = np.sort(df[buffer_col].dropna().unique())
        # 1. Plot heatmap
        if hasattr(self, 'optimization_heatmap') and pH_range is not None and org_range is not None:
            data = self.optimization_heatmap
            label_enc = data['label_enc']
            organic_col = data['organic_col']
            buffer_col = data['buffer_col']
            # Buat matrix prediksi hanya untuk kombinasi yang ada di data
            pred_matrix = np.full((len(pH_range), len(org_range)), np.nan)
            best_clf = results['classification']['best_model'] if 'classification' in results else None
            feat_cols = results['classification']['feature_cols'] if 'classification' in results else None
            for i, pH in enumerate(pH_range):
                for j, org in enumerate(org_range):
                    buf = 100 - org if buffer_col else 0
                    # Cek apakah kombinasi ini ada di data
                    if buffer_col:
                        exists = ((df['pH'] == pH) & (df[organic_col] == org) & (df[buffer_col] == buf)).any()
                    else:
                        exists = ((df['pH'] == pH) & (df[organic_col] == org)).any()
                    if not exists:
                        continue
                    feat_dict = {col: 0 for col in feat_cols}
                    feat_dict[organic_col] = org
                    if buffer_col:
                        feat_dict[buffer_col] = buf
                    feat_dict['pH'] = pH
                    for desc in ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'Peak_Area']:
                        if desc in feat_cols:
                            feat_dict[desc] = 0
                    X_df = pd.DataFrame([feat_dict], columns=feat_cols)
                    pred = best_clf.predict(X_df)[0]
                    pred_matrix[i, j] = pred
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(pred_matrix, aspect='auto', origin='lower',
                           extent=[org_range[0], org_range[-1], pH_range[0], pH_range[-1]],
                           cmap='viridis')
            cbar = plt.colorbar(im, ax=ax)
            if label_enc is not None:
                ticks = np.arange(len(label_enc.classes_))
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(label_enc.classes_)
            ax.set_xlabel(f"{organic_col} (%)")
            ax.set_ylabel("pH")
            ax.set_title("Prediksi Separation Quality (Heatmap)")
            fig.tight_layout()
            self.optimasi_heatmap_canvas = FigureCanvas(fig)
            self.optimasi_layout.insertWidget(1, self.optimasi_heatmap_canvas)
        # 2. Saran optimasi
        saran_text = ""
        if 'classification' in results and results['classification'].get('best_model') is not None and pH_range is not None and org_range is not None:
            try:
                best_clf = results['classification']['best_model']
                feat_cols = results['classification']['feature_cols']
                label_enc = results['classification'].get('label_encoder')
                solvents = [col for col in feat_cols if col.endswith('_%')]
                organic_col = solvents[0]
                buffer_col = solvents[1] if len(solvents) > 1 else None
                combos = []
                for pH in pH_range:
                    for org in org_range:
                        buf = 100 - org if buffer_col else 0
                        if buffer_col:
                            exists = ((df['pH'] == pH) & (df[organic_col] == org) & (df[buffer_col] == buf)).any()
                        else:
                            exists = ((df['pH'] == pH) & (df[organic_col] == org)).any()
                        if not exists:
                            continue
                        feat_dict = {col: 0 for col in feat_cols}
                        feat_dict[organic_col] = org
                        if buffer_col:
                            feat_dict[buffer_col] = buf
                        feat_dict['pH'] = pH
                        for desc in ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'Peak_Area']:
                            if desc in feat_cols:
                                feat_dict[desc] = 0
                        X_df = pd.DataFrame([feat_dict], columns=feat_cols)
                        pred = best_clf.predict(X_df)[0]
                        combos.append((pH, org, buf, pred))
                # Urutkan kelas terbaik sesuai prioritas
                if label_enc is not None:
                    class_priority = [
                        'Completely separated',
                        'Switching-fully separated',
                        'Switching-partialy separated',
                        'Partialy separated',
                        'Overlaping',
                        'No data'
                    ]
                    class_to_int = {c: label_enc.transform([c])[0] for c in class_priority if c in label_enc.classes_}
                    best_pred = None
                    best_combo = None
                    for class_name in class_priority:
                        class_val = class_to_int.get(class_name)
                        if class_val is None:
                            continue
                        filtered = [c for c in combos if c[3] == class_val]
                        if filtered:
                            best_combo = filtered[0]
                            best_pred = class_val
                            break
                    if best_combo is not None:
                        best_label = class_name
                        saran_text = f"pH Combination={best_combo[0]}, {organic_col}={best_combo[1]}%{f', {buffer_col}={best_combo[2]}%' if buffer_col else ''} is predicted to yield the best Separation Quality: {best_label}"
                    else:
                        saran_text = "Label encoder not found."
            except Exception as e:
                saran_text = f"[Error optimasi: {str(e)}]"
        self.optimasi_saran_text.setText(saran_text)
        # 3. Tabel komposisi terbaik
        # (Tampilkan 10 kombinasi terbaik)
        best_rows = []
        if hasattr(self, 'optimization_heatmap'):
            data = self.optimization_heatmap
            pH_range = data['pH_range']
            org_range = data['org_range']
            pred_matrix = data['pred_matrix']
            label_enc = data['label_enc']
            organic_col = data['organic_col']
            buffer_col = data['buffer_col']
            combos = []
            for i, pH in enumerate(pH_range):
                for j, org in enumerate(org_range):
                    pred = pred_matrix[i, j]
                    combos.append((pH, org, 100-org if buffer_col else 0, pred))
            combos.sort(key=lambda x: -x[3])
            for row in combos[:10]:
                pH, org, buf, pred = row
                if label_enc is not None:
                    pred_label = label_enc.inverse_transform([int(round(pred))])[0]
                else:
                    pred_label = str(pred)
                best_rows.append((f"{pH:.2f}", f"{org:.1f}", f"{buf:.1f}" if buffer_col else "-", pred_label))
        headers = ["pH", organic_col, buffer_col if buffer_col else "-", "Quality Prediction"]
        self.optimasi_best_table.setRowCount(len(best_rows))
        self.optimasi_best_table.setColumnCount(len(headers))
        self.optimasi_best_table.setHorizontalHeaderLabels(headers)
        for i, row in enumerate(best_rows):
            for j, val in enumerate(row):
                self.optimasi_best_table.setItem(i, j, QTableWidgetItem(str(val)))
        # 4. Tabel feature importance
        featimp_rows = []
        if 'classification' in results:
            best_clf = results['classification'].get('best_model')
            feat_cols = results['classification'].get('feature_cols')
            if best_clf is not None and feat_cols is not None:
                importance_attr = None
                if hasattr(best_clf, 'feature_importances_'):
                    importance_attr = best_clf.feature_importances_
                elif hasattr(best_clf, 'named_steps'):
                    for step in best_clf.named_steps.values():
                        if hasattr(step, 'feature_importances_'):
                            importance_attr = step.feature_importances_
                            break
                if importance_attr is not None:
                    indices = np.argsort(importance_attr)[::-1]
                    for rank, idx in enumerate(indices, 1):
                        featimp_rows.append((rank, feat_cols[idx], f"{importance_attr[idx]:.4f}"))
        self.optimasi_featimp_table.setRowCount(len(featimp_rows))
        self.optimasi_featimp_table.setColumnCount(3)
        self.optimasi_featimp_table.setHorizontalHeaderLabels(["Rank", "Feature", "Importance"])
        for i, row in enumerate(featimp_rows):
            for j, val in enumerate(row):
                self.optimasi_featimp_table.setItem(i, j, QTableWidgetItem(str(val)))
        # 5. Penjelasan data
        explain = []
        if featimp_rows:
            explain.append(f"Most influential factor for separation: {featimp_rows[0][1]}")
        if saran_text:
            explain.append(f"Optimization suggestion: {saran_text}")
        explain.append("The heatmap shows the predicted separation quality for each pH and composition combination.\n\nClass order (best to worst): Completely separated > Switching-fully separated > Switching-partialy separated > Partialy separated > Overlaping > No data.\n\nDarker color = lower class, lighter color = better separation quality.")
        explain.append("\nFor bar plots: Higher Accuracy, Precision, Recall, F1-score, and R² are better. Lower RMSE and MAE are better.")
        self.optimasi_explain_text.setText("\n".join(explain))

    def suggest_optimal_condition(self):
        """Suggest optimal buffer/organic/pH for best separation quality for input SMILES"""
        smiles = self.smiles_input.text().strip()
        if not smiles:
            msg = QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please enter a SMILES string!")
            msg.setWindowIcon(self._get_icon())
            msg.exec()
            return
        if not self.current_results:
            msg = QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No trained models available!")
            msg.setWindowIcon(self._get_icon())
            msg.exec()
            return
        try:
            # Compute descriptors for input SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                msg = QMessageBox(self)
                msg.setWindowTitle("Warning")
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Invalid SMILES string!")
                msg.setWindowIcon(self._get_icon())
                msg.exec()
                return
            descriptors = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
            }
            # Get all unique buffer/organic/pH from training data
            df = self.current_data['dataframe']
            feat_cols = self.current_results['classification']['feature_cols']
            solvents = [col for col in feat_cols if col.endswith('_%')]
            organic_col = solvents[0]
            buffer_col = solvents[1] if len(solvents) > 1 else None
            pH_vals = np.sort(df['pH'].dropna().unique())
            org_vals = np.sort(df[organic_col].dropna().unique())
            buf_vals = np.sort(df[buffer_col].dropna().unique()) if buffer_col else None
            # Get selected models
            reg_model_fullname = self.reg_model_combo.currentText()
            reg_model_key = self.reg_model_key_map.get(reg_model_fullname, reg_model_fullname)
            reg_models = self.current_results.get('regression', {}).get('all_models', {})
            clf_model_fullname = self.clf_model_combo.currentText()
            clf_model_key = self.clf_model_key_map.get(clf_model_fullname, clf_model_fullname)
            clf_models = self.current_results.get('classification', {}).get('all_models', {})
            label_enc = self.current_results.get('classification', {}).get('label_encoder')
            # Priority order for best class
            class_priority = [
                'Completely separated',
                'Switching-fully separated',
                'Switching-partialy separated',
                'Partialy separated',
                'Overlaping',
                'No data'
            ]
            class_to_int = {c: label_enc.transform([c])[0] for c in class_priority if c in label_enc.classes_} if label_enc else {}
            combos = []
            for pH in pH_vals:
                for org in org_vals:
                    buf = 100 - org if buffer_col else 0
                    if buffer_col:
                        exists = ((df['pH'] == pH) & (df[organic_col] == org) & (df[buffer_col] == buf)).any()
                    else:
                        exists = ((df['pH'] == pH) & (df[organic_col] == org)).any()
                    if not exists:
                        continue
                    feat_dict = {col: 0 for col in feat_cols}
                    feat_dict[organic_col] = org
                    if buffer_col:
                        feat_dict[buffer_col] = buf
                    feat_dict['pH'] = pH
                    for desc in ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA']:
                        if desc in feat_cols:
                            feat_dict[desc] = descriptors[desc]
                    # Predict separation quality
                    if clf_model_key in clf_models and label_enc is not None:
                        X_df = pd.DataFrame([feat_dict], columns=feat_cols)
                        pred = clf_models[clf_model_key].predict(X_df)[0]
                        combos.append((pH, org, buf, pred, feat_dict))
            # Find best combo by class priority
            best_combo = None
            best_label = None
            for class_name in class_priority:
                class_val = class_to_int.get(class_name)
                filtered = [c for c in combos if c[3] == class_val]
                if filtered:
                    best_combo = filtered[0]
                    best_label = class_name
                    break
            result_text = ""
            if best_combo is not None:
                pH, org, buf, pred, feat_dict = best_combo
                result_text += f"Optimal condition for best separation quality:\n"
                result_text += f"  - pH: {pH}\n  - {organic_col}: {org}%"
                if buffer_col:
                    result_text += f"\n  - {buffer_col}: {buf}%"
                result_text += f"\n  - Predicted Separation Quality: {best_label}"
                # Predict retention time if regression model available
                if reg_model_key in reg_models:
                    X_reg = np.array([[feat_dict.get(col, 0) for col in self.current_results['regression']['feature_cols']]])
                    pred_rt = reg_models[reg_model_key].predict(X_reg)[0]
                    result_text += f"\n  - Predicted Retention Time: {pred_rt:.2f} min"
                # Show accuracy for selected classification model
                clf_results_df = self.current_results['classification'].get('results_df')
                if clf_results_df is not None and 'Model' in clf_results_df.columns and 'Accuracy' in clf_results_df.columns:
                    acc_row = clf_results_df[clf_results_df['Model'] == clf_model_key]
                    if acc_row.shape[0] > 0:
                        acc = acc_row['Accuracy'].values[0]
                        result_text += f"\n  - Model accuracy (classification): {acc*100:.2f}%"
                # Show R2 for regression model
                reg_results_df = self.current_results.get('regression', {}).get('results_df')
                if reg_results_df is not None and 'Model' in reg_results_df.columns and 'R2' in reg_results_df.columns:
                    r2_row = reg_results_df[reg_results_df['Model'] == reg_model_key]
                    if r2_row.shape[0] > 0:
                        r2 = r2_row['R2'].values[0]
                        result_text += f"\n  - Regression R²: {r2:.3f}"
            else:
                result_text = "No optimal condition found for best separation quality in the training data."
            self.prediction_result.setText(result_text)
            self.log(f"Optimal condition suggested for SMILES: {smiles}")
        except Exception as e:
            error_msg = f"Error suggesting optimal condition: {str(e)}"
            self.log(error_msg)
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setIcon(QMessageBox.Critical)
            msg.setText(error_msg)
            msg.setWindowIcon(self._get_icon())
            msg.exec()
            self.status_label.setText("Error occurred")
            self.progress_bar.setValue(0)

    def run_pca_analysis(self):
        if not self.current_data:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
        df = self.current_data['dataframe']
        feat_cols = self.current_data['solvents'] + ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'pH']
        X = df[feat_cols].fillna(0).values
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_
        # Table: explained variance
        self.analysis_result_label.setText("PCA Results (Explained Variance)")
        self.analysis_result_table.setRowCount(len(explained_var))
        self.analysis_result_table.setColumnCount(2)
        self.analysis_result_table.setHorizontalHeaderLabels(["PC", "Explained Variance"])
        for i, var in enumerate(explained_var):
            self.analysis_result_table.setItem(i, 0, QTableWidgetItem(f"PC{i+1}"))
            self.analysis_result_table.setItem(i, 1, QTableWidgetItem(f"{var:.4f}"))
        # Text: loadings
        loadings = pd.DataFrame(pca.components_.T, index=feat_cols, columns=[f"PC{i+1}" for i in range(3)])
        self.analysis_result_text.setText("PCA Loadings:\n" + loadings.to_string())
        # Store for plotting
        if not hasattr(self, 'analysis_results'):
            self.analysis_results = {}
        self.analysis_results['PCA'] = {'X_pca': X_pca, 'explained_var': explained_var, 'loadings': loadings, 'labels': df.get('Separation_Quality', None)}
        # Update plot data
        if not hasattr(self, 'plot_widget'):
            return
        if not hasattr(self.plot_widget, 'plot_data'):
            self.plot_widget.plot_data = {}
        self.plot_widget.plot_data['PCA'] = self.analysis_results['PCA']
        self.plot_widget.update_plot_menu()
    def run_clustering_analysis(self):
        if not self.current_data:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
        df = self.current_data['dataframe']
        feat_cols = self.current_data['solvents'] + ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'pH']
        X = df[feat_cols].fillna(0).values
        Z = linkage(X, method='ward')
        clusterer = AgglomerativeClustering(n_clusters=2)
        labels = clusterer.fit_predict(X)
        self.analysis_result_label.setText("Hierarchical Clustering Results")
        self.analysis_result_table.setRowCount(len(labels))
        self.analysis_result_table.setColumnCount(1)
        self.analysis_result_table.setHorizontalHeaderLabels(["Cluster Label"])
        for i, lab in enumerate(labels):
            self.analysis_result_table.setItem(i, 0, QTableWidgetItem(str(lab)))
        self.analysis_result_text.setText("Dendrogram will be shown in Visualizations tab.")
        if not hasattr(self, 'analysis_results'):
            self.analysis_results = {}
        self.analysis_results['Clustering'] = {'Z': Z, 'labels': labels}
        if not hasattr(self, 'plot_widget'):
            return
        if not hasattr(self.plot_widget, 'plot_data'):
            self.plot_widget.plot_data = {}
        self.plot_widget.plot_data['Clustering'] = self.analysis_results['Clustering']
        self.plot_widget.update_plot_menu()
    def run_tsne_analysis(self):
        if not self.current_data:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
        df = self.current_data['dataframe']
        feat_cols = self.current_data['solvents'] + ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'pH']
        X = df[feat_cols].fillna(0).values
        tsne2 = TSNE(n_components=2, random_state=42)
        X_tsne2 = tsne2.fit_transform(X)
        tsne3 = TSNE(n_components=3, random_state=42)
        X_tsne3 = tsne3.fit_transform(X)
        self.analysis_result_label.setText("t-SNE Results")
        self.analysis_result_table.setRowCount(0)
        self.analysis_result_table.setColumnCount(0)
        self.analysis_result_text.setText("t-SNE 2D/3D results available in Visualizations tab.")
        if not hasattr(self, 'analysis_results'):
            self.analysis_results = {}
        self.analysis_results['TSNE'] = {'X_tsne2': X_tsne2, 'X_tsne3': X_tsne3, 'labels': df.get('Separation_Quality', None)}
        if not hasattr(self, 'plot_widget'):
            return
        if not hasattr(self.plot_widget, 'plot_data'):
            self.plot_widget.plot_data = {}
        self.plot_widget.plot_data['TSNE'] = self.analysis_results['TSNE']
        self.plot_widget.update_plot_menu()
    def run_umap_analysis(self):
        if not HAS_UMAP:
            QMessageBox.warning(self, "Warning", "UMAP is not installed!")
            return
        if not self.current_data:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
        df = self.current_data['dataframe']
        feat_cols = self.current_data['solvents'] + ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'pH']
        X = df[feat_cols].fillna(0).values
        umap2 = umap.UMAP(n_components=2, random_state=42)
        X_umap2 = umap2.fit_transform(X)
        umap3 = umap.UMAP(n_components=3, random_state=42)
        X_umap3 = umap3.fit_transform(X)
        self.analysis_result_label.setText("UMAP Results")
        self.analysis_result_table.setRowCount(0)
        self.analysis_result_table.setColumnCount(0)
        self.analysis_result_text.setText("UMAP 2D/3D results available in Visualizations tab.")
        if not hasattr(self, 'analysis_results'):
            self.analysis_results = {}
        self.analysis_results['UMAP'] = {'X_umap2': X_umap2, 'X_umap3': X_umap3, 'labels': df.get('Separation_Quality', None)}
        if not hasattr(self, 'plot_widget'):
            return
        if not hasattr(self.plot_widget, 'plot_data'):
            self.plot_widget.plot_data = {}
        self.plot_widget.plot_data['UMAP'] = self.analysis_results['UMAP']
        self.plot_widget.update_plot_menu()
    def run_qsrr_analysis(self):
        if not self.current_data:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
        df = self.current_data['dataframe']
        feat_cols = self.current_data['solvents'] + ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'pH']
        if 'Retention_Time_min' not in df.columns:
            QMessageBox.warning(self, "Warning", "No Retention_Time_min column in data!")
            return
        X = df[feat_cols].fillna(0).values
        y = df['Retention_Time_min'].values
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        coef = reg.coef_
        self.analysis_result_label.setText("QSRR Results")
        self.analysis_result_table.setRowCount(len(coef))
        self.analysis_result_table.setColumnCount(2)
        self.analysis_result_table.setHorizontalHeaderLabels(["Feature", "Coefficient"])
        for i, f in enumerate(feat_cols):
            self.analysis_result_table.setItem(i, 0, QTableWidgetItem(f))
            self.analysis_result_table.setItem(i, 1, QTableWidgetItem(f"{coef[i]:.4f}"))
        self.analysis_result_text.setText(f"QSRR R²: {r2:.3f}\nQSRR RMSE: {rmse:.3f}")
        if not hasattr(self, 'analysis_results'):
            self.analysis_results = {}
        self.analysis_results['QSRR'] = {'y_true': y, 'y_pred': y_pred, 'coef': coef, 'feat_cols': feat_cols, 'r2': r2, 'rmse': rmse}
        if not hasattr(self, 'plot_widget'):
            return
        if not hasattr(self.plot_widget, 'plot_data'):
            self.plot_widget.plot_data = {}
        self.plot_widget.plot_data['QSRR'] = self.analysis_results['QSRR']
        self.plot_widget.update_plot_menu()

    def show_help_app_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("HPLC Pipeline GUI - Help App")
        dlg.setWindowIcon(self._get_icon())
        dlg.setMinimumWidth(900)
        dlg.setMinimumHeight(600)
        layout = QVBoxLayout(dlg)
        from PySide6.QtWidgets import QTextBrowser, QPushButton, QHBoxLayout
        text = QTextBrowser()
        text.setOpenExternalLinks(True)
        text.setHtml(self.get_help_app_html())
        layout.addWidget(text)
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dlg.accept)
        btn_box.addStretch()
        btn_box.addWidget(ok_btn)
        layout.addLayout(btn_box)
        dlg.exec()


    def get_help_app_html(self):
        # This function returns a very detailed HTML help for the app, including theory, usage, and all menu/button explanations.
        return f'''
        <h1>HPLC Pipeline GUI - User Manual</h1>
        <hr>
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#intro">1. Introduction</a></li>
            <li><a href="#theory">2. Theoretical Background</a>
                <ul>
                    <li><a href="#hplc">2.1. HPLC Overview</a>
                        <ul>
                            <li><a href="#hplc_principle">2.1.1. Principle of HPLC</a></li>
                            <li><a href="#hplc_parameters">2.1.2. Key Parameters in HPLC</a></li>
                            <li><a href="#hplc_applications">2.1.3. Applications of HPLC</a></li>
                        </ul>
                    </li>
                    <li><a href="#qsrr">2.2. QSRR and Machine Learning Models</a>
                        <ul>
                            <li><a href="#qsrr_intro">2.2.1. What is QSRR?</a></li>
                            <li><a href="#qsrr_workflow">2.2.2. QSRR Workflow</a></li>
                            <li><a href="#qsrr_importance">2.2.3. Importance of QSRR</a></li>
                        </ul>
                    </li>
                    <li><a href="#regression">2.3. Regression Models</a>
                        <ul>
                            <li><a href="#regression_theory">2.3.1. Theory of Regression</a></li>
                            <li><a href="#regression_types">2.3.2. Types of Regression Models</a></li>
                            <li><a href="#regression_metrics">2.3.3. Regression Metrics</a></li>
                        </ul>
                    </li>
                    <li><a href="#classification">2.4. Classification Models</a>
                        <ul>
                            <li><a href="#classification_theory">2.4.1. Theory of Classification</a></li>
                            <li><a href="#classification_types">2.4.2. Types of Classification Models</a></li>
                            <li><a href="#classification_metrics">2.4.3. Classification Metrics</a></li>
                        </ul>
                    </li>
                    <li><a href="#descriptors">2.5. Molecular Descriptors</a>
                        <ul>
                            <li><a href="#desc_types">2.5.1. Types of Descriptors</a></li>
                            <li><a href="#desc_importance">2.5.2. Why Descriptors Matter</a></li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li><a href="#usage">3. Application Usage</a>
                <ul>
                    <li><a href="#mainwindow">3.1. Main Window Layout</a></li>
                    <li><a href="#dataloading">3.2. Data Loading</a></li>
                    <li><a href="#modeltraining">3.3. Model Training</a></li>
                    <li><a href="#prediction">3.4. Prediction</a></li>
                    <li><a href="#optimization">3.5. Optimization</a></li>
                    <li><a href="#analysis">3.6. Data Analysis</a></li>
                    <li><a href="#visualization">3.7. Visualization</a></li>
                    <li><a href="#saving">3.8. Saving and Loading</a></li>
                </ul>
            </li>
            <li><a href="#menus">4. Menu and Button Functions</a></li>
            <li><a href="#faq">5. FAQ & Troubleshooting</a></li>
        </ul>
        <hr>
        <h2 id="intro">1. Introduction</h2>
        <p>
            <b>HPLC Pipeline GUI</b> is a comprehensive, user-friendly software designed to facilitate chromatographic data analysis, machine learning model training, prediction, and optimization in High-Performance Liquid Chromatography (HPLC) workflows. It integrates data preprocessing, feature engineering, regression and classification modeling, and advanced visualization in a single graphical interface. This tool is suitable for both beginners and advanced users in analytical chemistry, pharmaceutical research, and related fields.
        </p>
        <hr>
        <h2 id="theory">2. Theoretical Background</h2>
        <h3 id="hplc">2.1. HPLC Overview</h3>
        <h4 id="hplc_principle">2.1.1. Principle of HPLC</h4>
        <p>
            High-Performance Liquid Chromatography (HPLC) is a powerful analytical technique used to separate, identify, and quantify components in a mixture. The separation is based on the differential interactions of analytes with the stationary phase (column) and the mobile phase (solvent).<br>
            <b>Basic Principle:</b> Compounds in a sample are injected into a stream of mobile phase and passed through a column packed with stationary phase. Each compound interacts differently with the stationary and mobile phases, resulting in different retention times (<i>t<sub>R</sub></i>), which is the time taken for a compound to elute from the column.<br>
            <b>Diagram:</b><br>
            <img src="data:image/png;base64,{HPLC}" alt="HPLC Diagram" width="800"><br>
            <i>Figure: Schematic of an HPLC system (Injector, Pump, Column, Detector)</i>
        </p>
        <h4 id="hplc_parameters">2.1.2. Key Parameters in HPLC</h4>
        <ul>
            <li><b>Retention Time (<i>t<sub>R</sub></i>):</b> The time a compound takes to pass through the column to the detector. It is a key parameter for identification.</li>
            <li><b>Mobile Phase Composition:</b> The ratio of buffer (aqueous) and organic solvent (e.g., acetonitrile, methanol). This affects analyte polarity and separation.</li>
            <li><b>pH:</b> The pH of the mobile phase can influence the ionization state of analytes, affecting retention and separation.</li>
            <li><b>Flow Rate:</b> The speed at which the mobile phase moves through the column.</li>
            <li><b>Column Temperature:</b> Can affect analyte interaction and separation efficiency.</li>
        </ul>
        <h4 id="hplc_applications">2.1.3. Applications of HPLC</h4>
        <ul>
            <li>Pharmaceutical analysis (drug purity, content uniformity)</li>
            <li>Environmental monitoring (pollutant analysis)</li>
            <li>Food and beverage quality control</li>
            <li>Biochemical and clinical research</li>
        </ul>
        <h3 id="qsrr">2.2. QSRR and Machine Learning Models</h3>
        <h4 id="qsrr_intro">2.2.1. What is QSRR?</h4>
        <p>
            <b>Quantitative Structure-Retention Relationship (QSRR)</b> is a modeling approach that relates the chemical structure of compounds (encoded as molecular descriptors) to their chromatographic retention times. The goal is to predict retention time or separation quality for new compounds based on their structure and experimental conditions.
        </p>
        <h4 id="qsrr_workflow">2.2.2. QSRR Workflow</h4>
        <ol>
            <li><b>Data Collection:</b> Gather experimental retention times and conditions for a set of compounds.</li>
            <li><b>Descriptor Calculation:</b> Compute molecular descriptors (e.g., MolWt, LogP, TPSA) from SMILES using cheminformatics tools like RDKit.</li>
            <li><b>Feature Engineering:</b> Combine descriptors with experimental variables (solvent %, pH, etc.).</li>
            <li><b>Model Training:</b> Use regression or classification algorithms to learn the relationship between features and retention/separation.</li>
            <li><b>Validation:</b> Evaluate model performance using metrics (RMSE, R², accuracy, etc.).</li>
            <li><b>Prediction:</b> Apply the model to predict retention/separation for new compounds or conditions.</li>
        </ol>
        <h4 id="qsrr_importance">2.2.3. Importance of QSRR</h4>
        <ul>
            <li>Reduces the need for extensive experimental work by enabling <b>in silico</b> predictions.</li>
            <li>Helps in method development and optimization.</li>
            <li>Facilitates understanding of how molecular structure and conditions affect chromatographic behavior.</li>
        </ul>
        <h3 id="regression">2.3. Regression Models</h3>
        <h4 id="regression_theory">2.3.1. Theory of Regression</h4>
        <p>
            Regression is a supervised machine learning technique used to predict a continuous outcome variable (e.g., retention time) from one or more input features (descriptors, solvent %, pH, etc.). The general form is:<br>
            <code>y = f(X) + ε</code><br>
            where <code>y</code> is the target, <code>X</code> is the feature vector, <code>f</code> is the model, and <code>ε</code> is the error term.<br>
            <b>Example:</b> Predicting retention time from molecular weight, LogP, and solvent composition.
        </p>
        <h4 id="regression_types">2.3.2. Types of Regression Models</h4>
        <ul>
            <li><b>Linear Regression (LR):</b> Assumes a linear relationship between features and target.<br>
                <code>y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ</code><br>
                <b>Objective:</b> Minimize the sum of squared errors:<br>
                <code>min Σ(yᵢ - Xᵢw)²</code>
            </li>
            <li><b>Ridge Regression:</b> Adds L2 regularization to penalize large coefficients.<br>
                <code>min Σ(yᵢ - Xᵢw)² + α||w||²</code>
            </li>
            <li><b>Lasso Regression:</b> Adds L1 regularization for feature selection.<br>
                <code>min Σ(yᵢ - Xᵢw)² + α||w||₁</code>
            </li>
            <li><b>Random Forest Regressor (RF):</b> An ensemble of decision trees. Each tree predicts a value, and the final prediction is the average.<br>
                <b>Advantage:</b> Handles non-linear relationships and feature interactions.<br>
                <b>Formula:</b> <code>ŷ = (1/T) Σfₜ(X)</code> where <code>fₜ</code> is the t-th tree.</li>
            <li><b>Gradient Boosting Regressor (GB):</b> Sequentially builds trees to correct errors of previous trees.<br>
                <b>Formula:</b> <code>ŷ = Σγₘhₘ(X)</code> where <code>hₘ</code> is the m-th weak learner.</li>
            <li><b>Support Vector Regression (SVR):</b> Finds a function within a margin of tolerance (epsilon-insensitive loss).<br>
                <b>Objective:</b> Minimize:<br>
                <code>(1/2)||w||² + C Σξᵢ</code> subject to <code>|yᵢ - f(Xᵢ)| ≤ ε + ξᵢ</code></li>
            <li><b>MLP Regressor:</b> Multi-layer perceptron (neural network) for regression.<br>
                <b>Formula:</b> <code>y = σ(W₂ · σ(W₁X + b₁) + b₂)</code> where <code>σ</code> is an activation function.</li>
            <li><b>KNN/DT:</b> K-Nearest Neighbors and Decision Tree regressors for non-parametric regression.</li>
        </ul>
        <h4 id="regression_metrics">2.3.3. Regression Metrics</h4>
        <ul>
            <li><b>Root Mean Squared Error (RMSE):</b> <code>RMSE = √[(1/N) Σ(yᵢ - ŷᵢ)²]</code></li>
            <li><b>Mean Absolute Error (MAE):</b> <code>MAE = (1/N) Σ|yᵢ - ŷᵢ|</code></li>
            <li><b>R² (Coefficient of Determination):</b> <code>R² = 1 - [Σ(yᵢ - ŷᵢ)²]/[Σ(yᵢ - ȳ)²]</code></li>
        </ul>
        <h3 id="classification">2.4. Classification Models</h3>
        <h4 id="classification_theory">2.4.1. Theory of Classification</h4>
        <p>
            Classification is a supervised learning task where the goal is to assign input data to one of several predefined categories (classes). In this app, classification is used to predict <b>separation quality</b> (e.g., 'Completely separated', 'Overlapping', etc.) based on features.<br>
            <b>Example:</b> Predicting whether a chromatographic condition will result in good or poor separation.
        </p>
        <h4 id="classification_types">2.4.2. Types of Classification Models</h4>
        <ul>
            <li><b>Random Forest Classifier (RFC):</b> Ensemble of decision trees, each voting for a class. The majority vote is the prediction.<br>
                <b>Formula:</b> <code>ŷ = mode(f₁(X), f₂(X), ..., fₜ(X))</code></li>
            <li><b>Gradient Boosting Classifier (GBC):</b> Sequentially builds trees to improve classification accuracy.</li>
            <li><b>Support Vector Classifier (SVC):</b> Finds the hyperplane that best separates classes in feature space.<br>
                <b>Objective:</b> Maximize the margin between classes.</li>
            <li><b>MLP Classifier:</b> Neural network for classification tasks.</li>
            <li><b>KNN/DT/LR:</b> K-Nearest Neighbors, Decision Tree, and Logistic Regression classifiers.</li>
        </ul>
        <h4 id="classification_metrics">2.4.3. Classification Metrics</h4>
        <ul>
            <li><b>Accuracy:</b> <code>(Number of correct predictions)/(Total predictions)</code></li>
            <li><b>Precision:</b> <code>TP/(TP + FP)</code> (True Positives / (True Positives + False Positives))</li>
            <li><b>Recall:</b> <code>TP/(TP + FN)</code> (True Positives / (True Positives + False Negatives))</li>
            <li><b>F1-score:</b> Harmonic mean of precision and recall.</li>
            <li><b>Confusion Matrix:</b> Table showing true vs. predicted classes.</li>
            <li><b>ROC Curve:</b> Plots True Positive Rate vs. False Positive Rate.</li>
            <li><b>PR Curve:</b> Plots Precision vs. Recall.</li>
        </ul>
        <h3 id="descriptors">2.5. Molecular Descriptors</h3>
        <h4 id="desc_types">2.5.1. Types of Descriptors</h4>
        <ul>
            <li><b>MolWt:</b> Molecular Weight. Sum of atomic weights.</li>
            <li><b>LogP:</b> Octanol-water partition coefficient. Indicates hydrophobicity.</li>
            <li><b>TPSA:</b> Topological Polar Surface Area. Related to polarity and hydrogen bonding.</li>
            <li><b>HBD:</b> Number of Hydrogen Bond Donors.</li>
            <li><b>HBA:</b> Number of Hydrogen Bond Acceptors.</li>
        </ul>
        <h4 id="desc_importance">2.5.2. Why Descriptors Matter</h4>
        <ul>
            <li>Descriptors encode chemical structure into numerical values usable by machine learning models.</li>
            <li>They capture properties like size, polarity, and hydrogen bonding, which influence chromatographic behavior.</li>
            <li>Proper selection of descriptors improves model accuracy and interpretability.</li>
        </ul>
        <hr>
        <h2 id="usage">3. Application Usage</h2>
        <h3 id="mainwindow">3.1. Main Window Layout</h3>
        <ul>
            <li><b>Left Panel:</b> Contains controls for data loading, model management, prediction, progress, and log. Each section is grouped for clarity.</li>
            <li><b>Right Panel (Tabs):</b> Contains multiple tabs:
                <ul>
                    <li><b>Data View:</b> Shows the loaded data table.</li>
                    <li><b>Model Results:</b> Shows model performance metrics.</li>
                    <li><b>Visualizations:</b> Shows plots for model comparison, metrics, and analysis.</li>
                    <li><b>Model Details:</b> Shows detailed model info, feature importance, and suggestions.</li>
                    <li><b>Optimization:</b> Shows heatmap, suggestions, and best conditions.</li>
                    <li><b>Analysis:</b> Run PCA, clustering, t-SNE, UMAP, and QSRR.</li>
                </ul>
            </li>
        </ul>
        <h3 id="dataloading">3.2. Data Loading</h3>
        <ol>
            <li>Click <b>Browse File</b> to select a CSV or Excel file containing HPLC data. The file should include columns for SMILES, retention time, separation quality, solvent %, and pH.</li>
            <li>Click <b>Load Data</b> to import and process the data. The app will compute molecular descriptors and encode features automatically.</li>
            <li>Loaded data appears in the <b>Data View</b> tab. Only the first 1000 rows are shown for large datasets.</li>
        </ol>
        <h3 id="modeltraining">3.3. Model Training</h3>
        <ol>
            <li>Click <b>Model &gt; Select Models...</b> to choose which regression and classification models to train. You can also set custom hyperparameters for each model.</li>
            <li>Click <b>Model &gt; Train Models</b> to start training. Progress is shown in the progress bar and log. The app will train all selected models using cross-validation and grid search.</li>
            <li>After training, results are shown in the <b>Model Results</b> tab, and plots are available in <b>Visualizations</b>. The best models are saved for later use.</li>
        </ol>
        <h3 id="prediction">3.4. Prediction</h3>
        <ol>
            <li>Enter a SMILES string in the <b>Prediction</b> section.</li>
            <li>Select a regression and classification model from the dropdowns.</li>
            <li>Click <b>Predict</b> to see predicted retention time and separation quality. The app computes descriptors for the input SMILES and uses the selected models for prediction.</li>
        </ol>
        <h3 id="optimization">3.5. Optimization</h3>
        <ol>
            <li>Click <b>Suggest Optimal Condition</b> to get the best buffer/organic/pH combination for the input SMILES, maximizing separation quality. The app evaluates all possible combinations in the training data and suggests the best one based on model predictions.</li>
            <li>Optimization results and heatmaps are shown in the <b>Optimization</b> tab, including a table of top combinations and feature importance ranking.</li>
        </ol>
        <h3 id="analysis">3.6. Data Analysis</h3>
        <ol>
            <li>Use the <b>Analysis</b> tab to run advanced analyses:
                <ul>
                    <li><b>PCA (Principal Component Analysis):</b> Reduces dimensionality and visualizes data structure.</li>
                    <li><b>Hierarchical Clustering:</b> Groups similar samples and shows dendrograms.</li>
                    <li><b>t-SNE/UMAP:</b> Nonlinear dimensionality reduction for visualization.</li>
                    <li><b>QSRR Analysis:</b> Linear regression of retention time vs. features, with coefficients and metrics.</li>
                </ul>
            </li>
            <li>Results are shown in tables and can be visualized in the <b>Visualizations</b> tab.</li>
        </ol>
        <h3 id="visualization">3.7. Visualization</h3>
        <ul>
            <li>Plots include model comparison (RMSE, R², accuracy), regression/classification metrics, PCA/t-SNE/UMAP, clustering dendrograms, and QSRR.</li>
            <li>Use <b>Save Image</b> to export plots as PNG, and <b>View Fullscreen</b> for detailed viewing.</li>
            <li>Customize plots with legend and grid options.</li>
        </ul>
        <h3 id="saving">3.8. Saving and Loading</h3>
        <ul>
            <li><b>Save Results:</b> Save model results, details, and plots to files for documentation or further analysis.</li>
            <li><b>Load Saved Models:</b> Load previously trained models for prediction without retraining.</li>
            <li><b>Load Training Data:</b> Restore a previous training session from a .pkl file, including data, models, and optimization context.</li>
        </ul>
        <hr>
        <h2 id="menus">4. Menu and Button Functions</h2>
        <h3>File Menu</h3>
        <ul>
            <li><b>Load Data File:</b> Open file dialog to select data file.</li>
            <li><b>Save Results:</b> Save current results and plots.</li>
            <li><b>Exit:</b> Close the application.</li>
        </ul>
        <h3>Model Menu</h3>
        <ul>
            <li><b>Select Models...</b> Choose which regression/classification models to train and set hyperparameters.</li>
            <li><b>Set Output Directory...</b> Choose where to save outputs.</li>
            <li><b>Train Models:</b> Start model training.</li>
            <li><b>Load Training Data...</b> Load a previous training context (.pkl).</li>
        </ul>
        <h3>Help Menu</h3>
        <ul>
            <li><b>Help App:</b> Show this detailed help dialog.</li>
            <li><b>About:</b> Show app and author information.</li>
        </ul>
        <h3>Left Panel Buttons</h3>
        <ul>
            <li><b>Browse File:</b> Select data file.</li>
            <li><b>Load Data:</b> Load and process data.</li>
            <li><b>Load Saved Models:</b> Load models from directory.</li>
            <li><b>Save Results:</b> Save results and plots.</li>
            <li><b>Predict:</b> Predict retention time and separation quality for input SMILES.</li>
            <li><b>Suggest Optimal Condition:</b> Find best buffer/organic/pH for input SMILES.</li>
            <li><b>Clear Log:</b> Clear the log area.</li>
        </ul>
        <h3>Tabs</h3>
        <ul>
            <li><b>Data View:</b> Shows loaded data table.</li>
            <li><b>Model Results:</b> Shows model performance metrics.</li>
            <li><b>Visualizations:</b> Shows plots for model comparison, metrics, and analysis.</li>
            <li><b>Model Details:</b> Shows detailed model info, feature importance, and suggestions.</li>
            <li><b>Optimization:</b> Shows heatmap, suggestions, and best conditions.</li>
            <li><b>Analysis:</b> Run PCA, clustering, t-SNE, UMAP, and QSRR.</li>
        </ul>
        <h3>Plot Controls</h3>
        <ul>
            <li><b>Save Image:</b> Save current plot as PNG.</li>
            <li><b>View Fullscreen:</b> Show plot in fullscreen dialog.</li>
            <li><b>Customize Plot:</b> Toggle legend/grid.</li>
        </ul>
        <hr>
        <h2 id="faq">5. FAQ & Troubleshooting</h2>
        <ul>
            <li><b>Q: The app crashes or freezes?</b><br>A: Check your data format, ensure all dependencies are installed, and avoid very large files.</li>
            <li><b>Q: Model training is slow?</b><br>A: Reduce the number of models or tune hyperparameters for faster training.</li>
            <li><b>Q: How to interpret feature importance?</b><br>A: Features with higher importance have more influence on the model's prediction.</li>
            <li><b>Q: What is the best separation quality?</b><br>A: 'Completely separated' is the best class. The app prioritizes this in optimization.</li>
            <li><b>Q: How do I prepare my data file?</b><br>A: Ensure your file has columns for SMILES, retention time, separation quality, solvent %, and pH. Use .csv or .xlsx format.</li>
            <li><b>Q: Can I use the app for other types of chromatography?</b><br>A: The app is optimized for HPLC but can be adapted for similar workflows with appropriate data.</li>
        </ul>
        <hr>
        <p style="font-size:10pt;color:gray;">For more information, contact Arif Maulana Azis | Titan Digitalsoft | titandigitalsoft@gmail.com</p>
        <p style="font-size:10pt;color:gray;">Version 1.0.0</p>
        <p style="font-size:10pt;color:gray;">Copyright 2025 Titan Digitalsoft</p>
        '''


def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("HPLC Pipeline GUI V1.0.0 | Titan Digitalsoft")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Titan Digitalsoft")
    
    # Create and show main window
    window = HPLCPipelineGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()