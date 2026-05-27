"""
Mining Rehabilitation & Environmental Compliance Verification Project
Phase 6: Machine Learning Training & Evaluation Pipeline - Realistic Milestones
Author: Senior Geospatial Data Scientist Portfolio Blueprint
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class RehabMLModelPipeline:
    def __init__(self, data_path: str, output_dir: str):
        """Initializes the ML training pipeline."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing engineered dataset at: {data_path}")
        
        self.df = pd.read_csv(data_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def prepare_data(self):
        print("[-] Preparing dataset for Machine Learning modeling...")
        
        # Look for case variations in columns
        col_mapping = {col.lower(): col for col in self.df.columns}
        mine_ndvi_col = col_mapping.get('mine_ndvi', 'mine_NDVI')
        mine_evi_col = col_mapping.get('mine_evi', 'mine_EVI')
        
        # ADJUSTED ECO-BENCHMARK: 0.40 represents Advanced Progressive Recovery Milestone
        self.df['target'] = (self.df['rehab_index'] >= 0.40).astype(int)
        
        print(f"[*] Target Distribution: {self.df['target'].value_counts().to_dict()}")
        
        self.feature_cols = [mine_ndvi_col, mine_evi_col, 'rehab_index', 'ndvi_velocity', 'ecosystem_divergence']
        X = self.df[self.feature_cols]
        y = self.df['target']
        
        if len(np.unique(y)) > 1:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"[+] Data split complete. Training rows: {self.X_train.shape[0]}, Testing rows: {self.X_test.shape[0]}")

    def train_and_evaluate(self):
        print("[-] Training Random Forest Classifier Ensemble...")
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        
        print("\n================== ML MODEL PERFORMANCE PERFORMANCE ==================")
        print(f"Accuracy Score: {acc * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        print("======================================================================\n")
        
        self._plot_feature_importances()

    def _plot_feature_importances(self):
        """Plots and saves feature importance matrix using explicit object-oriented canvas drawing."""
        importances = self.model.feature_importances_
        
        feat_df = pd.DataFrame({'Feature': self.feature_cols, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False)
        
        # Check if importances are all zero to print an early warning
        if feat_df['Importance'].sum() == 0:
            print("[⚠️ WARNING] All feature importances are 0.0! Bars will not appear on the plot.")
        
        plt.clf() 
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw the bar chart explicitly on the allocated axis
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=feat_df, 
            ax=ax, 
            hue='Feature', 
            palette='viridis', 
            legend=False
        )
        
        ax.set_title('Feature Importance Matrix: What Drives Environmental Verification?', fontsize=14, fontweight='bold')
        ax.set_xlabel('Statistical Importance Score', fontsize=11)
        ax.set_ylabel('Engineered Remote Sensing Feature', fontsize=11)
        
        plt.tight_layout()
        
        fig_out = os.path.join(self.output_dir, "ml_feature_importances.png")
        fig.savefig(fig_out, dpi=300, bbox_inches='tight')
        plt.close(fig) 
        print(f"[+] Feature importance evaluation graph saved successfully at: {fig_out}")
        
        # Safe probability assignment
        raw_probs = self.model.predict_proba(self.df[self.feature_cols])
        if raw_probs.shape[1] == 2:
            probabilities = raw_probs[:, 1]
        else:
            single_class = self.model.classes_[0]
            probabilities = np.ones(raw_probs.shape[0]) if single_class == 1 else np.zeros(raw_probs.shape[0])
        
        self.df['ml_prediction'] = self.model.predict(self.df[self.feature_cols])
        self.df['rehab_probability_score'] = (probabilities * 100).round(2)
        
        final_output_path = os.path.join(self.output_dir, "final_dashboard_metrics.csv")
        self.df.to_csv(final_output_path, index=False)
        print(f"[+] Master final dataset with ML predictions compiled at: {final_output_path}")

if __name__ == "__main__":
    dataset_input = os.path.join("data", "processed", "engineered_rehab_features.csv")
    processed_dir = os.path.join("data", "processed")
    
    pipeline = RehabMLModelPipeline(data_path=dataset_input, output_dir=processed_dir)
    pipeline.prepare_data()
    pipeline.train_and_evaluate()
    print("[*] Phase 6 Machine Learning Pipeline Complete.")