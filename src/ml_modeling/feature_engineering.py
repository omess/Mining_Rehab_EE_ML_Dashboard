"""
Mining Rehabilitation & Environmental Compliance Verification Project
Phase 3: Exploratory Data Analysis & Trajectory Modeling Engine
Author: Senior Geospatial Data Scientist Portfolio Blueprint
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RehabTrajectoryAnalyzer:
    def __init__(self, csv_path: str, output_dir: str):
        """Initializes the trajectory analysis engine."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing required zonal metrics CSV at: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean up column names if GEE exported them with system properties
        self.df.columns = [col.replace('system:index', '').strip('_') for col in self.df.columns]
        self.df = self.df.sort_values('year').reset_index(drop=True)

    def engineer_rehab_metrics(self) -> pd.DataFrame:
        """
        Calculates rolling baselines, structural divergence, 
        and the standardized Rehabilitation Index (RI).
        """
        print("[-] Engineering advanced temporal ecosystem features...")
        
        # Earth Engine might export columns as mine_NDVI or mine_ndvi. Let's make it case-insensitive.
        col_mapping = {col.lower(): col for col in self.df.columns}
        
        mine_ndvi_col = col_mapping.get('mine_ndvi')
        ref_ndvi_col = col_mapping.get('ref_ndvi')
        
        if not mine_ndvi_col or not ref_ndvi_col:
            raise KeyError(f"Could not find NDVI columns. Available columns: {list(self.df.columns)}")
        
        # 1. Identify the historical baseline minimum (maximum degradation point)
        min_mine_ndvi = self.df[mine_ndvi_col].min()
        
        # 2. Calculate the Rehabilitation Index (RI)
        self.df['rehab_index'] = (self.df[mine_ndvi_col] - min_mine_ndvi) / (self.df[ref_ndvi_col] - min_mine_ndvi)
        
        # 3. Calculate 3-year rolling velocity (the speed of change)
        self.df['ndvi_velocity'] = self.df[mine_ndvi_col].diff(periods=3) / 3
        
        # 4. Calculate divergence gap between mine and reference matrix
        self.df['ecosystem_divergence'] = self.df[ref_ndvi_col] - self.df[mine_ndvi_col]
        
        # Fill NaN values resulting from rolling windows with 0
        self.df = self.df.fillna(0)
        
        output_csv = os.path.join(self.output_dir, "engineered_rehab_features.csv")
        self.df.to_csv(output_csv, index=False)
        print(f"[+] Feature engineering matrix cached at: {output_csv}")
        return self.df

    def generate_portfolio_plots(self):
        """Generates industry-grade evaluation figures for portfolio display."""
        print("[-] Rendering time-series trajectory visualizations...")
        sns.set_theme(style="whitegrid")
        
        col_mapping = {col.lower(): col for col in self.df.columns}
        mine_ndvi_col = col_mapping.get('mine_ndvi')
        ref_ndvi_col = col_mapping.get('ref_ndvi')
        
        # Figure 1: Long-term NDVI comparison and divergence gap
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax1.plot(self.df['year'], self.df[ref_ndvi_col], color='green', marker='o', linewidth=2, label='Kakadu Reference Baseline')
        ax1.plot(self.df['year'], self.df[mine_ndvi_col], color='crimson', marker='s', linewidth=2, label='Ranger Mine Footprint')
        ax1.set_ylabel('NDVI Value', fontsize=12)
        ax1.set_title('Ranger Uranium Mine vs. Kakadu National Park: Canopy Trajectory', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower left')
        
        # Highlight the post-closure rehabilitation window (2021+)
        ax1.axvspan(2021, 2026, color='yellow', alpha=0.2, label='Closure & Progressive Rehab Phase')
        
        ax2.fill_between(self.df['year'], self.df['ecosystem_divergence'], color='purple', alpha=0.3, label='Ecosystem Divergence Gap')
        ax2.plot(self.df['year'], self.df['ecosystem_divergence'], color='purple', linewidth=1.5)
        ax2.set_ylabel('Divergence Delta', fontsize=12)
        ax2.set_xlabel('Operational Year', fontsize=12)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        fig1_path = os.path.join(self.output_dir, "rehab_trajectory_analysis.png")
        plt.savefig(fig1_path, dpi=300)
        plt.close()
        
        print(f"[+] Publication-quality visualization exported to: {fig1_path}")

if __name__ == "__main__":
    # Corrected relative paths to find data from the project root folder
    csv_input = os.path.join("data", "processed", "Ranger_Mine_Zonal_Rehab_Metrics_1995_2026.csv")
    processed_dir = os.path.join("data", "processed")
    
    analyzer = RehabTrajectoryAnalyzer(csv_path=csv_input, output_dir=processed_dir)
    analyzer.engineer_rehab_metrics()
    analyzer.generate_portfolio_plots()
    print("[*] Phase 3 Analysis Pipeline Complete.")