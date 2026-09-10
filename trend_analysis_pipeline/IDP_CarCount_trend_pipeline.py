import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import re
import geopandas as gpd
import os
import argparse
import sys


class TrendAnalyzer:
    """
    A class to perform linear regression trend analysis on car count data.
    """

    def __init__(self, indx, city_name, periods, car_counts, baseline, norm_car_counts,
                 idp=None, population_baseline=None, location_level=None):
        """
        Initialize the analyzer with data.

        Parameters:
        -----------
        indx : int
            Index identifier for the analysis
        city_name : str
            Name of the city being analyzed
        periods : list
            List of time period labels (e.g., ['Apr2012', 'May2012', ...])
        car_counts : list
            List of car counts corresponding to each period
        baseline : float or int
            Baseline car count for comparison
        norm_car_counts : list
            Normalized car counts
        idp : float, optional
            IDP (Internally Displaced Persons) value for this report
        population_baseline : float, optional
            Population baseline (admin1 level) for normalizing IDP
        location_level : str, optional
            Level of the report: 'city' or 'governorate' (from input data)
        """
        self.indx = indx
        self.city_name = city_name
        self.periods = periods
        self.car_counts = car_counts
        self.baseline = baseline
        self.norm_car_counts = norm_car_counts
        self.idp = idp if idp is not None else float('nan')
        self.location_level = location_level
        self.population_baseline = population_baseline if population_baseline is not None else float('nan')
        self._calculate_idp_normalized()

        valid_counts = [c for c in car_counts if c == c and c is not None]
        if len(valid_counts) < 2:
            raise ValueError(f"Insufficient valid data points for {city_name}. "
                f"Need at least 2 valid car counts, got {len(valid_counts)}. "
                f"This report should be filtered before creating TrendAnalyzer.")

        self.time_indices = calculate_time_indices(periods)

        self.df = pd.DataFrame({
            'indx': [indx] * len(periods),
            'Period': periods,
            'Car_Count': car_counts,
            'Norm_Car_Count': norm_car_counts,
            'Time_Index': self.time_indices
        })

        self.df['Relative_Change_Percent'] = ((self.df['Car_Count'] - baseline) / baseline) * 100

        self._perform_regression()

    def _calculate_idp_normalized(self):
        """Calculate normalized IDP using the admin1 population baseline."""
        if (self.idp == self.idp and
            self.population_baseline == self.population_baseline and
            self.population_baseline != 0):
            self.idp_norm = self.idp / self.population_baseline
        else:
            self.idp_norm = float('nan')

    def _perform_regression(self):
        """Perform linear regression analysis."""
        valid_mask = self.df['Car_Count'].notna() & self.df['Norm_Car_Count'].notna()
        df_valid = self.df[valid_mask].copy()
        self.n_valid_obs = len(df_valid)

        if len(df_valid) < 2:
            raise ValueError(f"Insufficient valid data points for regression in {self.city_name}. "
                f"After removing nan values, only {len(df_valid)} points remain. Need at least 3.")

        self.slope, self.intercept, self.r_value, self.p_value, self.std_err = stats.linregress(df_valid['Time_Index'], df_valid['Car_Count'])

        self.df['Predicted'] = self.slope * self.df['Time_Index'] + self.intercept
        self.df['Residual'] = self.df['Car_Count'] - self.df['Predicted']

        self.norm_slope, self.norm_intercept, norm_r, self.norm_p_value, self.norm_std_err = \
            stats.linregress(df_valid['Time_Index'], df_valid['Norm_Car_Count'])
        self.norm_r_squared = norm_r ** 2
        self.df['Norm_Predicted'] = self.norm_slope * self.df['Time_Index'] + self.norm_intercept

        self.df['Norm_Residual'] = self.df['Norm_Car_Count'] - self.df['Norm_Predicted']

        self.abs_slope_as_norm = self.slope / self.baseline
        self.slope_scale_diff = abs(self.abs_slope_as_norm - self.norm_slope)
        self.slope_scale_diff_pct = (self.slope_scale_diff / abs(self.norm_slope) * 100) if self.norm_slope != 0 else 0
        self.r2_diff = self.r_value**2 - self.norm_r_squared
        self.pval_diff = self.p_value - self.norm_p_value

        valid_data = self.df[self.df['Car_Count'].notna()].copy()

        if len(valid_data) >= 2:
            first_valid_idx = valid_data.index[0]
            last_valid_idx = valid_data.index[-1]

            first_time_idx = valid_data.loc[first_valid_idx, 'Time_Index']
            last_time_idx = valid_data.loc[last_valid_idx, 'Time_Index']

            original_first_idx = 0
            original_last_idx = len(self.df) - 1
            original_first_time_idx = self.df.loc[0, 'Time_Index']
            original_last_time_idx = self.df.loc[len(self.df)-1, 'Time_Index']
            
            # This is the C_LF ("last minus first") quantity.
            self.obs_diff_abs = valid_data.loc[last_valid_idx, 'Car_Count'] - valid_data.loc[first_valid_idx, 'Car_Count']
            self.obs_diff_abs_pct = self.obs_diff_abs / valid_data.loc[first_valid_idx, 'Car_Count'] * 100
            
            self.obs_diff_norm = valid_data.loc[last_valid_idx, 'Norm_Car_Count'] - valid_data.loc[first_valid_idx, 'Norm_Car_Count']
            self.obs_diff_norm_pct = self.obs_diff_norm / valid_data.loc[first_valid_idx, 'Norm_Car_Count'] * 100
            
            # This is the C_LR (regression-based) quantity.
            self.est_diff_abs = (self.slope * original_last_time_idx + self.intercept) - \
                                (self.slope * original_first_time_idx + self.intercept)
            self.est_diff_abs_pct = self.est_diff_abs / (self.slope * original_first_time_idx + self.intercept) * 100
            
            self.est_diff_norm = (self.norm_slope * original_last_time_idx + self.norm_intercept) - \
                                 (self.norm_slope * original_first_time_idx + self.norm_intercept)
            self.est_diff_norm_pct = self.est_diff_norm / (self.norm_slope * original_first_time_idx + self.norm_intercept) * 100

            self.obs_start_period = valid_data.loc[first_valid_idx, 'Period']
            self.obs_end_period = valid_data.loc[last_valid_idx, 'Period']
        else:
            self.obs_diff_abs = self.obs_diff_abs_pct = float('nan')
            self.obs_diff_norm = self.obs_diff_norm_pct = float('nan')
            self.est_diff_abs = self.est_diff_abs_pct = float('nan')
            self.est_diff_norm = self.est_diff_norm_pct = float('nan')
            self.obs_start_period = None
            self.obs_end_period = None

        # Actual vs Predicted correlation -- absolute (Pearson)
        valid_for_corr = self.df['Car_Count'].notna() & self.df['Predicted'].notna()
        if valid_for_corr.sum() >= 2:
            self.fit_pearson_abs, self.fit_pearson_abs_p = stats.pearsonr(
                self.df.loc[valid_for_corr, 'Car_Count'],
                self.df.loc[valid_for_corr, 'Predicted'])
        else:
            self.fit_pearson_abs = float('nan')
            self.fit_pearson_abs_p = float('nan')

        # Actual vs Predicted correlation -- normalized (Pearson)
        valid_for_corr_norm = self.df['Norm_Car_Count'].notna() & self.df['Norm_Predicted'].notna()
        if valid_for_corr_norm.sum() >= 2:
            self.fit_pearson_norm, self.fit_pearson_norm_p = stats.pearsonr(
                self.df.loc[valid_for_corr_norm, 'Norm_Car_Count'],
                self.df.loc[valid_for_corr_norm, 'Norm_Predicted'])
        else:
            self.fit_pearson_norm = float('nan')
            self.fit_pearson_norm_p = float('nan')

        valid_trend = self.df['Car_Count'].notna()
        if valid_trend.sum() >= 2:
            self.pearson_corr, self.pearson_pvalue = stats.pearsonr(
                self.df.loc[valid_trend, 'Time_Index'],
                self.df.loc[valid_trend, 'Car_Count'])
        else:
            self.pearson_corr = float('nan')
            self.pearson_pvalue = float('nan')

        # Max-Min difference (C_MM): positive if increasing (max after min),
        # negative if decreasing (max before min)
        valid_data = self.df[self.df['Car_Count'].notna()].copy()
        if len(valid_data) >= 2:
            max_idx = valid_data['Car_Count'].idxmax()
            min_idx = valid_data['Car_Count'].idxmin()
            max_val = valid_data.loc[max_idx, 'Car_Count']
            min_val = valid_data.loc[min_idx, 'Car_Count']
            max_time = valid_data.loc[max_idx, 'Time_Index']
            min_time = valid_data.loc[min_idx, 'Time_Index']

            if max_time > min_time:
                self.max_diff = max_val - min_val
            else:
                self.max_diff = -(max_val - min_val)

            self.max_diff_abs = abs(max_val - min_val)
            self.max_period = valid_data.loc[max_idx, 'Period']
            self.min_period = valid_data.loc[min_idx, 'Period']
            self.max_time_index = max_time
            self.min_time_index = min_time

            valid_norm = self.df[self.df['Norm_Car_Count'].notna()].copy()
            if len(valid_norm) >= 2:
                max_idx_norm = valid_norm['Norm_Car_Count'].idxmax()
                min_idx_norm = valid_norm['Norm_Car_Count'].idxmin()
                max_val_norm = valid_norm.loc[max_idx_norm, 'Norm_Car_Count']
                min_val_norm = valid_norm.loc[min_idx_norm, 'Norm_Car_Count']
                max_time_norm = valid_norm.loc[max_idx_norm, 'Time_Index']
                min_time_norm = valid_norm.loc[min_idx_norm, 'Time_Index']

                if max_time_norm > min_time_norm:
                    self.max_diff_norm = max_val_norm - min_val_norm
                else:
                    self.max_diff_norm = -(max_val_norm - min_val_norm)

                self.max_diff_norm_abs = abs(max_val_norm - min_val_norm)
            else:
                self.max_diff_norm = float('nan')
                self.max_diff_norm_abs = float('nan')
        else:
            self.max_diff = float('nan')
            self.max_diff_abs = float('nan')
            self.max_diff_norm = float('nan')
            self.max_diff_norm_abs = float('nan')
            self.max_period = None
            self.min_period = None
            self.max_time_index = None
            self.min_time_index = None

        if self.slope < 0:
            self.trend_direction = "DOWNWARD"
            self.trend_desc = "decreasing"
        elif self.slope > 0:
            self.trend_direction = "UPWARD"
            self.trend_desc = "increasing"
        else:
            self.trend_direction = "FLAT"
            self.trend_desc = "stable"

    def get_statistical_summary(self):
        """Return a dictionary of statistical results."""
        return {
            'indx': self.indx,
            'city': self.city_name,
            'location_level': self.location_level,
            'n_periods': len(self.df),
            'first_period': self.periods[0],
            'last_period': self.periods[-1],
            'baseline': self.baseline,
            'n_valid_obs': self.n_valid_obs,
            'slope': self.slope,
            'intercept': self.intercept,
            'r_value': self.r_value,
            'r_squared': self.r_value ** 2,
            'p_value': self.p_value,
            'std_error': self.std_err,
            'trend_direction': self.trend_direction,
            'is_significant': self.p_value < 0.05,
            'first_count': self.df['Car_Count'].iloc[0],
            'last_count': self.df['Car_Count'].iloc[-1],
            'total_change': self.df['Car_Count'].iloc[-1] - self.df['Car_Count'].iloc[0],
            'total_change_percent': ((self.df['Car_Count'].iloc[-1] / self.df['Car_Count'].iloc[0]) - 1) * 100,
            'pearson_corr': self.pearson_corr,
            'pearson_pvalue': self.pearson_pvalue,
            'norm_slope': self.norm_slope,
            'norm_intercept': self.norm_intercept,
            'norm_r_squared': self.norm_r_squared,
            'norm_p_value': self.norm_p_value,
            'norm_std_err': self.norm_std_err,
            'norm_is_significant': self.norm_p_value < 0.05,
        }

    def print_summary(self):
        """Print a formatted summary of the analysis."""
        print("=" * 80)
        print(f"LINEAR REGRESSION TREND ANALYSIS")
        print(f"City: {self.city_name}")
        print(f"Baseline: {self.baseline:,} cars")
        print("=" * 80)

        print("\n" + "-" * 80)
        print("DATA SUMMARY")
        print("-" * 80)
        print(self.df[['Period', 'Time_Index', 'Car_Count', 'Relative_Change_Percent', 'Norm_Car_Count']].to_string(index=False))

        print("\n" + "-" * 80)
        print("LINEAR REGRESSION RESULTS")
        print("-" * 80)
        print(f"Trend slope:           {self.slope:,.2f} cars per month")
        print(f"Y-intercept:           {self.intercept:,.2f} cars")
        print(f"R-squared (R2):        {self.r_value**2:.4f}")
        print(f"P-value:               {self.p_value:.4f}")
        print(f"Standard error:        {self.std_err:,.2f}")

        print("\n" + "-" * 80)
        print("INTERPRETATION")
        print("-" * 80)
        print(f"Trend direction:       {self.trend_direction}")
        print(f"Average change rate:   {self.slope:,.0f} cars per month ({self.trend_desc})")
        
        # Statistical significance
        if self.p_value < 0.01:
            sig_level = "Very strong (p < 0.01)"
        elif self.p_value < 0.05:
            sig_level = "Strong (p < 0.05)"
        elif self.p_value < 0.10:
            sig_level = "Moderate (p < 0.10)"
        else:
            sig_level = "Weak (p >= 0.10)"

        print(f"Statistical significance: {sig_level}")

        if self.r_value**2 > 0.9:
            fit_quality = "Excellent fit"
        elif self.r_value**2 > 0.7:
            fit_quality = "Good fit"
        elif self.r_value**2 > 0.5:
            fit_quality = "Moderate fit"
        else:
            fit_quality = "Weak fit"

        print(f"Model fit quality:     {fit_quality} (R2 = {self.r_value**2:.3f})")
        print(f"                       {self.r_value**2*100:.1f}% of variance explained by linear trend")

        total_change_pct = ((self.df['Car_Count'].iloc[-1] / self.df['Car_Count'].iloc[0]) - 1) * 100
        print(f"\nTotal observed change: {total_change_pct:+.1f}% over {len(self.df)} periods")
        print(f"From {self.df['Car_Count'].iloc[0]:,} to {self.df['Car_Count'].iloc[-1]:,} cars")

        print("\n" + "-" * 80)
        print("CORRELATION ANALYSIS")
        print("-" * 80)

        def interpret_correlation(corr):
            """Interpret correlation coefficient strength"""
            abs_corr = abs(corr)
            if abs_corr >= 0.9:
                return "Very strong"
            elif abs_corr >= 0.7:
                return "Strong"
            elif abs_corr >= 0.5:
                return "Moderate"
            elif abs_corr >= 0.3:
                return "Weak"
            else:
                return "Very weak"

        print(f"Pearson Correlation:   r = {self.pearson_corr:+.4f} (p = {self.pearson_pvalue:.4f})")
        print(f"                       {interpret_correlation(self.pearson_corr)} {'positive' if self.pearson_corr > 0 else 'negative'} linear relationship")
        print(f"                       {'Statistically significant' if self.pearson_pvalue < 0.05 else 'Not significant'}")

        print(f"\nNote: R2 ({self.r_value**2:.3f}) = Pearson correlation^2 ({self.pearson_corr**2:.3f})")
        print(f"      Both measure the same linear relationship")

        print("\n" + "-" * 80)
        print("CONCLUSION")
        print("-" * 80)
        print(f"The linear regression analysis shows a statistically {'significant' if self.p_value < 0.05 else 'non-significant'}")
        print(f"{self.trend_direction.lower()} trend in car counts for {self.city_name}.")
        print(f"On average, car counts are {self.trend_desc} by {abs(self.slope):,.0f} cars per month.")
        if self.p_value < 0.05:
            print(f"This trend is statistically significant (p = {self.p_value:.4f}),")
            print(f"meaning it is unlikely to be due to random variation.")
        else:
            print(f"However, the trend is not statistically significant at the 0.05 level")
            print(f"(p = {self.p_value:.4f}), suggesting caution in interpretation.")

    def create_visualization(self, output_path='trend_analysis.png'):
        """
        Create comprehensive visualization of the analysis.

        Parameters:
        -----------
        output_path : str
            Path where the visualization will be saved
        """
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.3, height_ratios=[2, 1, 0.8])

        fig.suptitle(f'Linear Regression Analysis - {self.city_name}\nBaseline: {self.baseline:,} cars',
                     fontsize=16, fontweight='bold')

        # ── Row 0 ────────────────────────────────────────────────────────────
        # Plot 1: Main scatter plot with the regression line (absolute counts)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(self.df['Time_Index'], self.df['Car_Count'], s=200, color='#2E86AB', zorder=3, label='Actual data', edgecolors='black', linewidth=2)
        ax1.plot(self.df['Time_Index'], self.df['Car_Count'], 'o-', color='#2E86AB', alpha=0.3, linewidth=2, markersize=0)

        regression_line = self.slope * self.df['Time_Index'] + self.intercept
        ax1.plot(self.df['Time_Index'], regression_line, 'r--', linewidth=3, label=f'Linear Reg: {self.slope:,.0f} cars/month')

        predict_error = np.sqrt(sum((self.df['Residual'])**2) / (len(self.df) - 2)) if len(self.df) > 2 else 0
        if predict_error > 0:
            ci = 1.96 * predict_error
            ax1.fill_between(self.df['Time_Index'], regression_line - ci, regression_line + ci, alpha=0.2, color='red', label='95% CI (Linear)')

        ax1.set_xlabel(f'Time Index (months from {self.periods[0]})',
                       fontweight='bold', fontsize=12)
        ax1.set_ylabel('Car Count', fontweight='bold', fontsize=12)
        ax1.set_title('Absolute Counts — Linear Regression', fontweight='bold', fontsize=13)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        equation_text = f'Linear: y = {self.slope:,.0f}x + {self.intercept:,.0f}\n'
        equation_text += f'R2 = {self.r_value**2:.3f}, p = {self.p_value:.4f}'
        ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Plot 2: Actual vs Predicted — absolute
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(self.df['Time_Index'], self.df['Car_Count'], s=100, color='#2E86AB', zorder=3, label='Actual', alpha=0.6)
        ax2.plot(self.df['Time_Index'], self.df['Predicted'], 'r--', linewidth=2.5, label='Linear Reg', marker='s', markersize=8)
        ax2.set_xlabel('Time Index (months)', fontweight='bold')
        ax2.set_ylabel('Car Count', fontweight='bold')
        ax2.set_title('Absolute Counts — Actual vs Predicted', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        # ── Row 1: NORMALIZED CAR COUNT REGRESSION ───────────────────────────
        norm_reg_line = self.norm_slope * self.df['Time_Index'] + self.norm_intercept

        # Plot 3: Normalized — Linear Regression
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(self.df['Time_Index'], self.df['Norm_Car_Count'], s=150, color='#2E86AB', zorder=3, edgecolors='black', linewidth=1.5, label='Normalized count')
        ax3.plot(self.df['Time_Index'], self.df['Norm_Car_Count'], 'o-', color='#2E86AB', alpha=0.3, linewidth=1.5, markersize=0)
        ax3.plot(self.df['Time_Index'], norm_reg_line, 'r--', linewidth=2.5, label=f'Linear Reg: {self.norm_slope:.4f}/month')
        ax3.axhline(y=1.0, color='green', linestyle=':', linewidth=2, label='Baseline (1.0)', alpha=0.8)

        norm_eq = f'Linear: y = {self.norm_slope:.4f}x + {self.norm_intercept:.4f}\n'
        norm_eq += f'R2 = {self.norm_r_squared:.3f},  p = {self.norm_p_value:.4f}'
        ax3.text(0.05, 0.95, norm_eq, transform=ax3.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax3.set_xlabel(f'Time Index (months from {self.periods[0]})', fontweight='bold')
        ax3.set_ylabel('Normalized Car Count (ratio to baseline)', fontweight='bold')
        ax3.set_title('Normalized Counts — Linear Regression', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Residuals for the normalized regression
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(self.df['Time_Index'], self.df['Norm_Residual'], s=100, color='purple',
                    zorder=3, edgecolors='black')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('Time Index (months)', fontweight='bold')
        ax4.set_ylabel('Normalized Residual (Actual - Predicted)', fontweight='bold')
        ax4.set_title('Residual Plot — Normalized Regression', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # ── Row 2: Summary box ───────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        sig_status = "YES - statistically significant" if self.p_value < 0.05 else "NO - not significant at p<0.05"

        summary_text = f"""
REGRESSION STATISTICS SUMMARY

Linear Regression (Absolute):
  Slope:           {self.slope:,.2f} cars/month
  Intercept:       {self.intercept:,.2f} cars
  R2:              {self.r_value**2:.4f}
  P-value:         {self.p_value:.4f}
  Significance:    {sig_status}

CORRELATION ANALYSIS:
  Pearson:   r = {self.pearson_corr:+.4f} (p = {self.pearson_pvalue:.4f}) {'Significant' if self.pearson_pvalue < 0.05 else 'Not significant'}

Period Coverage:        {self.periods[0]} to {self.periods[-1]} ({len(self.df)} periods)
Starting Value:         {self.df['Car_Count'].iloc[0]:,} cars
Ending Value:           {self.df['Car_Count'].iloc[-1]:,} cars
Total Change:           {self.df['Car_Count'].iloc[-1] - self.df['Car_Count'].iloc[0]:+,} cars ({((self.df['Car_Count'].iloc[-1]/self.df['Car_Count'].iloc[0])-1)*100:+.1f}%)

INTERPRETATION: The analysis reveals a {self.trend_direction.lower()} trend. Linear regression shows {abs(self.slope):,.0f} cars/month change.
{'This trend IS statistically significant.' if self.p_value < 0.05 else 'This trend is NOT statistically significant at p<0.05.'}
        """

        ax5.text(0.5, 0.5, summary_text, fontsize=9, family='monospace',
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")

    def export_results(self, data_csv_path='results_data.csv', model_csv_path='results_model.csv', summary_path='summary.txt'):
        """
        Export results to two CSV files and a text summary.

        Parameters:
        -----------
        data_csv_path  : str  -- period-level input values + predictions + residuals
        model_csv_path : str  -- model stats, comparison, observed/estimated differences
        summary_path   : str  -- text summary
        """
        data_df = self.df[['Period', 'Time_Index',
                            'Car_Count', 'Predicted', 'Residual', 'Relative_Change_Percent',
                            'Norm_Car_Count', 'Norm_Predicted', 'Norm_Residual']].copy()
        data_df['Predicted'] = data_df['Predicted'].round(0)
        data_df['Residual'] = data_df['Residual'].round(0)
        data_df['Norm_Predicted'] = data_df['Norm_Predicted'].round(6)
        data_df['Norm_Residual'] = data_df['Norm_Residual'].round(6)
        data_df.to_csv(data_csv_path, index=False)
        print(f"Data results saved to:  {data_csv_path}")

        rows = []

        # IDP information
        rows += [
            ('IDP Information', 'IDP value', self.idp, ''),
            ('IDP Information', 'Population baseline (admin1)', self.population_baseline, 'province-level population'),
            ('IDP Information', 'IDP normalized (admin1)', self.idp_norm, 'IDP / admin1 population'),
        ]
        
        # Absolute regression
        rows += [
            ('Absolute Regression', 'Slope (cars/month)', self.slope, ''),
            ('Absolute Regression', 'Intercept', self.intercept, ''),
            ('Absolute Regression', 'R2', self.r_value**2, ''),
            ('Absolute Regression', 'P-value', self.p_value, ''),
            ('Absolute Regression', 'Std Error', self.std_err, ''),
            ('Absolute Regression', 'Significant (p<0.05)', self.p_value < 0.05, ''),
        ]
        
        # Normalized regression
        rows += [
            ('Normalized Regression', 'Slope (fraction/month)', self.norm_slope, ''),
            ('Normalized Regression', 'Intercept', self.norm_intercept, ''),
            ('Normalized Regression', 'R2', self.norm_r_squared, ''),
            ('Normalized Regression', 'P-value', self.norm_p_value, ''),
            ('Normalized Regression', 'Std Error', self.norm_std_err, ''),
            ('Normalized Regression', 'Significant (p<0.05)', self.norm_p_value < 0.05, ''),
        ]
        
        # Model comparison
        rows += [
            ('Model Comparison', 'Abs slope / baseline (fraction/month)', self.abs_slope_as_norm, ''),
            ('Model Comparison', 'Norm slope (fraction/month)', self.norm_slope, ''),
            ('Model Comparison', 'Slope scale difference', self.slope_scale_diff, ''),
            ('Model Comparison', 'Slope scale difference (%)', self.slope_scale_diff_pct, ''),
            ('Model Comparison', 'R2 difference (abs - norm)', self.r2_diff, 'positive = absolute fits better'),
            ('Model Comparison', 'P-value difference (abs - norm)', self.pval_diff, 'positive = normalized more significant'),
        ]
        
        # Max-Min Difference (temporal ordering)
        rows += [
            ('Max-Min Difference', 'Max difference (signed)', self.max_diff, 'positive=increasing, negative=decreasing'),
            ('Max-Min Difference', 'Max difference (absolute)', self.max_diff_abs, ''),
            ('Max-Min Difference', 'Max period', self.max_period, ''),
            ('Max-Min Difference', 'Min period', self.min_period, ''),
            ('Max-Min Difference', 'Max time index', self.max_time_index, 'months from start'),
            ('Max-Min Difference', 'Min time index', self.min_time_index, 'months from start'),
        ]
        
        # Observed differences
        rows += [
            ('Observed Difference', 'Start period', self.periods[0], ''),
            ('Observed Difference', 'End period', self.periods[-1], ''),
            ('Observed Difference', 'Start value (absolute)', self.df["Car_Count"].iloc[0], ''),
            ('Observed Difference', 'End value (absolute)', self.df["Car_Count"].iloc[-1], ''),
            ('Observed Difference', 'Start value (normalized)', self.df["Norm_Car_Count"].iloc[0], ''),
            ('Observed Difference', 'End value (normalized)', self.df["Norm_Car_Count"].iloc[-1], ''),
            ('Observed Difference', 'Change (absolute)', self.obs_diff_abs, 'end - start'),
            ('Observed Difference', 'Change % (absolute)', self.obs_diff_abs_pct, 'relative to start'),
            ('Observed Difference', 'Change (normalized)', self.obs_diff_norm, 'end - start'),
            ('Observed Difference', 'Change % (normalized)', self.obs_diff_norm_pct, 'relative to start'),
        ]
        
        # Estimated differences
        reg_start_abs = self.slope * self.df['Time_Index'].iloc[0] + self.intercept
        reg_end_abs = self.slope * self.df['Time_Index'].iloc[-1] + self.intercept
        reg_start_norm = self.norm_slope * self.df['Time_Index'].iloc[0] + self.norm_intercept
        reg_end_norm = self.norm_slope * self.df['Time_Index'].iloc[-1] + self.norm_intercept
        rows += [
            ('Estimated Difference', 'Regression start (absolute)', reg_start_abs, self.periods[0]),
            ('Estimated Difference', 'Regression end (absolute)', reg_end_abs, self.periods[-1]),
            ('Estimated Difference', 'Regression start (normalized)', reg_start_norm, self.periods[0]),
            ('Estimated Difference', 'Regression end (normalized)', reg_end_norm, self.periods[-1]),
            ('Estimated Difference', 'Estimated change (absolute)', self.est_diff_abs, 'reg end - reg start'),
            ('Estimated Difference', 'Estimated change % (absolute)', self.est_diff_abs_pct, 'relative to reg start'),
            ('Estimated Difference', 'Estimated change (normalized)', self.est_diff_norm, 'reg end - reg start'),
            ('Estimated Difference', 'Estimated change % (normalized)', self.est_diff_norm_pct, 'relative to reg start'),
        ]
        
        # Actual vs Predicted correlation — absolute
        rows += [
            ('Fit Correlation (Absolute)', 'Pearson r', self.fit_pearson_abs, ''),
            ('Fit Correlation (Absolute)', 'Pearson p-value', self.fit_pearson_abs_p, ''),
            ('Fit Correlation (Absolute)', 'Pearson significant', self.fit_pearson_abs_p < 0.05 if self.fit_pearson_abs_p == self.fit_pearson_abs_p else False, ''),
            ('Fit Correlation (Absolute)', 'n periods', len(self.df), 'vector length used'),
        ]
        
        # Actual vs Predicted correlation — normalized
        rows += [
            ('Fit Correlation (Normalized)', 'Pearson r', self.fit_pearson_norm, ''),
            ('Fit Correlation (Normalized)', 'Pearson p-value', self.fit_pearson_norm_p, ''),
            ('Fit Correlation (Normalized)', 'Pearson significant', self.fit_pearson_norm_p < 0.05 if self.fit_pearson_norm_p == self.fit_pearson_norm_p else False, ''),
            ('Fit Correlation (Normalized)', 'n periods', len(self.df), 'vector length used'),
        ]

        model_df = pd.DataFrame(rows, columns=['Section', 'Metric', 'Value', 'Notes'])
        model_df.to_csv(model_csv_path, index=False)
        print(f"Model results saved to: {model_csv_path}")
        
        # Export summary to text file
        with open(summary_path, 'w') as f:
            f.write("LINEAR REGRESSION ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"City: {self.city_name}\n")
            f.write(f"Baseline: {self.baseline:,} cars\n")
            f.write(f"Analysis Period: {self.periods[0]} to {self.periods[-1]}\n\n")

            f.write("ABSOLUTE CAR COUNT REGRESSION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Linear Regression:\n")
            f.write(f"  Slope (trend rate):   {self.slope:,.2f} cars/month\n")
            f.write(f"  Y-intercept:          {self.intercept:,.2f} cars\n")
            f.write(f"  R-squared:            {self.r_value**2:.4f}\n")
            f.write(f"  P-value:              {self.p_value:.4f}\n")
            f.write(f"  Standard error:       {self.std_err:,.2f}\n\n")

            f.write("NORMALIZED CAR COUNT REGRESSION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Linear Regression:\n")
            f.write(f"  Slope (trend rate):   {self.norm_slope:.6f} (fraction of baseline/month)\n")
            f.write(f"  Y-intercept:          {self.norm_intercept:.6f}\n")
            f.write(f"  R-squared:            {self.norm_r_squared:.4f}\n")
            f.write(f"  P-value:              {self.norm_p_value:.4f}\n")
            f.write(f"  Standard error:       {self.norm_std_err:.6f}\n")
            f.write(f"  Significance:         {'YES (p < 0.05)' if self.norm_p_value < 0.05 else 'NO (p >= 0.05)'}\n\n")

            f.write("MODEL COMPARISON (Absolute vs Normalized)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Absolute slope:              {self.slope:.4f} cars/month\n")
            f.write(f"  Absolute slope (div baseline): {self.abs_slope_as_norm:.6f} fraction/month\n")
            f.write(f"  Normalized slope:            {self.norm_slope:.6f} fraction/month\n")
            f.write(f"  Slope scale difference:      {self.slope_scale_diff:.6f} ({self.slope_scale_diff_pct:.2f}%)\n")
            f.write(f"  {'-> Slopes agree' if self.slope_scale_diff_pct < 5 else '-> Slopes diverge -- outliers may affect absolute regression'}\n\n")
            f.write(f"  Absolute R2:                 {self.r_value**2:.4f}\n")
            f.write(f"  Normalized R2:               {self.norm_r_squared:.4f}\n")
            f.write(f"  R2 difference:               {self.r2_diff:+.4f}  "
                    f"({'absolute fits better' if self.r2_diff > 0 else 'normalized fits better' if self.r2_diff < 0 else 'identical fit'})\n\n")
            f.write(f"  Absolute p-value:            {self.p_value:.4f}\n")
            f.write(f"  Normalized p-value:          {self.norm_p_value:.4f}\n")
            f.write(f"  P-value difference:          {self.pval_diff:+.4f}  "
                    f"({'normalized more significant' if self.pval_diff > 0 else 'absolute more significant' if self.pval_diff < 0 else 'identical'})\n\n")

            f.write("OBSERVED & ESTIMATED DIFFERENCES\n")
            f.write("-" * 80 + "\n")
            if self.obs_start_period and self.obs_end_period:
                f.write(f"  Period (all data):     {self.periods[0]}  ->  {self.periods[-1]}\n")
                f.write(f"  Period (valid data):   {self.obs_start_period}  ->  {self.obs_end_period}\n\n")
            else:
                f.write(f"  Period:  {self.periods[0]}  ->  {self.periods[-1]}\n\n")

            f.write(f"  {'Metric':<40} {'Absolute':>15} {'Normalized':>15}\n")
            f.write(f"  {'-'*70}\n")
            f.write(f"  {'Start value':<40} "
                    f"{self.df['Car_Count'].iloc[0]:>15,.0f} "
                    f"{self.df['Norm_Car_Count'].iloc[0]:>15.4f}\n")
            f.write(f"  {'End value':<40} "
                    f"{self.df['Car_Count'].iloc[-1]:>15,.0f} "
                    f"{self.df['Norm_Car_Count'].iloc[-1]:>15.4f}\n")
            f.write(f"  {'Observed change':<40} "
                    f"{self.obs_diff_abs:>+15,.0f} "
                    f"{self.obs_diff_norm:>+15.4f}\n")
            f.write(f"  {'Observed change (%)':<40} "
                    f"{self.obs_diff_abs_pct:>+14.2f}% "
                    f"{self.obs_diff_norm_pct:>+14.2f}%\n\n")
            f.write(f"  {'Regression start (predicted)':<40} "
                    f"{self.slope * self.df['Time_Index'].iloc[0] + self.intercept:>15,.0f} "
                    f"{self.norm_slope * self.df['Time_Index'].iloc[0] + self.norm_intercept:>15.4f}\n")
            f.write(f"  {'Regression end (predicted)':<40} "
                    f"{self.slope * self.df['Time_Index'].iloc[-1] + self.intercept:>15,.0f} "
                    f"{self.norm_slope * self.df['Time_Index'].iloc[-1] + self.norm_intercept:>15.4f}\n")
            f.write(f"  {'Estimated change':<40} "
                    f"{self.est_diff_abs:>+15,.0f} "
                    f"{self.est_diff_norm:>+15.4f}\n")
            f.write(f"  {'Estimated change (%)':<40} "
                    f"{self.est_diff_abs_pct:>+14.2f}% "
                    f"{self.est_diff_norm_pct:>+14.2f}%\n\n")

            f.write("MAX-MIN DIFFERENCE (Temporal Ordering)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Max period: {self.max_period} (month {self.max_time_index})\n")
            f.write(f"  Min period: {self.min_period} (month {self.min_time_index})\n")
            f.write(f"  Max-Min difference (signed):   {self.max_diff:+,.0f} cars\n")
            f.write(f"  Max-Min difference (absolute): {self.max_diff_abs:,.0f} cars\n")
            if self.max_diff > 0:
                f.write(f"  -> Increasing trend (max comes after min in time)\n")
            elif self.max_diff < 0:
                f.write(f"  -> Decreasing trend (max comes before min in time)\n")
            else:
                f.write(f"  -> No clear trend (max and min at same time)\n")
            f.write("\n")

            f.write("FIT CORRELATION (Actual vs Predicted)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  n periods used: {len(self.df)}\n\n")

            def _sig(p): return 'Sig' if (p == p and p < 0.05) else 'Not sig'
            def _str(r):
                if r != r: return 'n/a'
                a = abs(r)
                if a >= 0.9: return 'Very strong'
                if a >= 0.7: return 'Strong'
                if a >= 0.5: return 'Moderate'
                if a >= 0.3: return 'Weak'
                return 'Very weak'

            for label, pr, pp in [
                ('ABSOLUTE  (Car_Count vs Predicted)', self.fit_pearson_abs, self.fit_pearson_abs_p),
                ('NORMALIZED (Norm_Car_Count vs Norm_Predicted)', self.fit_pearson_norm, self.fit_pearson_norm_p),
            ]:
                f.write(f"  {label}\n")
                f.write(f"  {'Pearson  r':<12}: {pr:+.4f}  p={pp:.4f}  {_sig(pp)}  [{_str(pr)}]\n\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Trend direction:        {self.trend_direction}\n")
            f.write(f"Absolute significance:  {'YES (p < 0.05)' if self.p_value < 0.05 else 'NO (p >= 0.05)'}\n")
            f.write(f"Normalized significance:{'YES (p < 0.05)' if self.norm_p_value < 0.05 else 'NO (p >= 0.05)'}\n")
            f.write(f"Absolute fit:           {self.r_value**2*100:.1f}% variance explained\n")
            f.write(f"Normalized fit:         {self.norm_r_squared*100:.1f}% variance explained\n\n")
            f.write("CONCLUSION\n")
            f.write("-" * 80 + "\n")
            f.write(f"The analysis shows a {self.trend_direction.lower()} trend of {abs(self.slope):,.0f} cars/month.\n")
            f.write(f"Normalized: {self.norm_slope:.6f} fraction of baseline per month "
                    f"({self.norm_slope*100:.4f}% of baseline/month).\n")
            if self.p_value < 0.05:
                f.write("Absolute trend is statistically significant.\n")
            else:
                f.write("Absolute trend is not statistically significant at p < 0.05.\n")
            if self.norm_p_value < 0.05:
                f.write("Normalized trend is statistically significant.\n")
            else:
                f.write("Normalized trend is not statistically significant at p < 0.05.\n")

        print(f"Summary report saved to: {summary_path}")


def analyze_from_dict(data_dict, output_dir='output'):
    """
    Analyze car count data from a dictionary.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with keys: 'city', 'periods', 'car_counts', 'baseline',
                              'idp' (optional), 'population_baseline' (optional)
    output_dir : str
        Directory where outputs will be saved

    Returns:
    --------
    TrendAnalyzer object
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    analyzer = TrendAnalyzer(
        indx=data_dict['indx'],
        city_name=data_dict['city'],
        location_level=data_dict.get('location_level', 'city'),
        periods=data_dict['periods'],
        car_counts=data_dict['car_counts'],
        baseline=data_dict['baseline'],
        norm_car_counts=data_dict['norm_car_counts'],
        idp=data_dict.get('idp', None),
        population_baseline=data_dict.get('population_baseline', None),
    )

    analyzer.print_summary()

    city_safe = data_dict['city'].replace(' ', '_').replace(',', '')
    analyzer.create_visualization(f"{output_dir}/{data_dict['indx']}_{city_safe}_visualization.png")
    analyzer.export_results(
        f"{output_dir}/{data_dict['indx']}_{city_safe}_data.csv",
        f"{output_dir}/{data_dict['indx']}_{city_safe}_model.csv",
        f"{output_dir}/{data_dict['indx']}_{city_safe}_summary.txt"
    )

    return analyzer


# ============================================================================
# POPULATION BASELINE UTILITIES (admin1 only)
# ============================================================================

def load_population_data(population_csv_path):
    """
    Load population baseline data from CSV and return a lookup dictionary.

    Parameters:
    -----------
    population_csv_path : str
        Path to population CSV with columns: name, total_pop, mean_density,
        max_density, area_km2, pop_density_adm

    Returns:
    --------
    dict : {city_name: total_pop}
    """
    pop_df = pd.read_csv(population_csv_path)
    pop_dict = dict(zip(pop_df['name'], pop_df['total_pop']))

    return pop_dict


# ============================================================================
# Load population baseline data (admin1 only)
# ============================================================================

population_dict = load_population_data(
    population_csv_path='syria_pop_density_output/Syria_admin1_baseline_population.csv'
)

def get_baseline_pop(row):
    return population_dict.get(row['City'], float('nan'))


def get_baseline_car_pop(row):
    return baseline_car_pop_dict.get(row['City'], float('nan'))


def parse_idp_number(idp_str):
    """Parse IDP number from various formats"""
    if pd.isna(idp_str):
        return np.nan

    idp_str = str(idp_str).replace(',', '').replace('"', '').strip()

    if 'to' in idp_str.lower():
        parts = re.findall(r'[+-]?\d+', idp_str)
        if len(parts) >= 2:
            return np.mean([float(p) for p in parts])

    match = re.search(r'([+-]?\d+)', idp_str)
    if match:
        return float(match.group(1))

    return np.nan


def parse_period_to_months(period_str):
    """
    Convert period string to a datetime for month-gap calculation.

    Examples:
    'Jan2014' -> datetime(2014, 1, 1)
    """
    try:
        for fmt in ['%b%Y', '%B%Y', '%m/%Y', '%Y-%m']:
            try:
                return datetime.strptime(period_str, fmt)
            except ValueError:
                continue
        return None
    except:
        return None


def calculate_time_indices(periods):
    """
    Calculate time indices based on actual month differences.

    Parameters:
    -----------
    periods : list
        List of period strings (e.g., ['Jan2014', 'Mar2014', 'Jun2014'])

    Returns:
    --------
    list : Time indices reflecting actual month gaps
    """
    dates = [parse_period_to_months(p) for p in periods]

    if None in dates:
        print("Warning: Could not parse all period strings. Using sequential indices.")
        return list(range(len(periods)))

    first_date = dates[0]
    time_indices = []

    for date in dates:
        months_diff = (date.year - first_date.year) * 12 + (date.month - first_date.month)
        time_indices.append(months_diff)

    return time_indices


# ============================================================================
# IDP CORRELATION ANALYSIS (integrated, admin1 population baseline only)
# ============================================================================

def calculate_idp_correlations(analyzers_list, idp_values, population_baseline):
    """
    Calculate correlations between car count changes and IDP.

    Parameters:
    -----------
    analyzers_list : list
        List of TrendAnalyzer objects from all reports
    idp_values : list or array
        IDP value for each report (same order as analyzers_list)
    population_baseline : list or array
        Population baseline for each report's city (same order as analyzers_list)

    Returns:
    --------
    dict : Dictionary with correlation results
    """
    est_change_abs = [a.est_diff_abs for a in analyzers_list]
    obs_change_abs = [a.obs_diff_abs for a in analyzers_list]
    est_change_norm = [a.est_diff_norm for a in analyzers_list]
    obs_change_norm = [a.obs_diff_norm for a in analyzers_list]
    max_diff = [a.max_diff for a in analyzers_list]
    max_diff_norm = [a.max_diff_norm for a in analyzers_list]

    idp_values = np.array(idp_values)
    population_baseline = np.array(population_baseline)
    est_change_abs = np.array(est_change_abs)
    obs_change_abs = np.array(obs_change_abs)
    est_change_norm = np.array(est_change_norm)
    obs_change_norm = np.array(obs_change_norm)
    max_diff = np.array(max_diff)
    max_diff_norm = np.array(max_diff_norm)

    idp_norm = np.array([a.idp_norm for a in analyzers_list])

    with np.errstate(divide='ignore', invalid='ignore'):
        idp_norm = np.where(np.isfinite(idp_norm), idp_norm, np.nan)

    def safe_corr(x, y):
        """Compute Pearson correlation after removing nan pairs."""
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        n_valid = len(x_clean)

        if n_valid < 2:
            return (float('nan'), float('nan'), n_valid)

        r, p_r = stats.pearsonr(x_clean, y_clean)

        return (r, p_r, n_valid)

    results = {}

    r1, p1, n1 = safe_corr(est_change_abs, idp_values)
    results['est_abs_vs_idp'] = {'pearson': r1, 'p_pearson': p1, 'n': n1}

    r2, p2, n2 = safe_corr(obs_change_abs, idp_values)
    results['obs_abs_vs_idp'] = {'pearson': r2, 'p_pearson': p2, 'n': n2}

    r3, p3, n3 = safe_corr(est_change_norm, idp_norm)
    results['est_norm_vs_idp_norm'] = {'pearson': r3, 'p_pearson': p3, 'n': n3}

    r4, p4, n4 = safe_corr(obs_change_norm, idp_norm)
    results['obs_norm_vs_idp_norm'] = {'pearson': r4, 'p_pearson': p4, 'n': n4}

    r5, p5, n5 = safe_corr(max_diff, idp_values)
    results['maxdiff_vs_idp'] = {'pearson': r5, 'p_pearson': p5, 'n': n5}

    r6, p6, n6 = safe_corr(max_diff_norm, idp_norm)
    results['maxdiff_norm_vs_idp_norm'] = {'pearson': r6, 'p_pearson': p6, 'n': n6}

    results['data'] = {
        'est_change_abs': est_change_abs,
        'obs_change_abs': obs_change_abs,
        'est_change_norm': est_change_norm,
        'obs_change_norm': obs_change_norm,
        'max_diff': max_diff,
        'max_diff_norm': max_diff_norm,
        'idp_values': idp_values,
        'idp_norm': idp_norm,
        'cities': [a.city_name for a in analyzers_list],
        'indices': [a.indx for a in analyzers_list]
    }

    return results


def print_idp_correlations(results, output_file=None):
    """Print formatted summary of IDP correlations and optionally write to file."""

    def sig(p): return 'Sig' if (p == p and p < 0.05) else 'Not sig'
    def strength(r):
        if r != r: return 'n/a'
        a = abs(r)
        if a >= 0.9: return 'Very strong'
        if a >= 0.7: return 'Strong'
        if a >= 0.5: return 'Moderate'
        if a >= 0.3: return 'Weak'
        return 'Very weak'

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("IDP CORRELATION ANALYSIS")
    lines.append(f"{'='*80}")
    lines.append(f"  Reports analyzed: {results['est_abs_vs_idp']['n']}")
    lines.append("")

    sections = [
        ('1. ESTIMATED CHANGE (ABSOLUTE) vs IDP', 'est_abs_vs_idp'),
        ('2. OBSERVED CHANGE (ABSOLUTE) vs IDP', 'obs_abs_vs_idp'),
        ('3. ESTIMATED CHANGE (NORMALIZED) vs IDP (NORMALIZED)', 'est_norm_vs_idp_norm'),
        ('4. OBSERVED CHANGE (NORMALIZED) vs IDP (NORMALIZED)', 'obs_norm_vs_idp_norm'),
        ('5. MAX-MIN DIFFERENCE vs IDP', 'maxdiff_vs_idp'),
        ('6. MAX-MIN DIFFERENCE (normalized) vs IDP (NORMALIZED)', 'maxdiff_norm_vs_idp_norm'),
    ]

    for title, key in sections:
        r = results[key]
        lines.append(f"  {title}")
        lines.append(f"  {'-'*70}")
        lines.append(f"  Pearson  r: {r['pearson']:+.4f}  p={r['p_pearson']:.4f}  {sig(r['p_pearson'])}  [{strength(r['pearson'])}]")
        lines.append("")

    for line in lines:
        print(line)

    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"IDP correlation summary saved to: {output_file}")


def plot_idp_correlations(results, output_path='idp_correlations.png'):
    """Create 2x2 scatter plots of car count changes vs IDP."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Car Count Changes vs IDP\n(Each point = one report)', fontsize=14, fontweight='bold')

    data = results['data']

    configs = [
        (0, 0, data['est_change_abs'], data['idp_values'], 'Estimated Change (Absolute)', 'IDP', 'est_abs_vs_idp'),
        (0, 1, data['obs_change_abs'], data['idp_values'], 'Observed Change (Absolute)', 'IDP', 'obs_abs_vs_idp'),
        (1, 0, data['est_change_norm'], data['idp_norm'], 'Estimated Change (Normalized)', 'IDP / Population', 'est_norm_vs_idp_norm'),
        (1, 1, data['obs_change_norm'], data['idp_norm'], 'Observed Change (Normalized)', 'IDP / Population', 'obs_norm_vs_idp_norm'),
    ]

    for row, col, y_data, x_data, y_label, x_label, key in configs:
        ax = axes[row, col]
        r = results[key]

        ax.scatter(x_data, y_data, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

        if (r['p_pearson'] < 0.1 and r['pearson'] == r['pearson'] and np.sum(~np.isnan(x_data) & ~np.isnan(y_data)) >= 2):
            try:
                mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_fit_data = x_data[mask]
                y_fit_data = y_data[mask]

                slope, intercept = np.polyfit(x_fit_data, y_fit_data, 1)
                x_fit = np.linspace(x_fit_data.min(), x_fit_data.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7, label=f"Trend (slope={slope:.2e})")
                ax.legend(fontsize=9)
            except (np.linalg.LinAlgError, ValueError):
                print('skipping, slope & intercept failed', np.linalg.LinAlgError, ValueError)
                pass

        r2 = r['pearson'] ** 2 if r['pearson'] == r['pearson'] else float('nan')
        ann = (f"r = {r['pearson']:+.3f}  (p={r['p_pearson']:.4f})\n"
               f"R2 = {r2:.3f}\n"
               f"n = {r['n']}")
        ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

        ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
        ax.set_title(f"{y_label} vs {x_label}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"IDP correlation plot saved to: {output_path}")


def export_idp_correlations(results, output_path='idp_correlations.csv'):
    """Export IDP correlation results to CSV."""

    rows = []
    for label, key in [
        ('Estimated Change (Abs) vs IDP', 'est_abs_vs_idp'),
        ('Observed Change (Abs) vs IDP', 'obs_abs_vs_idp'),
        ('Estimated Change (Norm) vs IDP (Norm)', 'est_norm_vs_idp_norm'),
        ('Observed Change (Norm) vs IDP (Norm)', 'obs_norm_vs_idp_norm'),
        ('Max-Min Difference vs IDP', 'maxdiff_vs_idp'),
        ('Max-Min Difference (Norm) vs IDP (Norm)', 'maxdiff_norm_vs_idp_norm'),
    ]:
        r = results[key]
        rows.append({
            'Comparison': label,
            'Pearson_r': r['pearson'],
            'Pearson_p': r['p_pearson'],
            'Pearson_sig': r['p_pearson'] < 0.05 if r['p_pearson'] == r['p_pearson'] else False,
            'n_reports': r['n']
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"IDP correlation CSV saved to: {output_path}")


def plot_absolute_vs_idp(results, output_path='absolute_vs_idp.png'):
    """Create 1x3 plot comparing observed/estimated/max-diff vs IDP."""

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('Car Count Changes vs IDP - Absolute Comparisons\n(Each point = one report)', fontsize=14, fontweight='bold')

    data = results['data']

    configs = [
        (0, data['obs_change_abs'], data['idp_values'], 'Observed Change (Absolute)', 'IDP', 'obs_abs_vs_idp'),
        (1, data['est_change_abs'], data['idp_values'], 'Estimated Change (Absolute)', 'IDP', 'est_abs_vs_idp'),
        (2, data['max_diff'], data['idp_values'], 'Max-Min Difference', 'IDP', 'maxdiff_vs_idp'),
    ]

    for idx, y_data, x_data, y_label, x_label, key in configs:
        ax = axes[idx]
        r = results[key]

        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[mask]
        y_clean = y_data[mask]

        if len(x_clean) < 1:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{y_label}\n(No valid data)", fontweight='bold')
            continue

        ax.scatter(x_clean, y_clean, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='No change')

        if (r['p_pearson'] < 0.1 and r['pearson'] == r['pearson'] and len(x_clean) >= 2):
            try:
                slope, intercept = np.polyfit(x_clean, y_clean, 1)
                x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7, label=f"Trend (slope={slope:.2e})")
            except (np.linalg.LinAlgError, ValueError):
                print("skipping .. ", np.linalg.LinAlgError, ValueError)
                pass

        if len(x_clean) >= 2:
            ax.legend(fontsize=9, loc='best')

        r2 = r['pearson'] ** 2 if r['pearson'] == r['pearson'] else float('nan')
        ann = (f"r = {r['pearson']:+.3f}  (p={r['p_pearson']:.4f})\n"
               f"R2 = {r2:.3f}\n"
               f"n = {r['n']}")
        ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

        ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
        if idx == 2:
            ax.set_ylabel('Max-Min Difference (cars)\n(+ve = increasing, -ve = decreasing)', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel(f'{y_label} (cars)', fontsize=11, fontweight='bold')

        ax.set_title(f"{y_label} vs {x_label}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Absolute vs IDP plot saved to: {output_path}")


def plot_norm_vs_norm_idp(results, output_path='norm_vs_norm_idp.png'):
    """Create 1x3 plot comparing normalized observed/estimated/max-diff vs normalized IDP."""

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('Normalized Changes vs Normalized IDP\n(Each point = one report)', fontsize=14, fontweight='bold')

    data = results['data']

    configs = [
        (0, data['obs_change_norm'], data['idp_norm'], 'Observed Change (Normalized)', 'IDP (Norm)', 'obs_norm_vs_idp_norm'),
        (1, data['est_change_norm'], data['idp_norm'], 'Estimated Change (Normalized)', 'IDP (Norm)', 'est_norm_vs_idp_norm'),
        (2, data['max_diff'], data['idp_norm'], 'Max-Min Difference (Norm)', 'IDP (Norm)', 'maxdiff_norm_vs_idp_norm'),
    ]

    for idx, y_data, x_data, y_label, x_label, key in configs:
        ax = axes[idx]
        r = results[key]

        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[mask]
        y_clean = y_data[mask]

        if len(x_clean) < 1:
            ax.text(0.5, 0.5, 'No valid data',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{y_label}\n(No valid data)", fontweight='bold')
            continue

        ax.scatter(x_clean, y_clean, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='No change')

        if (r['p_pearson'] < 0.1 and
            r['pearson'] == r['pearson'] and
            len(x_clean) >= 2):
            try:
                slope, intercept = np.polyfit(x_clean, y_clean, 1)
                x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7,
                        label=f"Trend (slope={slope:.2e})")
            except (np.linalg.LinAlgError, ValueError):
                print("skipping ..", np.linalg.LinAlgError, ValueError)
                pass

        if len(x_clean) >= 2:
            ax.legend(fontsize=9, loc='best')

        r2 = r['pearson'] ** 2 if r['pearson'] == r['pearson'] else float('nan')
        ann = (f"r = {r['pearson']:+.3f}  (p={r['p_pearson']:.4f})\n"
               f"R2 = {r2:.3f}\n"
               f"n = {r['n']}")
        ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

        ax.set_xlabel(x_label, fontsize=11, fontweight='bold')

        if idx == 2:
            ax.set_ylabel('Max-Min Difference (cars)\n(+ve = increasing, -ve = decreasing)', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel(f'{y_label} (fraction)', fontsize=11, fontweight='bold')

        ax.set_title(f"{y_label} vs {x_label}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Normalized vs Normalized IDP plot saved to: {output_path}")


def plot_car_count_vs_idp_absolute(results, output_path='car_count_vs_idp_absolute.png'):
    """
    Create 1x3 plot with swapped axes: Car count change (X) vs IDP (Y).

    Parameters:
    -----------
    results : dict
        Results from calculate_idp_correlations
    output_path : str
        Path to save the plot
    """

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    data = results['data']

    configs = [
        (0, data['obs_change_abs'], data['idp_values'], 'Observed Change (Absolute)', 'IDP', 'obs_abs_vs_idp', 'forestgreen'),
        (1, data['est_change_abs'], data['idp_values'], 'Estimated Change (Absolute)', 'IDP', 'est_abs_vs_idp', 'purple'),
        (2, data['max_diff'], data['idp_values'], 'Max-Min Difference (Absolute)', 'IDP', 'maxdiff_vs_idp', 'darkred'),
    ]

    for idx, x_data, y_data, x_label, y_label, key, color in configs:
        ax = axes[idx]
        r = results[key]

        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[mask]
        y_clean = y_data[mask]

        if len(x_clean) < 1:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{x_label}\n(No valid data)", fontweight='bold')
            continue

        ax.scatter(x_clean, y_clean, s=120, alpha=0.6, edgecolors='black', linewidth=1.5, color=color)

        if (r['pearson'] == r['pearson'] and len(x_clean) >= 2):
            try:
                slope, intercept = np.polyfit(x_clean, y_clean, 1)
                x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7, label=f"Trend (slope={slope:.2e})")
            except (np.linalg.LinAlgError, ValueError):
                pass

        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(fontsize=9, loc='best', framealpha=0.9)

        r2 = r['pearson'] ** 2 if r['pearson'] == r['pearson'] else float('nan')
        ann = (f"r = {r['pearson']:+.3f}  (p={r['p_pearson']:.4f})\nR2 = {r2:.3f}\nn = {r['n']}")
        ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1))

        ax.set_xlabel(f'{x_label} (cars)', fontsize=12, fontweight='bold')
        ax.set_ylabel('IDP', fontsize=12, fontweight='bold')

        ax.set_title(f'IDP vs {x_label}', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Car count vs IDP (absolute) plot saved to: {output_path}")


def plot_car_count_vs_idp_normalized(results, output_path='car_count_vs_idp_normalized.png'):
    """
    Create 1x3 plot with swapped axes: Car count change (X) vs IDP (Y). Normalized values.
    """

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    data = results['data']

    configs = [
        (0, data['obs_change_norm'], data['idp_norm'], 'Observed Change (Normalized)', 'IDP / Population', 'obs_norm_vs_idp_norm', 'forestgreen'),
        (1, data['est_change_norm'], data['idp_norm'], 'Estimated Change (Normalized)', 'IDP / Population', 'est_norm_vs_idp_norm', 'purple'),
        (2, data['max_diff_norm'], data['idp_norm'], 'Max-Min Difference (Normalized)', 'IDP / Population', 'maxdiff_norm_vs_idp_norm', 'darkred'),
    ]

    for idx, x_data, y_data, x_label, y_label, key, color in configs:
        ax = axes[idx]
        r = results[key]

        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[mask]
        y_clean = y_data[mask]

        if len(x_clean) < 1:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{x_label}\n(No valid data)", fontweight='bold')
            continue

        ax.scatter(x_clean, y_clean, s=120, alpha=0.6, edgecolors='black', linewidth=1.5, color=color)

        if (r['pearson'] == r['pearson'] and len(x_clean) >= 2):
            try:
                slope, intercept = np.polyfit(x_clean, y_clean, 1)
                x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7, label=f"Trend (slope={slope:.2e})")
            except (np.linalg.LinAlgError, ValueError):
                pass

        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(fontsize=9, loc='best', framealpha=0.9)

        r2 = r['pearson'] ** 2 if r['pearson'] == r['pearson'] else float('nan')
        ann = (f"r = {r['pearson']:+.3f}  (p={r['p_pearson']:.4f})\n"
               f"R2 = {r2:.3f}\n"
               f"n = {r['n']}")
        ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1))

        if idx == 2:
            ax.set_xlabel('Max-Min Difference (Normalized)', fontsize=12, fontweight='bold')
        else:
            ax.set_xlabel(f'{x_label}', fontsize=12, fontweight='bold')

        ax.set_ylabel('Normalized IDP', fontsize=12, fontweight='bold')

        ax.set_title(f'Normalized IDP vs {x_label}', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Car count vs IDP (normalized) plot saved to: {output_path}")


def plot_estimated_vs_observed(analyzers_list, output_path='estimated_vs_observed.png'):
    """
    Create scatter plots comparing estimated vs observed car count changes.
    Shows how well regression predictions match actual observations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('Model Predictions vs Actual Observations\n(Each point = one report)',
                 fontsize=14, fontweight='bold')

    est_abs = np.array([a.est_diff_abs for a in analyzers_list])
    obs_abs = np.array([a.obs_diff_abs for a in analyzers_list])
    max_diff = np.array([a.max_diff for a in analyzers_list])

    configs = [
        (0, est_abs, obs_abs, 'Absolute Change (cars)', 'Estimated vs Observed', 'abs'),
        (1, max_diff, obs_abs, 'Max-Min vs Observed (cars)', 'Max-Min vs Observed Change', 'maxdiff_vs_obs'),
        (2, est_abs, max_diff, 'Max-Min Difference (cars)', 'Regression Est. vs Max-Min', 'maxdiff'),
    ]

    for idx, est_data, obs_data, label, subtitle, key in configs:
        ax = axes[idx]

        mask = ~(np.isnan(est_data) | np.isnan(obs_data))
        est_clean = est_data[mask]
        obs_clean = obs_data[mask]

        if len(est_clean) < 2:
            ax.text(0.5, 0.5, 'Insufficient valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{label}\n(No valid data)", fontweight='bold')
            continue

        ax.scatter(est_clean, obs_clean, s=100, alpha=0.7, edgecolors='black', linewidth=1.5, label='Reports')

        min_val = min(est_clean.min(), obs_clean.min())
        max_val = max(est_clean.max(), obs_clean.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--',
               linewidth=2, alpha=0.7, label='Perfect prediction (y=x)')

        r, p = stats.pearsonr(est_clean, obs_clean)
        r2 = r ** 2

        slope, intercept = np.polyfit(est_clean, obs_clean, 1)
        fit_line = np.linspace(est_clean.min(), est_clean.max(), 100)
        fit_vals = slope * fit_line + intercept
        ax.plot(fit_line, fit_vals, 'b-', linewidth=2, alpha=0.7,
               label=f'Linear fit (R2={r2:.3f})')

        mae = np.mean(np.abs(est_clean - obs_clean))
        rmse = np.sqrt(np.mean((est_clean - obs_clean)**2))

        stats_text = (
            f"Pearson r = {r:+.3f} (p={p:.4f})\n"
            f"R2 = {r2:.3f}\n"
            f"MAE = {mae:.2e}\n"
            f"RMSE = {rmse:.2e}\n"
            f"n = {len(est_clean)}"
        )

        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

        if key == 'maxdiff':
            ax.set_xlabel('Estimated Change (regression)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Max-Min Difference (temporal)', fontsize=11, fontweight='bold')
        elif key == 'maxdiff_vs_obs':
            ax.set_xlabel('Max-Min Difference (temporal)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Observed Change (actual)', fontsize=11, fontweight='bold')
        else:
            ax.set_xlabel('Estimated Change (regression)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Observed Change (actual)', fontsize=11, fontweight='bold')

        ax.set_title(f"{label}\n{subtitle}", fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Estimated vs Observed plot saved to: {output_path}")


def analyze_idp_correlations(analyzers_list, idp_values, population_baseline, output_dir='output'):
    """
    Main entry point: calculate, print, plot, and export IDP correlations.

    Parameters:
    -----------
    analyzers_list : list
        List of TrendAnalyzer objects
    idp_values : list
        IDP value for each report (same order as analyzers_list)
    population_baseline : list
        Population baseline for each city (same order as analyzers_list)
    output_dir : str
        Directory to save outputs

    Returns:
    --------
    dict : Results dictionary
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = calculate_idp_correlations(analyzers_list, idp_values, population_baseline)
    print_idp_correlations(results, f"{output_dir}/idp_correlations_summary.txt")
    plot_idp_correlations(results, f"{output_dir}/idp_correlations.png")

    plot_absolute_vs_idp(results, f"{output_dir}/absolute_vs_idp.png")
    plot_car_count_vs_idp_absolute(results, f"{output_dir}/car_count_vs_idp_absolute.png")

    plot_norm_vs_norm_idp(results, f"{output_dir}/norm_vs_norm_idp.png")
    plot_car_count_vs_idp_normalized(results, f"{output_dir}/car_count_vs_idp_normalized.png")

    export_idp_correlations(results, f"{output_dir}/idp_correlations.csv")

    plot_estimated_vs_observed(analyzers_list, f"{output_dir}/estimated_vs_observed.png")

    return results


def export_analyzers_summary(analyzers_list, output_path='analyzers_summary.tsv'):
    """
    Export key metrics from all analyzers to a summary TSV file.

    Parameters:
    -----------
    analyzers_list : list
        List of TrendAnalyzer objects
    output_path : str
        Path to save the summary TSV

    Returns:
    --------
    pandas.DataFrame : Summary dataframe
    """
    rows = []
    for a in analyzers_list:
        row = {
            'indx': a.indx,
            'city': a.city_name,
            'first_period': a.obs_start_period if hasattr(a, 'obs_start_period') else a.periods[0],
            'last_period': a.obs_end_period if hasattr(a, 'obs_end_period') else a.periods[-1],
            'n_periods': len(a.periods),
            'n_valid_obs': a.n_valid_obs,
            'location_level': a.location_level,

            'baseline': a.baseline,
            'idp': a.idp,
            'population_admin1': a.population_baseline,
            'idp_norm_admin1': a.idp_norm,

            'slope': a.slope,
            'intercept': a.intercept,
            'r_squared': a.r_value ** 2,
            'r_value': a.r_value,
            'p_value': a.p_value,
            'trend_direction': a.trend_direction,

            'obs_diff_abs': a.obs_diff_abs,
            'obs_diff_abs_pct': a.obs_diff_abs_pct,
            'obs_diff_norm': a.obs_diff_norm,
            'obs_diff_norm_pct': a.obs_diff_norm_pct,

            'est_diff_abs': a.est_diff_abs,
            'est_diff_abs_pct': a.est_diff_abs_pct,
            'est_diff_norm': a.est_diff_norm,
            'est_diff_norm_pct': a.est_diff_norm_pct,

            'max_diff': a.max_diff,
            'max_diff_abs': a.max_diff_abs,
            'max_diff_norm': a.max_diff_norm,
            'max_period': a.max_period,
            'min_period': a.min_period,

            'norm_slope': a.norm_slope,
            'norm_r_squared': a.norm_r_squared,
            'norm_p_value': a.norm_p_value,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Analyzers summary exported to: {output_path}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="IDP - Car Count Correlation")
    parser.add_argument("--inp", type=str, required=True, help="Path to IDP reports TSV file")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory")
    return parser.parse_args()


# ============================================================================
# MAIN: Loop over all reports + overall correlation at the end
# ============================================================================

if __name__ == "__main__":

    args = parse_args()
    ou_dir = args.outdir
    directory_path = Path(ou_dir)
    directory_path.mkdir(parents=True, exist_ok=True)

    car_pop_baseline_files = '/export/sc2/nalemadi/Projects/IDP_SYR/grids_output_worldpopbased/100m/prewar_baseline_cars_count_grids/'
    all_entries = os.listdir(car_pop_baseline_files)
    all_entries.sort()

    baseline_car_pop_dict = {}
    for car_pop_baseline_file in all_entries:
        if car_pop_baseline_file.endswith(".gpkg"):
            gdf = gpd.read_file(car_pop_baseline_files + car_pop_baseline_file)
            car_pop_mean = int(round(gdf['mean_Ncars'].sum(), 0))
            baseline_car_pop_dict[car_pop_baseline_file.split('_baseline.gpkg')[0]] = car_pop_mean

    df = pd.read_csv(args.inp, sep='\t')
    df['baseline_car_pop'] = df.apply(get_baseline_car_pop, axis=1)

    print("Population baselines loaded:")
    for city, pop in population_dict.items():
        print(f"  {city}: {pop:,}")

    df['baseline_pop'] = df.apply(get_baseline_pop, axis=1)

    df['idp_num'] = df['#IDP'].apply(parse_idp_number)
    df['idp_abs'] = df['idp_num'].abs()

    df_ = df[['Year', 'Period', 'City', 'location level', 'Report Variation', '#IDP', 'idp_num', 'idp_abs', 'baseline_pop', 'CarCount Variation', 'car_count_diff', 'car_count_Ddiff', 'baseline_car_pop', 'car count']]

    # dealing with some cases where the population baseline is empty
    condition1 = df_['City'] == 'Afrin'
    df_.loc[condition1, 'baseline_pop'] = 78216.08

    condition2 = df_['City'] == 'Al-bab'
    df_.loc[condition2, 'baseline_pop'] = 135972.30

    # ===============================================================================================
    # ── 0. Filter Reports before processing ──────────────────────────

    df_ = df_[df_['City'] != 'Afrin']
    df_ = df_[df_['City'] != 'Al-bab']

    df_ = df_[df_['idp_abs'] > 100.0]

    df_.to_csv(f'{ou_dir}/check_df_.csv', index=True)
    # ===============================================================================================

    analyzers_list = []

    for idx, row in df_.iterrows():
        car_count = json.loads(row['car count'])
        valid_count = sum(1 for x in car_count['car_counts'] if x == x and x is not None)
        if valid_count < 2:
            print("Skipping...")
            continue

        city_dict = {
            "indx": idx,
            "city": row['City'],
            "car_count_diff": row['car_count_diff'],
            "baseline": row['baseline_car_pop'],
            "periods": car_count['period'],
            "car_counts": car_count['car_counts'],
            "norm_car_counts": [x/row['baseline_car_pop'] if x is not None else None for x in car_count['car_counts']],
            "idp": row['idp_num'],
            "population_baseline": row.get('baseline_pop', float('nan')),
            "location_level": row["location level"]
        }
        analyzer = analyze_from_dict(city_dict, output_dir=ou_dir)

        analyzers_list.append(analyzer)

        summary_stats = analyzer.get_statistical_summary()
        print(f"\nAnalysis complete!")
        print(f"  Trend:       {summary_stats['trend_direction']}")
        print(f"  Slope:       {summary_stats['slope']:.2f} cars/month")
        print(f"  Significant: {'Yes' if summary_stats['is_significant'] else 'No'}")

    print(f"\n{'='*80}")
    print(f"ALL REPORTS DONE ({len(analyzers_list)} reports analysed)")
    print(f"{'='*80}")

    # ── Create summary DataFrame with estimated differences ──────────────
    print("\nCreating summary DataFrame with estimated differences...")

    summary_data = []
    for analyzer in analyzers_list:
        summary_data.append({
            'indx': analyzer.indx,
            'city': analyzer.city_name,
            'n_periods': len(analyzer.df),
            'first_period': analyzer.obs_start_period if hasattr(analyzer, 'obs_start_period') else analyzer.periods[0],
            'last_period': analyzer.obs_end_period if hasattr(analyzer, 'obs_end_period') else analyzer.periods[-1],
            'baseline': analyzer.baseline,

            'obs_diff_abs': analyzer.obs_diff_abs,
            'obs_diff_abs_pct': analyzer.obs_diff_abs_pct,
            'obs_diff_norm': analyzer.obs_diff_norm,
            'obs_diff_norm_pct': analyzer.obs_diff_norm_pct,

            'est_diff_abs': analyzer.est_diff_abs,
            'est_diff_abs_pct': analyzer.est_diff_abs_pct,
            'est_diff_norm': analyzer.est_diff_norm,
            'est_diff_norm_pct': analyzer.est_diff_norm_pct,
            
            'max_diff': analyzer.max_diff,
            'max_diff_abs': analyzer.max_diff_abs,
            'max_period': analyzer.max_period,
            'min_period': analyzer.min_period,

            'slope': analyzer.slope,
            'intercept': analyzer.intercept,
            'r_squared': analyzer.r_value ** 2,
            'p_value': analyzer.p_value,
            'trend_direction': analyzer.trend_direction,

            'idp': analyzer.idp,
            'population_baseline_admin1': analyzer.population_baseline,
            'idp_norm_admin1': analyzer.idp_norm,
        })

    summary_df = pd.DataFrame(summary_data)

    summary_tsv_path = f'{ou_dir}/estimated_differences_summary.tsv'
    summary_df.to_csv(summary_tsv_path, sep='\t', index=False, float_format='%.6f')
    print(f"Summary TSV saved to: {summary_tsv_path}")
    print(f"  Contains {len(summary_df)} reports with estimated differences")

    summary_csv_path = f'{ou_dir}/estimated_differences_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f')
    print(f"Summary CSV saved to: {summary_csv_path}")

    print(f"\n{'='*80}")
    print("Exporting summary metrics from all analyzers...")
    print(f"{'='*80}")

    summary_df = export_analyzers_summary(
        analyzers_list,
        output_path=f'{ou_dir}/analyzers_summary.tsv'
    )

    print(f"Summary contains {len(summary_df)} cities with metrics:")
    print(f"  - Observed differences (obs_diff_abs, obs_diff_abs_pct)")
    print(f"  - Estimated differences (est_diff_abs, est_diff_abs_pct)")
    print(f"  - Max-min differences (max_diff)")
    print(f"  - Regression statistics (slope, r_squared, p_value)")
    print(f"  - IDP and population data (admin1 baseline)")

    # ── IDP correlation analysis ──────────────────────────────────────────
    idp_values = []
    population_baselines = []
    for a in analyzers_list:
        idp_values.append(a.idp)
        population_baselines.append(a.population_baseline)

    idp_arr = np.array(idp_values)
    pop_arr = np.array(population_baselines)
    idp_norm_arr = np.array([a.idp_norm for a in analyzers_list])

    print(f"\n{'='*80}")
    print("IDP DATA DIAGNOSTIC")
    print(f"{'='*80}")
    print(f"Total reports: {len(analyzers_list)}")
    print(f"Valid IDP values: {np.sum(~np.isnan(idp_arr))} / {len(analyzers_list)}")
    print(f"Valid population: {np.sum(~np.isnan(pop_arr))} / {len(analyzers_list)}")
    print(f"Valid IDP_norm:   {np.sum(~np.isnan(idp_norm_arr))} / {len(analyzers_list)}")

    print(f"\nFirst 5 samples:")
    print(f"{'City':<15} {'IDP':>12} {'Population':>12} {'IDP_norm':>12}")
    print("-" * 55)
    for i in range(min(5, len(analyzers_list))):
        a = analyzers_list[i]
        print(f"{a.city_name:<15} {a.idp:>12.0f} {a.population_baseline:>12.0f} {a.idp_norm:>12.6f}")

    if np.sum(~np.isnan(idp_norm_arr)) == 0:
        print("\nWARNING: All IDP normalized values are nan!")
        print("   Your df_['population_baseline'] column is all nan or 0")
        print("   You need to populate it from your population CSV")

    print(f"\n{'='*80}")
    print("Running IDP correlation analysis...")
    print(f"{'='*80}")

    idp_results = analyze_idp_correlations(
        analyzers_list=analyzers_list,
        idp_values=idp_values,
        population_baseline=population_baselines,
        output_dir=ou_dir
    )

    print("\nAll analyses complete!")
    print(f"\nOutput files generated:")
    print(f"  - Individual city reports: {ou_dir}/[indx]_[city]_*")
    print(f"  - IDP correlations: {ou_dir}/idp_correlations.*")
