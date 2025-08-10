import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime 
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import matplotlib.dates as mdates
import scipy.signal
import matplotlib.cm as cm
from itertools import cycle, islice
import matplotlib.colors as mcolors
warnings.filterwarnings('ignore')

class WeightedLocationPlotter:
    def __init__(self, input_dir, output_dir=None):
        self.input_dir = input_dir
        self.output_dir = output_dir or os.path.join(input_dir, "plots")
        
        # Create output directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['font.size'] = 10
        
        print(f"WeightedLocationPlotter initialized")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Note: This plotter displays pre-smoothed data from the output files (no additional smoothing applied)")

    # Find each subtype file in the input directory 
    def find_subtype_files(self):
        pattern = os.path.join(self.input_dir, "all_countries_*.tsv")
        files = glob.glob(pattern)
        
        # To extract the subtype names from filenames | mapping of each subtype to its own designated file
        subtype_files = {}
        for file in files:
            filename = os.path.basename(file)
            if filename.startswith("all_countries_") and filename.endswith(".tsv"):
                # Extract subtype from filename
                subtype = filename.replace("all_countries_", "").replace(".tsv", "")
                subtype_files[subtype] = file
        
        print(f"Found {len(subtype_files)} subtype files:")
        for subtype, file in subtype_files.items():
            print(f"  {subtype}: {file}")
        
        return subtype_files
    
    def load_and_validate_data(self, file_path):
        try:
            df = pd.read_csv(file_path, sep='\t')

            # Standardize column names (if there are any mismatch in column labelling)
            col_rename = {
                '#FluNet Cases/#GISAID Samples': '#FluNet_Cases/#GISAID_Samples',
                'Smoothed_#Cases/#Samples': 'Smoothed_#FluNet_Cases/#GISAID_Samples'
            }
            df = df.rename(columns=col_rename)

            # Check the required columns to plot the graphs (per subtype)
            required_columns = ['Collection_Year_Month', 'GISAID_Country', '#FluNet_Cases/#GISAID_Samples', 'Smoothed_#FluNet_Cases/#GISAID_Samples']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns in {file_path}: {missing_columns}")
                print(f"Available columns: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Convert Collection_Year_Month to datetime for proper sorting
            df['Collection_Year_Month'] = pd.to_datetime(df['Collection_Year_Month'] + '-01')

            # Clean data
            df = df.dropna(subset=['Collection_Year_Month', 'GISAID_Country', '#FluNet_Cases/#GISAID_Samples'])
            df = df[df['#FluNet_Cases/#GISAID_Samples'] >= 0]  # Remove negative values
            # Sort by country and Collection_Year_Month

            if (
                isinstance(df, pd.DataFrame)
                and not df.empty
                and 'GISAID_Country' in df.columns
                and 'Collection_Year_Month' in df.columns
            ):
                df = df.sort_values(by=['GISAID_Country', 'Collection_Year_Month'])

            # Print countries count safely
            gisaid_country_col = df['GISAID_Country']
            if not isinstance(gisaid_country_col, pd.Series):
                gisaid_country_col = pd.Series(gisaid_country_col)

            print(f"Countries: {gisaid_country_col.nunique()}")
            print(f"Loaded {len(df)} records from {file_path}")
            print(f"Date range: {df['Collection_Year_Month'].min()} to {df['Collection_Year_Month'].max()}")
            return df
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def calculate_accuracy_metrics(self, df):
        # Calculate accuracy metrics for both original and smoothed data (in plotted graphs)
        metrics = {}
        for country in df['GISAID_Country'].unique():
            country_data = df[df['GISAID_Country'] == country].copy()
            if len(country_data) < 2: # If less than 2 data points, accuracy metrics cannot be calculated
                continue

            # Get original weighted locations (by extracting the values of initial ratio)
            original_values = country_data['#FluNet_Cases/#GISAID_Samples'].values

            # Get smoothed values from the data (directly from the smoothed column)
            if 'Smoothed_#FluNet_Cases/#GISAID_Samples' in country_data.columns:
                smoothed_values = country_data['Smoothed_#FluNet_Cases/#GISAID_Samples'].values

            else:
                # If not available, skip smoothing
                smoothed_values = original_values

            # Calculate metrics for both original and smoothed data
            metrics[country] = {
                'original': {
                    'mean': np.mean(original_values),
                    'std': np.std(original_values),
                    'min': np.min(original_values),
                    'max': np.max(original_values),
                    'trend': np.polyfit(range(len(original_values)), original_values, 1)[0],
                    'fluctuation': np.std(original_values) / np.mean(original_values) if np.mean(original_values) > 0 else 0
                },
                'smoothed': {
                    'mean': np.mean(smoothed_values),
                    'std': np.std(smoothed_values),
                    'min': np.min(smoothed_values),
                    'max': np.max(smoothed_values),
                    'trend': np.polyfit(range(len(smoothed_values)), smoothed_values, 1)[0]
                },
                'count': len(original_values),
                'smoothing_effect': np.std(original_values) - np.std(smoothed_values)  # Reduction in volatility
            }
        return metrics

    def create_line_plot(self, df, subtype, metrics):
        # To create two separate line plots for a specific subtype: one for monthly smoothed, one for 200-day smoothed data
        # Load high accuracy country mappings to ensure ALL 173 countries are included
        try:
            mapping_file = os.path.join(self.input_dir, 'high_accuracy_country_mappings.tsv')
            
            if os.path.exists(mapping_file):
                mapping_df = pd.read_csv(mapping_file, sep='\t')
                # Check available columns and use appropriate one
                if 'GISAID_Country' in mapping_df.columns:
                    mapped_countries = set(mapping_df['GISAID_Country'].dropna().unique())
                elif 'Country' in mapping_df.columns:
                    mapped_countries = set(mapping_df['Country'].dropna().unique())
                elif 'country' in mapping_df.columns:
                    mapped_countries = set(mapping_df['country'].dropna().unique())
                else:
                    # Use first column as fallback
                    first_col = mapping_df.columns[0]
                    mapped_countries = set(mapping_df[first_col].dropna().unique())
                    print(f"Using column '{first_col}' as country column")
                print(f"Loaded {len(mapped_countries)} countries from {mapping_file}")
            else:
                print(f"Warning: {mapping_file} not found. Using all countries from data.")
                mapped_countries = None
        except Exception as e:
            print(f"Error loading country mappings: {e}. Using all countries from data.")
            mapped_countries = None
        
        # Filter data: exclude 'Unknown' and ensure we include ALL mapped countries present in data
        df_filtered = df[df['GISAID_Country'] != 'Unknown'].copy()
        
        # Get countries that are both in our data AND in the mapping file (if available)
        data_countries = set(df_filtered['GISAID_Country'].unique())
        
        if mapped_countries is not None:
            # Include countries that are in BOTH the data and the mapping file
            countries_to_include = data_countries.intersection(mapped_countries)
            missing_from_data = mapped_countries - data_countries
            missing_from_mapping = data_countries - mapped_countries
            
            print(f"Countries in data: {len(data_countries)}")
            print(f"Countries in mapping file: {len(mapped_countries)}")
            print(f"Countries to include in plot: {len(countries_to_include)}")
            if missing_from_data:
                print(f"Countries in mapping but missing from data ({len(missing_from_data)}): {sorted(list(missing_from_data))[:10]}{'...' if len(missing_from_data) > 10 else ''}")
            if missing_from_mapping:
                print(f"Countries in data but not in mapping ({len(missing_from_mapping)}): {sorted(list(missing_from_mapping))}")
            
            # Filter dataframe to include only mapped countries
            df = df_filtered[df_filtered['GISAID_Country'].isin(countries_to_include)].copy()
            countries = sorted(list(countries_to_include))  # Ensure consistent ordering
        else:
            # Fallback: use all countries except 'Unknown'
            df = df_filtered
            countries = sorted(df['GISAID_Country'].unique())
            print(f"Using all {len(countries)} countries from data (mapping file not available)")
        
        print(f"Final countries count for plotting: {len(countries)}")

        # Use a less vibrant, less saturated, but distinguishable color palette
        base_colors = list(plt.get_cmap('Set2')(np.linspace(0,1,8))) \
                    + list(plt.get_cmap('Pastel1')(np.linspace(0,1,9))) \
                    + list(plt.get_cmap('Pastel2')(np.linspace(0,1,8))) \
                    + list(plt.get_cmap('tab20c')(np.linspace(0,1,20))) \
                    + list(plt.get_cmap('Accent')(np.linspace(0,1,8)))
        
        # If not enough colors, cycle through the palette
        pastel_colors = [mcolors.to_hex(c[:3]) for c in base_colors]
        from itertools import cycle, islice
        pastel_colors = list(islice(cycle(pastel_colors), len(countries)))
        country_color_map = {country: pastel_colors[i % len(pastel_colors)] for i, country in enumerate(countries)}
        highlight_colors = pastel_colors[:8]

        # Identify significant countries for Graph 1 (Monthly Smoothed) - based on peaks and fluctuation trends
        country_peaks_1 = {}
        for country in countries:
            country_data = df[df['GISAID_Country'] == country].copy()
            if 'Smoothed_#FluNet_Cases/#GISAID_Samples' in country_data.columns and len(country_data) >= 3:
                smoothed = country_data['Smoothed_#FluNet_Cases/#GISAID_Samples'].values
                peaks, _ = scipy.signal.find_peaks(smoothed)
                country_peaks_1[country] = len(peaks)
            else:
                country_peaks_1[country] = 0

        sorted_by_peaks_1 = sorted(country_peaks_1.items(), key=lambda x: (-x[1], x[0]))
        top_n_1 = 8 if len(sorted_by_peaks_1) > 8 else len(sorted_by_peaks_1)
        significant_countries_1 = [c for c, n in sorted_by_peaks_1[:top_n_1]]

        # Fallback: if all have 0 peaks, use top N by std
        if all(n == 0 for c, n in sorted_by_peaks_1):
            stds = {c: df[df['GISAID_Country'] == c]['Smoothed_#FluNet_Cases/#GISAID_Samples'].std() for c in countries}
            sorted_by_std = sorted(stds.items(), key=lambda x: -x[1])
            significant_countries_1 = [c for c, _ in sorted_by_std[:top_n_1]]

        # Identify significant countries for Graph 2 (200-day smoothed)
        country_peaks_2 = {}
        for country in countries:
            country_data = df[df['GISAID_Country'] == country].copy()
            if '200_days_smoothed_#FluNet_Cases/#GISAID_Samples' in country_data.columns and len(country_data) >= 3:
                smoothed = country_data['200_days_smoothed_#FluNet_Cases/#GISAID_Samples'].values
                peaks, _ = scipy.signal.find_peaks(smoothed)
                country_peaks_2[country] = len(peaks)
            else:
                country_peaks_2[country] = 0

        sorted_by_peaks_2 = sorted(country_peaks_2.items(), key=lambda x: (-x[1], x[0]))
        top_n_2 = 8 if len(sorted_by_peaks_2) > 8 else len(sorted_by_peaks_2)
        significant_countries_2 = [c for c, n in sorted_by_peaks_2[:top_n_2]]

        # Fallback: if all have 0 peaks, use top N by std
        if all(n == 0 for c, n in sorted_by_peaks_2):
            stds = {c: df[df['GISAID_Country'] == c]['200_days_smoothed_#FluNet_Cases/#GISAID_Samples'].std() for c in countries}
            sorted_by_std = sorted(stds.items(), key=lambda x: -x[1])
            significant_countries_2 = [c for c, _ in sorted_by_std[:top_n_2]]

        # Graph 1: Monthly smoothing (to include the raw and smoothed versions in one PNG with two subplots)
        if 'Smoothed_#FluNet_Cases/#GISAID_Samples' in df.columns and '#FluNet_Cases/#GISAID_Samples' in df.columns:
            fig, (ax_raw, ax_smooth) = plt.subplots(2, 1, figsize=(18, 14), sharex=True, dpi=60)
            
            # Store country peak data for legends
            country_peak_data_raw = {}
            country_peak_data_smooth = {}
            
            for i, country in enumerate(countries):
                country_data = df[df['GISAID_Country'] == country].copy()
                if len(country_data) == 0:
                    continue

                color = country_color_map[country]
                linewidth = 3  # Embolden the line's width
                markersize = 5
                alpha = 1.0
                label = country

                # Top subplot: raw
                ax_raw.plot(
                    country_data['Collection_Year_Month'],
                    country_data['#FluNet_Cases/#GISAID_Samples'],
                    marker='o',
                    label=label,
                    color=color,
                    linewidth=linewidth,
                    markersize=markersize,
                    alpha=alpha,
                    markeredgecolor='white',
                    markeredgewidth=0.3
                )

                # Find MAXIMUM value for peak identification (raw data)
                yvals_raw = country_data['#FluNet_Cases/#GISAID_Samples'].values
                if len(yvals_raw) > 0:
                    idx_max_raw = np.argmax(yvals_raw)
                    x_max_raw = country_data['Collection_Year_Month'].iloc[idx_max_raw]
                    y_max_raw = yvals_raw[idx_max_raw]
                    country_peak_data_raw[country] = (x_max_raw, y_max_raw)

                # Add country label at the end of the line (raw data) - ONLY END LABELS
                if len(country_data) > 0:
                    last_x = country_data['Collection_Year_Month'].iloc[-1]
                    last_y = country_data['#FluNet_Cases/#GISAID_Samples'].iloc[-1]
                    ax_raw.text(last_x, last_y, f'  {country}', fontsize=7, fontweight='bold', 
                               color=color, ha='left', va='center', alpha=0.9, zorder=5)
                    
                # Bottom subplot: smoothed
                ax_smooth.plot(
                    country_data['Collection_Year_Month'],
                    country_data['Smoothed_#FluNet_Cases/#GISAID_Samples'],
                    marker='o',
                    label=label,
                    color=color,
                    linewidth=linewidth,
                    markersize=markersize,
                    alpha=alpha,
                    markeredgecolor='white',
                    markeredgewidth=0.3
                )

                # Find MAXIMUM value for peak identification (smoothed data)
                yvals_smooth = country_data['Smoothed_#FluNet_Cases/#GISAID_Samples'].values
                if len(yvals_smooth) > 0:
                    idx_max_smooth = np.argmax(yvals_smooth)
                    x_max_smooth = country_data['Collection_Year_Month'].iloc[idx_max_smooth]
                    y_max_smooth = yvals_smooth[idx_max_smooth]
                    country_peak_data_smooth[country] = (x_max_smooth, y_max_smooth)

                # Add country label at the end of the line (smoothed data) - ONLY END LABELS
                if len(country_data) > 0:
                    last_x = country_data['Collection_Year_Month'].iloc[-1]
                    last_y = country_data['Smoothed_#FluNet_Cases/#GISAID_Samples'].iloc[-1]
                    ax_smooth.text(last_x, last_y, f'  {country}', fontsize=7, fontweight='bold', 
                                  color=color, ha='left', va='center', alpha=0.9, zorder=5)
                    
            # Top subplot formatting (raw) - NO AUTO-SCALING, ACCURATE Y-AXIS
            ax_raw.set_ylabel('Raw #FluNet Cases/#GISAID_Samples', fontsize=12, fontweight='bold', color='black')
            ax_raw.set_title(f'Raw Weighted Location (Before Smoothing) - {subtype}', fontsize=14, fontweight='bold')
            ax_raw.set_xlabel('Collection Year Month', fontsize=12, fontweight='bold', color='black')
            ax_raw.tick_params(axis='x', rotation=60, labelsize=9, colors='black')
            ax_raw.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.8)
            
            # Set exact y-axis limits based on actual data range - NO AUTO-SCALING
            raw_data_values = df['#FluNet_Cases/#GISAID_Samples'].values
            raw_data_values = raw_data_values[~np.isnan(raw_data_values)]  # Remove NaN values
            if len(raw_data_values) > 0:
                y_min_exact = np.min(raw_data_values)
                y_max_exact = np.max(raw_data_values)
                y_margin = (y_max_exact - y_min_exact) * 0.05  # 5% margin for labels
                ax_raw.set_ylim(max(0, y_min_exact - y_margin), y_max_exact + y_margin)
            else:
                ax_raw.set_ylim(0, 1)
                
            ax_raw.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_raw.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            # Bottom subplot formatting (smoothed) - NO AUTO-SCALING, ACCURATE Y-AXIS
            ax_smooth.set_xlabel('Collection Year Month', fontsize=12, fontweight='bold', color='black')
            ax_smooth.set_ylabel('Smoothed #FluNet Cases/#GISAID Samples', fontsize=12, fontweight='bold', color='black')
            ax_smooth.set_title(f'Smoothed Weighted Location (Pre-smoothed data from output files) - {subtype}', fontsize=14, fontweight='bold')
            ax_smooth.tick_params(axis='x', rotation=60, labelsize=9, colors='black')
            ax_smooth.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.8)
            
            # Set exact y-axis limits based on actual smoothed data range - NO AUTO-SCALING
            smooth_data_values = df['Smoothed_#FluNet_Cases/#GISAID_Samples'].values
            smooth_data_values = smooth_data_values[~np.isnan(smooth_data_values)]  # Remove NaN values
            if len(smooth_data_values) > 0:
                y_min_exact = np.min(smooth_data_values)
                y_max_exact = np.max(smooth_data_values)
                y_margin = (y_max_exact - y_min_exact) * 0.05  # 5% margin for labels
                ax_smooth.set_ylim(max(0, y_min_exact - y_margin), y_max_exact + y_margin)
            else:
                ax_smooth.set_ylim(0, 1)
                
            ax_smooth.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_smooth.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            # Custom legend: show ALL countries except 'Unknown', with color, country name, and MAXIMUM peak coordinates
            from matplotlib.lines import Line2D
            legend_elements = []

            for i, country in enumerate(countries):
                color = country_color_map[country]
                if country in country_peak_data_smooth:
                    x, y = country_peak_data_smooth[country]
                    legend_label = f"{country}  ({x.strftime('%Y-%m')}, {y:.2f})"
                else:
                    legend_label = f"{country}"
                legend_elements.append(Line2D([0], [0], color=color, lw=3, label=legend_label))

            ax_smooth.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=True, fancybox=True, shadow=True, title='Country (Peak at x, y)')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.08, hspace=0.22, right=0.85)  # Added right margin for end labels
            plot_filename1 = f"{subtype}_smoothed_monthly_weighted_location.png" # The formatting's label to save the output file
            plot_path1 = os.path.join(self.output_dir, plot_filename1)
            plt.savefig(plot_path1, dpi=90, bbox_inches='tight', facecolor='white', edgecolor='black', 
                       format='png', pil_kwargs={'optimize': True, 'compress_level': 9})
            plt.close(fig)
            print(f"Saved plot: {plot_path1}")

        # Graph 2: 200-day smoothing (raw and smoothed in one PNG with two subplots) - similar to the Graph 1, but data is imported upon the implementation of 200-days sliding window
        if '200_days_smoothed_#FluNet_Cases/#GISAID_Samples' in df.columns and '200_days_#FluNet_Cases/#GISAID_Samples' in df.columns:
            fig, (ax_raw, ax_smooth) = plt.subplots(2, 1, figsize=(18, 14), sharex=True, dpi=60)
            
            # Store country peak data for legends
            country_peak_data_raw_200 = {}
            country_peak_data_smooth_200 = {}
            
            for i, country in enumerate(countries):
                country_data = df[df['GISAID_Country'] == country].copy()
                if len(country_data) == 0:
                    continue
                color = country_color_map[country]
                linewidth = 3  # Embolden the line width
                markersize = 5
                alpha = 1.0
                label = country

                # Top subplot: raw
                ax_raw.plot(
                    country_data['Collection_Year_Month'],
                    country_data['200_days_#FluNet_Cases/#GISAID_Samples'],
                    marker='o',
                    label=label,
                    color=color,
                    linewidth=linewidth,
                    markersize=markersize,
                    alpha=alpha,
                    markeredgecolor='white',
                    markeredgewidth=0.3
                )

                # Find MAXIMUM value for peak identification (raw 200-day data)
                yvals_raw_200 = country_data['200_days_#FluNet_Cases/#GISAID_Samples'].values
                if len(yvals_raw_200) > 0:
                    idx_max_raw_200 = np.argmax(yvals_raw_200)
                    x_max_raw_200 = country_data['Collection_Year_Month'].iloc[idx_max_raw_200]
                    y_max_raw_200 = yvals_raw_200[idx_max_raw_200]
                    country_peak_data_raw_200[country] = (x_max_raw_200, y_max_raw_200)

                # Add country label at the end of the line (raw 200-day data) - ONLY END LABELS
                if len(country_data) > 0:
                    last_x = country_data['Collection_Year_Month'].iloc[-1]
                    last_y = country_data['200_days_#FluNet_Cases/#GISAID_Samples'].iloc[-1]
                    ax_raw.text(last_x, last_y, f'  {country}', fontsize=7, fontweight='bold', 
                               color=color, ha='left', va='center', alpha=0.9, zorder=5)
                    
                # Bottom subplot: smoothed
                ax_smooth.plot(
                    country_data['Collection_Year_Month'],
                    country_data['200_days_smoothed_#FluNet_Cases/#GISAID_Samples'],
                    marker='o',
                    label=label,
                    color=color,
                    linewidth=linewidth,
                    markersize=markersize,
                    alpha=alpha,
                    markeredgecolor='white',
                    markeredgewidth=0.3
                )

                # Find MAXIMUM value for peak identification (smoothed 200-day data)
                yvals_smooth_200 = country_data['200_days_smoothed_#FluNet_Cases/#GISAID_Samples'].values
                if len(yvals_smooth_200) > 0:
                    idx_max_smooth_200 = np.argmax(yvals_smooth_200)
                    x_max_smooth_200 = country_data['Collection_Year_Month'].iloc[idx_max_smooth_200]
                    y_max_smooth_200 = yvals_smooth_200[idx_max_smooth_200]
                    country_peak_data_smooth_200[country] = (x_max_smooth_200, y_max_smooth_200)

                # Add country label at the end of the line (smoothed 200-day data) - ONLY END LABELS
                if len(country_data) > 0:
                    last_x = country_data['Collection_Year_Month'].iloc[-1]
                    last_y = country_data['200_days_smoothed_#FluNet_Cases/#GISAID_Samples'].iloc[-1]
                    ax_smooth.text(last_x, last_y, f'  {country}', fontsize=7, fontweight='bold', 
                                  color=color, ha='left', va='center', alpha=0.9, zorder=5)
                    
            # Top subplot formatting (raw) - NO AUTO-SCALING, ACCURATE Y-AXIS
            ax_raw.set_ylabel('Raw 200-day #FluNet Cases/#GISAID_Samples', fontsize=12, fontweight='bold', color='black')
            ax_raw.set_title(f'Raw Weighted Location (200-day, Before Smoothing) - {subtype}', fontsize=14, fontweight='bold')
            ax_raw.set_xlabel('Collection Year Month', fontsize=12, fontweight='bold', color='black')
            ax_raw.tick_params(axis='x', rotation=60, labelsize=9, colors='black')
            ax_raw.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.8)
            
            # Set exact y-axis limits based on actual 200-day raw data range - NO AUTO-SCALING
            raw_200_data_values = df['200_days_#FluNet_Cases/#GISAID_Samples'].values
            raw_200_data_values = raw_200_data_values[~np.isnan(raw_200_data_values)]  # Remove NaN values
            if len(raw_200_data_values) > 0:
                y_min_exact = np.min(raw_200_data_values)
                y_max_exact = np.max(raw_200_data_values)
                y_margin = (y_max_exact - y_min_exact) * 0.05  # 5% margin for labels
                ax_raw.set_ylim(max(0, y_min_exact - y_margin), y_max_exact + y_margin)
            else:
                ax_raw.set_ylim(0, 1)
                
            ax_raw.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_raw.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            # Bottom subplot formatting (smoothed) - NO AUTO-SCALING, ACCURATE Y-AXIS
            ax_smooth.set_xlabel('Collection Year Month', fontsize=12, fontweight='bold', color='black')
            ax_smooth.set_ylabel('Smoothed 200-day #FluNet Cases/#GISAID_Samples', fontsize=12, fontweight='bold', color='black')
            ax_smooth.set_title(f'Smoothed Weighted Location (200-day, Pre-Smoothed Data) - {subtype}', fontsize=14, fontweight='bold')
            ax_smooth.tick_params(axis='x', rotation=60, labelsize=9, colors='black')
            ax_smooth.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.8)
            
            # Set exact y-axis limits based on actual 200-day smoothed data range - NO AUTO-SCALING
            smooth_200_data_values = df['200_days_smoothed_#FluNet_Cases/#GISAID_Samples'].values
            smooth_200_data_values = smooth_200_data_values[~np.isnan(smooth_200_data_values)]  # Remove NaN values
            if len(smooth_200_data_values) > 0:
                y_min_exact = np.min(smooth_200_data_values)
                y_max_exact = np.max(smooth_200_data_values)
                y_margin = (y_max_exact - y_min_exact) * 0.05  # 5% margin for labels
                ax_smooth.set_ylim(max(0, y_min_exact - y_margin), y_max_exact + y_margin)
            else:
                ax_smooth.set_ylim(0, 1)
                
            ax_smooth.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_smooth.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            # Legends: show ALL countries except 'Unknown', with color, country name, and MAXIMUM peak coordinates
            from matplotlib.lines import Line2D
            legend_elements = []
            for i, country in enumerate(countries):
                color = country_color_map[country]
                if country in country_peak_data_smooth_200:
                    x, y = country_peak_data_smooth_200[country]
                    legend_label = f"{country}  ({x.strftime('%Y-%m')}, {y:.2f})"
                else:
                    legend_label = f"{country}"
                legend_elements.append(Line2D([0], [0], color=color, lw=3, label=legend_label))
            ax_smooth.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=True, fancybox=True, shadow=True, title='Country (Peak at x, y)')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.08, hspace=0.22, right=0.85)  # Added right margin for end labels
            plot_filename2 = f"{subtype}_200day_smoothed_weighted_location.png"
            plot_path2 = os.path.join(self.output_dir, plot_filename2)
            plt.savefig(plot_path2, dpi=90, bbox_inches='tight', facecolor='white', edgecolor='black',
                       format='png', pil_kwargs={'optimize': True, 'compress_level': 9})
            plt.close(fig)
            print(f"Saved plot: {plot_path2}")
            
    def create_summary_statistics(self, df, subtype, metrics):
        """Create summary statistics table for both original and smoothed data"""
        summary_data = []
        
        for country, country_metrics in metrics.items():
            # Original data statistics
            summary_data.append({
                'subtype': subtype,
                'country': country,
                'data_type': 'original',
                'mean_weighted_location': country_metrics['original']['mean'],
                'std_weighted_location': country_metrics['original']['std'],
                'min_weighted_location': country_metrics['original']['min'],
                'max_weighted_location': country_metrics['original']['max'],
                'trend_slope': country_metrics['original']['trend'],
                'data_points': country_metrics['count'],
                'smoothing_effect': 0.0
            })
            
            # Smoothed data statistics
            summary_data.append({
                'subtype': subtype,
                'country': country,
                'data_type': 'smoothed',
                'mean_weighted_location': country_metrics['smoothed']['mean'],
                'std_weighted_location': country_metrics['smoothed']['std'],
                'min_weighted_location': country_metrics['smoothed']['min'],
                'max_weighted_location': country_metrics['smoothed']['max'],
                'trend_slope': country_metrics['smoothed']['trend'],
                'data_points': country_metrics['count'],
                'smoothing_effect': country_metrics['smoothing_effect']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by country and data type
        summary_df = summary_df.sort_values(['country', 'data_type'])
        
        return summary_df
    
    def generate_accuracy_report(self, all_metrics):
        # (Optional) Generate accuracy report with pre-smoothed data analysis
        report = []
        report.append("=" * 80)
        report.append("WEIGHTED LOCATION TRENDS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Data source information
        report.append(f"Data Source Information:")
        report.append(f"  Source: Pre-Smoothed Data")
        report.append(f"  Note: No additional smoothing applied by this plotter")
        report.append(f"  Smoothing: Applied during data processing in final_revised_all_country_script.py")
        report.append("")
        
        # Overall statistics
        total_countries = set()
        total_subtypes = len(all_metrics)
        
        for subtype, metrics in all_metrics.items():
            total_countries.update(metrics.keys())
        
        report.append(f"Total subtypes analyzed: {total_subtypes}")
        report.append(f"Total unique countries: {len(total_countries)}")
        report.append("")
        
        # Subtype-specific statistics
        report.append("SUBTYPE ANALYSIS (Original vs Smoothed):")
        report.append("-" * 50)
        
        for subtype, metrics in all_metrics.items():
            if not metrics:
                continue
            
            original_means = [m['original']['mean'] for m in metrics.values()]
            smoothed_means = [m['smoothed']['mean'] for m in metrics.values()]
            original_stds = [m['original']['std'] for m in metrics.values()]
            smoothed_stds = [m['smoothed']['std'] for m in metrics.values()]
            smoothing_effects = [m['smoothing_effect'] for m in metrics.values()]
            
            report.append(f"\n{subtype}:")
            report.append(f"  Countries: {len(metrics)}")
            report.append(f"  Original - Mean: {np.mean(original_means):.3f} ± {np.std(original_means):.3f}")
            report.append(f"  Smoothed - Mean: {np.mean(smoothed_means):.3f} ± {np.std(smoothed_means):.3f}")
            report.append(f"  Average volatility reduction: {np.mean(smoothing_effects):.3f}")
            report.append(f"  Smoothing effectiveness: {(np.mean(smoothing_effects)/np.mean(original_stds)*100):.1f}% volatility reduction")
            
            # Countries with highest smoothing effect
            sorted_smoothing = sorted(metrics.items(), key=lambda x: x[1]['smoothing_effect'], reverse=True)
            report.append(f"  Most smoothed countries:")
            for country, country_metrics in sorted_smoothing[:3]:
                effect = country_metrics['smoothing_effect']
                original_std = country_metrics['original']['std']
                reduction_pct = (effect/original_std*100) if original_std > 0 else 0
                report.append(f"    {country}: {reduction_pct:.1f}% volatility reduction")
        
        return "\n".join(report)
    
    def plot_all_subtypes(self):
        """Main function to plot all subtypes with pre-smoothed data"""
        print("=== STARTING WEIGHTED LOCATION TREND PLOTTING ===")
        print("Displaying pre-smoothed data from output files (no additional smoothing applied)")
        
        # Find all subtype files
        subtype_files = self.find_subtype_files()
        
        if not subtype_files:
            print("No subtype files found!")
            return
        
        all_metrics = {}
        all_summaries = []
        
        # Process each subtype
        for subtype, file_path in subtype_files.items():
            print(f"\n=== PROCESSING {subtype} ===")
            
            # Load and validate data
            df = self.load_and_validate_data(file_path)
            
            if df.empty:
                print(f"No valid data for {subtype}")
                continue
            
            # Calculate metrics (including smoothed data)
            metrics = self.calculate_accuracy_metrics(df)
            all_metrics[subtype] = metrics
            
            # Create plot with both original and smoothed data
            plot_path = self.create_line_plot(df, subtype, metrics)
            
            # Create summary statistics
            summary_df = self.create_summary_statistics(df, subtype, metrics)
            all_summaries.append(summary_df)
            
            # Save summary statistics
            summary_file = os.path.join(self.output_dir, f"{subtype}_summary_statistics_smoothed.tsv")
            summary_df.to_csv(summary_file, sep='\t', index=False)
            print(f"Saved summary statistics: {summary_file}")
        
        # Combine all summaries
        if all_summaries:
            combined_summary = pd.concat(all_summaries, ignore_index=True)
            combined_file = os.path.join(self.output_dir, "all_subtypes_summary_smoothed.tsv")
            combined_summary.to_csv(combined_file, sep='\t', index=False)
            print(f"Saved combined summary: {combined_file}")
        
        # Generate accuracy report
        if all_metrics:
            report = self.generate_accuracy_report(all_metrics)
            report_file = os.path.join(self.output_dir, "trends_analysis_report_smoothed.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Saved analysis report: {report_file}")
            
            # Print preview
            print("\n=== ANALYSIS REPORT PREVIEW ===")
            print(report[:2000] + "..." if len(report) > 2000 else report)
        
        print(f"\n=== PLOTTING COMPLETE ===")
        print(f"Processed {len(subtype_files)} subtypes with pre-smoothed data")
        print(f"Plots and summaries saved to: {self.output_dir}")


def main():
    """Main function to run the plotting analysis with pre-smoothed data"""
    print("Weighted Location Trend Plotter")
    print("This will create line graphs showing both original and pre-smoothed trends from output files.")
    
    # Set input directory - change it according to the generated output file of the data's processing script
    input_dir = r"/Users/willnath/Desktop/analysis_scripts/all_countries_analysis_trial_crontab/" 
    
    # Initialize plotter (no additional smoothing applied)
    plotter = WeightedLocationPlotter(input_dir)
    
    # Generate all plots (run through the above-mentioned script)
    plotter.plot_all_subtypes() 


if __name__ == "__main__":
    main()