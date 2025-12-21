"""
Selected Best Plots - Patent Dataset Analysis
This file contains selected and improved plots for patent dataset visualization.
These plots have enhanced styling, better colors, and improved readability.
Includes general plots, AI analysis plots, and academic analysis plots.

Mariana Soares, December 2025 - Virginia Tech
CS5764 - Information Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy import stats
from prettytable import PrettyTable
from data_processing import READABLE_NAMES, get_cpc_major_category, get_us_region, flag_ai_patents, identify_academic_institutions

# Configure matplotlib for LaTeX-style rendering
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.labelcolor': 'darkred',   # Use dark red for x and y axis labels
    'axes.titlesize': 16,
    'axes.titlecolor': 'blue',      # Use blue for plot titles
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'text.usetex': False,  # Set to True if LaTeX is installed
    'mathtext.fontset': 'stix',  # LaTeX-like math font
})

# Setting up the plotting style with LaTeX-like appearance
sns.set_style("whitegrid", {
    'grid.color': '0.9',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
})


def sample_df(df, n=50000, random_state=42):
    """
    Sample a maximum of n rows for heavy plots (pairplot, swarm, etc.).
    This helps improve performance when plotting large datasets.
    
    Args:
        df: DataFrame to sample
        n: Maximum number of rows to return
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame if original has more than n rows, otherwise original DataFrame
    """
    if len(df) > n:
        return df.sample(n=n, random_state=random_state)
    return df


def get_label(col_name, readable_names):
    """
    Get readable label for a column name.
    
    Args:
        col_name: Column name to get label for
        readable_names: Dictionary mapping column names to readable labels
    
    Returns:
        Readable label if available, otherwise returns the column name
    """
    return readable_names.get(col_name, col_name) if readable_names else col_name


def apply_axis_colors(ax):
    """
    Apply consistent color styling to a single matplotlib Axes:
    - Blue for titles
    - Dark red for x and y axis labels
    """
    if ax is None:
        return
    # Axis labels
    if ax.xaxis.label:
        ax.xaxis.label.set_color('darkred')
    if ax.yaxis.label:
        ax.yaxis.label.set_color('darkred')
    # Axis title
    title_text = ax.get_title()
    if title_text:
        ax.title.set_color('blue')


def plot_feature_summary_table(df, readable_names, max_rows_for_plot=25):
    """
    Create a PrettyTable summary of numerical and categorical features and
    render a compact version as a matplotlib table.

    The PrettyTable (full list of features) is printed to the console,
    while the matplotlib table visualizes the first `max_rows_for_plot` rows.
    """
    if df is None or df.empty:
        print("Warning: DataFrame is empty. Cannot create feature summary table.")
        return

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    table = PrettyTable()
    table.field_names = [
        "Feature",
        "Type",
        "Non-Null",
        "Missing",
        "Unique",
        "Summary / Example",
    ]

    rows_for_plot = []

    # Helper to safely format long strings
    def _truncate(text, max_len=40):
        text = str(text)
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    # Numeric features
    for col in numeric_cols:
        s = df[col]
        non_null = s.notna().sum()
        missing = s.isna().sum()
        unique = s.nunique(dropna=True)
        if non_null > 0:
            summary = (
                f"mean={s.mean():.2f}, std={s.std():.2f}, "
                f"min={s.min():.2f}, max={s.max():.2f}"
            )
        else:
            summary = "All values missing"

        row = [
            get_label(col, readable_names),
            "Numeric",
            f"{non_null:,}",
            f"{missing:,}",
            f"{unique:,}",
            summary,
        ]
        table.add_row(row)
        rows_for_plot.append(row)

    # Categorical / boolean features
    for col in categorical_cols:
        s = df[col]
        non_null = s.notna().sum()
        missing = s.isna().sum()
        unique = s.nunique(dropna=True)
        if non_null > 0:
            vc = s.value_counts(dropna=True)
            top_val = vc.index[0]
            top_cnt = vc.iloc[0]
            summary = f"top={_truncate(top_val)}, freq={top_cnt:,}"
        else:
            summary = "All values missing"

        row = [
            get_label(col, readable_names),
            "Categorical",
            f"{non_null:,}",
            f"{missing:,}",
            f"{unique:,}",
            summary,
        ]
        table.add_row(row)
        rows_for_plot.append(row)

    # Print full PrettyTable to console
    print("\n" + "=" * 70)
    print("Feature Summary: Numerical and Categorical Variables")
    print("=" * 70)
    print(table)

    # Create matplotlib table (limited number of rows for readability)
    if not rows_for_plot:
        print("No features found to display.")
        return

    rows_for_plot = rows_for_plot[:max_rows_for_plot]

    fig_height = max(4, 0.35 * len(rows_for_plot) + 1)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")

    mpl_table = ax.table(
        cellText=rows_for_plot,
        colLabels=table.field_names,
        loc="center",
        cellLoc="left",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9)
    mpl_table.auto_set_column_width(col=list(range(len(table.field_names))))

    plt.title(
        "Feature Summary (First "
        f"{min(len(rows_for_plot), max_rows_for_plot)} Features)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    # Apply consistent axis colors for all axes in the figure
    for ax in fig.axes:
        apply_axis_colors(ax)
    plt.show()


def remove_outliers_iqr(df, column, lower_bound=None, upper_bound=None):
    """
    Remove outliers from a DataFrame using the Interquartile Range (IQR) method.
    This is a general function that can be used for any numerical column.
    
    Args:
        df: The pandas DataFrame to filter
        column: Name of the column to check for outliers
        lower_bound: Optional lower bound multiplier (default: 1.5 * IQR below Q1)
        upper_bound: Optional upper bound multiplier (default: 1.5 * IQR above Q3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found. Returning original DataFrame.")
        return df.copy()
    
    # Calculate quartiles and IQR
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    # Set default bounds if not provided
    if lower_bound is None:
        lower_bound = 1.5
    if upper_bound is None:
        upper_bound = 1.5
    
    # Calculate bounds
    lower = q1 - lower_bound * iqr
    upper = q3 + upper_bound * iqr
    
    # Filter out outliers
    df_clean = df[(df[column] >= lower) & (df[column] <= upper)].copy()
    
    removed_count = len(df) - len(df_clean)
    if removed_count > 0:
        print(f"Removed {removed_count:,} outliers ({removed_count/len(df)*100:.2f}%) from '{column}' column")
    
    return df_clean


def remove_claim_outliers_iqr(df):
    """
    Remove outliers from num_claims column using IQR method.
    Convenience wrapper for remove_outliers_iqr() specifically for claims.
    
    Args:
        df: The pandas DataFrame to filter
    
    Returns:
        DataFrame with claim outliers removed
    """
    return remove_outliers_iqr(df, 'num_claims')


def load_and_prepare_data_for_plots(filter_us=False):
    """
    Load the dataset and prepare it for plotting.
    Loads from cleaned_patents.csv which contains all the processed data.
    Optimized for large files with progress indication.
    
    Args:
        filter_us: If True, filters to only US patents. If False, keeps all countries.
                  Default is False to allow international comparisons in plots.
    
    Returns:
        df: Processed DataFrame
        readable_names: Dictionary mapping column names to readable labels
    """
    try:
        print("Loading and preparing dataset for plotting...")
        print("This may take a moment for large files...")
        
        # Try to use current_dataset.csv first (usually faster, already processed)
        import os
        if os.path.exists('data/current_dataset.csv'):
            print("Loading from current_dataset.csv...")
            df = pd.read_csv('data/current_dataset.csv', low_memory=False)
        elif os.path.exists('data/cleaned_patents.csv'):
            print("Loading from cleaned_patents.csv...")
        df = pd.read_csv('data/cleaned_patents.csv', low_memory=False)
        else:
            raise FileNotFoundError("No dataset file found")
        
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Converting patent_date to datetime for any date operations
        if 'patent_date' in df.columns:
            print("Converting dates...")
        df['patent_date'] = pd.to_datetime(df['patent_date'], format='%Y-%m-%d', errors='coerce')
        
        # Create derived date columns if needed
        if 'patent_date' in df.columns:
            if 'filing_year' not in df.columns:
                df['filing_year'] = df['patent_date'].dt.year
            if 'filing_quarter' not in df.columns:
                df['filing_quarter'] = df['patent_date'].dt.quarter
            if 'year_quarter' not in df.columns:
            df['year_quarter'] = df['filing_year'].astype(str) + ' Q' + df['filing_quarter'].astype(str)
            if 'grant_date' not in df.columns:
                df['grant_date'] = df['patent_date']
            if 'grant_year' not in df.columns:
                df['grant_year'] = df['patent_date'].dt.year
            if 'grant_month' not in df.columns:
                df['grant_month'] = df['patent_date'].dt.month
        
        # Ensure grant_date is datetime if it exists
        if 'grant_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['grant_date']):
            df['grant_date'] = pd.to_datetime(df['grant_date'], errors='coerce')
        
        # Create derived features if missing
        print("Creating derived features...")
        
        # Create cpc_major if missing
        if 'cpc_major' not in df.columns and 'cpc_section' in df.columns:
            df['cpc_major'] = df['cpc_section'].apply(get_cpc_major_category)
        
        # Create us_region if missing
        if 'us_region' not in df.columns and 'disambig_state' in df.columns:
            df['us_region'] = df['disambig_state'].apply(get_us_region)
        
        # Create patent_complexity if missing
        if 'patent_complexity' not in df.columns:
            if 'num_claims' in df.columns and 'num_figures' in df.columns:
                df['patent_complexity'] = (df['num_claims'] + df['num_figures']) / 2
        
        # Create total_citations if missing
        if 'total_citations' not in df.columns:
            if 'citation_count' in df.columns:
                foreign_citations = df.get('foreign_citation_count', pd.Series(0, index=df.index)).fillna(0)
                us_citations = df['citation_count'] - foreign_citations
                us_citations = us_citations.clip(lower=0)
                df['total_citations'] = us_citations + foreign_citations
        
        # Create figures_per_claim if missing
        if 'figures_per_claim' not in df.columns:
            if 'num_claims' in df.columns and 'num_figures' in df.columns:
                df['figures_per_claim'] = np.where(
                    df['num_claims'] > 0,
                    df['num_figures'] / df['num_claims'],
                    0
                )
        
        # Create citation_per_claim if missing
        if 'citation_per_claim' not in df.columns:
            if 'num_claims' in df.columns and 'total_citations' in df.columns:
                df['citation_per_claim'] = np.where(
                    df['num_claims'] > 0,
                    df['total_citations'] / df['num_claims'],
                    0
                )
        
        # Flag AI patents if missing
        if 'is_ai' not in df.columns or 'ai_bucket' not in df.columns:
            df = flag_ai_patents(df)
        
        # Identify academic institutions if missing
        if 'is_academic' not in df.columns or 'assignee_type_academic' not in df.columns:
            df = identify_academic_institutions(df)
        
        # Create assignee_category if missing
        if 'assignee_category' not in df.columns and 'assignee_type' in df.columns:
            assignee_type_numeric = pd.to_numeric(df['assignee_type'], errors='coerce')
            df['assignee_type_clean'] = assignee_type_numeric.map({
                2.0: 'US Company',
                3.0: 'Foreign Company', 
                4.0: 'US Individual',
                5.0: 'Foreign Individual',
                6.0: 'US Government',
                7.0: 'Foreign Government',
                8.0: 'Country Government',
                9.0: 'State Government'
            })
            df['assignee_category'] = df['assignee_type_clean'].replace({
                'US Company': 'Corporate',
                'Foreign Company': 'Corporate',
                'US Individual': 'Individual', 
                'Foreign Individual': 'Individual',
                'US Government': 'Government',
                'Foreign Government': 'Government',
                'Country Government': 'Government',
                'State Government': 'Government'
            })
        
        # Filtering to US by default (except when comparing countries)
        if filter_us:
            df = df[df['disambig_country'] == 'US'].copy()
            print(f"Filtered to {len(df):,} US patents")
        else:
            print(f"Keeping all {len(df):,} patents (all countries)")
        
        if 'patent_date' in df.columns and df['patent_date'].notna().any():
        print(f"Date range: {df['patent_date'].min()} to {df['patent_date'].max()}")
        
        print("Dataset ready for plotting!")
        
        return df, READABLE_NAMES
    
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. Please run data_processing.py first to generate cleaned data.")
        print(f"Looking for: data/current_dataset.csv or data/cleaned_patents.csv")
        return None, None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def plot_patent_trends_by_technology(df, readable_names):
    """
    Plot 1 - Line Plot: Patent Trends by Technology Category (CPC Classification)
    Improved version with better colors, larger fonts, and reduced spacing.
    """
    # Filter out missing CPC data
    df_tech = df[df['cpc_major'].notna()].copy()
    tech_trends = df_tech.groupby(['year_quarter', 'cpc_major']).size().unstack(fill_value=0)
    
    # Sort quarters chronologically
    tech_trends = tech_trends.sort_index()
    
    # Use a better color palette - tab10 provides 10 distinct colors, then cycle if needed
    num_categories = len(tech_trends.columns)
    if num_categories <= 10:
        # Use tab10 palette for up to 10 categories (very distinct colors)
        colors = plt.cm.tab10(np.linspace(0, 1, num_categories))
    elif num_categories <= 20:
        # Use Set2 for more categories (still distinct)
        colors = plt.cm.Set2(np.linspace(0, 1, num_categories))
    else:
        # Use husl palette for many categories (perceptually uniform)
        colors = sns.color_palette("husl", n_colors=num_categories)
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each technology category
    for i, tech in enumerate(tech_trends.columns):
        if tech_trends[tech].sum() > 0:  # Only plot categories with data
            ax.plot(range(len(tech_trends.index)), tech_trends[tech], 
                   marker='o', linewidth=2.5, markersize=7, 
                   label=tech, color=colors[i], alpha=0.9)
    
    # Set x-axis labels to quarters with reduced spacing
    ax.set_xticks(range(len(tech_trends.index)))
    ax.set_xticklabels(tech_trends.index, rotation=45, ha='right')
    
    # Set labels and title with larger fonts
    ax.set_xlabel('Quarter', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Patents Filed', fontsize=16, fontweight='bold')
    ax.set_title('Patent Trends by Technology Category (CPC Classification)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=11)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add legend with better positioning and larger font
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
             fontsize=12, framealpha=0.95, edgecolor='black', frameon=True)
    
    # Reduce spacing between x-axis points (tighten x-axis)
    ax.set_xlim(-0.3, len(tech_trends.index) - 0.7)
    
    # Adjust y-axis to reduce unnecessary whitespace
    y_min = tech_trends.values.min()
    y_max = tech_trends.values.max()
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - y_range * 0.05), y_max + y_range * 0.05)
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    print("Plot - Technology trends (total patents):")
    print(tech_trends.sum().sort_values(ascending=False))


def plot_top_companies_horizontal_bar(df, readable_names):
    """
    Horizontal bar chart showing top 10 companies with most patent counts.
    Better visualization than pie chart - shows exact values and easier to compare.
    """
    # Filter out missing assignee organization names
    df_companies = df[df['disambig_assignee_organization'].notna()].copy()
    
    # Get top 10 companies by patent count
    top_companies = df_companies['disambig_assignee_organization'].value_counts().head(10)
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart with color gradient
    bars = ax.barh(range(len(top_companies)), top_companies.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(top_companies))),
                   alpha=0.8)
    
    # Set y-axis labels (company names)
    ax.set_yticks(range(len(top_companies)))
    ax.set_yticklabels(top_companies.index, fontsize=13)
    
    # Set labels and title with larger fonts
    ax.set_xlabel('Number of Patents', fontsize=16, fontweight='bold')
    ax.set_title('Top 10 Patent Holders (2024-2025)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='x', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    max_value = top_companies.values.max()
    label_offset = max_value * 0.02  # 2% of max value for spacing
    for i, (bar, value) in enumerate(zip(bars, top_companies.values)):
        ax.text(value + label_offset, i, f'{value:,}', 
               va='center', fontweight='bold', fontsize=12)
    
    # Invert y-axis to show highest at top
    ax.invert_yaxis()
    
    # Adjust x-axis to accommodate labels
    x_max = top_companies.values.max()
    ax.set_xlim(0, x_max + x_max * 0.15)  # Add 15% padding for labels
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nTop 10 Patent Holders:")
    for i, (company, count) in enumerate(top_companies.items(), 1):
        pct = (count / len(df_companies)) * 100
        print(f"  {i}. {company}: {count:,} patents ({pct:.2f}%)")
    
    return top_companies


def plot_boxplot_before_after(df, readable_names):
    """
    Boxplot comparing the distribution of num_claims before and after outlier removal.
    Uses IQR method to remove outliers and displays both distributions on separate subplots
    with independent y-axes for better visual clarity.
    
    Args:
        df: The pandas DataFrame containing patent data
        readable_names: Dictionary mapping column names to readable labels
    
    Returns:
        DataFrame with outliers removed (cleaned data)
    """
    # Filter out missing num_claims data
    df_claims = df[df['num_claims'].notna()].copy()
    
    # Downsample if dataset is too large (> 20k rows)
    original_size = len(df_claims)
    if len(df_claims) > 20000:
        df_claims = df_claims.sample(20000, random_state=42)
        print(f"Downsampled from {original_size:,} to {len(df_claims):,} rows for faster rendering")
    
    # Calculate IQR and bounds for outlier removal
    q1 = df_claims['num_claims'].quantile(0.25)
    q3 = df_claims['num_claims'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Create "Before" dataset
    df_before = df_claims.copy()
    
    # Create "After" dataset (with outliers removed)
    df_after = df_claims[(df_claims['num_claims'] >= lower_bound) & 
                         (df_claims['num_claims'] <= upper_bound)].copy()
    
    # Create figure with two subplots (side by side) with independent y-axes
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
    
    # Create boxplot for "Before" dataset
    sns.boxplot(data=df_before, y='num_claims', ax=axes[0], color="#6baed6", width=0.6)
    axes[0].set_title("Before Outlier Removal", fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel("", fontsize=14)
    axes[0].set_ylabel("Number of Claims", fontsize=16, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=13)
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    axes[0].set_axisbelow(True)
    
    # Create boxplot for "After" dataset
    sns.boxplot(data=df_after, y='num_claims', ax=axes[1], color="#9e4c7e", width=0.6)
    axes[1].set_title("After Outlier Removal", fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel("", fontsize=14)
    axes[1].set_ylabel("Number of Claims", fontsize=16, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=13)
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    axes[1].set_axisbelow(True)
    
    # Add overall title
    plt.suptitle(
        "Boxplot Comparison of Number of Claims (Before vs After IQR Cleaning)",
        fontsize=18,
        fontweight='bold',
        y=0.98,
        color='blue',
    )
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    # Apply consistent axis colors to both subplots
    for a in axes:
        apply_axis_colors(a)
    plt.show()
    
    # Print summary statistics
    print("\nOutlier Removal Statistics:")
    print(f"  Before: {len(df_before):,} patents")
    print(f"  After: {len(df_after):,} patents")
    print(f"  Removed: {len(df_before) - len(df_after):,} outliers ({(len(df_before) - len(df_after))/len(df_before)*100:.2f}%)")
    print(f"\nIQR Method Details:")
    print(f"  Q1 (25th percentile): {q1:.2f}")
    print(f"  Q3 (75th percentile): {q3:.2f}")
    print(f"  IQR: {iqr:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    print(f"\nClaims Range:")
    print(f"  Before: {df_before['num_claims'].min():.0f} - {df_before['num_claims'].max():.0f}")
    print(f"  After: {df_after['num_claims'].min():.0f} - {df_after['num_claims'].max():.0f}")
    
    # Return the cleaned DataFrame (after outlier removal)
    return df_after


def plot_swarm_claims_by_cpc_section(df, readable_names, sort_by='claims'):
    """
    Plot 3 - Swarm Plot: Number of Claims by CPC Section
    Shows the distribution of claims across different technology categories.
    Outliers are removed using IQR method for better visualization.
    
    Args:
        df: DataFrame with patent data
        readable_names: Dictionary mapping column names to readable labels
        sort_by: How to sort CPC sections - 'claims' (by mean claims, descending) or 'alphabetical'
    """
    # Filter out missing CPC section data
    df_cpc = df[df['cpc_section'].notna()].copy()
    
    # Remove outliers from num_claims for better visualization
    df_cpc = remove_claim_outliers_iqr(df_cpc)
    
    # Sample data if too large for performance (swarm plots are slow with large datasets)
    if len(df_cpc) > 2000:
        df_cpc = df_cpc.sample(2000, random_state=42)
    
    # Determine order of CPC sections
    if sort_by == 'claims':
        # Sort by mean number of claims (descending)
        cpc_order = df_cpc.groupby('cpc_section')['num_claims'].mean().sort_values(ascending=False).index.tolist()
    else:
        # Sort alphabetically
        cpc_order = sorted(df_cpc['cpc_section'].unique())
    
    # Create a categorical type with the desired order
    df_cpc['cpc_section'] = pd.Categorical(df_cpc['cpc_section'], categories=cpc_order, ordered=True)
    
    # Create figure with adjusted size (wider to accommodate legend)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create swarm plot
    sns.swarmplot(data=df_cpc, x='cpc_section', y='num_claims', 
                 palette='husl', size=3, alpha=0.7, ax=ax)
    
    # Set labels and title with larger fonts
    ax.set_xlabel('CPC Section', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Claims', fontsize=16, fontweight='bold')
    ax.set_title('Swarm Plot: Number of Claims by CPC Section', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis range to 0-45 for better visualization
    ax.set_ylim(0, 45)
    
    # Create legend mapping CPC section letters to full names
    cpc_mapping = {
        'A': 'Human Necessities',
        'B': 'Operations/Transport',
        'C': 'Chemistry/Metallurgy',
        'D': 'Textiles/Paper',
        'E': 'Fixed Constructions',
        'F': 'Mechanical Engineering',
        'G': 'Physics/Computing',
        'H': 'Electricity/Electronics'
    }
    
    # Get unique sections in the data
    sections_in_data = [s for s in cpc_order if s in df_cpc['cpc_section'].values]
    legend_labels = [f"{section}: {cpc_mapping.get(section, 'Unknown')}" for section in sections_in_data]
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='none', edgecolor='none', label=label) for label in legend_labels]
    # Place legend slightly above the top-right corner of the axes to avoid covering data
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1.0, 1.15),
        fontsize=11,
        framealpha=0.95,
        edgecolor='black',
        frameon=True,
        title='CPC Section Legend',
        title_fontsize=12
    )
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nClaims Statistics by CPC Section:")
    cpc_stats = df_cpc.groupby('cpc_section')['num_claims'].agg(['mean', 'median', 'std', 'count'])
    for cpc in cpc_order:
        if cpc in cpc_stats.index:
            stats = cpc_stats.loc[cpc]
            full_name = cpc_mapping.get(cpc, 'Unknown')
            print(f"  {cpc} ({full_name}): Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, "
                  f"Std={stats['std']:.2f}, Count={stats['count']:.0f}")


def plot_strip_type_vs_citations(df, readable_names):
    """
    Plot 5 - Strip Plot: Citations by Patent Type
    Shows the distribution of citation counts across different patent types.
    """
    # Filter out missing data
    df_s = df[['citation_count', 'patent_type']].dropna()
    
    # Sample data if too large for performance
    df_s = sample_df(df_s, n=20000)
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create strip plot
    sns.stripplot(data=df_s, x='patent_type', y='citation_count', 
                 alpha=0.4, jitter=0.25, size=3, ax=ax, palette='Set2')
    
    # Set labels and title with larger fonts
    ax.set_xlabel(get_label('patent_type', readable_names), fontsize=16, fontweight='bold')
    ax.set_ylabel(get_label('citation_count', readable_names), fontsize=16, fontweight='bold')
    ax.set_title('Strip Plot: Citations by Patent Type', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nCitation Statistics by Patent Type:")
    type_stats = df_s.groupby('patent_type')['citation_count'].agg(['mean', 'median', 'std', 'count'])
    for ptype in type_stats.index:
        stats = type_stats.loc[ptype]
        print(f"  {ptype}: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, "
              f"Std={stats['std']:.2f}, Count={stats['count']:.0f}")


def plot_hexbin_inventors_vs_citations_us(df, readable_names):
    """
    Plot 6 - Hexbin Plot: Inventors vs Citation Count (US Patents)
    Shows the density relationship between number of inventors and citation counts.
    Uses hexbin to handle high-density regions effectively.
    """
    # Filter to US patents only and drop missing values
    df_h = df[(df['disambig_country'] == 'US')][['num_inventors', 'citation_count']].dropna()
    
    # Focus on main mass / remove extreme outliers for visualization
    df_h = df_h[(df_h['num_inventors'] <= 20) & (df_h['citation_count'] <= 500)]
    
    # Sample data if too large for performance
    df_h = sample_df(df_h, n=100000)
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create hexbin plot
    hb = ax.hexbin(
        df_h['num_inventors'],
        df_h['citation_count'],
        gridsize=(30, 40),
        mincnt=5,
        bins='log',
        cmap='viridis'
    )
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Number of Patents (log scale)', fontsize=14, fontweight='bold')
    cb.ax.tick_params(labelsize=12)
    
    # Set labels and title with larger fonts
    ax.set_xlabel(get_label('num_inventors', readable_names), fontsize=16, fontweight='bold')
    ax.set_ylabel(get_label('citation_count', readable_names), fontsize=16, fontweight='bold')
    ax.set_title('Hexbin Plot: Inventors vs Citation Count (US Patents)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nHexbin Plot Statistics:")
    print(f"  Total US patents plotted: {len(df_h):,}")
    print(f"  Inventors range: {df_h['num_inventors'].min():.0f} - {df_h['num_inventors'].max():.0f}")
    print(f"  Citations range: {df_h['citation_count'].min():.0f} - {df_h['citation_count'].max():.0f}")
    print(f"  Mean inventors: {df_h['num_inventors'].mean():.2f}")
    print(f"  Mean citations: {df_h['citation_count'].mean():.2f}")


def plot_contour_claims_figures_density(df, readable_names):
    """
    Plot 7 - Contour Plot: Density of Patent Figures vs Claims
    Shows the density relationship between number of claims and number of figures.
    Uses KDE with custom bandwidth for smooth contours and filled background for readability.
    """
    # Filter and prepare data
    cols = ['num_claims', 'num_figures']
    df_c = df[cols].dropna()
    
    # Clip to reasonable ranges for better density visibility
    # Most data lives under 40 claims and 60 figures
    df_c = df_c[(df_c['num_claims'] <= 40) & (df_c['num_figures'] <= 100)]
    
    # Sample data if too large for performance
    df_c = sample_df(df_c, n=30000)
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create KDE contour plot with custom bandwidth for smoothness
    # Using rainbow colormap for vibrant visualization (no fill, just contour lines)
    sns.kdeplot(
        data=df_c,
        x='num_claims',
        y='num_figures',
        fill=False,
        cmap="rainbow",
        thresh=0.1,
        levels=12,
        linewidths=1.5,
        bw_adjust=1.2,  # Smooths the KDE slightly for discrete data
        ax=ax
    )
    
    # Set labels and title with larger fonts
    ax.set_xlabel(get_label('num_claims', readable_names), fontsize=16, fontweight='bold')
    ax.set_ylabel(get_label('num_figures', readable_names), fontsize=16, fontweight='bold')
    
    # Main title
    ax.set_title('Contour Plot of Patent Figures vs Claims', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Add subtitle explaining the filtering (academic style)
    ax.text(0.5, 0.98, '(Clipped to ≤ 40 claims and < 60 figures for density clarity)',
            transform=ax.transAxes, ha='center', va='top', fontsize=12, 
            style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.3))
    
    # Clip axes to tighter ranges for better density visibility
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 60)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nContour Plot Statistics:")
    print(f"  Total patents plotted: {len(df_c):,}")
    print(f"  Claims range: {df_c['num_claims'].min():.0f} - {df_c['num_claims'].max():.0f}")
    print(f"  Figures range: {df_c['num_figures'].min():.0f} - {df_c['num_figures'].max():.0f}")
    print(f"  Mean claims: {df_c['num_claims'].mean():.2f}")
    print(f"  Mean figures: {df_c['num_figures'].mean():.2f}")


def plot_3d_scatter_figures_claims_complexity(df, readable_names):
    """
    Plot 8 - 3D Scatter Plot: Number of Figures vs Number of Claims vs Patent Complexity
    Shows the three-dimensional relationship between figures, claims, and complexity.
    Outliers in number of figures, number of claims, and complexity are removed using 
    IQR method for better visualization.
    """
    # Filter and prepare data
    cols = ['num_claims', 'num_figures', 'patent_complexity']
    df_3d = df[cols].dropna()
    
    # Remove outliers from all three variables for better visualization
    df_3d = remove_outliers_iqr(df_3d, 'num_figures')
    df_3d = remove_outliers_iqr(df_3d, 'num_claims')
    df_3d = remove_outliers_iqr(df_3d, 'patent_complexity')
    
    # Sample data if too large for performance
    df_3d = sample_df(df_3d, n=15000)
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D scatter plot
    scatter = ax.scatter(
        df_3d['num_figures'],
        df_3d['num_claims'],
        df_3d['patent_complexity'],
        c=df_3d['patent_complexity'],  # Color by complexity for better visualization
        cmap='viridis',
        alpha=0.6,
        s=20,
        edgecolors='black',
        linewidth=0.3
    )
    
    # Set labels with larger fonts
    ax.set_xlabel(get_label('num_figures', readable_names), fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(get_label('num_claims', readable_names), fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel(get_label('patent_complexity', readable_names), fontsize=14, fontweight='bold', labelpad=10)
    
    # Main title + subtitle in a single, two-line title so the subtitle is directly below
    ax.set_title(
        '3D Scatter: Figures vs Claims vs Patent Complexity\n'
        'Outliers removed using IQR method (all variables)',
        fontsize=18,
        fontweight='bold',
        pad=20,
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Patent Complexity', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Increase tick label sizes
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='z', labelsize=11)
    
    # Tight layout
    plt.tight_layout()
    # Apply consistent axis colors (3D axis)
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\n3D Scatter Plot Statistics:")
    print(f"  Total patents plotted: {len(df_3d):,}")
    print(f"  Claims range: {df_3d['num_claims'].min():.0f} - {df_3d['num_claims'].max():.0f}")
    print(f"  Figures range: {df_3d['num_figures'].min():.0f} - {df_3d['num_figures'].max():.0f}")
    print(f"  Complexity range: {df_3d['patent_complexity'].min():.2f} - {df_3d['patent_complexity'].max():.2f}")
    print(f"  Mean complexity: {df_3d['patent_complexity'].mean():.2f}")


def plot_ai_heatmap_by_region_type(df, readable_names):
    """
    Plot 9 - Heatmap: AI Patents by US Region × AI Type
    Shows where AI innovation is happening in the US by region and AI category.
    """
    # Filter to US patents with AI classification and known regions
    df_ai = df[
        (df['disambig_country'] == 'US') &
        (df['is_ai'] == True) &
        (df['us_region'].notna()) &
        (df['us_region'] != 'Unknown') &
        (df['ai_bucket'].notna())
    ].copy()
    
    if len(df_ai) == 0:
        print("Warning: No AI patent data found for heatmap.")
        return
    
    # Create pivot table: regions × AI types
    pivot = pd.crosstab(df_ai['us_region'], df_ai['ai_bucket'])
    
    # Sort regions in a logical order
    region_order = ['Northeast', 'Midwest', 'Southwest', 'Southeast', 'West']
    pivot = pivot.reindex([r for r in region_order if r in pivot.index], axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Number of Patents'},
                ax=ax, linewidths=0.5, linecolor='gray')
    
    # Set labels and title
    ax.set_xlabel('AI Category', fontsize=16, fontweight='bold')
    ax.set_ylabel('US Region', fontsize=16, fontweight='bold')
    ax.set_title('AI Patents by US Region × AI Type', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    # Apply consistent axis colors for both subplots
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nAI Patents by Region and Type:")
    print(pivot)
    print(f"\nTotal AI patents in heatmap: {pivot.sum().sum():,}")


def plot_violin_corporate_vs_academic_claims(df, readable_names):
    """
    Plot 10 - Violin Plot: Corporate vs Academic Claims Distribution
    Side-by-side violin plot comparing claims distribution between corporate and academic patents.
    Outliers are removed using IQR method for better visualization.
    """
    # Filter to corporate and academic only
    df_plot = df[
        (df['assignee_type_academic'].isin(['Corporate', 'Academic'])) &
        (df['num_claims'].notna())
    ].copy()
    
    if len(df_plot) == 0:
        print("Warning: No corporate/academic data found. Run data_processing.py first.")
        return
    
    # Remove outliers from num_claims for better visualization
    df_plot = remove_outliers_iqr(df_plot, 'num_claims')
    
    # Sample if too large
    df_plot = sample_df(df_plot, n=20000)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create violin plot
    sns.violinplot(data=df_plot, x='assignee_type_academic', y='num_claims',
                  palette=['#2E86AB', '#A23B72'], ax=ax, cut=0)
    
    # Set labels and title
    ax.set_xlabel('Assignee Type', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Claims', fontsize=16, fontweight='bold')
    ax.set_title('Corporate vs Academic: Claims Distribution', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print statistics
    print("\nClaims Statistics by Assignee Type:")
    stats = df_plot.groupby('assignee_type_academic')['num_claims'].agg(['mean', 'median', 'std', 'count'])
    for atype in stats.index:
        s = stats.loc[atype]
        print(f"  {atype}: Mean={s['mean']:.2f}, Median={s['median']:.2f}, "
              f"Std={s['std']:.2f}, Count={s['count']:.0f}")


def plot_kde_complexity_corp_vs_academic(df, readable_names):
    """
    Plot 11 - KDE Plot: Corporate vs Academic Patent Complexity
    Patent complexity distribution for Corporate vs Academic assignees.
    Uses IQR-based trimming to remove extreme outliers and produce a smooth KDE.
    """
    cols = ['patent_complexity', 'assignee_category']
    df_k = df[cols].dropna()
    
    # Keep only corporate vs academic
    df_k = df_k[df_k['assignee_category'].isin(['Corporate', 'Academic'])].copy()
    
    if len(df_k) == 0:
        print("Warning: No corporate/academic data found. Run data_processing.py first.")
        return
    
    # IQR-based trimming on complexity (avoid crazy tails)
    q1 = df_k['patent_complexity'].quantile(0.25)
    q3 = df_k['patent_complexity'].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = max(q1 - 1.5 * iqr, 0)  # complexity shouldn't go negative
    
    df_k = df_k[(df_k['patent_complexity'] >= lower) &
                (df_k['patent_complexity'] <= upper)].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Nice palette with good contrast
    palette = {
        'Corporate': 'tab:blue',
        'Academic': 'tab:purple'
    }
    
    sns.kdeplot(
        data=df_k,
        x='patent_complexity',
        hue='assignee_category',
        fill=True,
        common_norm=False,   # keep group densities separate
        alpha=0.6,
        linewidth=2,
        palette=palette,
        ax=ax
    )
    
    # Set labels and title
    ax.set_xlabel(get_label('patent_complexity', readable_names), fontsize=16, fontweight='bold')
    ax.set_ylabel('Density', fontsize=16, fontweight='bold')
    ax.set_title('Corporate vs Academic: Patent Complexity Density (KDE)',
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(title='Assignee Type', fontsize=12, title_fontsize=13, loc='upper right')
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print statistics
    print("\nPatent Complexity Statistics by Assignee Type:")
    stats = df_k.groupby('assignee_category')['patent_complexity'].agg(['mean', 'median', 'std', 'count'])
    for atype in stats.index:
        s = stats.loc[atype]
        print(f"  {atype}: Mean={s['mean']:.2f}, Median={s['median']:.2f}, "
              f"Std={s['std']:.2f}, Count={s['count']:.0f}")
    print(f"\nNote: Outliers removed using IQR method (Q1-1.5*IQR to Q3+1.5*IQR)")


def plot_histogram_cpc_by_assignee_type(df, readable_names):
    """
    Plot 12 - Histogram: CPC Technology Profile by Assignee Type
    Shows technology focus differences between corporate and academic patents using grouped bar chart.
    """
    # Filter to corporate and academic with CPC data
    df_plot = df[
        (df['assignee_type_academic'].isin(['Corporate', 'Academic'])) &
        (df['cpc_major'].notna())
    ].copy()
    
    if len(df_plot) == 0:
        print("Warning: No corporate/academic data found. Run data_processing.py first.")
        return
    
    # Create pivot table: assignee type × CPC major (counts)
    pivot = pd.crosstab(df_plot['assignee_type_academic'], df_plot['cpc_major'])
    
    # Normalize by row to show percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    
    # Prepare data for grouped bar chart
    cpc_categories = pivot.columns.tolist()
    x = np.arange(len(cpc_categories))
    width = 0.35  # Width of bars
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create grouped bars
    corporate_pct = pivot_pct.loc['Corporate', cpc_categories].values if 'Corporate' in pivot_pct.index else np.zeros(len(cpc_categories))
    academic_pct = pivot_pct.loc['Academic', cpc_categories].values if 'Academic' in pivot_pct.index else np.zeros(len(cpc_categories))
    
    bars1 = ax.bar(x - width/2, corporate_pct, width, label='Corporate', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, academic_pct, width, label='Academic', color='#A23B72', alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('CPC Major Category', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold')
    ax.set_title('CPC Technology Profile by Assignee Type', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(cpc_categories, rotation=45, ha='right')
    
    # Add legend
    ax.legend(fontsize=13, title='Assignee Type', title_fontsize=14, loc='upper right')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print statistics
    print("\nCPC Technology Profile by Assignee Type (Counts):")
    print(pivot)
    print("\nCPC Technology Profile by Assignee Type (Percentages):")
    print(pivot_pct.round(1))


def plot_boxen_claims_by_cpc_and_assignee(df, readable_names):
    """
    Plot 13 - Multivariate Boxen Plot: Corporate vs Academic Claims by CPC Category
    Claims distribution by CPC major category, split by Corporate vs Academic.
    Outliers are trimmed using the IQR method for readability.
    """
    cols = ['num_claims', 'cpc_major', 'assignee_category']
    df_b = df[cols].dropna()
    
    # Keep only corporate vs academic assignees
    df_b = df_b[df_b['assignee_category'].isin(['Corporate', 'Academic'])].copy()
    
    if len(df_b) == 0:
        print("Warning: No corporate/academic data found. Run data_processing.py first.")
        return
    
    # IQR-based trimming to remove extreme outliers
    q1 = df_b['num_claims'].quantile(0.25)
    q3 = df_b['num_claims'].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df_b = df_b[df_b['num_claims'] <= upper].copy()
    
    # Order CPC major categories by median claims for better readability
    order = (
        df_b.groupby('cpc_major')['num_claims']
            .median()
            .sort_values()
            .index.tolist()
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create boxen plot
    sns.boxenplot(
        data=df_b,
        x='cpc_major',
        y='num_claims',
        hue='assignee_category',
        order=order,
        k_depth='proportion',
        linewidth=0.8,
        palette=['#2E86AB', '#A23B72'],
        ax=ax
    )
    
    # Set labels and title
    ax.set_xlabel('CPC Major Category', fontsize=16, fontweight='bold')
    ax.set_ylabel(get_label('num_claims', readable_names), fontsize=16, fontweight='bold')
    ax.set_title('Claims Distribution by CPC Category: Corporate vs Academic (Boxen)',
                fontsize=18, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    plt.xticks(rotation=35, ha='right')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(title='Assignee Type', fontsize=12, title_fontsize=13, loc='upper right')
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print statistics
    print("\nClaims Statistics by CPC Category and Assignee Type:")
    stats = df_b.groupby(['cpc_major', 'assignee_category'])['num_claims'].agg(['mean', 'median', 'count'])
    print(stats.round(2))


def plot_violin_corporate_vs_academic_patent_volume(df, readable_names):
    """
    Plot 17 - Violin Plot: Corporate vs Academic Patent Volume
    Compares the distribution of patent-related metrics between corporate and academic patents.
    Uses patent complexity as a proxy for patent "volume" or "size".
    """
    # Filter to corporate and academic only
    df_plot = df[
        (df['assignee_type_academic'].isin(['Corporate', 'Academic'])) &
        (df['patent_complexity'].notna())
    ].copy()
    
    if len(df_plot) == 0:
        print("Warning: No corporate/academic data found. Run data_processing.py first.")
        return
    
    # Remove outliers from patent_complexity for better visualization
    df_plot = remove_outliers_iqr(df_plot, 'patent_complexity')
    
    # Sample if too large
    df_plot = sample_df(df_plot, n=20000)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create violin plot
    sns.violinplot(data=df_plot, x='assignee_type_academic', y='patent_complexity',
                  palette=['#2E86AB', '#A23B72'], ax=ax, cut=0)
    
    # Set labels and title
    ax.set_xlabel('Assignee Type', fontsize=16, fontweight='bold')
    ax.set_ylabel(get_label('patent_complexity', readable_names), fontsize=16, fontweight='bold')
    ax.set_title('Corporate vs Academic: Patent Complexity Distribution', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print statistics
    print("\nPatent Complexity Statistics by Assignee Type:")
    stats = df_plot.groupby('assignee_type_academic')['patent_complexity'].agg(['mean', 'median', 'std', 'count'])
    for atype in stats.index:
        s = stats.loc[atype]
        print(f"  {atype}: Mean={s['mean']:.2f}, Median={s['median']:.2f}, "
              f"Std={s['std']:.2f}, Count={s['count']:.0f}")


def plot_area_claims_by_us_region_over_months(df, readable_names):
    """
    Plot 15 - Area Plot: Total Claims by US Region over Months
    Stacked area plot showing total number of claims per month by US region.
    Uses year-month combination (2024-01 through 2025-06) to show all 18 months.
    Filters out Unknown regions.
    """
    # Filter to US patents with known regions
    df_us = df[
        (df['disambig_country'] == 'US') & 
        (df['us_region'].notna()) & 
        (df['us_region'] != 'Unknown') &
        (df['grant_date'].notna()) &
        (df['num_claims'].notna())
    ].copy()
    
    if len(df_us) == 0:
        print("Warning: No US region data found for area plot.")
        return
    
    # Create year-month column from grant_date (e.g., '2024-01', '2024-02', ..., '2025-06')
    # This ensures we show all 18 months instead of just 12 month numbers
    df_us['year_month'] = df_us['grant_date'].dt.to_period('M').astype(str)
    
    # Group by year-month and region, sum claims
    grp = df_us.groupby(['year_month', 'us_region'])['num_claims'].sum().unstack(fill_value=0)
    grp = grp.sort_index()  # Sort by year-month chronologically
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for stacked area plot
    months = grp.index.values
    regions = grp.columns
    
    # Create stacked area plot
    # Convert year_month strings to numeric positions for plotting
    month_positions = range(len(months))
    # Use a more vibrant palette (husl) to better match swarm/bar plots
    vibrant_colors = sns.color_palette("husl", n_colors=len(regions))
    ax.stackplot(
        month_positions,
        [grp[reg].values for reg in regions],
        labels=regions,
        alpha=0.8,
        colors=vibrant_colors,
    )
    
    # Add total line on top
    ax.plot(month_positions, grp.sum(axis=1), linewidth=2, color='black', alpha=0.8, 
           label='Total', linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('Month (Year-Month)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Total Number of Claims', fontsize=16, fontweight='bold')
    ax.set_title('Total Number of Claims by US Region over Months (Jan 2024 - Jun 2025)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis ticks to show all months with year-month labels
    # Rotate labels for better readability
    ax.set_xticks(month_positions)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_xlim(-0.5, len(months) - 0.5)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(title='US Region', bbox_to_anchor=(1.02, 1), loc='upper left', 
             fontsize=12, title_fontsize=13, framealpha=0.95, edgecolor='black', frameon=True)
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nTotal Claims by US Region over Months:")
    print(grp)
    print(f"\nTotal claims across all regions: {grp.sum().sum():,.0f}")


def plot_stacked_bar_cpc_by_region(df, readable_names):
    """
    Plot 16 - Stacked Bar Plot: CPC Major Category by US Region
    Stacked bar plot showing CPC major category distribution by US region.
    Only uses US patents with a known region (excludes Unknown).
    """
    # Filter to US patents with known regions and CPC data (excluding 'Unknown' CPC major)
    df_us = df[
        (df['disambig_country'] == 'US') &
        (df['us_region'].notna()) &
        (df['us_region'] != 'Unknown') &
        (df['cpc_major'].notna()) &
        (df['cpc_major'] != 'Unknown')
    ].copy()
    
    if len(df_us) == 0:
        print("Warning: No US region data found for stacked bar plot.")
        return
    
    # Create pivot table: region × CPC major
    pivot = df_us.pivot_table(
        index='us_region',
        columns='cpc_major',
        values='patent_id',
        aggfunc='count',
        fill_value=0
    )
    
    # Order regions in a logical order
    region_order = ['Northeast', 'Midwest', 'Southwest', 'Southeast', 'West']
    pivot = pivot.reindex([r for r in region_order if r in pivot.index], axis=0).dropna(how='all')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create stacked bars
    bottom = np.zeros(len(pivot.index))
    # Use original Set2 palette for clear, soft categorical colors
    colors = plt.cm.Set2(np.linspace(0, 1, len(pivot.columns)))
    
    for i, col in enumerate(pivot.columns):
        ax.bar(
            pivot.index,
            pivot[col],
            bottom=bottom,
            label=col,
            color=colors[i],
            alpha=0.8,
        )
        bottom += pivot[col].values
    
    # Set labels and title
    ax.set_xlabel('US Region', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Patents', fontsize=16, fontweight='bold')
    ax.set_title('CPC Major Category Distribution by US Region (Stacked)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(title=get_label('cpc_major', readable_names), bbox_to_anchor=(1.02, 1), 
             loc='upper left', fontsize=11, title_fontsize=12, framealpha=0.95, 
             edgecolor='black', frameon=True)
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nCPC Major Category Distribution by US Region:")
    print(pivot)
    print(f"\nTotal patents: {pivot.sum().sum():,}")


def plot_pie_chart_us_region_patents(df, readable_names):
    """
    Plot 18 - Pie Chart: Percentage of Patents per US Region
    Pie chart showing the distribution of patents across US regions.
    Filters out Unknown regions and displays percentages with clear labels.
    """
    # Filter to US patents with known regions
    df_us = df[
        (df['disambig_country'] == 'US') &
        (df['us_region'].notna()) &
        (df['us_region'] != 'Unknown')
    ].copy()
    
    if len(df_us) == 0:
        print("Warning: No US region data found for pie chart.")
        return
    
    # Count patents by region
    region_counts = df_us['us_region'].value_counts()
    
    # Order regions in a logical order (geographical)
    region_order = ['Northeast', 'Midwest', 'Southwest', 'Southeast', 'West']
    region_counts = region_counts.reindex([r for r in region_order if r in region_counts.index])
    
    # Calculate percentages
    total_patents = region_counts.sum()
    percentages = (region_counts / total_patents * 100).round(2)
    
    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for each region (using viridis colormap like the 3D scatter plot)
    # Viridis provides beautiful, perceptually uniform colors from purple/blue to yellow
    colors = plt.cm.viridis(np.linspace(0, 1, len(region_counts)))
    
    # Create pie chart with autopct to show percentages
    # startangle=90 makes the first slice start at the top
    wedges, texts, autotexts = ax.pie(
        region_counts.values,
        labels=region_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 13, 'fontweight': 'bold'},
        pctdistance=0.85,  # Distance of percentage labels from center
        labeldistance=1.05  # Distance of region labels from center
    )
    
    # Enhance percentage text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    # Enhance region label readability
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight('bold')
    
    # Add title
    ax.set_title('Percentage of Patents per US Region', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Add a text box with total count
    total_text = f'Total US Patents: {total_patents:,}'
    ax.text(0, -1.2, total_text, ha='center', va='center',
           fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Ensure equal aspect ratio so pie is circular
    ax.axis('equal')
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nPatent Distribution by US Region:")
    print(f"{'Region':<15} {'Count':<12} {'Percentage':<12}")
    print("-" * 40)
    for region in region_counts.index:
        count = region_counts[region]
        pct = percentages[region]
        print(f"{region:<15} {count:>10,} {pct:>10.2f}%")
    print("-" * 40)
    print(f"{'Total':<15} {total_patents:>10,} {100.00:>10.2f}%")


def plot_rug_claims_northeast(df, readable_names):
    """
    Plot 19 - Rug Plot: Number of Claims for Northeast Region (US)
    Histogram + KDE + rug plot combo showing the distribution of claims.
    Improved version with density scaling, reduced bins, limited x-axis, and sampled rug.
    """
    # Filter to Northeast US patents
    df_ne = df[
        (df['disambig_country'] == 'US') &
        (df['us_region'] == 'Northeast') &
        (df['num_claims'].notna())
    ].copy()
    
    if len(df_ne) == 0:
        print("Warning: No Northeast US region data found for rug plot.")
        return
    
    # Sample if too large
    df_ne = sample_df(df_ne, n=10000)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram with density stat to match KDE scale
    sns.histplot(
        df_ne['num_claims'],
        bins=25,  # Reduced from 40 for less spiky appearance
        kde=True,
        stat='density',  # Use density to match KDE scale
        ax=ax,
        alpha=0.7,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )
    
    # Sample rug plot to reduce clutter in the tail (use 500 points)
    rug_sample = df_ne['num_claims'].sample(min(500, len(df_ne)), random_state=42)
    sns.rugplot(x=rug_sample, height=0.05, alpha=0.5, ax=ax, color='darkred')
    
    # Limit x-axis to 0-60 for better readability (where >95% of data are)
    ax.set_xlim(0, 60)
    
    # Set labels and title
    ax.set_xlabel(get_label('num_claims', readable_names), fontsize=16, fontweight='bold')
    ax.set_ylabel('Density', fontsize=16, fontweight='bold')  # Changed from 'Frequency' to 'Density'
    ax.set_title('Number of Claims for Northeast Region (US) with Rug Plot', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nNumber of Claims Statistics for Northeast Region:")
    stats = df_ne['num_claims'].describe()
    print(f"  Count: {stats['count']:.0f}")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['50%']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Min: {stats['min']:.2f}")
    print(f"  Max: {stats['max']:.2f}")
    print(f"  Q1: {stats['25%']:.2f}")
    print(f"  Q3: {stats['75%']:.2f}")
    print(f"\nNote: X-axis limited to 0-60 for readability (covers >95% of data)")


def get_continent(country_code):
    """
    Helper function to map country codes to continents.
    Returns continent name for major countries, 'Other' for others.
    """
    if pd.isna(country_code):
        return 'Unknown'
    
    country_code = str(country_code).upper()
    
    # Major country to continent mapping
    continent_map = {
        'US': 'North America',
        'CA': 'North America',
        'MX': 'North America',
        'GB': 'Europe',
        'DE': 'Europe',
        'FR': 'Europe',
        'IT': 'Europe',
        'ES': 'Europe',
        'NL': 'Europe',
        'SE': 'Europe',
        'CH': 'Europe',
        'JP': 'Asia',
        'CN': 'Asia',
        'KR': 'Asia',
        'TW': 'Asia',
        'IN': 'Asia',
        'SG': 'Asia',
        'AU': 'Oceania',
        'NZ': 'Oceania',
        'BR': 'South America',
        'AR': 'South America',
        'ZA': 'Africa',
        'IL': 'Asia',
        'RU': 'Europe',
    }
    
    return continent_map.get(country_code, 'Other')


def plot_four_pies(df, readable_names):
    """
    Plot 20 - 4-Grid Pie Chart:
    1. US Region distribution
    2. Continent distribution (from country codes)
    3. Assignee Type distribution
    4. CPC Major Category distribution
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # -----------------------
    # Pie 1: US Region
    # -----------------------
    df_us_regions = df[
        (df['disambig_country'] == 'US') &
        (df['us_region'].notna()) &
        (df['us_region'] != 'Unknown')
    ].copy()
    
    if len(df_us_regions) > 0:
        region_counts = df_us_regions['us_region'].value_counts()
        # Order regions logically
        region_order = ['Northeast', 'Midwest', 'Southwest', 'Southeast', 'West']
        region_counts = region_counts.reindex([r for r in region_order if r in region_counts.index])
        
        colors1 = plt.cm.viridis(np.linspace(0, 1, len(region_counts)))
        wedges1, texts1, autotexts1 = axes[0, 0].pie(
            region_counts.values,
            labels=region_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors1,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        axes[0, 0].set_title('% of Patents by US Region', fontsize=14, fontweight='bold', pad=10)
    else:
        axes[0, 0].text(0.5, 0.5, 'No US Region Data', ha='center', va='center', fontsize=12)
        axes[0, 0].set_title('% of Patents by US Region', fontsize=14, fontweight='bold', pad=10)
    
    # -----------------------
    # Pie 2: Continent
    # -----------------------
    df_cont = df[df['disambig_country'].notna()].copy()
    if len(df_cont) > 0:
        df_cont['continent'] = df_cont['disambig_country'].apply(get_continent)
        cont_counts = df_cont['continent'].value_counts()
        
        colors2 = plt.cm.viridis(np.linspace(0, 1, len(cont_counts)))
        wedges2, texts2, autotexts2 = axes[0, 1].pie(
            cont_counts.values,
            labels=cont_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors2,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        axes[0, 1].set_title('% of Patents by Continent', fontsize=14, fontweight='bold', pad=10)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Country Data', ha='center', va='center', fontsize=12)
        axes[0, 1].set_title('% of Patents by Continent', fontsize=14, fontweight='bold', pad=10)
    
    # -----------------------
    # Pie 3: Assignee Type
    # -----------------------
    df_ass = df[df['assignee_category'].notna()].copy()
    if len(df_ass) > 0:
        ass_counts = df_ass['assignee_category'].value_counts()
        
        colors3 = plt.cm.viridis(np.linspace(0, 1, len(ass_counts)))
        wedges3, texts3, autotexts3 = axes[1, 0].pie(
            ass_counts.values,
            labels=ass_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors3,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        for autotext in autotexts3:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        axes[1, 0].set_title('% of Patents by Assignee Type', fontsize=14, fontweight='bold', pad=10)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Assignee Data', ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('% of Patents by Assignee Type', fontsize=14, fontweight='bold', pad=10)
    
    # -----------------------
    # Pie 4: CPC Major Category
    # -----------------------
    df_cpc = df[df['cpc_major'].notna()].copy()
    if len(df_cpc) > 0:
        cpc_counts = df_cpc['cpc_major'].value_counts()
        
        colors4 = plt.cm.viridis(np.linspace(0, 1, len(cpc_counts)))
        wedges4, texts4, autotexts4 = axes[1, 1].pie(
            cpc_counts.values,
            labels=cpc_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors4,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        for autotext in autotexts4:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        axes[1, 1].set_title('% of Patents by CPC Major Category', fontsize=14, fontweight='bold', pad=10)
    else:
        axes[1, 1].text(0.5, 0.5, 'No CPC Data', ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('% of Patents by CPC Major Category', fontsize=14, fontweight='bold', pad=10)
    
    # Add overall title
    fig.suptitle('Patent Distribution Overview: Four Key Dimensions', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # Print summary statistics
    print("\nFour-Pie Chart Summary Statistics:")
    if len(df_us_regions) > 0:
        print("\nUS Region Distribution:")
        for region, count in region_counts.items():
            pct = (count / region_counts.sum() * 100)
            print(f"  {region}: {count:,} ({pct:.2f}%)")
    
    if len(df_cont) > 0:
        print("\nContinent Distribution:")
        for continent, count in cont_counts.items():
            pct = (count / cont_counts.sum() * 100)
            print(f"  {continent}: {count:,} ({pct:.2f}%)")
    
    if len(df_ass) > 0:
        print("\nAssignee Type Distribution:")
        for atype, count in ass_counts.items():
            pct = (count / ass_counts.sum() * 100)
            print(f"  {atype}: {count:,} ({pct:.2f}%)")
    
    if len(df_cpc) > 0:
        print("\nCPC Major Category Distribution:")
        for cpc, count in cpc_counts.items():
            pct = (count / cpc_counts.sum() * 100)
            print(f"  {cpc}: {count:,} ({pct:.2f}%)")


def plot_four_pies_overview(df, readable_names):
    """
    New 4x4-style overview pie chart figure:
      1. % of patents by US region
      2. % of patents by continent
      3. % of patents: US vs Rest of World
      4. % of patents: Academic vs Corporate (from academic dataset, if available)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # -----------------------
    # Pie 1: % patents per US region
    # -----------------------
    df_us_regions = df[
        (df['disambig_country'] == 'US') &
        (df['us_region'].notna()) &
        (df['us_region'] != 'Unknown')
    ].copy()

    if len(df_us_regions) > 0:
        region_counts = df_us_regions['us_region'].value_counts()
        region_order = ['Northeast', 'Midwest', 'Southwest', 'Southeast', 'West']
        region_counts = region_counts.reindex(
            [r for r in region_order if r in region_counts.index]
        ).dropna()

        # Softer qualitative palette without bright yellow
        colors1 = sns.color_palette("Set2", n_colors=len(region_counts))
        wedges1, texts1, autotexts1 = axes[0, 0].pie(
            region_counts.values,
            labels=region_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors1,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
        )
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        axes[0, 0].set_title(
            '% of Patents by US Region',
            fontsize=14,
            fontweight='bold',
            pad=10,
        )
    else:
        axes[0, 0].text(
            0.5, 0.5, 'No US Region Data',
            ha='center', va='center', fontsize=12,
        )
        axes[0, 0].set_title(
            '% of Patents by US Region',
            fontsize=14,
            fontweight='bold',
            pad=10,
        )

    # -----------------------
    # Pie 2: % patents per continent
    # -----------------------
    df_cont = df[df['disambig_country'].notna()].copy()
    if len(df_cont) > 0:
        df_cont['continent'] = df_cont['disambig_country'].apply(get_continent)
        # Keep only top 5 continents by patent count
        cont_counts = df_cont['continent'].value_counts().head(5)

        colors2 = sns.color_palette("Set2", n_colors=len(cont_counts))
        # Draw a clean pie and move both names and percentages into the legend
        wedges2, _ = axes[0, 1].pie(
            cont_counts.values,
            labels=None,
            startangle=90,
            colors=colors2,
        )

        # Build legend labels with explicit percentages
        total = cont_counts.sum()
        legend_labels = [
            f"{name} ({count / total * 100:.1f}%)"
            for name, count in cont_counts.items()
        ]

        # Add legend outside the pie
        axes[0, 1].legend(
            wedges2,
            legend_labels,
            title='Continent',
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=10,
            title_fontsize=11,
            frameon=True,
        )

        axes[0, 1].set_title(
            '% of Patents by Continent',
            fontsize=14,
            fontweight='bold',
            pad=10,
        )
    else:
        axes[0, 1].text(
            0.5, 0.5, 'No Country Data',
            ha='center', va='center', fontsize=12,
        )
        axes[0, 1].set_title(
            '% of Patents by Continent',
            fontsize=14,
            fontweight='bold',
            pad=10,
        )

    # -----------------------
    # Pie 3: % patents US vs Rest of World
    # -----------------------
    if 'disambig_country' in df.columns:
        total_count = len(df)
        us_count = (df['disambig_country'] == 'US').sum()
        non_us_count = total_count - us_count

        labels3 = ['US', 'Rest of World']
        values3 = [us_count, non_us_count]

        colors3 = sns.color_palette("Set2", n_colors=len(labels3))
        wedges3, texts3, autotexts3 = axes[1, 0].pie(
            values3,
            labels=labels3,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors3,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
        )
        for autotext in autotexts3:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        axes[1, 0].set_title(
            '% of Patents: US vs Rest of World',
            fontsize=14,
            fontweight='bold',
            pad=10,
        )
    else:
        axes[1, 0].text(
            0.5, 0.5, 'No Country Data',
            ha='center', va='center', fontsize=12,
        )
        axes[1, 0].set_title(
            '% of Patents: US vs Rest of World',
            fontsize=14,
            fontweight='bold',
            pad=10,
        )

    # -----------------------
    # Pie 4: % patents Academic vs Corporate
    # -----------------------
    if 'assignee_type_academic' in df.columns:
        df_ac = df[
            df['assignee_type_academic'].isin(['Corporate', 'Academic'])
        ].copy()

        if len(df_ac) > 0:
            ac_counts = df_ac['assignee_type_academic'].value_counts()
            colors4 = sns.color_palette("Set2", n_colors=len(ac_counts))
            wedges4, texts4, autotexts4 = axes[1, 1].pie(
                ac_counts.values,
                labels=ac_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors4,
                textprops={'fontsize': 11, 'fontweight': 'bold'},
            )
            for autotext in autotexts4:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            axes[1, 1].set_title(
                '% of Patents: Academic vs Corporate',
                fontsize=14,
                fontweight='bold',
                pad=10,
            )
        else:
            axes[1, 1].text(
                0.5, 0.5, 'No Academic/Corporate Data',
                ha='center', va='center', fontsize=12,
            )
            axes[1, 1].set_title(
                '% of Patents: Academic vs Corporate',
                fontsize=14,
                fontweight='bold',
                pad=10,
            )
    else:
        axes[1, 1].text(
            0.5, 0.5, 'Academic dataset not available',
            ha='center', va='center', fontsize=12,
        )
        axes[1, 1].set_title(
            '% of Patents: Academic vs Corporate',
            fontsize=14,
            fontweight='bold',
            pad=10,
        )

    # Overall title
    fig.suptitle(
        'Patent Share Overview: Regions, Continents, US vs World, Academic vs Corporate',
        fontsize=18,
        fontweight='bold',
        y=0.98,
        color='blue',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Apply consistent axis colors for all subplots
    for ax in axes.flat:
        apply_axis_colors(ax)
    plt.show()


def plot_count_cpc_major(df, readable_names):
    """
    Plot 21 - Count Plot: Number of Patents by CPC Major Category
    Shows how many patents fall into each CPC major technology category.
    """
    # Filter out missing CPC data and remove 'Unknown' category
    df_cpc = df[(df['cpc_major'].notna()) & (df['cpc_major'] != 'Unknown')].copy()
    
    if len(df_cpc) == 0:
        print("Warning: No CPC major category data found.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Order by count (descending)
    order = (
        df_cpc['cpc_major']
        .value_counts()
        .sort_values(ascending=False)
        .index
    )
    
    # Create count plot
    sns.countplot(
        data=df_cpc,
        x='cpc_major',
        order=order,
        ax=ax,
        palette='viridis'
    )
    
    # Set labels and title
    ax.set_title('Count Plot: Number of Patents by CPC Major Category',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(get_label('cpc_major', readable_names), fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Patents', fontsize=16, fontweight='bold')
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', labelrotation=35, labelsize=13)
    plt.setp(ax.get_xticklabels(), ha='right')
    ax.tick_params(axis='y', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Annotate counts on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height):,}',
                    (p.get_x() + p.get_width()/2., height),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    # Apply consistent axis colors to all figure axes
    for ax in fig.axes:
        apply_axis_colors(ax)
    plt.show()
    
    # Print statistics
    print("\nCount Plot Statistics (patents per CPC major):")
    counts = df_cpc['cpc_major'].value_counts()
    for cpc, count in counts.items():
        pct = (count / len(df_cpc)) * 100
        print(f"  {cpc}: {count:,} ({pct:.2f}%)")


def plot_pair_structural_variables(df_academic, readable_names):
    """
    Plot 22 - Pair Plot: Structural Variables (Corporate vs Academic)
    Shows relationships between num_claims, num_figures, citation_count, and patent_complexity,
    colored by assignee_type_academic (Corporate vs Academic).
    """
    if 'assignee_type_academic' not in df.columns:
        print("Warning: Academic dataset not available. Run data_processing.py first.")
        return
    
    cols = ['num_claims', 'num_figures', 'citation_count', 'patent_complexity',
            'assignee_type_academic']
    df_p = df[cols].dropna()
    
    # Filter to Corporate and Academic only
    df_p = df_p[df_p['assignee_type_academic'].isin(['Corporate', 'Academic'])].copy()
    
    if len(df_p) == 0:
        print("Warning: No corporate/academic data found.")
        return
    
    # Downsample for readability
    df_p = df_p.sample(n=min(3000, len(df_p)), random_state=42)
    
    print("\nPair Plot Sample Size:", len(df_p))
    
    # Create pair plot
    g = sns.pairplot(
        data=df_p,
        vars=['num_claims', 'num_figures', 'citation_count', 'patent_complexity'],
        hue='assignee_type_academic',
        diag_kind='kde',
        corner=True,
        plot_kws={'alpha': 0.4, 's': 10},
        palette=['#2E86AB', '#A23B72']
    )
    
    g.fig.suptitle(
        'Pair Plot: Claims, Figures, Citations, Complexity (Corporate vs Academic)',
        fontsize=18,
        fontweight='bold',
        y=1.02,
        color='blue',
    )
    
    plt.tight_layout()
    # Apply consistent axis colors to all pairplot axes
    for row in g.axes:
        for ax in row:
            if ax is not None:
                apply_axis_colors(ax)
    plt.show()
    
    # Print correlation statistics
    print("\nCorrelation Matrix (Corporate vs Academic):")
    for assignee_type in ['Corporate', 'Academic']:
        subset = df_p[df_p['assignee_type_academic'] == assignee_type]
        corr = subset[['num_claims', 'num_figures', 'citation_count', 'patent_complexity']].corr()
        print(f"\n{assignee_type} Patents:")
        print(corr.round(3))


def plot_qq_num_claims(df, readable_names):
    """
    Plot 23 - QQ-Plot: Number of Claims vs Theoretical Normal Distribution
    Checks whether the cleaned claims distribution is close to normal after IQR-based outlier removal.
    """
    # Filter out missing claims data
    claims = df['num_claims'].dropna()
    
    if len(claims) == 0:
        print("Warning: No claims data found.")
        return
    
    # IQR-based cleaning (same logic as boxplot)
    q1, q3 = claims.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    claims_clean = claims[(claims >= lower) & (claims <= upper)]
    
    print("\nQQ-Plot Statistics for num_claims (cleaned):")
    print(f"  Count: {len(claims_clean):,}")
    print(f"  Mean:  {claims_clean.mean():.2f}")
    print(f"  Std:   {claims_clean.std():.2f}")
    print(f"  Range: {claims_clean.min():.0f} - {claims_clean.max():.0f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Create QQ-plot
    stats.probplot(claims_clean, dist="norm", plot=ax)
    
    # Set labels and title
    ax.set_title('QQ-Plot: Number of Claims (After IQR Cleaning)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Theoretical Quantiles', fontsize=16, fontweight='bold')
    # Explicitly set ylabel to override any default from probplot
    ax.set_ylabel('Sample Quantiles', fontsize=16, fontweight='bold')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()


def plot_reg_claims_vs_citations(df, readable_names):
    """
    Plot 24 - Regression Plot: Citation Count vs Number of Claims
    Shows relationship between number of claims and citation_count with linear regression fit.
    Tests whether patents with more claims tend to receive more citations.
    """
    cols = ['num_claims', 'citation_count']
    df_r = df[cols].dropna()
    
    if len(df_r) == 0:
        print("Warning: No claims/citation data found.")
        return
    
    # Limit extreme citation outliers for readability (e.g., <= 200)
    df_r = df_r[df_r['citation_count'] <= 200]
    
    # Downsample to keep plot readable
    df_r = df_r.sample(n=min(15000, len(df_r)), random_state=42)
    
    print("\nRegplot Sample Size:", len(df_r))
    correlation = df_r['num_claims'].corr(df_r['citation_count'])
    print(f"Correlation (Pearson) between claims and citations: {correlation:.3f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create regression plot
    sns.regplot(
        data=df_r,
        x='num_claims',
        y='citation_count',
        scatter_kws={'alpha': 0.25, 's': 10},
        line_kws={'linewidth': 2, 'color': 'red'},
        ax=ax
    )
    
    # Set labels and title
    ax.set_title('Regression Plot: Citation Count vs Number of Claims',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(get_label('num_claims', readable_names), fontsize=16, fontweight='bold')
    ax.set_ylabel('Citation Count (capped at 200)', fontsize=16, fontweight='bold')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(ax)
    plt.show()


def plot_joint_claims_citations(df, readable_names):
    """
    Plot 25 - Joint Plot: Claims vs Citations (KDE + Scatter)
    Combines scatter plot with KDE density contours for num_claims vs citation_count.
    Shows both density structure and individual points.
    """
    cols = ['num_claims', 'citation_count']
    df_j = df[cols].dropna()
    
    if len(df_j) == 0:
        print("Warning: No claims/citation data found.")
        return
    
    # Focus on bulk of distribution
    df_j = df_j[df_j['citation_count'] <= 200]
    df_j = df_j[df_j['num_claims'] <= 40]
    df_j = df_j.sample(n=min(20000, len(df_j)), random_state=42)
    
    print("\nJoint Plot Sample Size:", len(df_j))
    
    # Create joint plot with KDE
    g = sns.jointplot(
        data=df_j,
        x='num_claims',
        y='citation_count',
        kind='kde',
        fill=True,
        thresh=0.05,
        cmap='Blues'
    )
    
    # Overlay scatter on top of KDE
    g.plot_joint(
        sns.scatterplot,
        alpha=0.15,
        s=8,
        color='black'
    )
    
    # Set labels
    g.ax_joint.set_xlabel(get_label('num_claims', readable_names), fontsize=14, fontweight='bold')
    g.ax_joint.set_ylabel('Citation Count (≤ 200)', fontsize=14, fontweight='bold')
    
    # Set title
    g.fig.suptitle(
        'Joint Plot: Claims vs Citations (KDE + Scatter)',
        fontsize=18,
        fontweight='bold',
        y=0.98,  # Move title down slightly to fit in plot
        color='blue',
    )
    
    plt.tight_layout()
    # Apply consistent axis colors to the joint axes
    apply_axis_colors(g.ax_joint)
    plt.show()
    
    # Print statistics
    print("\nJoint Plot Statistics:")
    print(f"  Claims range: {df_j['num_claims'].min():.0f} - {df_j['num_claims'].max():.0f}")
    print(f"  Citations range: {df_j['citation_count'].min():.0f} - {df_j['citation_count'].max():.0f}")
    print(f"  Correlation: {df_j['num_claims'].corr(df_j['citation_count']):.3f}")


def plot_clustermap_cpc_by_region(df, readable_names):
    """
    Plot 26 - Cluster Map: CPC Major Category Distribution by US Region
    Clustered heatmap showing which regions have similar technology mixes.
    Values are normalized patent counts (row-wise percentages).
    """
    # Keep only US patents with region and CPC data
    df_us = df[
        (df['disambig_country'] == 'US') &
        (df['us_region'].notna()) &
        (df['us_region'] != 'Unknown') &
        (df['cpc_major'].notna())
    ].copy()
    
    if len(df_us) == 0:
        print("Warning: No US region/CPC data found for cluster map.")
        return
    
    # Create pivot table: region × CPC major → counts
    pivot = (
        df_us.pivot_table(
            index='us_region',
            columns='cpc_major',
            values='patent_id',  # Use patent_id instead of publication_number
            aggfunc='count',
            fill_value=0
        )
    )
    
    # Convert to row-wise percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    
    print("\nCluster Map Data (row-wise % of patents per CPC major):")
    print(pivot_pct.round(1).to_string())
    
    # Create clustered heatmap
    g = sns.clustermap(
        pivot_pct,
        cmap='viridis',
        standard_scale=1,  # scale by column for clustering comparability
        linewidths=0.5,
        figsize=(12, 8),
        cbar_kws={'label': 'Percentage of Patents'}
    )
    
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=12)
    g.ax_heatmap.set_xlabel('CPC Major Category', fontsize=14, fontweight='bold')
    g.ax_heatmap.set_ylabel('US Region', fontsize=14, fontweight='bold')
    
    g.fig.suptitle('Cluster Map: CPC Technology Mix by US Region',
                   fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    # Apply consistent axis colors
    apply_axis_colors(g.ax_heatmap)
    plt.show()


# ============================================================================
# AI ANALYSIS PLOTS
# ============================================================================

def plot_ai_vs_non_ai_over_quarters(df, readable_names):
    """
    Line plot: AI vs Non-AI patent counts over time (year_quarter).
    Shows the trend of AI patents compared to non-AI patents across quarters.
    """
    # Filter out rows without year_quarter or is_ai information
    df_q = df.dropna(subset=['year_quarter', 'is_ai'])
    
    # Group by year_quarter and is_ai to count patents
    counts = (
        df_q
        .groupby(['year_quarter', 'is_ai'])['patent_id']
        .count()
        .reset_index(name='patent_count')
        .sort_values('year_quarter')
    )
    
    # Create nicer legend labels
    counts['AI label'] = counts['is_ai'].map({True: 'AI patents', False: 'Non-AI patents'})
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot lines for AI and non-AI patents
    for ai_type in [True, False]:
        data = counts[counts['is_ai'] == ai_type]
        if len(data) > 0:
            label = 'AI patents' if ai_type else 'Non-AI patents'
            color = '#2E86AB' if ai_type else '#6A8080'  # Blue for AI, gray for non-AI
            marker = 'o' if ai_type else 's'  # Circle for AI, square for non-AI
            ax.plot(range(len(data)), data['patent_count'], 
                   marker=marker, linewidth=2.5, markersize=8, 
                   label=label, color=color, alpha=0.9)
    
    # Set x-axis labels
    unique_quarters = sorted(counts['year_quarter'].unique())
    ax.set_xticks(range(len(unique_quarters)))
    ax.set_xticklabels(unique_quarters, rotation=45, ha='right')
    
    # Set labels and title with larger fonts
    ax.set_xlabel('Year-Quarter', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Patents', fontsize=16, fontweight='bold')
    ax.set_title('AI vs Non-AI Patent Trends over Time', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add legend with better positioning and larger font
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
             fontsize=12, framealpha=0.95, edgecolor='black', frameon=True)
    
    # Reduce spacing between x-axis points
    ax.set_xlim(-0.3, len(unique_quarters) - 0.7)
    
    # Adjust y-axis to reduce unnecessary whitespace
    y_min = counts['patent_count'].min()
    y_max = counts['patent_count'].max()
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - y_range * 0.05), y_max + y_range * 0.05)
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nAI vs Non-AI Patent Counts by Quarter:")
    pivot_counts = counts.pivot(index='year_quarter', columns='AI label', values='patent_count')
    print(pivot_counts)


def plot_ai_vs_non_ai_stacked_bar(df, readable_names):
    """
    Stacked bar chart: Composition of AI vs non-AI patents per quarter.
    Shows the proportion of AI and non-AI patents in each quarter.
    """
    # Filter out rows without year_quarter or is_ai information
    df_q = df.dropna(subset=['year_quarter', 'is_ai'])
    
    # Group by year_quarter and is_ai to count patents
    counts = (
        df_q
        .groupby(['year_quarter', 'is_ai'])['patent_id']
        .count()
        .reset_index(name='patent_count')
        .sort_values('year_quarter')
    )
    
    # Pivot to create stacked bar data
    pivot_counts = counts.pivot(index='year_quarter', columns='is_ai', values='patent_count').fillna(0)
    pivot_counts.columns = ['Non-AI patents', 'AI patents']
    pivot_counts = pivot_counts.sort_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create stacked bar chart
    x_pos = range(len(pivot_counts))
    bottom = np.zeros(len(pivot_counts))
    
    # Plot non-AI patents first (bottom)
    ax.bar(x_pos, pivot_counts['Non-AI patents'], 
          label='Non-AI patents', color='#6A8080', alpha=0.8)
    bottom += pivot_counts['Non-AI patents'].values
    
    # Plot AI patents on top
    ax.bar(x_pos, pivot_counts['AI patents'], 
          bottom=bottom, label='AI patents', color='#2E86AB', alpha=0.8)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pivot_counts.index, rotation=45, ha='right')
    
    # Set labels and title with larger fonts
    ax.set_xlabel('Year-Quarter', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Patents', fontsize=16, fontweight='bold')
    ax.set_title('AI vs Non-AI Patent Composition by Quarter (Stacked)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Add legend with better positioning and larger font
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
             fontsize=12, framealpha=0.95, edgecolor='black', frameon=True)
    
    # Reduce spacing between bars
    ax.set_xlim(-0.5, len(pivot_counts) - 0.5)
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nAI vs Non-AI Patent Composition by Quarter:")
    print(pivot_counts)
    print("\nPercentage of AI patents per quarter:")
    total_per_quarter = pivot_counts.sum(axis=1)
    ai_pct = (pivot_counts['AI patents'] / total_per_quarter * 100).round(2)
    for quarter, pct in ai_pct.items():
        print(f"  {quarter}: {pct:.2f}%")


def plot_ai_only_corr_heatmap(df, readable_names):
    """
    Correlation heatmap for AI patents only.
    Focuses on structural vs impact features to understand relationships
    between patent characteristics in AI patents.
    """
    # Filter to AI patents only
    df_ai = df[df['is_ai']].copy()
    
    # Select columns for correlation analysis
    cols = [
        'num_claims',
        'num_figures',
        'num_inventors',
        'num_assignees',
        'citation_count',
        'total_citations',
        'patent_complexity',
        'citation_per_claim',
        'foreign_citation_ratio'
    ]
    
    # Filter to columns that exist and drop rows with missing values
    existing_cols = [col for col in cols if col in df_ai.columns]
    df_corr = df_ai[existing_cols].dropna()
    
    if len(df_corr) == 0:
        print("Warning: No data available for correlation heatmap after filtering.")
        return
    
    # Calculate correlation matrix
    corr = df_corr.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with annotations
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                cbar=True, square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1, center=0)
    
    # Set title and labels with larger fonts
    ax.set_title('Correlation Heatmap – AI Patents Only', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Use readable names for labels if available
    if readable_names:
        labels = [readable_names.get(col, col) for col in corr.columns]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
    
    plt.tight_layout()
    apply_axis_colors(ax)
    plt.show()
    
    print(f"\nCorrelation matrix computed for {len(df_corr):,} AI patents")


def plot_pca_analysis(df, readable_names):
    """
    Principal Component Analysis (PCA) plot for patent dataset.
    Uses the same numeric columns as the correlation heatmap for consistency.
    PCA is calculated on: num_claims, num_figures, num_inventors, patent_complexity, total_citations.
    Points are color-coded by assignee_category (Corporate/Academic/Government/Individual) 
    or top countries for visualization purposes only - this does NOT affect the PCA calculation.
    """
    # Select columns for PCA (same as correlation heatmap)
    cols = [
        'num_claims',
        'num_figures',
        'num_inventors',
        'patent_complexity',
        'total_citations'
    ]
    
    # Filter to columns that exist and drop rows with missing values
    existing_cols = [col for col in cols if col in df.columns]
    if len(existing_cols) < 2:
        print("Warning: Need at least 2 numeric columns for PCA. Available columns:", existing_cols)
        return
    
    df_pca = df[existing_cols].dropna()
    
    if len(df_pca) == 0:
        print("Warning: No data available for PCA after filtering.")
        return
    
    # Sample if dataset is too large for performance
    if len(df_pca) > 20000:
        df_pca = df_pca.sample(n=20000, random_state=42)
        print(f"Sampled {len(df_pca):,} rows for PCA visualization")
    
    # Standardize the data (center and scale)
    subset_centered = df_pca[existing_cols] - df_pca[existing_cols].mean(axis=0)
    subset_scaled = subset_centered / subset_centered.std(axis=0)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(subset_scaled.values)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        pca_components,
        columns=['PC1', 'PC2'],
        index=df_pca.index
    )
    
    # Add color column - ONLY use assignee_category (NO countries allowed)
    # NOTE: This is ONLY for color-coding visualization - it does NOT affect PCA calculation
    # PCA is calculated purely on numeric features: num_claims, num_figures, num_inventors, patent_complexity, total_citations
    if 'assignee_category' in df.columns:
        pca_df['color_col'] = df.loc[pca_df.index, 'assignee_category'].fillna('Unknown')
        color_col_name = 'Assignee Category'
        print(f"Color-coding by Assignee Category: {sorted(pca_df['color_col'].unique())}")
    else:
        # If assignee_category not available, use single color (no grouping, NO countries)
        pca_df['color_col'] = 'All Patents'
        color_col_name = 'All Patents'
        print(f"Note: assignee_category not available. Using single color for all patents (countries NOT used).")
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    total_explained = explained_variance.sum()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique categories for coloring
    unique_categories = pca_df['color_col'].unique()
    colors = sns.color_palette("husl", len(unique_categories))
    color_map = dict(zip(unique_categories, colors))
    
    # Plot each category
    for category in unique_categories:
        mask = pca_df['color_col'] == category
        ax.scatter(
            pca_df.loc[mask, 'PC1'],
            pca_df.loc[mask, 'PC2'],
            label=category,
            alpha=0.6,
            s=20,
            c=[color_map[category]]
        )
    
    # Set labels and title
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance explained)', 
                  fontsize=16, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance explained)', 
                  fontsize=16, fontweight='bold')
    ax.set_title(f'Principal Component Analysis (PCA) of Patent Features\n'
                 f'Total Variance Explained: {total_explained:.1%}',
                 fontsize=18, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(title=color_col_name, fontsize=10, title_fontsize=12, 
              loc='best', framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    apply_axis_colors(ax)
    plt.show()
    
    # Print PCA statistics
    print(f"\n{'='*70}")
    print("PCA Analysis Results")
    print(f"{'='*70}")
    print(f"Number of components: 2")
    print(f"Number of features: {len(existing_cols)}")
    print(f"Features used: {', '.join([get_label(col, readable_names) for col in existing_cols])}")
    print(f"\nExplained Variance:")
    print(f"  PC1: {explained_variance[0]:.2%}")
    print(f"  PC2: {explained_variance[1]:.2%}")
    print(f"  Total: {total_explained:.2%}")
    print(f"\nComponent Loadings:")
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=existing_cols
    )
    for col in existing_cols:
        print(f"  {get_label(col, readable_names)}:")
        print(f"    PC1: {loadings_df.loc[col, 'PC1']:.3f}")
        print(f"    PC2: {loadings_df.loc[col, 'PC2']:.3f}")
    print(f"\nSample size: {len(pca_df):,} patents")
    print(f"{'='*70}")


def plot_scatter_matrix(df, readable_names):
    """
    Scatter plot matrix (pairplot) for patent dataset.
    Uses the same numeric columns as the correlation heatmap for consistency.
    Shows pairwise relationships between all selected numeric variables.
    """
    # Select columns for scatter matrix (same as correlation heatmap)
    cols = [
        'num_claims',
        'num_figures',
        'num_inventors',
        'patent_complexity',
        'total_citations'
    ]
    
    # Filter to columns that exist
    existing_cols = [col for col in cols if col in df.columns]
    if len(existing_cols) < 2:
        print("Warning: Need at least 2 numeric columns for scatter matrix. Available columns:", existing_cols)
        return
    
    df_pair = df[existing_cols].dropna()
    
    if len(df_pair) == 0:
        print("Warning: No data available for scatter matrix after filtering.")
        return
    
    # Sample if dataset is too large for performance
    if len(df_pair) > 10000:
        df_pair = df_pair.sample(n=10000, random_state=42)
        print(f"Sampled {len(df_pair):,} rows for scatter matrix visualization")
    
    # Add color column - ONLY use assignee_category (NO countries)
    # NOTE: This is ONLY for color-coding visualization
    if 'assignee_category' in df.columns:
        df_pair['color_col'] = df.loc[df_pair.index, 'assignee_category'].fillna('Unknown')
        color_col_name = 'Assignee Category'
        print(f"Color-coding by Assignee Category: {sorted(df_pair['color_col'].unique())}")
    else:
        # If assignee_category not available, use single color (no grouping, NO countries)
        df_pair['color_col'] = 'All Patents'
        color_col_name = 'All Patents'
        print(f"Note: assignee_category not available. Using single color for all patents (countries NOT used).")
    
    # Create readable column names for display
    display_cols = [get_label(col, readable_names) for col in existing_cols]
    df_pair_display = df_pair[existing_cols + ['color_col']].copy()
    df_pair_display.columns = display_cols + ['color_col']
    
    # Create pairplot
    print(f"\nCreating scatter plot matrix with {len(existing_cols)} variables...")
    print("This may take a moment for large datasets...")
    
    g = sns.pairplot(
        df_pair_display,
        vars=display_cols,
        hue='color_col',
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 15},
        diag_kws={'alpha': 0.7, 'fill': True},
        corner=False
    )
    
    # Set title with proper styling (blue, bold) and adjust position to fit
    g.fig.suptitle('Scatter Plot Matrix: Pairwise Relationships Between Patent Features',
                   fontsize=18, fontweight='bold', color='blue', y=0.98)
    
    # Apply axis label styling (dark red) to all subplots
    for ax in g.axes.flat:
        if ax is not None:
            # Set axis label colors to dark red
            if ax.xaxis.label:
                ax.xaxis.label.set_color('darkred')
            if ax.yaxis.label:
                ax.yaxis.label.set_color('darkred')
            # Increase label font size
            ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout with more space for title
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("Scatter Plot Matrix Summary")
    print(f"{'='*70}")
    print(f"Variables included: {len(existing_cols)}")
    print(f"Variables: {', '.join([get_label(col, readable_names) for col in existing_cols])}")
    print(f"Color coding: {color_col_name}")
    print(f"Sample size: {len(df_pair):,} patents")
    print(f"{'='*70}")


def plot_kde_complexity_ai_vs_non_ai(df, readable_names):
    """
    KDE comparison of patent_complexity for AI vs Non-AI patents.
    Shows the distribution of patent complexity scores for both groups.
    """
    # Filter to required columns and drop missing values
    df2 = df[['patent_complexity', 'is_ai']].dropna().copy()
    
    if len(df2) == 0:
        print("Warning: No data available for KDE plot after filtering.")
        return
    
    # Create AI label column for better legend
    df2['AI label'] = df2['is_ai'].map({True: 'AI patents', False: 'Non-AI patents'})
    
    # Sample data if too large for performance
    df2 = sample_df(df2, n=50000)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create KDE plot with filled areas
    sns.kdeplot(
        data=df2,
        x='patent_complexity',
        hue='AI label',
        fill=True,
        alpha=0.5,
        linewidth=2.5,
        ax=ax
    )
    
    # Set title and labels with larger fonts
    ax.set_title('KDE of Patent Complexity – AI vs Non-AI', 
                fontsize=18, fontweight='bold', pad=20)
    xlabel = readable_names.get('patent_complexity', 'Patent Complexity') if readable_names else 'Patent Complexity'
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel('Density', fontsize=16, fontweight='bold')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Improve legend
    ax.legend(fontsize=12, framealpha=0.95, edgecolor='black', frameon=True)
    
    plt.tight_layout()
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nPatent Complexity Statistics:")
    for ai_type in [True, False]:
        label = 'AI patents' if ai_type else 'Non-AI patents'
        data = df2[df2['is_ai'] == ai_type]['patent_complexity']
        if len(data) > 0:
            print(f"  {label}:")
            print(f"    Mean: {data.mean():.2f}")
            print(f"    Median: {data.median():.2f}")
            print(f"    Std: {data.std():.2f}")


def plot_swarm_ai_complexity_by_assignee_category(df, readable_names):
    """
    Swarm plot: patent complexity for AI patents by assignee category.
    Shows the distribution of complexity scores for AI patents across
    different assignee types (Corporate, Individual, Government).
    """
    # Filter to AI patents only
    df_ai = df[df['is_ai']].copy()
    
    # Select required columns and drop missing values
    df_swarm = df_ai[['patent_complexity', 'assignee_category']].dropna()
    
    if len(df_swarm) == 0:
        print("Warning: No data available for swarm plot after filtering.")
        return
    
    # Sample data if too large for performance
    df_swarm = sample_df(df_swarm, n=8000)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create swarm plot
    sns.swarmplot(
        data=df_swarm,
        x='assignee_category',
        y='patent_complexity',
        size=3,
        alpha=0.7,
        ax=ax
    )
    
    # Set title and labels with larger fonts
    ax.set_title('AI Patent Complexity by Assignee Category', 
                fontsize=18, fontweight='bold', pad=20)
    xlabel = readable_names.get('assignee_category', 'Assignee Category') if readable_names else 'Assignee Category'
    ylabel = readable_names.get('patent_complexity', 'Patent Complexity') if readable_names else 'Patent Complexity'
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics by category
    print("\nAI Patent Complexity by Assignee Category:")
    for category in df_swarm['assignee_category'].unique():
        cat_data = df_swarm[df_swarm['assignee_category'] == category]['patent_complexity']
        if len(cat_data) > 0:
            print(f"  {category}:")
            print(f"    Count: {len(cat_data):,}")
            print(f"    Mean: {cat_data.mean():.2f}")
            print(f"    Median: {cat_data.median():.2f}")


# ============================================================================
# ACADEMIC ANALYSIS PLOTS
# ============================================================================

def plot_academic_percentage_over_time(df, readable_names):
    """
    Line plot showing academic patent percentage over quarters.
    Shows the trend of academic patent percentage across time periods.
    """
    # Checking if required columns exist
    if 'year_quarter' not in df.columns or 'assignee_type_academic' not in df.columns:
        print("Warning: Required columns not found for academic percentage plot.")
        return
    
    # Filtering out rows without required data
    df_q = df.dropna(subset=['year_quarter', 'assignee_type_academic'])
    
    # Calculating academic percentage over time
    academic_trends = df_q.groupby('year_quarter').agg({
        'assignee_type_academic': lambda x: (x == 'Academic').sum(),
        'patent_id': 'count'
    })
    academic_trends.columns = ['academic_count', 'total_count']
    academic_trends['academic_percentage'] = (
        academic_trends['academic_count'] / academic_trends['total_count'] * 100
    )
    academic_trends = academic_trends.sort_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot line
    quarters = list(range(len(academic_trends)))
    ax.plot(quarters, academic_trends['academic_percentage'], 
           marker='o', linewidth=2.5, markersize=8, 
           color='#2E86AB', alpha=0.9)
    
    # Set x-axis labels
    ax.set_xticks(quarters)
    ax.set_xticklabels(academic_trends.index, rotation=45, ha='right')
    
    # Set labels and title with larger fonts
    ax.set_xlabel('Year-Quarter', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage of Patents from Academic Institutions', fontsize=16, fontweight='bold')
    ax.set_title('Academic Patent Percentage Over Time', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Reduce spacing between x-axis points
    ax.set_xlim(-0.3, len(academic_trends) - 0.7)
    
    # Adjust y-axis to reduce unnecessary whitespace
    y_min = academic_trends['academic_percentage'].min()
    y_max = academic_trends['academic_percentage'].max()
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - y_range * 0.1), y_max + y_range * 0.1)
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nAcademic Patent Percentage by Quarter:")
    for quarter, row in academic_trends.iterrows():
        print(f"  {quarter}: {row['academic_percentage']:.2f}% ({row['academic_count']:.0f} / {row['total_count']:.0f})")


def plot_academic_vs_corporate_trends(df, readable_names):
    """
    Line plot comparing absolute counts of academic vs corporate patents over time.
    Shows the trend of both academic and corporate patents across quarters.
    """
    # Checking if required columns exist
    if 'year_quarter' not in df.columns or 'assignee_type_academic' not in df.columns:
        print("Warning: Required columns not found for academic vs corporate trends plot.")
        return
    
    # Filtering out rows without required data
    df_q = df.dropna(subset=['year_quarter', 'assignee_type_academic'])
    
    # Grouping by quarter and assignee type
    counts = (
        df_q
        .groupby(['year_quarter', 'assignee_type_academic'])['patent_id']
        .count()
        .reset_index(name='patent_count')
        .sort_values('year_quarter')
    )
    
    # Filtering to Academic and Corporate only
    counts = counts[counts['assignee_type_academic'].isin(['Academic', 'Corporate'])]
    
    # Pivoting for easier plotting
    pivot_counts = counts.pivot(index='year_quarter', columns='assignee_type_academic', values='patent_count').fillna(0)
    pivot_counts = pivot_counts.sort_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot lines for Academic and Corporate
    quarters = list(range(len(pivot_counts)))
    if 'Academic' in pivot_counts.columns:
        ax.plot(quarters, pivot_counts['Academic'], 
               marker='o', linewidth=2.5, markersize=8, 
               label='Academic', color='#2E86AB', alpha=0.9)
    if 'Corporate' in pivot_counts.columns:
        ax.plot(quarters, pivot_counts['Corporate'], 
               marker='s', linewidth=2.5, markersize=8, 
               label='Corporate', color='#6A8080', alpha=0.9)
    
    # Set x-axis labels
    ax.set_xticks(quarters)
    ax.set_xticklabels(pivot_counts.index, rotation=45, ha='right')
    
    # Set labels and title with larger fonts
    ax.set_xlabel('Year-Quarter', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Patents', fontsize=16, fontweight='bold')
    ax.set_title('Academic vs Corporate Patent Trends Over Time', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add legend with better positioning and larger font
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
             fontsize=12, framealpha=0.95, edgecolor='black', frameon=True)
    
    # Reduce spacing between x-axis points
    ax.set_xlim(-0.3, len(pivot_counts) - 0.7)
    
    # Adjust y-axis to reduce unnecessary whitespace
    y_min = pivot_counts.values.min()
    y_max = pivot_counts.values.max()
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - y_range * 0.05), y_max + y_range * 0.05)
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nAcademic vs Corporate Patent Counts by Quarter:")
    print(pivot_counts)


def plot_academic_by_region(df, readable_names):
    """
    Bar chart showing academic percentage by US region.
    Only plots if US region data is available.
    """
    # Checking if required columns exist
    if 'us_region' not in df.columns or 'assignee_type_academic' not in df.columns:
        print("Warning: Required columns not found for academic by region plot.")
        return
    
    # Filtering to US data and excluding Unknown region
    us_data = df[(df['disambig_country'] == 'US') & (df['us_region'] != 'Unknown')].copy()
    
    if len(us_data) == 0:
        print("Warning: No US region data available for plotting.")
        return
    
    # Calculating academic percentage by region
    academic_by_region = pd.crosstab(
        us_data['us_region'], 
        us_data['assignee_type_academic'], 
        normalize='index'
    ) * 100
    
    if 'Academic' not in academic_by_region.columns:
        print("Warning: No academic data found for region plot.")
        return
    
    academic_pct = academic_by_region['Academic'].sort_values(ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(range(len(academic_pct)), academic_pct.values, 
                 color='#2E86AB', alpha=0.8)
    
    # Set x-axis labels
    ax.set_xticks(range(len(academic_pct)))
    ax.set_xticklabels(academic_pct.index, rotation=45, ha='right')
    
    # Set labels and title with larger fonts
    ax.set_xlabel('US Region', fontsize=16, fontweight='bold')
    ax.set_ylabel('Academic Patent Percentage', fontsize=16, fontweight='bold')
    ax.set_title('Academic Patent Percentage by US Region', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (region, pct) in enumerate(academic_pct.items()):
        ax.text(i, pct + 0.5, f'{pct:.1f}%', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nAcademic Patent Percentage by US Region:")
    for region, pct in academic_pct.items():
        print(f"  {region}: {pct:.2f}%")


def plot_academic_by_technology(df, readable_names):
    """
    Bar chart showing academic percentage by CPC major category.
    Sorted by academic percentage in descending order.
    """
    # Checking if required columns exist
    if 'cpc_major' not in df.columns or 'assignee_type_academic' not in df.columns:
        print("Warning: Required columns not found for academic by technology plot.")
        return
    
    # Filtering to data with CPC major category
    cpc_data = df[df['cpc_major'].notna()].copy()
    
    if len(cpc_data) == 0:
        print("Warning: No CPC major category data available for plotting.")
        return
    
    # Calculating academic percentage by technology
    academic_by_tech = pd.crosstab(
        cpc_data['cpc_major'],
        cpc_data['assignee_type_academic'],
        normalize='index'
    ) * 100
    
    if 'Academic' not in academic_by_tech.columns:
        print("Warning: No academic data found for technology plot.")
        return
    
    academic_pct = academic_by_tech['Academic'].sort_values(ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.barh(range(len(academic_pct)), academic_pct.values, 
                  color='#2E86AB', alpha=0.8)
    
    # Set y-axis labels
    ax.set_yticks(range(len(academic_pct)))
    ax.set_yticklabels(academic_pct.index)
    
    # Set labels and title with larger fonts
    ax.set_xlabel('Academic Patent Percentage', fontsize=16, fontweight='bold')
    ax.set_ylabel('Technology Category (CPC Major)', fontsize=16, fontweight='bold')
    ax.set_title('Academic Patent Percentage by Technology Category', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid with subtle appearance
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (tech, pct) in enumerate(academic_pct.items()):
        ax.text(pct + 0.5, i, f'{pct:.1f}%', 
               ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Tight layout to reduce whitespace
    plt.tight_layout()
    apply_axis_colors(ax)
    plt.show()
    
    # Print summary statistics
    print("\nAcademic Patent Percentage by Technology Category:")
    for tech, pct in academic_pct.items():
        print(f"  {tech}: {pct:.2f}%")


def create_all_ai_plots(df, readable_names):
    """Create all AI analysis visualizations"""
    
    print("\n" + "="*70)
    print("Creating AI Analysis Plots")
    print("="*70)
    
    # Check if AI data exists
    if 'is_ai' not in df.columns:
        print("\nError: 'is_ai' column not found in dataset.")
        print("Please run data_processing.py first to generate AI flags.")
        return
    
    ai_count = df['is_ai'].sum() if 'is_ai' in df.columns else 0
    total_count = len(df)
    ai_pct = (ai_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\nDataset Statistics:")
    print(f"  - Total patents: {total_count:,}")
    print(f"  - AI patents: {ai_count:,} ({ai_pct:.2f}%)")
    print(f"  - Non-AI patents: {total_count - ai_count:,} ({100 - ai_pct:.2f}%)")
    
    # 1. Line plot: AI vs Non-AI trends
    print("\n1. Creating AI vs Non-AI Trends (Line Plot)...")
    plot_ai_vs_non_ai_over_quarters(df, readable_names)
    
    # 2. Stacked bar: AI vs Non-AI composition
    print("\n2. Creating AI vs Non-AI Composition (Stacked Bar)...")
    plot_ai_vs_non_ai_stacked_bar(df, readable_names)
    
    # 3. AI-only correlation heatmap
    print("\n3. Creating AI-only Correlation Heatmap...")
    plot_ai_only_corr_heatmap(df, readable_names)
    
    # 4. KDE: AI vs Non-AI complexity
    print("\n4. Creating KDE: Patent Complexity (AI vs Non-AI)...")
    plot_kde_complexity_ai_vs_non_ai(df, readable_names)
    
    # 5. Swarm: AI complexity by assignee category
    print("\n5. Creating Swarm Plot: AI Complexity by Assignee Category...")
    plot_swarm_ai_complexity_by_assignee_category(df, readable_names)
    
    print("\n" + "="*70)
    print("All AI analysis plots displayed!")
    print("="*70)


def create_all_academic_plots(df, readable_names):
    """Create all academic analysis visualizations"""
    
    print("\n" + "="*70)
    print("Creating Academic Analysis Plots")
    print("="*70)
    
    # Check if academic classification exists
    if 'assignee_type_academic' not in df.columns:
        print("\nError: 'assignee_type_academic' column not found in dataset.")
        print("Please run data_processing.py first to generate academic classification.")
        return
    
    # Print summary statistics
    academic_count = (df['assignee_type_academic'] == 'Academic').sum()
    total_count = len(df)
    academic_pct = (academic_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\nDataset Statistics:")
    print(f"  - Total patents: {total_count:,}")
    print(f"  - Academic patents: {academic_count:,} ({academic_pct:.2f}%)")
    print(f"  - Corporate patents: {(df['assignee_type_academic'] == 'Corporate').sum():,}")
    
    # 1. Academic percentage over time
    print("\n1. Creating Academic Percentage Over Time plot...")
    plot_academic_percentage_over_time(df, readable_names)
    
    # 2. Academic vs Corporate trends
    print("\n2. Creating Academic vs Corporate Trends plot...")
    plot_academic_vs_corporate_trends(df, readable_names)
    
    # 3. Academic by region (if US data available)
    if 'us_region' in df.columns and (df['disambig_country'] == 'US').any():
        print("\n3. Creating Academic by Region plot...")
        plot_academic_by_region(df, readable_names)
    
    # 4. Academic by technology
    if 'cpc_major' in df.columns:
        print("\n4. Creating Academic by Technology plot...")
        plot_academic_by_technology(df, readable_names)
    
    print("\n" + "="*70)
    print("All academic analysis plots displayed!")
    print("="*70)


def create_all_selected_plots(df, readable_names):
    """Create all selected and improved plots"""
    
    print("\n" + "="*70)
    print("Creating Selected Best Plots")
    print("="*70)
    
    # Plot 1 - Line Plot: Technology Evolution
    print("\nPlot 1 - Creating Technology Evolution plot...")
    plot_patent_trends_by_technology(df, readable_names)
    
    # Plot 2 - Horizontal Bar Chart: Top Companies
    print("\nPlot 2 - Creating Top Companies Horizontal Bar Chart...")
    plot_top_companies_horizontal_bar(df, readable_names)
    
    # Plot 3 - Boxplot: Before vs After Outlier Removal
    print("\nPlot 3 - Creating Boxplot: Before vs After Outlier Removal...")
    df_cleaned = plot_boxplot_before_after(df, readable_names)
    
    # Plot 4 - Swarm Plot: Claims by CPC Section (using cleaned data)
    print("\nPlot 4 - Creating Swarm Plot: Claims by CPC Section...")
    plot_swarm_claims_by_cpc_section(df_cleaned, readable_names)
    
    # Plot 5 - Strip Plot: Citations by Patent Type
    print("\nPlot 5 - Creating Strip Plot: Citations by Patent Type...")
    plot_strip_type_vs_citations(df, readable_names)
    
    # Plot 6 - Hexbin Plot: Inventors vs Citations (US Patents)
    print("\nPlot 6 - Creating Hexbin Plot: Inventors vs Citations (US Patents)...")
    plot_hexbin_inventors_vs_citations_us(df, readable_names)
    
    # Plot 7 - Contour Plot: Density of Claims vs Figures
    print("\nPlot 7 - Creating Contour Plot: Density of Claims vs Figures...")
    plot_contour_claims_figures_density(df, readable_names)
    
    # Plot 8 - 3D Scatter: Figures vs Claims vs Complexity
    print("\nPlot 8 - Creating 3D Scatter: Figures vs Claims vs Complexity...")
    plot_3d_scatter_figures_claims_complexity(df, readable_names)
    
    # Plot 9 - AI Heatmap: Region × AI Type
    print("\nPlot 9 - Creating AI Heatmap: Region × AI Type...")
    plot_ai_heatmap_by_region_type(df, readable_names)
    
    # Academic plots (if academic data is available)
    if 'assignee_type_academic' in df.columns:
        # Plot 10 - Violin: Corporate vs Academic Claims
        print("\nPlot 10 - Creating Violin Plot: Corporate vs Academic Claims...")
        plot_violin_corporate_vs_academic_claims(df, readable_names)
        
        # Plot 11 - KDE Plot: Corporate vs Academic Patent Complexity
        print("\nPlot 11 - Creating KDE Plot: Corporate vs Academic Patent Complexity...")
        plot_kde_complexity_corp_vs_academic(df, readable_names)
        
        # Plot 12 - Histogram: CPC by Assignee Type
        print("\nPlot 12 - Creating Histogram: CPC Technology Profile by Assignee Type...")
        plot_histogram_cpc_by_assignee_type(df, readable_names)
        
        # Plot 13 - Multivariate Boxen Plot: Corporate vs Academic Claims by CPC Category
        print("\nPlot 13 - Creating Multivariate Boxen Plot: Corporate vs Academic Claims by CPC Category...")
        plot_boxen_claims_by_cpc_and_assignee(df, readable_names)
    else:
        print("\nWarning: Academic dataset not available. Skipping academic comparison plots.")
        print("Run data_processing.py first to generate academic classification.")
    
    # Plot 15 - Area Plot: Total Claims by US Region over Months
    print("\nPlot 15 - Creating Area Plot: Total Claims by US Region over Months...")
    plot_area_claims_by_us_region_over_months(df, readable_names)
    
    # Plot 16 - Stacked Bar Plot: CPC Major Category by US Region
    print("\nPlot 16 - Creating Stacked Bar Plot: CPC Major Category by US Region...")
    plot_stacked_bar_cpc_by_region(df, readable_names)
    
    # Plot 18 - Pie Chart: Percentage of Patents per US Region
    print("\nPlot 18 - Creating Pie Chart: Percentage of Patents per US Region...")
    plot_pie_chart_us_region_patents(df, readable_names)
    
    # Plot 19 - Rug Plot: Number of Claims for Northeast Region (US)
    print("\nPlot 19 - Creating Rug Plot: Number of Claims for Northeast Region (US)...")
    plot_rug_claims_northeast(df, readable_names)
    
    # Plot 20 - Four-Pie Chart: Distribution Overview
    print("\nPlot 20 - Creating Four-Pie Chart: Distribution Overview...")
    plot_four_pies(df, readable_names)
    
    # Plot 17 - Violin Plot: Corporate vs Academic Patent Volume
    if 'assignee_type_academic' in df.columns:
        print("\nPlot 17 - Creating Violin Plot: Corporate vs Academic Patent Volume...")
        plot_violin_corporate_vs_academic_patent_volume(df, readable_names)
    
    # Plot 21 - Count Plot: Number of Patents by CPC Major Category
    print("\nPlot 21 - Creating Count Plot: Number of Patents by CPC Major Category...")
    plot_count_cpc_major(df, readable_names)
    
    # Plot 22 - Pair Plot: Structural Variables (Corporate vs Academic)
    if 'assignee_type_academic' in df.columns:
        print("\nPlot 22 - Creating Pair Plot: Structural Variables (Corporate vs Academic)...")
        plot_pair_structural_variables(df, readable_names)
    
    # Plot 23 - QQ-Plot: Number of Claims vs Theoretical Normal
    print("\nPlot 23 - Creating QQ-Plot: Number of Claims vs Theoretical Normal...")
    plot_qq_num_claims(df, readable_names)
    
    # Plot 24 - Regression Plot: Citation Count vs Number of Claims
    print("\nPlot 24 - Creating Regression Plot: Citation Count vs Number of Claims...")
    plot_reg_claims_vs_citations(df, readable_names)
    
    # Plot 25 - Joint Plot: Claims vs Citations (KDE + Scatter)
    print("\nPlot 25 - Creating Joint Plot: Claims vs Citations (KDE + Scatter)...")
    plot_joint_claims_citations(df, readable_names)
    
    # Plot 26 - Cluster Map: CPC Major Category Distribution by US Region
    print("\nPlot 26 - Creating Cluster Map: CPC Major Category Distribution by US Region...")
    plot_clustermap_cpc_by_region(df, readable_names)
    
    # Plot 27 - Feature Summary Table
    print("\nPlot 27 - Creating Feature Summary Table...")
    plot_feature_summary_table(df, readable_names)
    
    # Plot 28 - Four-Pie Overview
    print("\nPlot 28 - Creating Four-Pie Overview...")
    plot_four_pies_overview(df, readable_names)
    
    # Plot 29 - PCA Analysis
    print("\nPlot 29 - Creating PCA Analysis plot...")
    plot_pca_analysis(df, readable_names)
    
    # Plot 30 - Scatter Plot Matrix
    print("\nPlot 30 - Creating Scatter Plot Matrix...")
    plot_scatter_matrix(df, readable_names)
    
    print("\n" + "="*70)
    print("All selected plots displayed!")
    print("="*70)


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate selected patent analysis plots')
    parser.add_argument(
        '--plot',
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        help=('Plot number to generate (1-13, 15-26, 27 for feature summary table, '
              '28 for four-pie overview, 29 for PCA analysis, 30 for scatter plot matrix). '
              'If not specified, generates all plots.')
    )
    parser.add_argument(
        '--ai',
        action='store_true',
        help='Generate all AI analysis plots'
    )
    parser.add_argument(
        '--academic',
        action='store_true',
        help='Generate all academic analysis plots'
    )
    args = parser.parse_args()
    
    print("Loading and preparing dataset...")
    # Loading all countries (not filtering to US) to allow international comparisons
    df, readable_names = load_and_prepare_data_for_plots(filter_us=False)
    
    if df is None:
        print("Error: Could not load dataset. Exiting.")
        exit(1)
    
    if args.plot:
        # Run specific plot
        print(f"\n{'='*70}")
        print(f"Generating Plot {args.plot} only...")
        print(f"{'='*70}")
        
        if args.plot == 1:
            plot_patent_trends_by_technology(df, readable_names)
        elif args.plot == 2:
            plot_top_companies_horizontal_bar(df, readable_names)
        elif args.plot == 3:
            plot_boxplot_before_after(df, readable_names)
        elif args.plot == 4:
            # For plot 4, we need cleaned data (outliers removed)
            df_cleaned = remove_claim_outliers_iqr(df)
            plot_swarm_claims_by_cpc_section(df_cleaned, readable_names)
        elif args.plot == 5:
            plot_strip_type_vs_citations(df, readable_names)
        elif args.plot == 6:
            plot_hexbin_inventors_vs_citations_us(df, readable_names)
        elif args.plot == 7:
            plot_contour_claims_figures_density(df, readable_names)
        elif args.plot == 8:
            plot_3d_scatter_figures_claims_complexity(df, readable_names)
        elif args.plot == 9:
            plot_ai_heatmap_by_region_type(df, readable_names)
        elif args.plot in [10, 11, 12, 13]:
            # Check if academic data is available
            if 'assignee_type_academic' not in df.columns:
                print("Error: Academic dataset not available. Run data_processing.py first.")
            else:
                if args.plot == 10:
                    plot_violin_corporate_vs_academic_claims(df, readable_names)
                elif args.plot == 11:
                    plot_kde_complexity_corp_vs_academic(df, readable_names)
                elif args.plot == 12:
                    plot_histogram_cpc_by_assignee_type(df, readable_names)
                elif args.plot == 13:
                    plot_boxen_claims_by_cpc_and_assignee(df, readable_names)
        elif args.plot == 15:
            plot_area_claims_by_us_region_over_months(df, readable_names)
        elif args.plot == 16:
            plot_stacked_bar_cpc_by_region(df, readable_names)
        elif args.plot == 17:
            # Check if academic data is available
            if 'assignee_type_academic' not in df.columns:
                print("Error: Academic dataset not available. Run data_processing.py first.")
            else:
                plot_violin_corporate_vs_academic_patent_volume(df, readable_names)
        elif args.plot == 18:
            plot_pie_chart_us_region_patents(df, readable_names)
        elif args.plot == 19:
            plot_rug_claims_northeast(df, readable_names)
        elif args.plot == 20:
            plot_four_pies(df, readable_names)
        elif args.plot == 21:
            plot_count_cpc_major(df, readable_names)
        elif args.plot == 22:
            # Check if academic data is available
            if 'assignee_type_academic' not in df.columns:
                print("Error: Academic dataset not available. Run data_processing.py first.")
            else:
                plot_pair_structural_variables(df, readable_names)
        elif args.plot == 23:
            plot_qq_num_claims(df, readable_names)
        elif args.plot == 24:
            plot_reg_claims_vs_citations(df, readable_names)
        elif args.plot == 25:
            plot_joint_claims_citations(df, readable_names)
        elif args.plot == 26:
            plot_clustermap_cpc_by_region(df, readable_names)
        elif args.plot == 27:
            plot_feature_summary_table(df, readable_names)
        elif args.plot == 28:
            plot_four_pies_overview(df, readable_names)
        elif args.plot == 29:
            plot_pca_analysis(df, readable_names)
        elif args.plot == 30:
            plot_scatter_matrix(df, readable_names)
        
        print(f"\n{'='*70}")
        print(f"Plot {args.plot} displayed!")
        print(f"{'='*70}")
    elif args.ai:
        # Run all AI plots
        create_all_ai_plots(df, readable_names)
    elif args.academic:
        # Run all academic plots
        create_all_academic_plots(df, readable_names)
    else:
        # Run all plots
        create_all_selected_plots(df, readable_names)

