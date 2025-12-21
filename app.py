from dash import Dash, html, dcc, callback, Output, Input, State
from dash import dash_table

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from scipy import stats
import warnings
import os
from typing import List

READABLE_NAMES = {
    'patent_id': 'Patent ID',
    'patent_type': 'Patent Type',
    'patent_date': 'Patent Date',
    'wipo_kind': 'WIPO Kind',
    'num_claims': 'Number of Claims',
    'grant_year': 'Grant Year',
    'num_figures': 'Number of Figures',
    'citation_count': 'Citation Count',
    'foreign_citation_count': 'Foreign Citations',
    'num_assignees': 'Number of Assignees',
    'assignee_type': 'Assignee Type',
    'disambig_city': 'City',
    'disambig_state': 'State',
    'disambig_country': 'Country',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'cpc_section': 'CPC Section',
    'cpc_class': 'CPC Class',
    'cpc_subclass': 'CPC Subclass',
    'cpc_group': 'CPC Group',
    'cpc_type': 'CPC Type',
    'disambig_assignee_organization': 'Organization',
    'location_id': 'Location ID',
    'num_assignees_from_tsv': 'Assignees Count',
    'num_inventors': 'Number of Inventors',
    'us_region': 'US Region',
    'cpc_major': 'CPC Major Category',
    'assignee_type_clean': 'Assignee Type',
    'assignee_category': 'Assignee Category',
    'filing_year': 'Filing Year',
    'filing_quarter': 'Filing Quarter',
    'year_quarter': 'Year-Quarter',
    'us_citations': 'US Citations',
    'total_citations': 'Total Citations',
    'patent_complexity': 'Patent Complexity',
    'citation_per_claim': 'Citations per Claim',
    'foreign_citation_ratio': 'Foreign Citation Ratio',
    'figures_per_claim': 'Figures per Claim',
    'month': 'Month',
    'grant_date': 'Grant Date',
    'quarter': 'Grant Quarter',
    'year': 'Grant Year',
    'grant_month': 'Grant Month',
    'is_ai': 'Is AI Patent',
    'ai_bucket': 'AI Category',
    'is_academic': 'Is Academic Institution',
    'is_academic_enhanced': 'Is Academic (Enhanced)',
    'assignee_type_academic': 'Assignee Type (Academic)'
}

warnings.filterwarnings(
    "ignore",
    message="Parsing dates involving a day of month without a year specified is ambiguious",
    category=DeprecationWarning,
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

STYLES = {
    'primary_bg': '#F5F8FA',
    'secondary_bg': 'white',
    'primary_text': '#2F4F4F',
    'secondary_text': '#6A8080',
    'accent_blue': '#87CEEB',
    'accent_green': '#A2D9CE',
    'border_color': '#E0E6EB',
    'font_family': 'Segoe UI, Roboto, sans-serif',
}


df = None
original_length = 0

app_dir = os.path.dirname(os.path.abspath(__file__))
local_path = os.path.join(app_dir, 'preprocessed_sample.csv')

try:
    if os.path.exists(local_path):
        df = pd.read_csv(local_path, low_memory=False)
        print(f"✓ Loaded preprocessed sample dataset from local file ({len(df):,} rows)")
    else:
        print(f"⚠ Local file not found: {local_path}")
        df = None
except Exception as e:
    print(f"⚠ Error loading preprocessed sample dataset from local file: {e}")
    df = None

if df is not None and not df.empty:
    if 'patent_date' in df.columns:
        df['patent_date'] = pd.to_datetime(df['patent_date'], errors='coerce')
    if 'grant_date' in df.columns:
        df['grant_date'] = pd.to_datetime(df['grant_date'], errors='coerce')
    original_length = len(df)
else:
    df = None
    original_length = 0

if df is None or df.empty:
    data = {
        'patent_id': [f'P{i:03}' for i in range(50)],
        'patent_date': pd.to_datetime(pd.date_range(start='2024-01-01', periods=50, freq='W')),
        'disambig_country': ['US', 'DE', 'JP', 'GB', 'CN'] * 10,
        'cpc_class': ['A61K', 'G06F', 'H04L', 'C07D', 'B29C'] * 10,
        'num_claims': [i % 20 + 5 for i in range(50)],
        'citation_count': [i % 30 + 1 for i in range(50)],
        'num_figures': [i % 10 + 1 for i in range(50)],
    }
    df = pd.DataFrame(data)
    df['filing_year'] = df['patent_date'].dt.year
    df['filing_quarter'] = df['patent_date'].dt.quarter
    df['year_quarter'] = df['filing_year'].astype(str) + ' Q' + df['filing_quarter'].astype(str)
    original_length = len(df)

if 'abstract_length' not in df.columns:
    df['abstract_length'] = df['num_figures'].fillna(0) * 10

country_column = 'disambig_country'
assignee_category_column = 'assignee_category'
year_column = 'filing_year'
quarter_column = 'year_quarter'

numeric_cols = sorted(
    [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['year', 'grant_month']]
)
categorical_cols = sorted(
    [
        c
        for c in df.columns
        if c not in numeric_cols
        and df[c].dtype == 'object'
        and c not in ['patent_title', 'summary_text']
    ]
)

EXCLUDED_NUMERIC_COLS = [
    'patent_id', 'location_id', 'location_id_inventor',
    'abstract_length', 'assignee_type',
    'us_citations', 'grant_quarter', 'quarter',
    'grant_year', 'year', 'grant_month', 'month',
    'filing_year', 'filing_quarter',
    'assignee_count', 'num_assignees_from_tsv',
    'num_assignees', 'citation_count',
    'latitude', 'longitude',
    'foreign_citation_ratio', 'figures_per_claim', 'citation_per_claim'
]

suitable_numeric_cols = [c for c in numeric_cols if c not in EXCLUDED_NUMERIC_COLS]

EXCLUDED_CATEGORICAL_COLS = [
    'patent_id', 'cpc_group', 'cpc_subclass', 'cpc_type',
    'gender_code', 'location_id', 'inventor_location_id', 'location_id_inventor'
]

suitable_categorical_cols = [
    c for c in categorical_cols 
    if c.lower() not in [excluded.lower() for excluded in EXCLUDED_CATEGORICAL_COLS]
]

TOP_10_CATEGORICAL_COLS = [
    'organization', 'disambig_assignee_organization',
    'city', 'disambig_city',
    'country', 'disambig_country',
    'state', 'disambig_state',
    'us_region'
]

if assignee_category_column in df.columns:
    assignee_type_options = [
        {'label': str(i), 'value': str(i)} 
        for i in sorted(df[assignee_category_column].dropna().unique())
    ]
else:
    assignee_type_options = []

quarter_options = [{'label': str(i), 'value': str(i)} for i in sorted(df[quarter_column].dropna().unique())]

quarter_values = [opt['value'] for opt in quarter_options]
quarter_marks = {
    i: {
        'label': q,
        'style': {'color': STYLES['secondary_text'], 'fontSize': '0.8em'},
    }
    for i, q in enumerate(quarter_values)
}



def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=message,
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(color=STYLES['secondary_text'], family=STYLES['font_family'], size=14),
        xaxis={'visible': False},
        yaxis={'visible': False},
    )
    return fig


def make_region_area_plot(df_in: pd.DataFrame) -> go.Figure:
    if 'disambig_country' not in df_in.columns or 'us_region' not in df_in.columns:
        return empty_figure("US regional information not available.")
    df_us = df_in[
        (df_in['disambig_country'] == 'US')
        & df_in['us_region'].notna()
        & (df_in['us_region'] != 'Unknown')
        & df_in['num_claims'].notna()
    ].copy()
    if 'grant_date' not in df_us.columns:
        if 'patent_date' in df_us.columns:
            df_us['grant_date'] = df_us['patent_date']
        else:
            return empty_figure("Grant date not available for regional plot.")
    df_us['year_month'] = df_us['grant_date'].dt.to_period('M').astype(str)
    grp = df_us.groupby(['year_month', 'us_region'])['num_claims'].sum().reset_index()
    if grp.empty:
        return empty_figure("No US regional claims data.")
    fig = px.area(
        grp,
        x='year_month',
        y='num_claims',
        color='us_region',
        title='Total Number of Claims by US Region over Months',
    )
    fig.update_layout(
        xaxis_title='Year-Month',
        yaxis_title='Total Claims',
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def make_ai_heatmap(df_in: pd.DataFrame) -> go.Figure:
    required_cols = ['disambig_country', 'is_ai', 'us_region', 'ai_bucket']
    if not all(c in df_in.columns for c in required_cols):
        return empty_figure("AI regional information not available.")
    df_ai = df_in[
        (df_in['disambig_country'] == 'US')
        & (df_in['is_ai'] == True)
        & df_in['us_region'].notna()
        & (df_in['us_region'] != 'Unknown')
        & df_in['ai_bucket'].notna()
    ].copy()
    if df_ai.empty:
        return empty_figure("No AI patents found for heatmap.")
    pivot = pd.crosstab(df_ai['us_region'], df_ai['ai_bucket'])
    fig = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="YlOrRd",
        title="AI Patents by US Region × AI Type",
    )
    fig.update_layout(
        xaxis_title="AI Category",
        yaxis_title="US Region",
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def make_patent_map(df_in: pd.DataFrame, map_scope: str, selected_state: str = None, selected_country: str = None) -> go.Figure:
    """
    Create a choropleth map showing patent distribution.
    Supports US state map with optional state filtering, or specific country view.
    
    Args:
        df_in: Filtered dataframe
        map_scope: 'us' or 'country'
        selected_state: Optional state code to filter (e.g., 'CA', 'NY') - for US scope
        selected_country: Optional country code to show (e.g., 'US', 'DE', 'JP') - for country scope
    """
    if 'disambig_country' not in df_in.columns:
        return empty_figure("Country information not available.")
    
    df_map = df_in.copy()
    
    # Apply state filter if provided
    if selected_state and selected_state != 'ALL':
        if 'disambig_state' not in df_map.columns:
            return empty_figure("State information not available.")
        df_map = df_map[df_map['disambig_state'].str.upper() == selected_state.upper()]
    
    
    if df_map.empty:
        return empty_figure("No data available for selected filters.")
    
    if map_scope == 'country' and selected_country:
        # Show specific country on world map
        if 'disambig_country' not in df_map.columns:
            return empty_figure("Country information not available.")
        
        # Filter to selected country
        df_country = df_map[df_map['disambig_country'] == selected_country].copy()
        if df_country.empty:
            return empty_figure(f"No data available for {selected_country}.")
        
        # Create country-level map showing only the selected country
        country_counts = pd.DataFrame({
            'country': [selected_country],
            'patent_count': [len(df_country)]
        })
        
        fig = px.choropleth(
            country_counts,
            locations='country',
            locationmode='country names',
            color='patent_count',
            color_continuous_scale='Viridis',
            title=f'Number of Patents in {selected_country}: {len(df_country):,}',
            labels={'patent_count': 'Number of Patents', 'country': 'Country'},
            hover_data={'country': True, 'patent_count': True},
        )
        
        # Customize hover template
        fig.update_traces(
            hovertemplate='<b>%{location}</b><br>Number of Patents: %{z:,}<extra></extra>'
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                bgcolor='rgba(0,0,0,0)',
                center=dict(lon=0, lat=0),  # Center on selected country if possible
                scope='world',
            ),
            paper_bgcolor=STYLES['secondary_bg'],
            plot_bgcolor=STYLES['secondary_bg'],
            font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
        )
        
    else:  # US map (default)
        if 'disambig_state' not in df_map.columns:
            return empty_figure("US state information not available.")
        
        # Filter to US only
        df_us = df_map[df_map['disambig_country'] == 'US'].copy()
        
        if df_us.empty:
            return empty_figure("No US patent data available.")
        
        # If a specific state is selected, show that state only
        if selected_state and selected_state != 'ALL':
            df_us = df_us[df_us['disambig_state'].str.upper() == selected_state.upper()]
            if df_us.empty:
                return empty_figure(f"No data available for {selected_state}.")
            
            # Show the selected state only
            state_count = len(df_us)
            title = f'Number of Patents in {selected_state}: {state_count:,}'
            
            # Create a single-state map by showing only that state
            state_counts = pd.DataFrame({
                'state': [selected_state.upper()],
                'patent_count': [len(df_us)]
            })
            
            # Get cities for this state
            if 'disambig_city' in df_us.columns:
                cities_with_patents = df_us[df_us['disambig_city'].notna()]['disambig_city'].value_counts()
                city_info = cities_with_patents.to_dict()
                state_counts['cities_info'] = [city_info]
            else:
                state_counts['cities_info'] = [{}]
        else:
            # Count patents per state for all states and collect city information
            state_counts = df_us['disambig_state'].value_counts().reset_index()
            state_counts.columns = ['state', 'patent_count']
            
            # For each state, collect city information
            if 'disambig_city' in df_us.columns:
                cities_info_list = []
                for state in state_counts['state']:
                    state_data = df_us[df_us['disambig_state'].str.upper() == state.upper()]
                    cities_with_patents = state_data[state_data['disambig_city'].notna()]['disambig_city'].value_counts()
                    cities_info_list.append(cities_with_patents.to_dict())
                state_counts['cities_info'] = cities_info_list
            else:
                state_counts['cities_info'] = [{}] * len(state_counts)
            
            title = 'Number of Patents per US State'
        
        state_counts['state'] = state_counts['state'].str.upper()
        
        # Format cities info for hover display
        def format_cities_info(cities_dict):
            if not cities_dict:
                return "No city data"
            # Sort by patent count (descending) and take top 10
            sorted_cities = sorted(cities_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            city_lines = [f"{city}: {count:,} patents" for city, count in sorted_cities]
            if len(cities_dict) > 10:
                city_lines.append(f"... and {len(cities_dict) - 10} more cities")
            return "<br>".join(city_lines)
        
        state_counts['cities_text'] = state_counts['cities_info'].apply(format_cities_info)
        
        # Calculate better color range based on data distribution
        min_count = state_counts['patent_count'].min()
        max_count = state_counts['patent_count'].max()
        # Use a more compressed range to make lower values more visible
        # This helps differentiate states with lower patent counts
        if max_count > min_count:
            color_max = max(min_count + (max_count - min_count) * 0.3, max_count * 0.2)
        else:
            color_max = max_count
        
        fig = px.choropleth(
            state_counts,
            locations='state',
            locationmode='USA-states',
            color='patent_count',
            color_continuous_scale='Blues',
            range_color=[min_count, color_max],  
            scope='usa',
            title=title,
            labels={'patent_count': 'Number of Patents', 'state': 'State'},
            hover_data={'state': True, 'patent_count': True, 'cities_text': True},
        )
        
        fig.update_layout(
            geo=dict(
                bgcolor='rgba(0,0,0,0)',
                lakecolor='rgba(0,0,0,0)',
                landcolor='rgba(0,0,0,0)',
                showlakes=False,
                showland=False,
                subunitcolor='white',
                showframe=False,
            ),
        )
        
        # Customize hover template to show cities
        fig.update_traces(
            hovertemplate='<b>%{location}</b><br>' +
                         'Total Patents: %{z:,}<br>' +
                         '<b>Cities with Patents:</b><br>%{customdata[0]}<extra></extra>',
            customdata=state_counts[['cities_text']].values
        )
    
    fig.update_layout(
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    
    # Update hover template
    if map_scope == 'world':
        fig.update_traces(
            hovertemplate='<b>%{location}</b><br>Patents: %{z:,}<extra></extra>'
        )
    else:
        fig.update_traces(
            hovertemplate='<b>%{location}</b><br>Patents: %{z:,}<extra></extra>'
        )
    
    return fig


def make_pca_scatter(df_in: pd.DataFrame, numeric_for_pca: List[str]) -> go.Figure:
    cols = [c for c in numeric_for_pca if c in df_in.columns]
    if len(cols) < 2:
        return empty_figure("Select at least two numeric columns for PCA.")
    subset = df_in[cols].dropna()
    if subset.empty:
        return empty_figure("No data available for PCA with current filters.")
    # Standardize roughly by centering
    subset_centered = subset - subset.mean(axis=0)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(subset_centered.values)
    pca_df = pd.DataFrame(
        comps,
        columns=['PC1', 'PC2'],
        index=subset.index,
    )
    pca_df['patent_id'] = df_in.loc[pca_df.index, 'patent_id'] if 'patent_id' in df_in.columns else pca_df.index
    color_col = 'assignee_category' if 'assignee_category' in df_in.columns else country_column
    pca_df[color_col] = df_in.loc[pca_df.index, color_col] if color_col in df_in.columns else "All"
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color=color_col,
        hover_name='patent_id',
        title="PCA (2D) of Selected Numeric Features",
    )
    var_explained = pca.explained_variance_ratio_.sum()
    fig.update_layout(
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
        xaxis_title="PC1",
        yaxis_title="PC2",
        title=f"PCA (2D) of Selected Numeric Features (explained variance ≈ {var_explained:.2%})",
    )
    return fig


def make_overview_cpc_count(filtered_df: pd.DataFrame, scope: str, selected_state: str = None) -> go.Figure:
    """
    Overview figure: count plot of CPC major category vs number of patents.
    Scope:
      - 'ALL': use full filtered_df
      - 'US': restrict to US patents
      - 'US_REGIONS': US patents by region (grouped bars)
    """
    if 'cpc_major' not in filtered_df.columns:
        return empty_figure("CPC major category information not available.")

    # Apply scope-specific restriction
    df_plot = filtered_df.copy()
    title_suffix = ""

    if scope == 'US':
        if 'disambig_country' not in df_plot.columns:
            return empty_figure("Country information not available for US-only view.")
        df_plot = df_plot[df_plot['disambig_country'] == 'US']
        if selected_state and selected_state != 'ALL':
            if 'disambig_state' in df_plot.columns:
                df_plot = df_plot[df_plot['disambig_state'] == selected_state]
                title_suffix = f" ({selected_state})"
            else:
                title_suffix = " (US only)"
        else:
            title_suffix = " (US only)"
    elif scope == 'US_REGIONS':
        required_cols = ['disambig_country', 'us_region']
        if not all(c in df_plot.columns for c in required_cols):
            return empty_figure("US regional information not available.")
        df_plot = df_plot[
            (df_plot['disambig_country'] == 'US')
            & df_plot['us_region'].notna()
            & (df_plot['us_region'] != 'Unknown')
        ]
        title_suffix = " (US regions)"

    if df_plot.empty:
        return empty_figure("No data available for the selected scope / filters.")

    if scope == 'US_REGIONS':
        grouped = (
            df_plot.groupby(['cpc_major', 'us_region'])['patent_id']
            .nunique()
            .reset_index(name='num_patents')
        )
        fig = px.bar(
            grouped,
            x='cpc_major',
            y='num_patents',
            color='us_region',
            barmode='group',
            title="Patent Major Categories Overview" + title_suffix,
        )
        fig.update_layout(legend_title_text="US Region")
    else:
        grouped = (
            df_plot.groupby('cpc_major')['patent_id']
            .nunique()
            .reset_index(name='num_patents')
        )
        # Use different color schemes based on scope
        if scope == 'US':
            # Use green tones for US-only view
            color_palette = px.colors.qualitative.Set2
        else:
            # Use blue/purple tones for whole dataset view
            color_palette = px.colors.qualitative.Pastel
        fig = px.bar(
            grouped,
            x='cpc_major',
            y='num_patents',
            color='cpc_major',
            color_discrete_sequence=color_palette,
            title="Patent Major Categories Overview" + title_suffix,
        )
        fig.update_layout(showlegend=False)  # Hide legend since each bar is a different category

    fig.update_layout(
        xaxis_title=READABLE_NAMES.get('cpc_major', 'CPC Major Category'),
        yaxis_title="Number of Patents",
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
        xaxis_tickangle=35,
    )
    return fig


def make_histogram(df_in: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    if column not in df_in.columns or df_in[column].dropna().empty:
        return empty_figure("No data available for histogram.")
    fig = px.histogram(
        df_in,
        x=column,
        nbins=bins,
        marginal="rug",
        title=f"Histogram of {READABLE_NAMES.get(column, column)}",
        color_discrete_sequence=['#636EFA'],
    )
    fig.update_traces(marker_color='#636EFA', marker_line_color='#4C63D2', marker_line_width=1)
    fig.update_layout(
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
    )
    return fig


def make_box_before_after(df_full: pd.DataFrame, column: str, iqr_multiplier: float = 1.5) -> go.Figure:
    series = df_full[column].dropna()
    if series.empty:
        return empty_figure("No data available for boxplot.")

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    cleaned = series[(series >= lower) & (series <= upper)]

    # Create subplots with independent y-axes
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Before Outlier Removal", "After IQR Cleaning"),
        horizontal_spacing=0.15,
    )
    
    fig.add_trace(
        go.Box(y=series, name="Before", marker_color=STYLES['accent_blue'], showlegend=False),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(y=cleaned, name="After (IQR cleaned)", marker_color=STYLES['accent_green'], showlegend=False),
        row=1, col=2
    )

    # Update y-axes independently - each subplot gets its own range
    fig.update_yaxes(title_text=READABLE_NAMES.get(column, column), row=1, col=1)
    fig.update_yaxes(title_text=READABLE_NAMES.get(column, column), row=1, col=2)
    
    # Update x-axes to remove tick labels (they're not needed for single boxplots)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)

    fig.update_layout(
        title=f"Outlier Cleaning via IQR on {READABLE_NAMES.get(column, column)}",
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
    )
    return fig


def make_qq_plot(df_in: pd.DataFrame, column: str) -> go.Figure:
    series = df_in[column].dropna()
    if series.empty:
        return empty_figure("No data available for QQ-plot.")

    sorted_vals = np.sort(series.values)
    n = len(sorted_vals)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.quantile(np.random.normal(loc=0, scale=1, size=n * 5), probs)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=sorted_vals,
            mode="markers",
            marker=dict(color='#19D3F3', size=6, opacity=0.7),  # Cyan
            name="Sample Quantiles",
        )
    )
    min_ax = min(theoretical.min(), sorted_vals.min())
    max_ax = max(theoretical.max(), sorted_vals.max())
    fig.add_trace(
        go.Scatter(
            x=[min_ax, max_ax],
            y=[min_ax, max_ax],
            mode="lines",
            line=dict(color="#FF6692", dash="dash", width=2),  # Pink-red
            name="Reference Line",
        )
    )
    fig.update_layout(
        title=f"QQ-Plot for {READABLE_NAMES.get(column, column)}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
    )
    return fig


def make_histogram_kde(df_in: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    """Create histogram with KDE overlay"""
    from scipy import stats
    
    series = df_in[column].dropna()
    if series.empty:
        return empty_figure(f"No data for {column}.")
    
    # Compute histogram manually to ensure proper density normalization
    counts, bin_edges = np.histogram(series, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram bars
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        width=bin_width * 0.9,  # Slight gap between bars
        marker_color='#00CC96',
        marker_line_color='#00A67E',
        marker_line_width=1,
        opacity=0.7,
        name='Histogram'
    ))
    
    # Compute and add KDE curve
    try:
        if len(series) < 2:
            raise ValueError("Not enough data points for KDE")
        
        kde = stats.gaussian_kde(series)
        
        # Create x range for KDE (slightly wider than data range)
        x_min, x_max = series.min(), series.max()
        x_range = np.linspace(x_min, x_max, 200)
        
        # Evaluate KDE
        y_kde = kde(x_range)
        
        # Ensure KDE values are valid
        valid_mask = np.isfinite(y_kde) & (y_kde >= 0)
        if valid_mask.sum() > 0:
            x_kde = x_range[valid_mask]
            y_kde_valid = y_kde[valid_mask]
            
            # Add KDE curve
            fig.add_trace(go.Scatter(
                x=x_kde,
                y=y_kde_valid,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2.5),
                hovertemplate='KDE: %{y:.4f}<extra></extra>'
            ))
    except Exception as e:
        print(f"Warning: KDE computation failed for {column}: {e}")
    
    fig.update_layout(
        title=f"Histogram with KDE: {READABLE_NAMES.get(column, column)}",
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
        xaxis_title=READABLE_NAMES.get(column, column),
        yaxis_title='Density',
        showlegend=True,
        legend=dict(x=0.7, y=0.95),
        bargap=0.1,
    )
    return fig


def make_regression_plot(df_in: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Create regression plot with scatter and regression line"""
    from sklearn.linear_model import LinearRegression
    
    # Check if columns exist
    if x_col not in df_in.columns or y_col not in df_in.columns:
        return empty_figure("Selected columns not found in dataset.")
    
    # Check if columns are numeric
    if not pd.api.types.is_numeric_dtype(df_in[x_col]) or not pd.api.types.is_numeric_dtype(df_in[y_col]):
        return empty_figure("Regression plot requires numeric columns.")
    
    df_plot = df_in[[x_col, y_col]].dropna()
    if df_plot.empty:
        return empty_figure("No data available for regression plot.")
    
    # Sample if too large
    if len(df_plot) > 15000:
        df_plot = df_plot.sample(n=15000, random_state=42)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df_plot[x_col],
        y=df_plot[y_col],
        mode='markers',
        marker=dict(
            color='#EF553B',
            opacity=0.6,
            size=5
        ),
        name='Data Points'
    ))
    
    # Calculate and add regression line
    X = df_plot[[x_col]].values
    y = df_plot[y_col].values
    reg = LinearRegression().fit(X, y)
    x_line = np.linspace(df_plot[x_col].min(), df_plot[x_col].max(), 100)
    y_line = reg.predict(x_line.reshape(-1, 1))
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name='Regression Line',
        line=dict(color='#FF6B35', width=3)
    ))
    
    fig.update_layout(
        title=f"Regression Plot: {READABLE_NAMES.get(y_col, y_col)} vs {READABLE_NAMES.get(x_col, x_col)}",
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
        xaxis_title=READABLE_NAMES.get(x_col, x_col),
        yaxis_title=READABLE_NAMES.get(y_col, y_col),
        showlegend=True,
    )
    return fig


def make_joint_plot(df_in: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    df_plot = df_in[[x_col, y_col]].dropna()
    if df_plot.empty:
        return empty_figure("No data available for joint plot.")
    # Sample if too large
    if len(df_plot) > 20000:
        df_plot = df_plot.sample(n=20000, random_state=42)
    # Create scatter plot with marginal histograms
    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        marginal_x="histogram",
        marginal_y="histogram",
        title=f"Joint Plot: {READABLE_NAMES.get(y_col, y_col)} vs {READABLE_NAMES.get(x_col, x_col)}",
        opacity=0.6,
        color_discrete_sequence=['#AB63FA'],
    )
    fig.update_traces(marker_color='#AB63FA')
    fig.update_layout(
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
    )
    return fig


def make_rug_plot(df_in: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    """Create histogram with KDE and rug plot"""
    series = df_in[column].dropna()
    if series.empty:
        return empty_figure(f"No data for {column}.")
    # Sample for rug plot if too large
    rug_sample = series.sample(n=min(500, len(series)), random_state=42) if len(series) > 500 else series
    fig = px.histogram(
        df_in,
        x=column,
        nbins=bins,
        histnorm='density',
        title=f"Rug Plot: {READABLE_NAMES.get(column, column)}",
    )
    rug_sample = series.sample(n=min(500, len(series)), random_state=42) if len(series) > 500 else series
    fig.add_trace(go.Scatter(
        x=rug_sample.values,
        y=[0] * len(rug_sample),
        mode='markers',
        marker=dict(size=3, color='darkred', opacity=0.5),
        name='Rug',
        showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
        xaxis_title=READABLE_NAMES.get(column, column),
        yaxis_title='Density',
    )
    return fig


def make_3d_scatter(df_in: pd.DataFrame, x_col: str, y_col: str, z_col: str = None) -> go.Figure:
    """Create 3D scatter plot"""
    if z_col is None:
        # Use a third column if available, otherwise use x_col again
        numeric_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
        z_col = [c for c in numeric_cols if c not in [x_col, y_col]]
        if z_col:
            z_col = z_col[0]
        else:
            z_col = x_col
    cols = [x_col, y_col, z_col]
    df_plot = df_in[cols].dropna()
    if df_plot.empty:
        return empty_figure("No data available for 3D scatter plot.")
    if len(df_plot) > 15000:
        df_plot = df_plot.sample(n=15000, random_state=42)
    fig = go.Figure(data=go.Scatter3d(
        x=df_plot[x_col],
        y=df_plot[y_col],
        z=df_plot[z_col],
        mode='markers',
        marker=dict(
            size=5,
            color=df_plot[z_col],
            colorscale='Viridis',
            opacity=0.6,
            showscale=True,
            colorbar=dict(title=READABLE_NAMES.get(z_col, z_col)),
        ),
    ))
    fig.update_layout(
        title=f"3D Scatter: {READABLE_NAMES.get(x_col, x_col)} vs {READABLE_NAMES.get(y_col, y_col)} vs {READABLE_NAMES.get(z_col, z_col)}",
        scene=dict(
            xaxis_title=READABLE_NAMES.get(x_col, x_col),
            yaxis_title=READABLE_NAMES.get(y_col, y_col),
            zaxis_title=READABLE_NAMES.get(z_col, z_col),
        ),
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
    )
    return fig


def make_contour_plot(df_in: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Create contour plot (density plot) with contour lines on white background"""
    from scipy.stats import gaussian_kde
    
    df_plot = df_in[[x_col, y_col]].dropna()
    if df_plot.empty:
        return empty_figure("No data available for contour plot.")
    
    # Sample if too large
    if len(df_plot) > 30000:
        df_plot = df_plot.sample(n=30000, random_state=42)
    
    # Clip to reasonable ranges for better density visibility (like selected_plots.py)
    if x_col == 'num_claims' and y_col == 'num_figures':
        df_plot = df_plot[(df_plot[x_col] <= 40) & (df_plot[y_col] <= 100)]
        x_max, y_max = 40, 100
    else:
        x_max = df_plot[x_col].max()
        y_max = df_plot[y_col].max()
    
    # Prepare data for KDE
    x_data = df_plot[x_col].values
    y_data = df_plot[y_col].values
    
    # Create grid for contour plot
    x_min, x_max_plot = df_plot[x_col].min(), x_max
    y_min, y_max_plot = df_plot[y_col].min(), y_max
    
    # Create meshgrid
    x_grid = np.linspace(x_min, x_max_plot, 100)
    y_grid = np.linspace(y_min, y_max_plot, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute KDE
    try:
        kde = gaussian_kde(np.vstack([x_data, y_data]))
        # Adjust bandwidth (similar to bw_adjust=1.2 in seaborn)
        kde.set_bandwidth(kde.factor * 1.2)
        
        # Evaluate KDE on grid
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)
    except:
        # Fallback if KDE fails
        return empty_figure("Unable to compute density for contour plot.")
    
    # Create contour plot with lines only (no fill)
    fig = go.Figure(data=go.Contour(
        x=x_grid,
        y=y_grid,
        z=Z,
        colorscale='Rainbow',
        contours=dict(
            showlines=True,
            showlabels=True,
            labelfont=dict(size=10, color='black'),
            start=Z.min(),
            end=Z.max(),
            size=(Z.max() - Z.min()) / 12,  # 12 levels like selected_plots.py
            coloring='lines'  # Only show lines, no fill
        ),
        line=dict(width=1.5),  # Line width like selected_plots.py
        showscale=False,  # Hide colorbar for cleaner look
    ))
    
    # Set white background
    fig.update_layout(
        title=f"Contour Plot: {READABLE_NAMES.get(y_col, y_col)} vs {READABLE_NAMES.get(x_col, x_col)}",
        xaxis_title=READABLE_NAMES.get(x_col, x_col),
        yaxis_title=READABLE_NAMES.get(y_col, y_col),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
        xaxis=dict(
            range=[x_min, x_max_plot],
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            range=[y_min, y_max_plot],
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
    )
    return fig


def make_hexbin_plot(df_in: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Create hexbin plot using density heatmap with hexagonal bins"""
    df_plot = df_in[[x_col, y_col]].dropna()
    if df_plot.empty:
        return empty_figure("No data available for hexbin plot.")
    
    # Sample if too large
    if len(df_plot) > 100000:
        df_plot = df_plot.sample(n=100000, random_state=42)
    
    # Use density heatmap with hexagonal appearance
    # Plotly doesn't have native hexbin, so we use a 2D histogram with more bins
    # and adjust the appearance to look more like hexbins
    fig = go.Figure(data=go.Histogram2d(
        x=df_plot[x_col],
        y=df_plot[y_col],
        histfunc='count',
        colorscale='Viridis',
        nbinsx=30,
        nbinsy=40,
        colorbar=dict(title="Count"),
        # Make it look more like hexbins by using a smoother appearance
        autobinx=False,
        autobiny=False,
    ))
    
    fig.update_layout(
        title=f"Hexbin Plot: {READABLE_NAMES.get(y_col, y_col)} vs {READABLE_NAMES.get(x_col, x_col)}",
        xaxis_title=READABLE_NAMES.get(x_col, x_col),
        yaxis_title=READABLE_NAMES.get(y_col, y_col),
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
    )
    return fig


def make_correlation_heatmap(df_in: pd.DataFrame, selected_numeric: List[str]) -> go.Figure:
    if not selected_numeric:
        return empty_figure("Select at least one numeric column for correlation.")
    subset = df_in[selected_numeric].dropna()
    if subset.empty:
        return empty_figure("No data available for correlation heatmap.")
    corr = subset.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",  # Red-Blue diverging scale for correlation
        title="Correlation Heatmap",
    )
    fig.update_layout(
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
    )
    return fig



app.layout = html.Div(
    style={
        'fontFamily': STYLES['font_family'],
        'backgroundColor': STYLES['primary_bg'],
        'minHeight': '100vh',
        'color': STYLES['primary_text'],
        'display': 'flex',
        'flexDirection': 'column',
        'padding': '20px',
    },
    children=[
        # --- Project Description Header with Global Filters ---
        html.Div(
            style={
                'backgroundColor': STYLES['secondary_bg'],
                'padding': '15px 20px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                'border': f'1px solid {STYLES["border_color"]}',
                'marginBottom': '20px',
                'textAlign': 'center',
            },
            children=[
                html.H1(
                    children="USPTO Patent Analysis (2024-2025)",
                    style={
                        'fontSize': '1.8em',
                        'fontWeight': '700',
                        'color': STYLES['primary_text'],
                        'marginBottom': '8px',
                        'marginTop': '0',
                    },
                ),
                html.P(
                    children="Mariana Soares | Virginia Tech",
                    style={
                        'fontSize': '0.95em',
                        'color': STYLES['secondary_text'],
                        'margin': '0 0 20px 0',
                        'fontStyle': 'italic',
                    },
                ),
                # --- Global filters (shared across tabs) ---
                html.Div(
                    style={
                        'display': 'flex',
                        'flexWrap': 'wrap',
                        'gap': '15px',
                        'justifyContent': 'center',
                        'alignItems': 'flex-start',
                    },
                    children=[
                        html.Div(
                            style={'minWidth': '180px', 'flex': '1', 'maxWidth': '250px'},
                            children=[
                                html.Label(
                                    "Dataset Scope",
                                    style={
                                        'fontWeight': 'bold',
                                        'marginBottom': '5px',
                                        'display': 'block',
                                        'fontSize': '0.9em',
                                    },
                                ),
                                dcc.Dropdown(
                                    id='global-scope',
                                    options=[
                                        {'label': 'Whole dataset', 'value': 'ALL'},
                                        {'label': 'US only', 'value': 'US'},
                                        {'label': 'US regions', 'value': 'US_REGIONS'},
                                    ],
                                    value='ALL',
                                    clearable=False,
                                    style={'fontSize': '0.9em'},
                                ),
                            ],
                        ),
                        html.Div(
                            style={'minWidth': '180px', 'flex': '1', 'maxWidth': '250px'},
                            children=[
                                html.Label(
                                    "Assignee Type",
                                    style={
                                        'fontWeight': 'bold',
                                        'marginBottom': '5px',
                                        'display': 'block',
                                        'fontSize': '0.9em',
                                    },
                                ),
                                dcc.Dropdown(
                                    id='global-assignee-type',
                                    options=[{'label': 'All Assignee Types', 'value': 'ALL'}] + assignee_type_options,
                                    value='ALL',
                                    clearable=False,
                                    style={'fontSize': '0.9em'},
                                ),
                            ],
                        ),
                        html.Div(
                            style={'minWidth': '250px', 'flex': '2', 'maxWidth': '400px'},
                            children=[
                                html.Label(
                                    "Quarter Range",
                                    style={
                                        'fontWeight': 'bold',
                                        'marginBottom': '5px',
                                        'display': 'block',
                                        'fontSize': '0.9em',
                                    },
                                ),
                                dcc.RangeSlider(
                                    id='global-quarter-range',
                                    min=0,
                                    max=max(len(quarter_values) - 1, 0),
                                    value=[0, max(len(quarter_values) - 1, 0)],
                                    marks=quarter_marks,
                                    step=1,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # --- Main Tabs ---
        dcc.Tabs(
            id='main-tabs',
            value='tab-overview',
            children=[
                dcc.Tab(
                    label='Overview & Data',
                    value='tab-overview',
                    children=[
                        html.Br(),
                        # Scope is controlled globally now (global-scope); no extra control here.
        html.Div(
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'gap': '20px',
                'marginBottom': '20px',
            },
            children=[
                html.Div(
                    style={
                        'flex': '1',
                        'minWidth': '500px',
                        'backgroundColor': STYLES['secondary_bg'],
                        'padding': '20px',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                        'border': f'1px solid {STYLES["border_color"]}',
                    },
                    children=[
                        html.Div(
                            style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px', 'flexWrap': 'wrap'},
                            children=[
                                html.Div(
                                    style={'flex': '1', 'minWidth': '150px'},
                                    children=[
                                        html.Label("Map Scope", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id='overview-map-scope',
                                            options=[
                                                {'label': 'United States', 'value': 'us'},
                                            ],
                                            value='us',
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id='overview-state-filter-container',
                                    style={'flex': '1', 'minWidth': '150px', 'display': 'block'},
                                    children=[
                                        html.Label("State", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id='overview-state-filter',
                                            options=[{'label': 'All States', 'value': 'ALL'}],
                                            value='ALL',
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id='overview-country-filter-container',
                                    style={'flex': '1', 'minWidth': '150px', 'display': 'none'},
                                    children=[
                                        html.Label("Country", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id='overview-country-filter',
                                            options=[{'label': 'Select a country', 'value': None}],
                                            value=None,
                                            clearable=True,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Loading(
                            type="default",
                            children=dcc.Graph(
                                id='overview-map',
                                style={'height': '500px'},
                            ),
                        ),
                    ],
                ),
                html.Div(
                    style={
                        'flex': '1',
                        'minWidth': '400px',
                        'backgroundColor': STYLES['secondary_bg'],
                        'padding': '20px',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                        'border': f'1px solid {STYLES["border_color"]}',
                    },
                    children=[
                        html.H3("Patent Categories by CPC Major", style={'marginBottom': '15px'}),
                        dcc.Loading(
                            type="default",
                            children=dcc.Graph(
                                id='overview-cpc-bar',
                                style={'height': '500px'},
                            ),
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'gap': '20px',
            },
            children=[
                html.Div(
                    style={
                        'flex': '1',
                        'minWidth': '320px',
                        'backgroundColor': STYLES['secondary_bg'],
                        'padding': '20px',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                        'border': f'1px solid {STYLES["border_color"]}',
                    },
                    children=[
                        html.Div(
                            style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '10px'},
                            children=[
                                html.H3("Dataset Summary", style={'margin': 0}),
                                html.Button(
                                    "Download CSV",
                                    id="download-csv-button",
                                    n_clicks=0,
                                    style={
                                        'backgroundColor': STYLES['accent_blue'],
                                        'color': 'white',
                                        'border': 'none',
                                        'padding': '8px 16px',
                                        'borderRadius': '6px',
                                        'cursor': 'pointer',
                                        'fontSize': '12px',
                                        'fontWeight': '500',
                                        'fontFamily': STYLES['font_family'],
                                    },
                                ),
                            ],
                        ),
                        html.P(
                            "Note: This dashboard uses a sample dataset of 50,000 patents for optimal performance.",
                            style={
                                'fontSize': '11px',
                                'color': STYLES['secondary_text'],
                                'fontStyle': 'italic',
                                'marginBottom': '10px',
                                'marginTop': '5px',
                            },
                        ),
                        dcc.Download(id="download-csv"),
                        dash_table.DataTable(
                            id='overview-summary-table',
                            style_table={'maxHeight': '400px', 'overflowY': 'auto'},
                            style_cell={
                                'fontFamily': STYLES['font_family'],
                                'fontSize': 12,
                                'textAlign': 'left',
                            },
                        ),
                    ],
                ),
            ],
        ),
                    ],
                ),
                dcc.Tab(
                    label='Cleaning & Outliers',
                    value='tab-cleaning',
                    children=[
                        html.Br(),
                        html.Div(
                            style={
                                'display': 'flex',
                                'gap': '20px',
                                'flexWrap': 'wrap',
                            },
                            children=[
                                # Left side: Outlier detection controls (stacked)
                                html.Div(
                                    style={
                                        'backgroundColor': STYLES['secondary_bg'],
                                        'padding': '20px',
                                        'borderRadius': '12px',
                                        'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                                        'border': f'1px solid {STYLES["border_color"]}',
                                        'flex': '1',
                                        'minWidth': '300px',
                                        'maxWidth': '400px',
                                    },
                                    children=[
                                        html.H3("Outlier Detection & Removal (IQR)", style={'marginTop': '0'}),
                                        html.Div(
                                            style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'},
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Label("Numeric column", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                        dcc.Dropdown(
                                                            id='cleaning-column',
                                                            options=[
                                                                {'label': READABLE_NAMES.get(c, c), 'value': c}
                                                                for c in suitable_numeric_cols
                                                            ],
                                                            value='num_claims'
                                                            if 'num_claims' in suitable_numeric_cols
                                                            else (suitable_numeric_cols[0] if suitable_numeric_cols else None),
                                                            clearable=False,
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label("IQR multiplier", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                        dcc.Slider(
                                                            id='cleaning-iqr-multiplier',
                                                            min=0.5,
                                                            max=3.0,
                                                            step=0.5,
                                                            value=1.5,
                                                            marks={i: str(i) for i in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]},
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label("Method", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                        html.Div(
                                                            style={'fontSize': '0.85em', 'color': STYLES['secondary_text'], 'marginBottom': '8px'},
                                                            children="IQR: Shows before/after boxplots with outlier removal. Raw: Shows histogram with rug plot of original data without cleaning."
                                                        ),
                                                        dcc.RadioItems(
                                                            id='cleaning-method',
                                                            options=[
                                                                {'label': 'IQR Outlier Removal (Before/After Boxplots)', 'value': 'IQR'},
                                                                {'label': 'Raw Data Only (Histogram + Rug Plot - No Cleaning)', 'value': 'NONE'},
                                                            ],
                                                            value='IQR',
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                # Right side: Plot
                                html.Div(
                                    style={
                                        'flex': '2',
                                        'minWidth': '500px',
                                    },
                                    children=[
                                        dcc.Loading(
                                            children=dcc.Graph(
                                                id='cleaning-boxplot',
                                                style={'height': '450px'},
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label='Normality Tests & Transformations',
                    value='tab-normality',
                    children=[
                        html.Br(),
                        # Section 1: Column & Test Controls (top row)
                        html.Div(
                            style={
                                'backgroundColor': STYLES['secondary_bg'],
                                'padding': '20px',
                                'borderRadius': '12px',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                                'border': f'1px solid {STYLES["border_color"]}',
                                'marginBottom': '20px',
                            },
                            children=[
                                html.Div(
                                    style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'},
                                    children=[
                                        html.Div(
                                            style={'minWidth': '200px', 'flex': '1'},
                                            children=[
                                                html.Label("Numeric column", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                dcc.Dropdown(
                                                    id='normality-column',
                                                    options=[
                                                        {'label': READABLE_NAMES.get(c, c), 'value': c}
                                                        for c in suitable_numeric_cols
                                                    ],
                                                    value='num_claims'
                                                    if 'num_claims' in suitable_numeric_cols
                                                    else (suitable_numeric_cols[0] if suitable_numeric_cols else None),
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={'minWidth': '200px', 'flex': '1'},
                                            children=[
                                                html.Label("Normality Tests", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                dcc.Dropdown(
                                                    id='normality-test',
                                                    options=[
                                                        {'label': 'Shapiro–Wilk test', 'value': 'shapiro'},
                                                        {'label': 'K-S test', 'value': 'ks'},
                                                    ],
                                                    value='shapiro',
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={'minWidth': '200px', 'flex': '1'},
                                            children=[
                                                html.Label("Transformation", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                dcc.Dropdown(
                                                    id='normality-transform',
                                                    options=[
                                                        {'label': 'Raw', 'value': 'raw'},
                                                        {'label': 'Log', 'value': 'log'},
                                                        {'label': 'Sqrt', 'value': 'sqrt'},
                                                    ],
                                                    value='raw',
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                # Section 2: Test Results Box
                                html.Div(
                                    id='normality-results',
                                    style={
                                        'marginTop': '20px',
                                        'padding': '15px',
                                        'backgroundColor': STYLES['primary_bg'],
                                        'borderRadius': '8px',
                                        'border': f'1px solid {STYLES["border_color"]}',
                                    },
                                ),
                            ],
                        ),
                        # Section 3: Plots (all 3 in same row)
                        html.Div(
                            style={'display': 'flex', 'gap': '20px', 'width': '100%'},
                            children=[
                                # A. QQ Plot (Raw Data)
                                html.Div(
                                    style={'flex': '1', 'width': '100%'},
                                    children=[
                                        html.H4("A. QQ Plot (Raw Data)", style={
                                            'marginBottom': '10px',
                                            'fontSize': '14px',
                                            'fontWeight': 'bold',
                                            'color': STYLES['primary_text']
                                        }),
                                        dcc.Loading(
                                            children=dcc.Graph(
                                                id='normality-qq-raw',
                                                style={'height': '500px', 'width': '100%'},
                                            ),
                                        ),
                                    ],
                                ),
                                # B. Histogram (Raw Data)
                                html.Div(
                                    style={'flex': '1', 'width': '100%'},
                                    children=[
                                        html.H4("B. Histogram (Raw Data)", style={
                                            'marginBottom': '10px',
                                            'fontSize': '14px',
                                            'fontWeight': 'bold',
                                            'color': STYLES['primary_text']
                                        }),
                                        dcc.Loading(
                                            children=dcc.Graph(
                                                id='normality-hist-raw',
                                                style={'height': '500px', 'width': '100%'},
                                            ),
                                        ),
                                    ],
                                ),
                                # C. Transformation comparison
                                html.Div(
                                    style={'flex': '1', 'width': '100%'},
                                    children=[
                                        html.Div(
                                            id='transformation-plot-label',
                                            style={
                                                'marginBottom': '10px',
                                                'fontSize': '14px',
                                                'fontWeight': 'bold',
                                                'color': STYLES['primary_text']
                                            },
                                        ),
                                        dcc.Loading(
                                            children=dcc.Graph(
                                                id='normality-transformed',
                                                style={'height': '500px', 'width': '100%'},
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label='Numeric Plots',
                    value='tab-numeric',
                    children=[
                        html.Br(),
                        html.Div(
                            style={
                                'display': 'flex',
                                'gap': '20px',
                                'flexWrap': 'wrap',
                            },
                            children=[
                                # Left side: Filtering controls (stacked)
                                html.Div(
                                    style={
                                        'backgroundColor': STYLES['secondary_bg'],
                                        'padding': '20px',
                                        'borderRadius': '12px',
                                        'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                                        'border': f'1px solid {STYLES["border_color"]}',
                                        'flex': '1',
                                        'minWidth': '300px',
                                        'maxWidth': '400px',
                                    },
                                    children=[
                                        html.H3("Plots for Numerical Features", style={'marginTop': '0'}),
                                        html.Div(
                                            style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'},
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Label("X (numeric)", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                        dcc.Dropdown(
                                                            id='numeric-x',
                                                            options=[
                                                                {'label': READABLE_NAMES.get(c, c), 'value': c}
                                                                for c in suitable_numeric_cols
                                                            ],
                                                            value='num_claims'
                                                            if 'num_claims' in suitable_numeric_cols
                                                            else (suitable_numeric_cols[0] if suitable_numeric_cols else None),
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label("Y (numeric, optional)", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                        dcc.Dropdown(
                                                            id='numeric-y',
                                                            options=[
                                                                {'label': READABLE_NAMES.get(c, c), 'value': c}
                                                                for c in suitable_numeric_cols
                                                            ],
                                                            value='num_figures'
                                                            if 'num_figures' in suitable_numeric_cols
                                                            else (suitable_numeric_cols[1] if len(suitable_numeric_cols) > 1 else None),
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label("Plot type", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                        dcc.Dropdown(
                                                            id='numeric-plot-type',
                                                            options=[
                                                                {'label': 'Histogram', 'value': 'hist'},
                                                                {'label': 'Histogram with KDE (Dist Plot)', 'value': 'hist_kde'},
                                                                {'label': 'Scatter', 'value': 'scatter'},
                                                                {'label': 'Regression Plot', 'value': 'regplot'},
                                                                {'label': 'Joint Plot (KDE + Scatter)', 'value': 'joint'},
                                                                {'label': 'Rug Plot', 'value': 'rug'},
                                                                {'label': 'QQ-Plot', 'value': 'qq'},
                                                                {'label': '3D Scatter', 'value': '3d'},
                                                                {'label': 'Contour Plot', 'value': 'contour'},
                                                                {'label': 'Hexbin', 'value': 'hexbin'},
                                                                {'label': 'Boxplot (Before/After)', 'value': 'box_before_after'},
                                                                {'label': 'Correlation Heatmap', 'value': 'corr_heatmap'},
                                                            ],
                                                            value='scatter',
                                                            clearable=False,
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    id='numeric-z-container',
                                                    style={'display': 'none'},
                                                    children=[
                                                        html.Label("Z (numeric, for 3D plots)", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                        dcc.Dropdown(
                                                            id='numeric-z',
                                                            options=[
                                                                {'label': READABLE_NAMES.get(c, c), 'value': c}
                                                                for c in suitable_numeric_cols
                                                            ],
                                                            value='patent_complexity'
                                                            if 'patent_complexity' in suitable_numeric_cols
                                                            else (suitable_numeric_cols[2] if len(suitable_numeric_cols) > 2 else None),
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                # Right side: Plot
                                html.Div(
                                    style={
                                        'flex': '2',
                                        'minWidth': '500px',
                                    },
                                    children=[
                                        dcc.Loading(
                                            children=dcc.Graph(
                                                id='numeric-graph',
                                                style={'height': '460px'},
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label='Categorical Plots',
                    value='tab-categorical',
                    children=[
                        html.Br(),
                        html.Div(
                            style={
                                'display': 'flex',
                                'gap': '20px',
                                'flexWrap': 'wrap',
                            },
                            children=[
                                # Left side: Filtering controls (stacked)
                                html.Div(
                                    style={
                                        'backgroundColor': STYLES['secondary_bg'],
                                        'padding': '20px',
                                        'borderRadius': '12px',
                                        'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                                        'border': f'1px solid {STYLES["border_color"]}',
                                        'flex': '1',
                                        'minWidth': '300px',
                                        'maxWidth': '400px',
                                    },
                                    children=[
                                        html.H3("Plots for Categorical Features", style={'marginTop': '0'}),
                                        html.Div(
                                            style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'},
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Label("Categorical feature", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                        dcc.Dropdown(
                                                            id='categorical-col',
                                                            options=[
                                                                {'label': READABLE_NAMES.get(c, c), 'value': c}
                                                                for c in suitable_categorical_cols
                                                            ],
                                                            value='cpc_major'
                                                            if 'cpc_major' in suitable_categorical_cols
                                                            else (suitable_categorical_cols[0] if suitable_categorical_cols else None),
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label("Numeric feature (optional)", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                        dcc.Dropdown(
                                                            id='categorical-numeric',
                                                            options=[
                                                                {'label': READABLE_NAMES.get(c, c), 'value': c}
                                                                for c in suitable_numeric_cols
                                                            ],
                                                            value='num_claims'
                                                            if 'num_claims' in suitable_numeric_cols
                                                            else (suitable_numeric_cols[0] if suitable_numeric_cols else None),
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Label("Plot type", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                        dcc.Dropdown(
                                                            id='categorical-plot-type',
                                                            options=[
                                                                {'label': 'Bar (mean)', 'value': 'bar'},
                                                                {'label': 'Bar (count)', 'value': 'count'},
                                                                {'label': 'Stacked Bar', 'value': 'stacked_bar'},
                                                                {'label': 'Grouped Bar', 'value': 'grouped_bar'},
                                                                {'label': 'Pie Chart', 'value': 'pie'},
                                                                {'label': 'Box', 'value': 'box'},
                                                                {'label': 'Violin', 'value': 'violin'},
                                                                {'label': 'Strip Plot', 'value': 'strip'},
                                                                {'label': 'Swarm Plot', 'value': 'swarm'},
                                                            ],
                                                            value='bar',
                                                            clearable=False,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                # Right side: Plot
                                html.Div(
                                    style={
                                        'flex': '2',
                                        'minWidth': '500px',
                                    },
                                    children=[
                                        dcc.Loading(
                                            children=dcc.Graph(
                                                id='categorical-graph',
                                                style={'height': '460px'},
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label='Statistics',
                    value='tab-stats',
                    children=[
                        html.Br(),
                        html.Div(
                                            style={
                                'backgroundColor': STYLES['secondary_bg'],
                                'padding': '20px',
                                'borderRadius': '12px',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                                'border': f'1px solid {STYLES["border_color"]}',
                                'marginBottom': '20px',
                            },
                            children=[
                                html.H3("Descriptive Statistics & Correlations"),
                                html.Div(
                                    style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'},
                                    children=[
                                        html.Div(
                                            style={'minWidth': '260px', 'flex': '1'},
                                            children=[
                                                html.Label("Numeric columns for correlation"),
                                                dcc.Dropdown(
                                                    id='stats-corr-cols',
                                                    options=[
                                                        {'label': READABLE_NAMES.get(c, c), 'value': c}
                                                        for c in suitable_numeric_cols
                                                    ],
                                                    value=suitable_numeric_cols,
                                                    multi=True,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                                html.Div(
                            style={'display': 'flex', 'gap': '20px', 'width': '100%'},
                                    children=[
                                html.Div(
                                    style={'flex': '1', 'width': '100%'},
                                    children=[
                                        dcc.Loading(
                                            children=dcc.Graph(
                                                id='stats-corr-heatmap',
                                                style={'height': '600px', 'width': '100%'},
                                            ),
                                        ),
                                    ],
                                ),
                        html.Div(
                            style={
                                        'flex': '1',
                                        'width': '100%',
                                'backgroundColor': STYLES['secondary_bg'],
                                        'padding': '15px',
                                'borderRadius': '12px',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                                'border': f'1px solid {STYLES["border_color"]}',
                            },
                            children=[
                                        html.H4("Key numeric summary (current filters)", style={'fontSize': '14px', 'marginBottom': '10px'}),
                                        dash_table.DataTable(
                                            id='stats-summary-table',
                                            style_table={'maxHeight': '550px', 'overflowY': 'auto', 'width': '100%'},
                                            style_cell={
                                                'fontFamily': STYLES['font_family'],
                                                'fontSize': 11,
                                                'textAlign': 'left',
                                                'padding': '4px 6px',
                                                'minWidth': '60px',
                                                'maxWidth': '80px',
                                                'whiteSpace': 'normal',
                                                'overflow': 'hidden',
                                                'textOverflow': 'ellipsis',
                                            },
                                            style_data_conditional=[
                                                {
                                                    'if': {'column_id': 'index'},
                                                    'minWidth': '80px',
                                                    'maxWidth': '100px',
                                                }
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label='PCA Visualization',
                    value='tab-pca',
                    children=[
                        html.Br(),
                        html.Div(
                            style={
                                'backgroundColor': STYLES['secondary_bg'],
                                'padding': '20px',
                                'borderRadius': '12px',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                                'border': f'1px solid {STYLES["border_color"]}',
                                'marginBottom': '20px',
                            },
                            children=[
                                html.H3("Principal Component Analysis (PCA)", style={'marginTop': '0'}),
                                html.Div(
                                    style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'},
                                    children=[
                                        html.Div(
                                            style={'minWidth': '300px', 'flex': '1'},
                                            children=[
                                                html.Label("Numeric columns for PCA", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                                                html.P(
                                                    "Select multiple numeric columns to perform PCA dimensionality reduction.",
                                                    style={
                                                        'fontSize': '12px',
                                                        'color': STYLES['secondary_text'],
                                                        'marginBottom': '8px',
                                                        'fontStyle': 'italic',
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id='pca-cols',
                                                    options=[
                                                        {'label': READABLE_NAMES.get(c, c), 'value': c}
                                                        for c in suitable_numeric_cols
                                                    ],
                                                    value=[
                                                        c
                                                        for c in ['num_claims', 'num_figures', 'patent_complexity']
                                                        if c in suitable_numeric_cols
                                                    ],
                                                    multi=True,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style={'width': '100%'},
                            children=[
                                dcc.Loading(
                                    children=dcc.Graph(
                                        id='pca-scatter',
                                        style={'height': '600px', 'width': '100%'},
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label='Regional & AI',
                    value='tab-regional',
                    children=[
                        html.Br(),
                        html.Div(
                            style={
                                'backgroundColor': STYLES['secondary_bg'],
                                'padding': '20px',
                                'borderRadius': '12px',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.03)',
                                'border': f'1px solid {STYLES["border_color"]}',
                                'marginBottom': '20px',
                            },
                            children=[
                                html.H3("Regional Trends & AI Analysis"),
                            ],
                        ),
                        html.Div(
                            style={'display': 'flex', 'gap': '20px', 'width': '100%'},
                            children=[
                                html.Div(
                                    style={'flex': '1', 'minWidth': 0},
                                    children=[
                                        dcc.Loading(
                                            type="default",
                                            children=dcc.Graph(
                                                id='regional-area',
                                                style={'height': '420px'},
                                            ),
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={'flex': '1', 'minWidth': 0},
                                    children=[
                                        dcc.Loading(
                                            type="default",
                                            children=dcc.Graph(
                                                id='ai-heatmap',
                                                style={'height': '420px'},
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


###############################################################################
# 5. Callbacks
###############################################################################


def apply_global_filters(
    df_in: pd.DataFrame,
    scope_value: str,
    assignee_type_value: str,
    quarter_range: List[int],
) -> pd.DataFrame:
    if df_in is None:
        return pd.DataFrame()
    result = df_in.copy()
    # Scope dropdown replaces explicit country filter:
    # - ALL: no restriction
    # - US: restrict to US patents
    # - US_REGIONS: restrict to US patents with known region
    if scope_value == 'US':
        if 'disambig_country' in result.columns:
            result = result[result['disambig_country'] == 'US']
    elif scope_value == 'US_REGIONS':
        if 'disambig_country' in result.columns and 'us_region' in result.columns:
            result = result[
                (result['disambig_country'] == 'US')
                & result['us_region'].notna()
                & (result['us_region'] != 'Unknown')
            ]
    if assignee_type_value and assignee_type_value != 'ALL' and assignee_category_column in result.columns:
        result = result[result[assignee_category_column] == assignee_type_value]
    # Quarter slider filtering using global quarter_values list
    if quarter_range and len(quarter_range) == 2 and quarter_values:
        q_min, q_max = quarter_range
        q_min = max(0, int(q_min))
        q_max = min(len(quarter_values) - 1, int(q_max))
        if q_min <= q_max:
            allowed = set(quarter_values[q_min : q_max + 1])
            result = result[result[quarter_column].isin(allowed)]
    return result


@callback(
    Output('overview-summary-table', 'data'),
    Output('overview-summary-table', 'columns'),
    Input('global-scope', 'value'),
    Input('global-assignee-type', 'value'),
    Input('global-quarter-range', 'value'),
)
def update_overview_summary(scope_value, global_assignee_type, quarter_range):
    # Overview summary table is a fixed snapshot from the full dataset (for the report)
    summary_rows = [
        {
            'Section': 'Basic Information',
            'Detail': 'Total patents: 554,777; Total columns: 49',
        },
        {
            'Section': 'Patent Type Distribution',
            'Detail': 'utility: 481,260 (86.75%)',
        },
        {
            'Section': 'Patent Type Distribution',
            'Detail': 'design: 71,609 (12.91%)',
        },
        {
            'Section': 'Patent Type Distribution',
            'Detail': 'plant: 1,210 (0.22%)',
        },
        {
            'Section': 'Patent Type Distribution',
            'Detail': 'reissue: 698 (0.13%)',
        },
        {
            'Section': 'Assignee Category Distribution',
            'Detail': 'Corporate: 504,068 (98.94%)',
        },
        {
            'Section': 'Assignee Category Distribution',
            'Detail': 'Individual: 3,249 (0.64%)',
        },
        {
            'Section': 'Assignee Category Distribution',
            'Detail': 'Government: 2,160 (0.42%)',
        },
    ]
    columns = [
        {'name': 'Section', 'id': 'Section'},
        {'name': 'Detail', 'id': 'Detail'},
    ]
    return summary_rows, columns


@callback(
    Output("download-csv", "data"),
    Input("download-csv-button", "n_clicks"),
    State('global-scope', 'value'),
    State('global-assignee-type', 'value'),
    State('global-quarter-range', 'value'),
    prevent_initial_call=True,
)
def download_csv(n_clicks, scope_value, global_assignee_type, quarter_range):
    if n_clicks == 0:
        return None
    
    filtered = apply_global_filters(df, scope_value or 'ALL', global_assignee_type, quarter_range)
    
    if filtered.empty:
        return None
    
    csv_string = filtered.to_csv(index=False)
    return dict(content=csv_string, filename="filtered_patents.csv")


@callback(
    Output('cleaning-boxplot', 'figure'),
    Input('cleaning-column', 'value'),
    Input('cleaning-iqr-multiplier', 'value'),
    Input('cleaning-method', 'value'),
    Input('global-scope', 'value'),
    Input('global-assignee-type', 'value'),
    Input('global-quarter-range', 'value'),
)
def update_cleaning_plot(col, iqr_mult, method, scope_value, global_assignee_type, quarter_range):
    filtered = apply_global_filters(df, scope_value or 'ALL', global_assignee_type, quarter_range)
    if filtered.empty or col is None:
        return empty_figure("No data after filters.")
    if method == 'NONE':
        return make_histogram(filtered, col)
    return make_box_before_after(filtered, col, iqr_multiplier=iqr_mult or 1.5)


@callback(
    Output('normality-qq-raw', 'figure'),
    Output('normality-hist-raw', 'figure'),
    Output('normality-transformed', 'figure'),
    Output('transformation-plot-label', 'children'),
    Output('normality-results', 'children'),
    Input('normality-column', 'value'),
    Input('normality-transform', 'value'),
    Input('normality-test', 'value'),
    Input('global-scope', 'value'),
    Input('global-assignee-type', 'value'),
    Input('global-quarter-range', 'value'),
)
def update_normality_plots(col, transform, test_type, scope_value, global_assignee_type, quarter_range):
    filtered = apply_global_filters(df, scope_value or 'ALL', global_assignee_type, quarter_range)
    if filtered.empty or col is None:
        empty = empty_figure("No data after filters.")
        return empty, empty, empty, html.Div(), html.Div()
    series = filtered[col].dropna()
    if series.empty:
        empty = empty_figure("No data for selected column.")
        return empty, empty, empty, html.Div(), html.Div()
    
    # Raw data plots (always show original data)
    raw_df = filtered.copy()
    raw_qq = make_qq_plot(raw_df, col)
    raw_hist = make_histogram(raw_df, col, bins=30)
    
    # Apply transformation for transformed plot
    original_series = series.copy()
    transformed_series = series.copy()
    transform_label = "B. Transformation Comparison"
    
    if transform == 'log':
        transformed_series = np.log1p(transformed_series.clip(lower=0))
        transform_label = "B. After applying Log(1+x) transformation"
    elif transform == 'sqrt':
        transformed_series = np.sqrt(transformed_series.clip(lower=0))
        transform_label = "B. After applying Sqrt(x) transformation"
    else:
        transform_label = "B. Raw Data (No Transformation)"
    
    transformed_df = filtered.copy()
    transformed_df[col] = transformed_series
    
    transformed_plot = make_histogram(transformed_df, col, bins=30)
    
    test_series = transformed_series.copy()
    is_normal = False
    test_name = ""
    p_value = None
    statistic = None
    
    if test_type == 'shapiro':
        test_name = "Shapiro–Wilk test"
        if len(test_series) > 5000:
            test_series = test_series.sample(n=5000, random_state=42)
        statistic, p_value = stats.shapiro(test_series)
        is_normal = p_value > 0.05
    elif test_type == 'ks':
        test_name = "K-S test"
        if len(test_series) > 5000:
            test_series = test_series.sample(n=5000, random_state=42)
        normalized = (test_series - test_series.mean()) / test_series.std()
        statistic, p_value = stats.kstest(normalized, 'norm')
        is_normal = p_value > 0.05
    
    # Box-Cox / Yeo-Johnson suggestion
    power_transform_lambda = None
    transform_method = None
    
    # Try Box-Cox (requires positive values)
    if (original_series > 0).all():
        try:
            bc_data = original_series.values
            bc_data = bc_data[bc_data > 0]
            if len(bc_data) > 0:
                bc_transformed, bc_lambda = stats.boxcox(bc_data)
                power_transform_lambda = bc_lambda
                transform_method = "Box-Cox"
        except:
            pass
    
    if power_transform_lambda is None:
        try:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            yj_data = original_series.values.reshape(-1, 1)
            pt.fit(yj_data)
            power_transform_lambda = pt.lambdas_[0]
            transform_method = "Yeo-Johnson"
        except:
            pass
    
    # Update transform label with Box-Cox suggestion if available
    if power_transform_lambda is not None and transform != 'raw':
        transform_label = f"B. After applying {transform.upper()} transformation (Recommended: {transform_method} λ={power_transform_lambda:.2f})"
    elif power_transform_lambda is not None:
        transform_label = f"B. Raw Data (Recommended: {transform_method} λ={power_transform_lambda:.2f})"
    
    test_results_col = []
    boxcox_col = []
    
    if p_value is not None:
        interpretation = "data is normal" if is_normal else "data is not normal"
        color = STYLES.get('accent_green', '#28a745') if is_normal else STYLES.get('accent_red', '#dc3545')
        
        test_results_col.append(
            html.H5(f"{test_name} Results", style={'marginBottom': '10px', 'fontSize': '14px', 'fontWeight': 'bold'})
        )
        test_results_col.append(
            html.Div([
                html.Div(f"Statistic: {statistic:.4f}", style={'marginBottom': '8px', 'fontSize': '12px'}),
                html.Div(f"p-value: {p_value:.4f}", style={'marginBottom': '8px', 'fontSize': '12px'}),
                html.Div(
                    f"Interpretation: {interpretation}",
                    style={'color': color, 'fontWeight': 'bold', 'fontSize': '12px'}
                ),
            ])
        )
    
    # Box-Cox / Yeo-Johnson suggestion (right column)
    if power_transform_lambda is not None:
        boxcox_col.append(
            html.H5("Box-Cox / Yeo-Johnson suggestion:", style={'marginBottom': '10px', 'fontSize': '14px', 'fontWeight': 'bold'})
        )
        boxcox_col.append(
            html.Div(
                f"Recommended λ = {power_transform_lambda:.2f} (moderate power transform)",
                style={'fontSize': '12px', 'color': STYLES.get('secondary_text', '#666')}
            )
        )
    
    results_display = html.Div(
        style={'display': 'flex', 'gap': '20px', 'width': '100%'},
        children=[
            html.Div(
                style={'flex': '1'},
                children=test_results_col if test_results_col else [html.Div()]
            ),
            html.Div(
                style={'flex': '1'},
                children=boxcox_col if boxcox_col else [html.Div()]
            ),
        ]
    )
    label_display = html.H4(transform_label, style={
        'marginBottom': '10px',
        'fontSize': '14px',
        'fontWeight': 'bold',
        'color': STYLES['primary_text']
    })
    
    return raw_qq, raw_hist, transformed_plot, label_display, results_display


@callback(
    Output('numeric-z-container', 'style'),
    Input('numeric-plot-type', 'value'),
)
def toggle_z_axis_dropdown(plot_type):
    if plot_type == '3d':
        return {'display': 'block'}
    return {'display': 'none'}


@callback(
    Output('numeric-graph', 'figure'),
    Input('numeric-x', 'value'),
    Input('numeric-y', 'value'),
    Input('numeric-z', 'value'),
    Input('numeric-plot-type', 'value'),
    Input('global-scope', 'value'),
    Input('global-assignee-type', 'value'),
    Input('global-quarter-range', 'value'),
)
def update_numeric_graph(x_col, y_col, z_col, plot_type, scope_value, global_assignee_type, quarter_range):
    filtered = apply_global_filters(df, scope_value or 'ALL', global_assignee_type, quarter_range)
    if filtered.empty or x_col is None:
        return empty_figure("No data after filters.")
    
    # Single variable plots (only need x_col)
    if plot_type == 'hist':
        return make_histogram(filtered, x_col)
    elif plot_type == 'hist_kde':
        return make_histogram_kde(filtered, x_col)
    elif plot_type == 'rug':
        return make_rug_plot(filtered, x_col)
    elif plot_type == 'qq':
        return make_qq_plot(filtered, x_col)
    elif plot_type == 'box_before_after':
        return make_box_before_after(filtered, x_col)
    elif plot_type == 'corr_heatmap':
        numeric_cols = [c for c in suitable_numeric_cols if c in filtered.columns]
        if len(numeric_cols) < 2:
            return empty_figure("Need at least 2 numeric columns for correlation heatmap.")
        return make_correlation_heatmap(filtered, numeric_cols)
    
    # Two variable plots (need both x_col and y_col)
    if not y_col:
        return empty_figure(f"Plot type '{plot_type}' requires both X and Y columns to be selected.")
    
    if plot_type == 'scatter':
        fig = px.scatter(
            filtered,
            x=x_col,
            y=y_col,
            title=f"Scatter: {READABLE_NAMES.get(x_col, x_col)} vs {READABLE_NAMES.get(y_col, y_col)}",
            opacity=0.6,
            color_discrete_sequence=['#FFA15A'],  # Orange
        )
        fig.update_traces(marker_color='#FFA15A')
        fig.update_layout(
            paper_bgcolor=STYLES['secondary_bg'],
            plot_bgcolor=STYLES['secondary_bg'],
            font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
            xaxis_title=READABLE_NAMES.get(x_col, x_col),
            yaxis_title=READABLE_NAMES.get(y_col, y_col),
        )
        return fig
    elif plot_type == 'regplot':
        return make_regression_plot(filtered, x_col, y_col)
    elif plot_type == 'joint':
        return make_joint_plot(filtered, x_col, y_col)
    elif plot_type == 'contour':
        return make_contour_plot(filtered, x_col, y_col)
    elif plot_type == 'hexbin':
        return make_hexbin_plot(filtered, x_col, y_col)
    elif plot_type == '3d':
        if not z_col:
            numeric_cols = [c for c in suitable_numeric_cols if c in filtered.columns and c not in [x_col, y_col]]
            z_col = numeric_cols[0] if numeric_cols else x_col
        return make_3d_scatter(filtered, x_col, y_col, z_col)
    elif plot_type == 'violin':
        fig = px.violin(
            filtered,
            x=x_col,
            y=y_col,
            points="all",
            title=f"Violin: {READABLE_NAMES.get(y_col, y_col)} by {READABLE_NAMES.get(x_col, x_col)}",
            color_discrete_sequence=px.colors.qualitative.Set2,  # Colorful palette
        )
        fig.update_layout(
            paper_bgcolor=STYLES['secondary_bg'],
            plot_bgcolor=STYLES['secondary_bg'],
            font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
            xaxis_title=READABLE_NAMES.get(x_col, x_col),
            yaxis_title=READABLE_NAMES.get(y_col, y_col),
        )
        return fig
    elif plot_type == 'box':
        fig = px.box(
            filtered,
            x=x_col,
            y=y_col,
            title=f"Box: {READABLE_NAMES.get(y_col, y_col)} by {READABLE_NAMES.get(x_col, x_col)}",
            color_discrete_sequence=px.colors.qualitative.Pastel,  # Pastel colors
        )
        fig.update_layout(
            paper_bgcolor=STYLES['secondary_bg'],
            plot_bgcolor=STYLES['secondary_bg'],
            font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
            xaxis_title=READABLE_NAMES.get(x_col, x_col),
            yaxis_title=READABLE_NAMES.get(y_col, y_col),
        )
        return fig
    else:
        return make_histogram(filtered, x_col)


@callback(
    Output('categorical-graph', 'figure'),
    Input('categorical-col', 'value'),
    Input('categorical-numeric', 'value'),
    Input('categorical-plot-type', 'value'),
    Input('global-scope', 'value'),
    Input('global-assignee-type', 'value'),
    Input('global-quarter-range', 'value'),
)
def update_categorical_graph(cat_col, num_col, plot_type, scope_value, global_assignee_type, quarter_range):
    filtered = apply_global_filters(df, scope_value or 'ALL', global_assignee_type, quarter_range)
    if filtered.empty or cat_col is None:
        return empty_figure("Select a categorical feature.")

    # Filter to top 10 values for specific categorical columns
    # Check if cat_col matches any of the top 10 columns (exact match or contains)
    if cat_col and cat_col in filtered.columns:
        cat_col_lower = cat_col.lower()
        should_filter_top10 = any(
            top_col.lower() == cat_col_lower or 
            top_col.lower() in cat_col_lower or 
            cat_col_lower in top_col.lower()
            for top_col in TOP_10_CATEGORICAL_COLS
        )
        if should_filter_top10:
            top_values = filtered[cat_col].value_counts().head(10).index.tolist()
            filtered = filtered[filtered[cat_col].isin(top_values)]

    # Plots that don't require numeric column
    if plot_type == 'count':
        count_data = filtered[cat_col].value_counts().reset_index()
        count_data.columns = [cat_col, 'count']
        fig = px.bar(
            count_data,
            x=cat_col,
            y='count',
            title=f"Count of {READABLE_NAMES.get(cat_col, cat_col)}",
            color_discrete_sequence=px.colors.qualitative.Set3,  # Colorful palette
        )
    elif plot_type == 'pie':
        count_data = filtered[cat_col].value_counts().reset_index()
        count_data.columns = [cat_col, 'count']
        fig = px.pie(
            count_data,
            names=cat_col,
            values='count',
            title=f"Distribution of {READABLE_NAMES.get(cat_col, cat_col)}",
            color_discrete_sequence=px.colors.qualitative.Pastel,  # Pastel colors for pie
        )
    # Plots that require numeric column
    elif num_col is None:
        return empty_figure(f"Plot type '{plot_type}' requires both categorical and numeric features.")
    elif plot_type == 'bar':
        grouped = filtered.groupby(cat_col)[num_col].mean().reset_index()
        fig = px.bar(
            grouped,
            x=cat_col,
            y=num_col,
            title=f"Mean {READABLE_NAMES.get(num_col, num_col)} by {READABLE_NAMES.get(cat_col, cat_col)}",
            color_discrete_sequence=px.colors.qualitative.Set2,  # Colorful palette
        )
    elif plot_type == 'stacked_bar':
        # For stacked bar, we need a second categorical column
        # Use assignee_category if available, otherwise use a default grouping
        if 'assignee_category' in filtered.columns:
            grouped = filtered.groupby([cat_col, 'assignee_category'])[num_col].mean().reset_index()
            fig = px.bar(
                grouped,
                x=cat_col,
                y=num_col,
                color='assignee_category',
                barmode='stack',
                title=f"Stacked Bar: {READABLE_NAMES.get(num_col, num_col)} by {READABLE_NAMES.get(cat_col, cat_col)}",
                color_discrete_sequence=px.colors.qualitative.Set3,  # Colorful palette
            )
        else:
            # Fallback to regular bar if no second category available
            grouped = filtered.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(
                grouped,
                x=cat_col,
                y=num_col,
                title=f"Mean {READABLE_NAMES.get(num_col, num_col)} by {READABLE_NAMES.get(cat_col, cat_col)}",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
    elif plot_type == 'grouped_bar':
        # For grouped bar, we need a second categorical column
        # Use assignee_category if available, otherwise use a default grouping
        if 'assignee_category' in filtered.columns:
            grouped = filtered.groupby([cat_col, 'assignee_category'])[num_col].mean().reset_index()
            fig = px.bar(
                grouped,
                x=cat_col,
                y=num_col,
                color='assignee_category',
                barmode='group',
                title=f"Grouped Bar: {READABLE_NAMES.get(num_col, num_col)} by {READABLE_NAMES.get(cat_col, cat_col)}",
                color_discrete_sequence=px.colors.qualitative.Pastel,  # Pastel colors
            )
        else:
            # Fallback to regular bar if no second category available
            grouped = filtered.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(
                grouped,
                x=cat_col,
                y=num_col,
                title=f"Mean {READABLE_NAMES.get(num_col, num_col)} by {READABLE_NAMES.get(cat_col, cat_col)}",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
    elif plot_type == 'box':
        fig = px.box(
            filtered,
            x=cat_col,
            y=num_col,
            title=f"Boxplot of {READABLE_NAMES.get(num_col, num_col)} by {READABLE_NAMES.get(cat_col, cat_col)}",
            color_discrete_sequence=px.colors.qualitative.Pastel,  # Pastel colors
        )
    elif plot_type == 'violin':
        fig = px.violin(
            filtered,
            x=cat_col,
            y=num_col,
            box=True,
            points="all",
            title=f"Violin of {READABLE_NAMES.get(num_col, num_col)} by {READABLE_NAMES.get(cat_col, cat_col)}",
            color_discrete_sequence=px.colors.qualitative.Set2,  # Colorful palette
        )
    elif plot_type == 'strip':
        # Sample if too large for performance
        plot_data = filtered.copy()
        if len(plot_data) > 10000:
            plot_data = plot_data.sample(n=10000, random_state=42)
        fig = px.strip(
            plot_data,
            x=cat_col,
            y=num_col,
            title=f"Strip Plot: {READABLE_NAMES.get(num_col, num_col)} by {READABLE_NAMES.get(cat_col, cat_col)}",
            color_discrete_sequence=px.colors.qualitative.Set1,  # Bright colors
        )
    elif plot_type == 'swarm':
        # Plotly doesn't have native swarm plot, so we'll use strip with jitter
        # Sample if too large for performance
        plot_data = filtered.copy()
        if len(plot_data) > 5000:
            plot_data = plot_data.sample(n=5000, random_state=42)
        fig = px.strip(
            plot_data,
            x=cat_col,
            y=num_col,
            stripmode='overlay',
            title=f"Swarm Plot: {READABLE_NAMES.get(num_col, num_col)} by {READABLE_NAMES.get(cat_col, cat_col)}",
            color_discrete_sequence=px.colors.qualitative.Dark2,  # Darker colors
        )
    else:
        # Fallback to bar
        grouped = filtered.groupby(cat_col)[num_col].mean().reset_index()
        fig = px.bar(
            grouped,
            x=cat_col,
            y=num_col,
            title=f"Mean {READABLE_NAMES.get(num_col, num_col)} by {READABLE_NAMES.get(cat_col, cat_col)}",
        )
    
    fig.update_layout(
        paper_bgcolor=STYLES['secondary_bg'],
        plot_bgcolor=STYLES['secondary_bg'],
        font=dict(family=STYLES['font_family'], size=12, color=STYLES['primary_text']),
        xaxis_tickangle=35,
    )
    return fig


@callback(
    Output('stats-corr-heatmap', 'figure'),
    Output('stats-summary-table', 'data'),
    Output('stats-summary-table', 'columns'),
    Input('stats-corr-cols', 'value'),
    Input('global-scope', 'value'),
    Input('global-assignee-type', 'value'),
    Input('global-quarter-range', 'value'),
)
def update_stats_tab(cols, scope_value, global_assignee_type, quarter_range):
    filtered = apply_global_filters(df, scope_value or 'ALL', global_assignee_type, quarter_range)
    cols = cols or []
    heatmap_fig = make_correlation_heatmap(filtered, cols)

    if filtered.empty or not cols:
        return heatmap_fig, [], []

    desc = filtered[cols].describe().reset_index()
    # Format numeric columns to 2 decimal places
    for col in desc.columns:
        if col != 'index':
            if desc[col].dtype in [np.float64, np.float32]:
                desc[col] = desc[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '')
            elif desc[col].dtype in [np.int64, np.int32]:
                desc[col] = desc[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '')
    columns = [{'name': c, 'id': c} for c in desc.columns]
    data = desc.to_dict('records')
    return heatmap_fig, data, columns


@callback(
    Output('regional-area', 'figure'),
    Output('ai-heatmap', 'figure'),
    Input('global-scope', 'value'),
    Input('global-assignee-type', 'value'),
    Input('global-quarter-range', 'value'),
)
def update_regional_ai(scope_value, global_assignee_type, quarter_range):
    filtered = apply_global_filters(df, scope_value or 'ALL', global_assignee_type, quarter_range)
    if filtered.empty:
        empty = empty_figure("No data after filters.")
        return empty, empty
    area_fig = make_region_area_plot(filtered)
    ai_fig = make_ai_heatmap(filtered)
    return area_fig, ai_fig


@callback(
    Output('pca-scatter', 'figure'),
    Input('pca-cols', 'value'),
    Input('global-scope', 'value'),
    Input('global-assignee-type', 'value'),
    Input('global-quarter-range', 'value'),
)
def update_pca(pca_cols, scope_value, global_assignee_type, quarter_range):
    filtered = apply_global_filters(df, scope_value or 'ALL', global_assignee_type, quarter_range)
    if filtered.empty:
        return empty_figure("No data after filters.")
    pca_fig = make_pca_scatter(filtered, pca_cols or [])
    return pca_fig


@callback(
    Output('overview-state-filter-container', 'style'),
    Output('overview-country-filter-container', 'style'),
    Input('overview-map-scope', 'value'),
)
def toggle_overview_filters(map_scope):
    """Show/hide state filter based on map scope"""
    # Always show state filter for US map
    return (
        {'flex': '1', 'minWidth': '150px', 'display': 'block'},
        {'flex': '1', 'minWidth': '150px', 'display': 'none'}  # Country filter always hidden
    )


@callback(
    Output('overview-state-filter', 'options'),
    Input('overview-map-scope', 'value'),
    Input('global-scope', 'value'),
    Input('global-assignee-type', 'value'),
    Input('global-quarter-range', 'value'),
)
def update_overview_state_options(map_scope, scope_value, global_assignee_type, quarter_range):
    if map_scope != 'us':
        return [{'label': 'All States', 'value': 'ALL'}]
    
    filtered = apply_global_filters(df, scope_value or 'ALL', global_assignee_type, quarter_range)
    if filtered.empty or 'disambig_state' not in filtered.columns:
        return [{'label': 'All States', 'value': 'ALL'}]
    
    df_us = filtered[filtered['disambig_country'] == 'US'].copy()
    if df_us.empty:
        return [{'label': 'All States', 'value': 'ALL'}]
    
    states = sorted(df_us['disambig_state'].dropna().unique())
    options = [{'label': 'All States', 'value': 'ALL'}]
    options.extend([{'label': state, 'value': state} for state in states])
    return options


@callback(
    Output('overview-map', 'figure'),
    Output('overview-cpc-bar', 'figure'),
    Input('overview-map-scope', 'value'),
    Input('overview-state-filter', 'value'),
    Input('global-scope', 'value'),
    Input('global-assignee-type', 'value'),
    Input('global-quarter-range', 'value'),
)
def update_overview_map_and_bar(map_scope, selected_state, scope_value, global_assignee_type, quarter_range):
    """Update both the map and bar plot based on filters"""
    filtered = apply_global_filters(df, scope_value or 'ALL', global_assignee_type, quarter_range)
    if filtered.empty:
        empty = empty_figure("No data after filters.")
        return empty, empty
    
    # Show US map with optional state filter
    map_fig = make_patent_map(
        filtered,
        'us',
        selected_state or 'ALL'
    )
    
    # Prepare data for bar plot
    filtered_for_bar = filtered
    bar_scope = 'US'
    bar_state = selected_state if selected_state and selected_state != 'ALL' else None
    
    # Update bar plot with filtered data
    bar_fig = make_overview_cpc_count(filtered_for_bar, bar_scope, bar_state)
    
    return map_fig, bar_fig


if __name__ == '__main__':
    port = int(os.environ.get("PORT", "8000"))
    app.run_server(debug=True, host='0.0.0.0', port=port)