import pandas as pd
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

COLORS = {
    'primary_bg': '#F5F8FA',  
    'secondary_bg': 'white',  
    'primary_text': '#2F4F4F',
    'secondary_text': '#6A8080',
    'accent_blue': '#87CEEB',  
    'accent_green': '#A2D9CE',  
    'border_color': '#E0E6EB',  
}

AI_SUBCLASS_PREFIXES = [
    'G06N',
    'G06K',
    'G06T',
    'G10L',
]

AI_GROUP_PREFIXES = [
    'G06F 17',
    'G06F 18',
    'G06F 19',
    'G06F 40',
]

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


def flag_ai_patents(df):
    df = df.copy()
    
    cpc_sub = df['cpc_subclass'].fillna('').astype(str).str.replace(' ', '')
    cpc_group = df['cpc_group'].fillna('').astype(str)
    
    ai_mask_sub = np.zeros(len(df), dtype=bool)
    for prefix in AI_SUBCLASS_PREFIXES:
        ai_mask_sub |= cpc_sub.str.startswith(prefix)
    
    ai_mask_group = np.zeros(len(df), dtype=bool)
    for prefix in AI_GROUP_PREFIXES:
        ai_mask_group |= cpc_group.str.startswith(prefix)
    
    df['is_ai'] = ai_mask_sub | ai_mask_group
    
    ai_bucket = np.full(len(df), 'Non-AI', dtype=object)
    
    ai_bucket[np.where(cpc_sub.str.startswith('G06N'))] = 'AI models (G06N)'
    ai_bucket[np.where(cpc_sub.str.startswith('G06K'))] = 'Pattern recognition (G06K)'
    ai_bucket[np.where(cpc_sub.str.startswith('G06T'))] = 'Vision / graphics (G06T)'
    ai_bucket[np.where(cpc_sub.str.startswith('G10L'))] = 'Speech / audio (G10L)'
    ai_bucket[np.where(cpc_group.str.startswith('G06F 17'))] = 'IR / search (G06F 17)'
    ai_bucket[np.where(cpc_group.str.startswith('G06F 40'))] = 'NLP / text (G06F 40)'
    ai_bucket[np.where(cpc_group.str.startswith('G06F 18'))] = 'Machine translation (G06F 18)'
    ai_bucket[np.where(cpc_group.str.startswith('G06F 19'))] = 'Scientific models (G06F 19)'
    
    df['ai_bucket'] = ai_bucket
    
    return df


def identify_academic_institutions(df, assignee_col='disambig_assignee_organization'):
    df = df.copy()
    
    if assignee_col not in df.columns:
        print(f"Warning: Column '{assignee_col}' not found. Skipping academic identification.")
        df['is_academic'] = False
        df['is_academic_enhanced'] = False
        df['assignee_type_academic'] = 'Corporate'
        return df
    
    # University keywords
    university_keywords = [
        'university', 'univ\\b', 'universidad', 'université', 'universiteit', 'università',
        'college', 'school of', 'medical school', 'law school', 'business school',
        'institute of technology', 'polytechnic', 'technical college',
        'academy', 'seminary', 'conservatory'
    ]
    
    research_keywords = [
        'research institute', 'research center', 'research centre', 'research foundation',
        'national laboratory', 'national lab', 'medical center', 'cancer center',
        'institut de recherche', 'research hospital', 'medical institute'
    ]
    
    known_academic = [
        '\\bmit\\b', '\\bcmu\\b', '\\basu\\b', '\\busc\\b', '\\bucla\\b', '\\bucsb\\b',
        '\\bcaltech\\b', '\\bstanford\\b', '\\bharvard\\b', '\\byale\\b', '\\bpenn\\b',
        '\\bpurdue\\b', '\\bnyu\\b', '\\bbu\\b', '\\bbc\\b', 'georgia tech',
        'johns hopkins', 'mayo clinic', 'cleveland clinic', 'mass general'
    ]
    
    all_academic_patterns = university_keywords + research_keywords + known_academic
    academic_regex = '|'.join(all_academic_patterns)
    
    is_academic = df[assignee_col].str.lower().str.contains(
        academic_regex, 
        na=False, 
        regex=True
    )
    
    corporate_false_positives = [
        'microsoft university', 'google university', 'apple university',
        'corporate university', 'company university', 'training institute'
    ]
    corporate_regex = '|'.join(corporate_false_positives)
    
    is_corporate_training = df[assignee_col].str.lower().str.contains(
        corporate_regex,
        na=False,
        regex=True
    )
    
    df['is_academic'] = is_academic & ~is_corporate_training
    
    state_university_patterns = [
        'state university', 'state college', 'university of [state]',
        'university of california', 'university of texas', 'university of florida',
        'ohio state', 'penn state', 'michigan state', 'arizona state'
    ]
    
    medical_academic_patterns = [
        'medical college', 'school of medicine', 'health science',
        'hospital', 'clinic', 'medical center'
    ]
    
    international_patterns = [
        'technische universität', 'école', 'instituto', 'technion',
        'eth zurich', 'max planck', 'fraunhofer', 'cnrs'
    ]
    
    additional_academic_regex = '|'.join(
        state_university_patterns + medical_academic_patterns + international_patterns
    )
    
    additional_academic = df[assignee_col].str.lower().str.contains(
        additional_academic_regex,
        na=False,
        regex=True
    )
    
    df['is_academic_enhanced'] = df['is_academic'] | additional_academic
    
    df['assignee_type_academic'] = 'Corporate'
    
    df.loc[df['is_academic_enhanced'], 'assignee_type_academic'] = 'Academic'
    
    if 'assignee_category' in df.columns:
        individual_mask = df['assignee_category'].isin(['Individual', 4, 5, '4', '5'])
        df.loc[individual_mask, 'assignee_type_academic'] = 'Individual'
        
        government_mask = df['assignee_category'].isin(['Government', 6, 7, 8, 9, '6', '7', '8', '9'])
        df.loc[government_mask, 'assignee_type_academic'] = 'Government'
    
    manual_government_corrections = [
        'national institutes of health', 'centers for disease control',
        'department of veterans affairs', 'air force', 'navy', 'army'
    ]
    
    for institution in manual_government_corrections:
        mask = df[assignee_col].str.lower().str.contains(
            institution, na=False
        )
        df.loc[mask, 'assignee_type_academic'] = 'Government'
    
    return df


def get_us_region(state_code):
    if pd.isna(state_code):
        return 'Unknown'
    
    state_code = str(state_code).upper()
    
    west = ['CA', 'WA', 'OR', 'NV', 'AK', 'HI', 'CO', 'ID', 'MT', 'UT', 'WY']
    southwest = ['TX', 'AZ', 'NM', 'OK', 'AR', 'LA']
    southeast = ['FL', 'GA', 'SC', 'NC', 'VA', 'WV', 'KY', 'TN', 'AL', 'MS']
    northeast = ['NY', 'NJ', 'PA', 'CT', 'RI', 'MA', 'VT', 'NH', 'ME', 'DE', 'MD']
    midwest = ['IL', 'IN', 'OH', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS']
    
    if state_code in west:
        return 'West'
    elif state_code in southwest:
        return 'Southwest'
    elif state_code in southeast:
        return 'Southeast'
    elif state_code in northeast:
        return 'Northeast'
    elif state_code in midwest:
        return 'Midwest'
    else:
        return 'Unknown'


def get_cpc_major_category(cpc_section):
    if pd.isna(cpc_section):
        return 'Unknown'
    
    cpc_section = str(cpc_section).upper()
    
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
    
    return cpc_mapping.get(cpc_section, 'Unknown')


def load_and_prepare_data(filter_us=True):
    try:
        df = pd.read_csv('data/current_dataset.csv', low_memory=False)
        
        df['patent_date'] = pd.to_datetime(df['patent_date'], format='%Y-%m-%d', errors='coerce')
        df['filing_year'] = df['patent_date'].dt.year
        df['filing_quarter'] = df['patent_date'].dt.quarter
        df['year_quarter'] = df['filing_year'].astype(str) + ' Q' + df['filing_quarter'].astype(str)
        df['month'] = df['patent_date'].dt.to_period('M')
        df['grant_date'] = df['patent_date']
        df['quarter'] = df['grant_date'].dt.quarter
        df['year'] = df['grant_date'].dt.year
        df['grant_month'] = df['grant_date'].dt.month
        
        df['figures_per_claim'] = np.where(
            df['num_claims'] > 0,
            df['num_figures'] / df['num_claims'],
            0
        )
        
        df['us_citations'] = df['citation_count'] - df['foreign_citation_count'].fillna(0)
        df['us_citations'] = df['us_citations'].clip(lower=0)
        df['us_region'] = df['disambig_state'].apply(get_us_region)
        df['cpc_major'] = df['cpc_section'].apply(get_cpc_major_category)
        
        foreign_citations = df['foreign_citation_count'].fillna(0)
        df['total_citations'] = df['us_citations'] + foreign_citations
        df['patent_complexity'] = (df['num_claims'] + df['num_figures']) / 2
        df['citation_per_claim'] = np.where(
            df['num_claims'] > 0,
            df['total_citations'] / df['num_claims'],
            0
        )
        df['foreign_citation_ratio'] = np.where(
            df['total_citations'] > 0,
            foreign_citations / df['total_citations'],
            0
        )
        
        if 'assignee_type' in df.columns:
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
        
        df = flag_ai_patents(df)
        df = identify_academic_institutions(df)
        
        columns_to_drop = [
            'withdrawn', 
            'filename',
            'summary_text',
            'gender_code',
            'location_id_inventor',
            'patent_title'
        ]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop)
            print(f"Dropped columns: {existing_columns_to_drop}")
        
        initial_count = len(df)
        if filter_us:
            df = df[df['disambig_country'] == 'US'].copy()
            filtered_count = len(df)
            print(f"Dataset loaded successfully: {filtered_count:,} US patents (filtered from {initial_count:,} total)")
        else:
            print(f"Dataset loaded successfully: {initial_count:,} patents (all countries)")
        
        print(f"Date range: {df['patent_date'].min()} to {df['patent_date'].max()}")
        print(f"Quarters available: {sorted(df['year_quarter'].unique())}")
        
        if filter_us:
            print(f"US Regions: {df['us_region'].value_counts().to_dict()}")
        elif 'disambig_country' in df.columns and df['disambig_country'].eq('US').any():
            us_data = df[df['disambig_country'] == 'US']
            if len(us_data) > 0:
                print(f"US Regions (from {len(us_data)} US patents): {us_data['us_region'].value_counts().to_dict()}")
        
        print(f"\n{'='*70}")
        print("Dataset Statistics Summary")
        print(f"{'='*70}")
        
        print(f"\nBasic Information:")
        print(f"  - Total patents: {len(df):,}")
        print(f"  - Total columns: {len(df.columns)}")
        
        if 'patent_type' in df.columns:
            print(f"\nPatent Type Distribution:")
            type_counts = df['patent_type'].value_counts()
            for ptype, count in type_counts.items():
                pct = (count / len(df)) * 100
                print(f"  - {ptype}: {count:,} ({pct:.2f}%)")
        
        # WIPO kind distribution
        if 'wipo_kind' in df.columns:
            wipo_data = df[df['wipo_kind'].notna()]
            if len(wipo_data) > 0:
                print(f"\nWIPO Kind Distribution ({len(wipo_data):,} patents with WIPO kind):")
                wipo_counts = wipo_data['wipo_kind'].value_counts()
                for kind, count in wipo_counts.items():
                    pct = (count / len(wipo_data)) * 100
                    print(f"  - {kind}: {count:,} ({pct:.2f}%)")
        
        if 'disambig_country' in df.columns:
            print(f"\nTop 10 Countries by Patent Count:")
            country_counts = df['disambig_country'].value_counts().head(10)
            for country, count in country_counts.items():
                pct = (count / len(df)) * 100
                print(f"  - {country}: {count:,} ({pct:.2f}%)")
        
        if 'us_region' in df.columns and not df[df['us_region'] != 'Unknown'].empty:
            us_patents = df[df['us_region'] != 'Unknown']
            if len(us_patents) > 0:
                print(f"\nUS Regional Distribution ({len(us_patents):,} US patents):")
                region_counts = us_patents['us_region'].value_counts()
                for region, count in region_counts.items():
                    pct = (count / len(us_patents)) * 100
                    print(f"  - {region}: {count:,} ({pct:.2f}%)")
        
        if 'disambig_state' in df.columns:
            state_data = df[df['disambig_state'].notna()]
            if len(state_data) > 0:
                print(f"\nTop 10 US States by Patent Count:")
                state_counts = state_data['disambig_state'].value_counts().head(10)
                for state, count in state_counts.items():
                    pct = (count / len(state_data)) * 100
                    print(f"  - {state}: {count:,} ({pct:.2f}%)")
        
        # CPC category distribution
        if 'cpc_major' in df.columns:
            cpc_data = df[df['cpc_major'].notna()]
            if len(cpc_data) > 0:
                print(f"\nCPC Major Category Distribution ({len(cpc_data):,} patents with CPC):")
                cpc_counts = cpc_data['cpc_major'].value_counts()
                for category, count in cpc_counts.items():
                    pct = (count / len(cpc_data)) * 100
                    print(f"  - {category}: {count:,} ({pct:.2f}%)")
        
        if 'assignee_category' in df.columns:
            assignee_data = df[df['assignee_category'].notna()]
            if len(assignee_data) > 0:
                print(f"\nAssignee Category Distribution ({len(assignee_data):,} patents with assignee info):")
                assignee_counts = assignee_data['assignee_category'].value_counts()
                for category, count in assignee_counts.items():
                    pct = (count / len(assignee_data)) * 100
                    print(f"  - {category}: {count:,} ({pct:.2f}%)")
        
        if 'is_ai' in df.columns:
            ai_data = df[df['is_ai'] == True]
            non_ai_data = df[df['is_ai'] == False]
            total_patents = len(df)
            ai_count = len(ai_data)
            ai_pct = (ai_count / total_patents) * 100 if total_patents > 0 else 0
            print(f"\nAI Patent Statistics:")
            print(f"  - Total AI patents: {ai_count:,} ({ai_pct:.2f}% of dataset)")
            print(f"  - Non-AI patents: {len(non_ai_data):,} ({100 - ai_pct:.2f}% of dataset)")
            
            if 'ai_bucket' in df.columns:
                ai_categories = df[df['is_ai'] == True]['ai_bucket'].value_counts()
                if len(ai_categories) > 0:
                    print(f"  - AI Category Breakdown:")
                    for category, count in ai_categories.items():
                        if category != 'Non-AI':
                            pct = (count / ai_count) * 100 if ai_count > 0 else 0
                            print(f"    {category}: {count:,} ({pct:.2f}%)")
        
        if 'citation_count' in df.columns:
            citation_data = df[df['citation_count'].notna()]
            if len(citation_data) > 0:
                print(f"\nCitation Statistics ({len(citation_data):,} patents with citation data):")
                print(f"  - Mean citations per patent: {citation_data['citation_count'].mean():.2f}")
                print(f"  - Median citations per patent: {citation_data['citation_count'].median():.2f}")
                print(f"  - Max citations: {citation_data['citation_count'].max():.2f}")
                zero_citations = (citation_data['citation_count'] == 0).sum()
                zero_pct = (zero_citations / len(citation_data)) * 100
                print(f"  - Patents with 0 citations: {zero_citations:,} ({zero_pct:.2f}%)")
                if 'total_citations' in df.columns:
                    total_citation_data = df[df['total_citations'].notna() & (df['total_citations'] > 0)]
                    if len(total_citation_data) > 0:
                        print(f"  - Mean total citations (US + foreign): {total_citation_data['total_citations'].mean():.2f}")
        
        # Patent complexity statistics
        if 'patent_complexity' in df.columns:
            complexity_data = df[df['patent_complexity'].notna()]
            if len(complexity_data) > 0:
                print(f"\nPatent Complexity Statistics:")
                print(f"  - Mean complexity score: {complexity_data['patent_complexity'].mean():.2f}")
                print(f"  - Median complexity score: {complexity_data['patent_complexity'].median():.2f}")
                if 'num_claims' in df.columns:
                    claims_data = df[df['num_claims'].notna()]
                    if len(claims_data) > 0:
                        print(f"  - Mean number of claims: {claims_data['num_claims'].mean():.2f}")
                        print(f"  - Median number of claims: {claims_data['num_claims'].median():.2f}")
                if 'num_figures' in df.columns:
                    figures_data = df[df['num_figures'].notna()]
                    if len(figures_data) > 0:
                        print(f"  - Mean number of figures: {figures_data['num_figures'].mean():.2f}")
                        print(f"  - Median number of figures: {figures_data['num_figures'].median():.2f}")
        
        print(f"\nTime-Based Statistics:")
        if 'year_quarter' in df.columns:
            quarterly_counts = df['year_quarter'].value_counts().sort_index()
            print(f"  - Patents per quarter:")
            for quarter, count in quarterly_counts.items():
                print(f"    {quarter}: {count:,}")
        if 'filing_year' in df.columns:
            year_counts = df['filing_year'].value_counts().sort_index()
            print(f"  - Patents per year:")
            for year, count in year_counts.items():
                print(f"    {year}: {count:,}")
        
        print(f"\nMissing Data Summary:")
        key_columns = ['citation_count', 'cpc_section', 'disambig_state', 'assignee_type', 
                      'num_claims', 'num_figures', 'disambig_country']
        for col in key_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                if missing_count > 0:
                    print(f"  - {col}: {missing_count:,} missing ({missing_pct:.2f}%)")
                else:
                    print(f"  - {col}: No missing data")
        
        print(f"\n{'='*70}")
        
        return df
    
    except FileNotFoundError:
        print("Error: 'data/current_dataset.csv' not found.")
        print("Please run datasets_merging.py first to create the merged dataset.")
        return None


def save_cleaned_data(df, output_path='data/cleaned_patents.csv'):
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved successfully to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        return False


if __name__ == '__main__':
    df = load_and_prepare_data(filter_us=False)
    
    if df is not None:
        print("\n" + "="*70)
        print("Data Summary:")
        print("="*70)
        print(f"Total rows: {len(df):,}")
        print(f"Total columns: {len(df.columns)}")
        
        # Check for columns without readable names
        missing_readable = [col for col in df.columns if col not in READABLE_NAMES]
        if missing_readable:
            print(f"\n⚠️  Warning: {len(missing_readable)} columns without readable names:")
            for col in missing_readable:
                print(f"  - {col}")
        else:
            print(f"\n✓ All {len(df.columns)} columns have readable names")
        
        print(f"\nColumn names: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nBasic statistics:")
        print(df.describe())
        
        print("\n" + "="*70)
        save_cleaned_data(df, 'data/cleaned_patents.csv')

