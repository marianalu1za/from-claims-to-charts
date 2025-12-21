"""
Patent Dataset Merging Script
This script merges multiple TSV data files with the raw patents CSV file.
This creates a comprehensive current_dataset.csv file that will be used as the starting point
for data engineering in the rest of the project.
"""

# Importing necessary libraries for data manipulation and file handling
import pandas as pd
from pathlib import Path
import warnings

# Suppressing warnings to keep the output clean during data processing
warnings.filterwarnings('ignore')


def load_base_dataset(file_path='data/raw_patents.csv'):
    """
    This function loads the base raw patents dataset.
    This serves as the foundation for merging additional data.
    
    Args:
        file_path: Path to the raw patents CSV file
    
    Returns:
        A pandas DataFrame with the base dataset, or None if file not found
    """
    try:
        # Loading the CSV file with low_memory=False to handle mixed data types properly
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Loaded base dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Filtering to ensure only patents within the target date range are included
        # Target timeframe: January 1, 2024 to June 30, 2025 (end of Q2 2025)
        if 'patent_date' in df.columns:
            # Parsing patent_date to datetime format for filtering
            df['patent_date'] = pd.to_datetime(df['patent_date'], format='%Y-%m-%d', errors='coerce')
            # Filtering to the target date range
            start_date = pd.to_datetime('2024-01-01')
            end_date = pd.to_datetime('2025-06-30')
            initial_count = len(df)
            df = df[(df['patent_date'] >= start_date) & (df['patent_date'] <= end_date)].copy()
            filtered_count = len(df)
            print(f"Date filtering: {initial_count} -> {filtered_count} patents (2024-01-01 to 2025-06-30)")
            if filtered_count < initial_count:
                print(f"  Removed {initial_count - filtered_count} patents outside date range")
        else:
            print("Warning: 'patent_date' column not found. Cannot filter by date range.")
        
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading base dataset: {e}")
        return None


def merge_assignee_data(df, file_path='data/g_assignee_disambiguated.tsv'):
    """
    This function merges assignee disambiguated data with the base dataset.
    This adds general assignee information including organization names and assignee types.
    Individual assignee names are not included as only general categories are needed.
    
    Args:
        df: Base dataframe to merge with
        file_path: Path to the assignee TSV file
    
    Returns:
        Merged dataframe with assignee information added
    """
    try:
        # Loading the assignee TSV file
        assignee_df = pd.read_csv(file_path, sep='\t', low_memory=False)
        print(f"\nLoaded assignee data: {len(assignee_df)} rows")
        
        # Checking if there are multiple rows per patent_id
        patent_counts = assignee_df['patent_id'].value_counts()
        max_assignees = patent_counts.max()
        print(f"Max assignees per patent: {max_assignees}")
        
        # Selecting only the columns we care about: general types and categories
        # Excluding individual name fields (first/last name) and assignee_id
        columns_to_keep = ['patent_id', 'assignee_type', 'disambig_assignee_organization', 'location_id']
        assignee_df = assignee_df[columns_to_keep]
        
        if max_assignees > 1:
            # If multiple assignees per patent, aggregating the data
            # Taking the first assignee type and location, but aggregating organizations
            assignee_agg = assignee_df.groupby('patent_id').agg({
                'assignee_type': 'first',  # Taking first assignee type
                'disambig_assignee_organization': lambda x: '; '.join(x.dropna().astype(str).unique()),
                'location_id': 'first'
            }).reset_index()
            # Adding count of assignees
            assignee_counts = assignee_df.groupby('patent_id').size().reset_index(name='num_assignees_from_tsv')
            assignee_agg = assignee_agg.merge(assignee_counts, on='patent_id', how='left')
            assignee_df = assignee_agg
        else:
            # If single assignee, just adding a count column
            assignee_df['num_assignees_from_tsv'] = 1
        
        # Merging with base dataset using left join to preserve all patents
        # Only merging if assignee_type column doesn't already exist or if we want to update it
        merge_cols = ['patent_id'] + [col for col in assignee_df.columns if col != 'patent_id' and col not in df.columns]
        if len(merge_cols) > 1:
            df = df.merge(assignee_df[merge_cols], on='patent_id', how='left', suffixes=('', '_assignee'))
        
        print(f"Merged assignee data: {df['patent_id'].notna().sum()} patents matched")
        return df
    
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping assignee data merge.")
        return df
    except Exception as e:
        print(f"Error merging assignee data: {e}")
        return df


def merge_summary_text(df, file_path='data/g_brf_sum_text_2025.tsv'):
    """
    This function merges brief summary text data with the base dataset.
    This adds summary text information for each patent.
    
    Args:
        df: Base dataframe to merge with
        file_path: Path to the summary text TSV file
    
    Returns:
        Merged dataframe with summary text added
    """
    try:
        # Loading the summary text TSV file
        summary_df = pd.read_csv(file_path, sep='\t', low_memory=False)
        print(f"\nLoaded summary text data: {len(summary_df)} rows")
        
        # Merging with base dataset using left join
        df = df.merge(summary_df, on='patent_id', how='left', suffixes=('', '_summary'))
        
        print(f"Merged summary text: {df['summary_text'].notna().sum()} patents with summary text")
        return df
    
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping summary text merge.")
        return df
    except Exception as e:
        print(f"Error merging summary text: {e}")
        return df


def merge_inventor_data(df, file_path='data/g_inventor_disambiguated.tsv'):
    """
    This function merges inventor disambiguated data with the base dataset.
    This adds general inventor information like gender codes and location, but not individual names.
    Only general categories and types are included, not specific inventor names.
    
    Args:
        df: Base dataframe to merge with
        file_path: Path to the inventor TSV file
    
    Returns:
        Merged dataframe with inventor information added
    """
    try:
        # Loading the inventor TSV file
        inventor_df = pd.read_csv(file_path, sep='\t', low_memory=False)
        print(f"\nLoaded inventor data: {len(inventor_df)} rows")
        
        # Checking if there are multiple rows per patent_id (multiple inventors)
        patent_counts = inventor_df['patent_id'].value_counts()
        max_inventors = patent_counts.max()
        print(f"Max inventors per patent: {max_inventors}")
        
        # Selecting only general categories, excluding individual names
        # Keeping only gender_code and location_id as general categories
        columns_to_keep = ['patent_id', 'gender_code', 'location_id']
        inventor_df = inventor_df[columns_to_keep]
        
        if max_inventors > 1:
            # If multiple inventors per patent, aggregating the data
            # Aggregating gender codes and taking first location
            inventor_agg = inventor_df.groupby('patent_id').agg({
                'gender_code': lambda x: '; '.join(x.dropna().astype(str).unique()),
                'location_id': 'first'
            }).reset_index()
            # Adding count of inventors
            inventor_counts = inventor_df.groupby('patent_id').size().reset_index(name='num_inventors')
            inventor_agg = inventor_agg.merge(inventor_counts, on='patent_id', how='left')
            inventor_df = inventor_agg
        else:
            # If single inventor, just adding a count column
            inventor_df['num_inventors'] = 1
        
        # Merging with base dataset using left join
        df = df.merge(inventor_df, on='patent_id', how='left', suffixes=('', '_inventor'))
        
        print(f"Merged inventor data: {df['patent_id'].notna().sum()} patents matched")
        return df
    
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping inventor data merge.")
        return df
    except Exception as e:
        print(f"Error merging inventor data: {e}")
        return df


def merge_patent_data(df, file_path='data/g_patent.tsv'):
    """
    This function merges additional patent data with the base dataset.
    This adds patent title and other patent-specific information.
    
    Args:
        df: Base dataframe to merge with
        file_path: Path to the patent TSV file
    
    Returns:
        Merged dataframe with additional patent information added
    """
    try:
        # Loading the patent TSV file
        patent_df = pd.read_csv(file_path, sep='\t', low_memory=False)
        print(f"\nLoaded patent data: {len(patent_df)} rows")
        
        # Selecting only columns that don't already exist in the base dataset
        # or that might have additional information
        columns_to_merge = ['patent_id']
        for col in patent_df.columns:
            if col != 'patent_id' and col not in df.columns:
                columns_to_merge.append(col)
            elif col == 'patent_title' and 'patent_title' not in df.columns:
                columns_to_merge.append(col)
        
        if len(columns_to_merge) > 1:
            patent_df_subset = patent_df[columns_to_merge]
            # Merging with base dataset using left join
            df = df.merge(patent_df_subset, on='patent_id', how='left', suffixes=('', '_patent'))
            print(f"Merged patent data: {len(columns_to_merge)-1} new columns added")
        else:
            print("No new columns to merge from patent data")
        
        return df
    
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping patent data merge.")
        return df
    except Exception as e:
        print(f"Error merging patent data: {e}")
        return df


def clean_merged_dataset(df):
    """
    This function cleans the merged dataset by handling duplicate columns and data types.
    This ensures the final dataset is ready for use in data engineering.
    
    Args:
        df: Merged dataframe to clean
    
    Returns:
        Cleaned dataframe
    """
    # Removing duplicate columns that might have been created during merge
    # Keeping the first occurrence of each column name
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Ensuring patent_id is the first column for easier reference
    if 'patent_id' in df.columns:
        cols = ['patent_id'] + [col for col in df.columns if col != 'patent_id']
        df = df[cols]
    
    print(f"\nCleaned dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def save_current_dataset(df, output_path='data/current_dataset.csv'):
    """
    This function saves the merged and cleaned dataset to a CSV file.
    This creates the current_dataset.csv file that will be used as the starting point
    for data engineering in the rest of the project.
    
    Args:
        df: Dataframe to save
        output_path: Path where the dataset should be saved
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Creating the data directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Saving the dataframe to CSV
        # Using index=False to avoid saving the index as a column
        df.to_csv(output_path, index=False)
        print(f"\n{'='*70}")
        print(f"Current dataset saved successfully to {output_path}")
        print(f"{'='*70}")
        return True
    except Exception as e:
        print(f"Error saving current dataset: {e}")
        return False


def main():
    """
    This is the main function that orchestrates the entire merging process.
    This loads the base dataset, merges all additional data files, cleans the result,
    and saves it as current_dataset.csv.
    """
    print("="*70)
    print("Patent Dataset Merging Script")
    print("="*70)
    
    # Loading the base dataset
    df = load_base_dataset('data/raw_patents.csv')
    if df is None:
        print("Cannot proceed without base dataset. Exiting.")
        return
    
    # Merging all additional data files
    df = merge_assignee_data(df)
    df = merge_summary_text(df)
    df = merge_inventor_data(df)
    df = merge_patent_data(df)
    
    # Cleaning the merged dataset
    df = clean_merged_dataset(df)
    
    # Printing summary statistics
    print("\n" + "="*70)
    print("Merge Summary:")
    print("="*70)
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Saving the current dataset
    save_current_dataset(df, 'data/current_dataset.csv')
    
    print("\n" + "="*70)
    print("Merging complete!")
    print("="*70)


# Running the main function when the script is executed
if __name__ == '__main__':
    main()

