import pandas as pd
import streamlit as st
import pandas as pd
import re

def handle_missing_values(df, strategy='drop'):
    """Handles missing values in a DataFrame."""
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        for col in df.select_dtypes(include=['number']).columns:
            fill_val = df[col].mean()
            if pd.api.types.is_integer_dtype(df[col].dtype):
                fill_val = int(round(fill_val))  
            df[col] = df[col].fillna(fill_val)
    elif strategy == 'median':
        for col in df.select_dtypes(include=['number']).columns:
            fill_val = df[col].median()
            if pd.api.types.is_integer_dtype(df[col].dtype):
                fill_val = int(round(fill_val))
            df[col] = df[col].fillna(fill_val)
    elif strategy == 'most_frequent':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def remove_duplicates(df):
    """Removes duplicate rows from a DataFrame.
    """
    return df.drop_duplicates()

def convert_data_type(df, column, target_dtype):
    """Converts a column to a specified target data type (safe conversion)."""
    if target_dtype == 'datetime':
        df[column] = pd.to_datetime(df[column], errors='coerce')
    else:
        try:
            if target_dtype in ['int', 'float']:
 
                df[column] = pd.to_numeric(df[column], errors='coerce')

                if target_dtype == 'int':
                    df[column] = df[column].astype('Int64')  
                else:
                    df[column] = df[column].astype(float)

            else:
                df[column] = df[column].astype(target_dtype)

        except Exception as e:
            st.warning(f"Could not convert column '{column}' to {target_dtype}. Error: {e}")

    return df


def rename_columns(df, rename_dict=None):
    """Renames columns in the DataFrame based on a dictionary mapping.
    """
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df.head(2)

def clean_special(df, column, mode="remove_special", keep_part=None, remove_chars=None):
    """
        Cleaning strategy:
        - "remove_special" : remove special characters (!?@#$%^&* etc.)
        - "remove_custom" : remove user-specified characters (remove_chars)
        - "keep_before"   : keep text before a delimiter (keep_part)
        - "keep_after"    : keep text after a delimiter (keep_part)
    """

    def _clean(val):
        if pd.isna(val):
            return val
        val = str(val)

        if mode == "remove_special":
            return re.sub(r"[^a-zA-Z0-9\s]", "", val)

        elif mode == "remove_custom" and remove_chars:
            return re.sub(f"[{re.escape(remove_chars)}]", "", val)

        elif mode == "keep_before" and keep_part:
            return val.split(keep_part)[0]

        elif mode == "keep_after" and keep_part:
            return val.split(keep_part)[-1]

        return val  

    df[column] = df[column].apply(_clean)
    return df

def check_unique_values(df, columns):
    """Returns unique values for each selected column.
    """
    unique_dict = {}
    for col in columns:
        if col in df.columns:
            unique_dict[col] = df[col].unique().tolist()
    return unique_dict

def drop_selected_columns(df, columns):
    """Drops the specified columns from the DataFrame.
    """
    return df.drop(columns=columns, errors="ignore")