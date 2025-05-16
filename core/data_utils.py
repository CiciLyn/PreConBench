"""
Data reading and processing module for medical consultation simulation.

This module provides functionality for:
1. Reading patient data from Excel files
2. Processing patient information
3. Assigning character traits to patients
"""

from typing import Dict, List, Any
import pandas as pd
from data.prompts import patient
from pprint import pprint


def read_patient_data(patient_path: str, patient_num: int) -> pd.DataFrame:
    """Read and process patient data from Excel file.

    Args:
        patient_path: Path to the Excel file containing patient data
        patient_num: Number of patients to read

    Returns:
        DataFrame containing processed patient data with character assignments
    """

    if patient_path.endswith(".json"):
        # Read patient data
        p_df = pd.read_json(patient_path)
    else:
        # Read patient data
        p_df = pd.read_excel(patient_path)
    
    p_df = p_df.iloc[:patient_num]

    # Get character templates
    p_characters_all, p_cog_cla_all = patient.p_characters()
    # partial_p_characters = patient.partial_p_characters(p_characters_all, p_cog_cla_all)
    partial_p_characters_ablation_study = patient.partial_p_characters_ablation_study(p_characters_all, p_cog_cla_all) # OOA only. For controlled_experiemnt.

    # Process easy and hard records

    record_df = _process_record_data(p_df)
    # return _assign_characters(record_df, partial_p_characters)
    return _assign_characters(record_df, partial_p_characters_ablation_study)


def _process_record_data(df: pd.DataFrame) -> pd.DataFrame:

    """Process patient data for a specific record type (easy/hard).

    Args:
        df: Input DataFrame containing patient data
        record_type: Type of record ('easy' or 'hard')

    Returns:
        Processed DataFrame for the specified record type
    """
    # Define column mappings
    base_cols = ["科室", "病人性别", "病人年龄", "record_difficulty_level", "word_count_level"]  # 病人没有名字. 

    record_cols = {
        "处理后主诉": [
            "process_info",
        ],
        "处理后现病史": [
            "process_info",
        ],
        "处理后既往史": [
            "process_info",
        ],
    }

    # Select and rename columns
    selected_cols = base_cols + list(record_cols.keys())
    record_df = df[selected_cols]
    
    # Process and rename columns
    for old_col, new_cols in record_cols.items():
        if old_col in record_df.columns:
            col_name = old_col.replace("处理后", "")
            record_df[col_name] = record_df[old_col].apply(lambda x: x['process_info'] if isinstance(x, dict) else x)
            record_df = record_df.drop(columns=[old_col])
            
    pprint(record_df.head(5))
    return record_df


def _assign_characters(df: pd.DataFrame, characters: Dict[str, str]) -> pd.DataFrame:
    """Assign character traits to patients.

    Args:
        df: DataFrame containing patient data
        characters: Dictionary mapping character labels to descriptions

    Returns:
        DataFrame with character assignments for each patient
    """
    expanded_rows = []

    # Create new rows for each patient-character combination
    for _, row in df.iterrows():
        for char_key, char_value in characters.items():
            new_row = row.copy()
            new_row["character_label"] = char_key
            new_row["character"] = char_value
            expanded_rows.append(new_row)

    # Create new DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df.reset_index(drop=True)
