import pandas as pd
import ast

def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

def normalize_description(desc):
    # Replace non-ASCII characters, e.g., â€™ with their appropriate replacements
    desc = desc.replace('â€™', "'")  # Replace â€™ with a standard single quote
    return desc

def clean_tabular_data(raw_data):
    # Make a copy of the raw_data to avoid SettingWithCopy warning
    data_copy = raw_data.copy()

    # Remove rows with missing ratings
    rating_columns = [
        'Cleanliness_rating',
        'Accuracy_rating',
        'Communication_rating',
        'Location_rating',
        'Check-in_rating',
        'Value_rating'
    ]
    data_copy = data_copy.dropna(subset=rating_columns)

    # Combine and clean description strings
    data_copy['Description'] = data_copy['Description'].fillna('')
    data_copy['Description'] = data_copy['Description'].apply(normalize_description)
    data_copy['Description'] = data_copy['Description'].apply(lambda desc: ' '.join([d.strip() for d in safe_literal_eval(desc) if d.strip() != '']))
    data_copy['Description'] = data_copy['Description'].str.replace('About this space', '')

    # Set default values of 1 for empty feature columns
    default_columns = ["guests", "beds", "bathrooms", "bedrooms"]
    data_copy.loc[:, default_columns] = data_copy[default_columns].fillna(1)

    return data_copy

if __name__ == "__main__":
    # Load the raw data using pandas
    input_file = 'listing.csv'
    raw_data = pd.read_csv(input_file)

    # Call clean_tabular_data on the raw data
    processed_data = clean_tabular_data(raw_data)

    # Save the processed data as clean_tabular_data.csv in the same folder
    output_file = 'clean_tabular_data.csv'
    processed_data.to_csv(output_file, index=False)

    print("Data cleaning and processing complete. Processed data saved to", output_file)






