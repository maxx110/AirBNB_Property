import pandas as pd
import ast

class DataCleaner:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def safe_literal_eval(self, value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    def normalize_description(self, desc):
        desc = desc.replace('â€™', "'")  # Replace â€™ with a standard single quote
        return desc

    def clean(self):
        # Load the raw data using pandas
        raw_data = pd.read_csv(self.input_file)

        # Remove rows with missing ratings
        rating_columns = [
            'Cleanliness_rating',
            'Accuracy_rating',
            'Communication_rating',
            'Location_rating',
            'Check-in_rating',
            'Value_rating'
        ]
        raw_data = raw_data.dropna(subset=rating_columns)

        # Combine and clean description strings
        raw_data['Description'] = raw_data['Description'].fillna('')
        raw_data['Description'] = raw_data['Description'].apply(self.normalize_description)
        raw_data['Description'] = raw_data['Description'].apply(
            lambda desc: ' '.join([d.strip() for d in self.safe_literal_eval(desc) if d.strip() != '']))
        raw_data['Description'] = raw_data['Description'].str.replace('About this space', '')

        # Set default values of 1 for empty feature columns
        default_columns = ["guests", "beds", "bathrooms", "bedrooms"]
        raw_data.loc[:, default_columns] = raw_data[default_columns].fillna(1)

        # Save the processed data as clean_tabular_data.csv in the same folder
        raw_data.to_csv(self.output_file, index=False)

        print("Data cleaning and processing complete. Processed data saved to", self.output_file)

if __name__ == "__main__":
    input_file = 'listing.csv'
    output_file = 'clean_tabular_data.csv'

    cleaner = DataCleaner(input_file, output_file)
    cleaner.clean()







