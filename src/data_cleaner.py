import pandas as pd

def clean_spam_data(input_file='spam.csv', output_file='spam_cleaned.csv'):
    # 1. Load the dataset using pandas with 'latin-1' encoding.
    df = pd.read_csv(input_file, encoding='latin-1')

    # 2. Remove unnecessary columns.
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    
    # 3. Convert 'ham' and 'spam' labels to 0 and 1.
    # This maps the string values to integer representations.
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # 4. Drop any rows where the message is missing to prevent errors later.
    df.dropna(subset=['message'], inplace=True)

    # 5. Save the cleaned DataFrame to a new CSV file.
    df.to_csv(output_file, index=False)
    
    print(f"--- Successfully cleaned data and saved to {output_file} ---")
    print("Head of the cleaned data:")
    print(df.head())

    return df


# --- Main Execution ---
if __name__ == '__main__':
    clean_spam_data()
