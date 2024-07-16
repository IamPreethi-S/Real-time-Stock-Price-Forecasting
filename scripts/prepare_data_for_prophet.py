import pandas as pd

def prepare_data_for_prophet(input_csv, output_csv):
    # Load the enhanced data
    df = pd.read_csv(input_csv)
    
    # Ensure the data is in the correct format
    df = df[['ds', 'y']]
    
    # Check for any missing values
    df['y'].fillna(method='ffill', inplace=True)
    
    # Save the correctly formatted data for Prophet
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    input_csv = '../data/enhanced_stock_data.csv'
    output_csv = '../data/prophet_ready_data.csv'
    prepare_data_for_prophet(input_csv, output_csv)
    
    # Load and print the first few rows to verify
    df = pd.read_csv(output_csv)
    print(df.head())
