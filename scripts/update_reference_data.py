"""
Script to complete the reference_activities.csv with all environmental metrics
"""
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def update_reference_data():
    """Update reference activities with comprehensive environmental metrics."""
    
    csv_path = Path(__file__).parent.parent / "data" / "reference_activities.csv"
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Check which columns exist
    required_columns = [
        'Activity', 'Avg_CO2_Emission(kg/day)', 'Category',
        'Water_Usage(L/day)', 'Energy_Usage(kWh/day)', 
        'Waste_Generation(kg/day)', 'Land_Use(m2)', 'Pollution_Index'
    ]
    
    # Add missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            if col == 'Water_Usage(L/day)':
                df[col] = 0.0
            elif col == 'Energy_Usage(kWh/day)':
                df[col] = 0.0
            elif col == 'Waste_Generation(kg/day)':
                df[col] = 0.0
            elif col == 'Land_Use(m2)':
                df[col] = 0.0
            elif col == 'Pollution_Index':
                df[col] = 0
    
    # Fill NaN values based on CO2 emissions (reasonable estimates)
    for idx, row in df.iterrows():
        if pd.isna(row['Water_Usage(L/day)']):
            co2 = row['Avg_CO2_Emission(kg/day)']
            category = row['Category']
            
            # Estimate water based on category and CO2
            if category == 'Transport':
                df.at[idx, 'Water_Usage(L/day)'] = round(co2 * 0.5, 1)
                df.at[idx, 'Energy_Usage(kWh/day)'] = round(co2 * 0.3, 1)
                df.at[idx, 'Waste_Generation(kg/day)'] = round(co2 * 0.02, 2)
                df.at[idx, 'Pollution_Index'] = min(int(co2 * 8), 100)
            elif category == 'Household':
                df.at[idx, 'Water_Usage(L/day)'] = round(co2 * 8, 1)
                df.at[idx, 'Energy_Usage(kWh/day)'] = round(co2 * 1.5, 1)
                df.at[idx, 'Waste_Generation(kg/day)'] = round(co2 * 0.03, 2)
                df.at[idx, 'Pollution_Index'] = min(int(co2 * 7), 100)
            elif category == 'Food':
                df.at[idx, 'Water_Usage(L/day)'] = round(co2 * 250, 1)
                df.at[idx, 'Energy_Usage(kWh/day)'] = round(co2 * 0.2, 1)
                df.at[idx, 'Waste_Generation(kg/day)'] = round(co2 * 0.05, 2)
                df.at[idx, 'Pollution_Index'] = min(int(co2 * 6), 100)
            elif category == 'Lifestyle':
                df.at[idx, 'Water_Usage(L/day)'] = round(co2 * 15, 1)
                df.at[idx, 'Energy_Usage(kWh/day)'] = round(co2 * 0.4, 1)
                df.at[idx, 'Waste_Generation(kg/day)'] = round(co2 * 0.04, 2)
                df.at[idx, 'Pollution_Index'] = min(int(co2 * 5), 100)
            elif category == 'Energy':
                df.at[idx, 'Water_Usage(L/day)'] = round(co2 * 10, 1)
                df.at[idx, 'Energy_Usage(kWh/day)'] = round(co2 * 2.0, 1)
                df.at[idx, 'Waste_Generation(kg/day)'] = round(co2 * 0.02, 2)
                df.at[idx, 'Pollution_Index'] = min(int(co2 * 9), 100)
            
            # Set land use to 0 by default (special cases already set)
            if pd.isna(row['Land_Use(m2)']):
                df.at[idx, 'Land_Use(m2)'] = 0.0
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"✓ Updated {csv_path}")
    print(f"✓ Total activities: {len(df)}")
    print(f"✓ Columns: {', '.join(df.columns)}")
    print("\nSample data:")
    print(df.head(10))
    
if __name__ == "__main__":
    update_reference_data()
