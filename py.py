import os
import csv
import glob

def extract_and_consolidate_data():
    # Find all hist_t_*.csv files
    csv_files = glob.glob('hist_t_*.csv')
    csv_files.sort()  # Sort files alphabetically
    
    output_data = []
    
    for file_path in csv_files:
        # Extract filename
        filename = os.path.basename(file_path)
        
        # Read CSV data
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            
            # Add filename header
            output_data.append(filename)
            output_data.append("theta_deg,counts")
            
            # Filter and extract data where theta_deg < 5
            for row in reader:
                theta_deg = float(row['theta_deg'])
                if theta_deg < 5.0:
                    output_data.append(f"{theta_deg},{row['counts']}")
            
            # Add empty line between files
            output_data.append("")
    
    # Write to consolidated output file
    with open('consolidated_data.csv', 'w') as output_file:
        for line in output_data:
            output_file.write(line + '\n')
    
    print(f"Data extracted from {len(csv_files)} files and saved to consolidated_data.csv")
    print(f"Files processed: {', '.join(csv_files)}")

if __name__ == "__main__":
    extract_and_consolidate_data()