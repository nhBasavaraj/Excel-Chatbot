import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = 'WORK/files/Facebook_data_V3.csv'

# Read the CSV file with the correct separator
df = pd.read_csv(csv_file_path, sep=',', encoding="Windows-1252")

# Create a new file to store the sentences
output_file_path = 'WORK/files/csv-txt-facebook.txt'

# Open the file in write mode
with open(output_file_path, 'w') as output_file:
    # Iterate over rows in the DataFrame
    for _, row in df.iterrows():
        # Create a sentence for each row
        sentence = (
        f"The ad set named '{row['Ad Set Name']}' with the treatment '{row['Treatment']}' "
        f"and targeted at '{row['Location']}' displayed the ad named '{row['Ad name']}'. "
        f"It achieved {row['Results']} results of the '{row['Result type']}' type, "
        f"reaching {row['Reach']} people and generating {row['Impressions']} impressions. "
        f"The cost per result was approximately ₹{row['Cost per result']:.2f}, "
        f"with a total expenditure of ₹{row['Amount spent (INR)']:.2f}. "
        f"The ad received {row['Clicks (all)']} clicks, resulting in a click-through rate (CTR) "
        f"of approximately {row['CTR (all) (click through rate)']:.2%}, "
        f"and the cost per click (CPC) was approximately ₹{row['CPC (All) (cost per click)']:.2f}.\n")

        # Write the sentence to the output file
        output_file.write(sentence)

print(f"The sentences have been saved to {output_file_path}.")


