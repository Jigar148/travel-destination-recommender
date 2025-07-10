import pandas as pd

# Load travel_data.csv
df = pd.read_csv("travel_data.csv")

# Get all unique destinations
destinations = df["DESTINATION"].dropna().unique()

# Sort and print them
destinations = sorted(destinations)
print("All Unique Destinations in travel_data.csv:\n")
for dest in destinations:
    print("-", dest)
