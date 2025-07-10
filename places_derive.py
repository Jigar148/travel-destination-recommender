import csv
import random

# Simulated extended data for demonstration
def generate_place_info(place):
    distances = range(1, 50)  # km from city center
    best_seasons = ['Winter', 'Monsoon', 'Spring', 'Autumn', 'Summer']
    transport_modes = ['By Road', 'By Train', 'By Air', 'Cable Car', 'By Boat']

    descriptions = {
        "Temple": "A famous and spiritual temple known for religious significance.",
        "Fort": "A historical fort offering insights into past architecture.",
        "Lake": "A scenic lake, great for boating and photography.",
        "Palace": "A royal palace reflecting rich heritage and art.",
        "Beach": "A relaxing beach with golden sands and gentle waves.",
        "Garden": "A beautifully maintained garden with exotic plants.",
        "Museum": "An informative museum showcasing local culture and history.",
        "Hill": "A picturesque hill with trekking and panoramic views.",
        "Park": "A recreational park ideal for family and kids.",
        "Cave": "An ancient cave site with archaeological significance."
    }

    # Infer keyword-based description
    description = "A must-visit destination with unique attractions."
    for key in descriptions:
        if key.lower() in place.lower():
            description = descriptions[key]
            break

    return {
        'Distance (km)': random.randint(2, 40),
        'Best Time to Visit': random.choice(best_seasons),
        'How to Reach': random.choice(transport_modes),
        'Description': description
    }

# Input and output files
input_file = 'places_of_interest.csv'
output_file = 'places_enhanced.csv'

# Read and process
with open(input_file, 'r') as csvfile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(csvfile)
    writer = csv.writer(outfile)
    
    # Write header
    writer.writerow(['Destination', 'Place of Interest', 'Distance (km)', 'Best Time to Visit', 'How to Reach', 'Description'])

    for row in reader:
        if not row or row[0].strip() == "Destination":
            continue  # skip empty or header
        dest, place = row
        info = generate_place_info(place)
        writer.writerow([dest, place, info['Distance (km)'], info['Best Time to Visit'], info['How to Reach'], info['Description']])

print("✅ Enhanced CSV with place info generated as 'places_enhanced.csv'.")
