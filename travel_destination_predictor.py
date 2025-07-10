import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="✈️ Travel Destination Recommender", layout="centered")
st.title("✈️ Travel Destination Recommender")
st.markdown("Find the perfect travel destination based on your preferences!")

# --- Load Data ---
@st.cache_data
def load_travel_data():
    return pd.read_csv("travel_data.csv")

@st.cache_data
def load_hotel_data():
    return pd.read_csv("hotels.csv")

@st.cache_data
def load_places_data():
    df1 = pd.read_csv("places_of_interest.csv")
    df2 = pd.read_csv("places_enhanced.csv")
    return pd.concat([df1, df2], ignore_index=True)

places_df = load_places_data()
travel_df = load_travel_data()
hotels_df = load_hotel_data()

# Normalize for matching
hotels_df['DESTINATION'] = hotels_df['DESTINATION'].str.strip().str.lower()
places_df['DESTINATION'] = places_df['DESTINATION'].str.strip().str.lower()

# Amount binning
bins = [0, 1000, 5000, 10000, 20000, 60000]
labels = ['Budget', 'Economy', 'Standard', 'Premium', 'Luxury']
travel_df['AMOUNT_RANGE'] = pd.cut(travel_df['AMOUNT_USD'], bins=bins, labels=labels)

# Train model
def train_model(df):
    label_encoders = {}
    df_encoded = df.copy()

    for column in ['CLIMATE', 'SEASON', 'TRAVEL_TYPE', 'GROUP_TYPE', 'DURATION', 'AMOUNT_RANGE']:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    target_le = LabelEncoder()
    df_encoded['DESTINATION'] = target_le.fit_transform(df['DESTINATION'])

    X = df_encoded.drop(['DESTINATION', 'AMOUNT_USD'], axis=1)
    y = df_encoded['DESTINATION']

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    return model, label_encoders, target_le

model, encoders, dest_encoder = train_model(travel_df)

# User Input
with st.form("user_input_form"):
    climate = st.selectbox("🌤️ Preferred Climate", travel_df["CLIMATE"].unique())
    season = st.selectbox("🍂 Preferred Season", travel_df["SEASON"].unique())
    travel_type = st.selectbox("🛃 Type of Travel", travel_df["TRAVEL_TYPE"].unique())
    group_type = st.selectbox("👨‍👩‍👧‍👦 Group Type", travel_df["GROUP_TYPE"].unique())
    duration = st.selectbox("🗓️ Trip Duration", travel_df["DURATION"].unique())
    amount = st.number_input("💰 Budget (USD)", min_value=300, max_value=60000, step=100)
    submitted = st.form_submit_button("🔍 Find Destination")

# On Submit
if submitted:
    try:
        with st.spinner("🔎 Finding the perfect destination..."):
            amount_range = pd.cut([amount], bins=bins, labels=labels)[0]
            if pd.isna(amount_range):
                st.warning("⚠️ Budget is out of range. Try adjusting your budget.")
                st.stop()

            # Encode Inputs
            input_data = {
                'CLIMATE': [climate],
                'SEASON': [season],
                'TRAVEL_TYPE': [travel_type],
                'GROUP_TYPE': [group_type],
                'DURATION': [duration],
                'AMOUNT_RANGE': [amount_range]
            }

            encoded_input = {}
            for feature, value in input_data.items():
                encoder = encoders[feature]
                val = value[0]
                if val in encoder.classes_:
                    encoded_input[feature] = [encoder.transform([val])[0]]
                else:
                    fallback = pd.Series(encoder.classes_).mode()[0]
                    encoded_input[feature] = [encoder.transform([fallback])[0]]
                    st.warning(f"⚠️ Unrecognized `{feature}`: `{val}`. Using `{fallback}` instead.")

            input_df = pd.DataFrame(encoded_input)
            prediction = model.predict(input_df)
            destination = dest_encoder.inverse_transform(prediction)[0].strip()
            if not destination:
                st.warning("⚠️ No destination found. Try different inputs.")
                st.stop()
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
    else:
        # --- Output Section ---
        st.success(f"🌍 **Recommended Destination:** `{destination}`")
        destination_key = destination.lower()

        # --- Hotels ---
        st.subheader(f"🏨 Hotels in {destination}")
        hotels_in_city = hotels_df[hotels_df["DESTINATION"] == destination_key]

        if hotels_in_city.empty:
            st.warning("🚫 No hotels found.")
        else:
            hotels_in_city = hotels_in_city.sort_values(by="STARS", ascending=False)
            cols = st.columns(2)
            for idx, (_, hotel) in enumerate(hotels_in_city.iterrows()):
                with cols[idx % 2].expander(f"🏨 {hotel['HOTEL_TYPE']} — {hotel['ROOM_TYPE']}"):
                    st.write(f"⭐ **Stars:** {hotel['STARS']}")
                    st.write(f"💸 **Price/Night:** ${hotel['PRICE_PER_NIGHT']}")

        # --- Map View ---
        if "LAT" in hotels_in_city.columns and "LON" in hotels_in_city.columns and not hotels_in_city[['LAT', 'LON']].dropna().empty:
            st.subheader("🗺️ Hotel Locations")
            st.map(hotels_in_city.rename(columns={"LAT": "lat", "LON": "lon"}))

        # --- Places ---
        st.subheader(f"📍 Places to Visit in {destination}")
        places = places_df[places_df["DESTINATION"] == destination_key]
        places = places.dropna(subset=["Place of Interest", "Distance (km)", "Best Time to Visit", "How to Reach", "Description"])
        places = places.drop_duplicates(subset=["Place of Interest"], keep="first")
        places["Distance (km)"] = pd.to_numeric(places["Distance (km)"], errors="coerce")
        places = places.sort_values(by="Distance (km)")

        if places.empty:
            st.warning("🚫 No places of interest found.")
        else:
            for _, row in places.iterrows():
                with st.expander(f"📍 {row['Place of Interest']}"):
                    st.write(f"📏 **Distance:** {row['Distance (km)']} km")
                    st.write(f"🕒 **Best Time:** {row['Best Time to Visit']}")
                    st.write(f"🚌 **How to Reach:** {row['How to Reach']}")
                    st.write(f"📝 {row['Description']}")

        # --- Itinerary ---
        st.subheader("🗓️ Suggested Itinerary")
        try:
            days = int(duration.split()[0])
        except:
            days = 3
        for day in range(1, days + 1):
            with st.expander(f"📅 Day {day} Plan"):
                st.markdown("- 🌄 Morning: Explore major attraction")
                st.markdown("- 🍽️ Afternoon: Try local restaurants & visit another spot")
                st.markdown("- 🌆 Evening: Leisure walk or cultural show")

        # --- Budget ---
        st.subheader("💸 Estimated Budget")
        hotel_cost = hotels_in_city['PRICE_PER_NIGHT'].mean() * days if not hotels_in_city.empty else 0
        misc_cost = 50 * days
        total_cost = hotel_cost + misc_cost
        st.info(f"**Hotel:** ${hotel_cost:.2f} | **Other:** ${misc_cost:.2f} | **Total:** ${total_cost:.2f}")

        # --- Travel Tips ---
        st.subheader("🧳 Travel Tips")
        if season.lower() == "winter":
            st.markdown("- 🧥 Pack warm clothing & moisturizers")
        elif season.lower() == "summer":
            st.markdown("- ☀️ Bring sunscreen, hats & water")
        else:
            st.markdown("- 🌍 Pack according to local weather")

        # --- Booking Links ---
        st.subheader("🔗 Quick Links")
        st.markdown("- ✈️ [Book Flights](https://www.google.com/travel/flights)")
        st.markdown("- 🏨 [Find Hotels](https://www.booking.com/)")
