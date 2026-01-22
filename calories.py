#Import Libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scipy.sparse import hstack

#Read File
df = pd.read_csv('Calories.csv')
print(" Data loaded successfully!")
print(df.head())


# Clean the 'Calories' column
df['Calories'] = df['Calories'].astype(str).str.replace('cal', '').str.strip().astype(int)

# Extract grams from 'Serving'
def extract_grams(serving):
    if pd.isna(serving):
        return None
    match = re.search(r'\((\d+)\s*g\)', str(serving))
    if match:
        return int(match.group(1))
    return None

df['Grams'] = df['Serving'].apply(extract_grams)

# Calculate calories per 100g
df['Cal_per_100g'] = df.apply(
    lambda row: (row['Calories'] / row['Grams']) * 100 if row['Grams'] else None,
    axis=1
)

# Drop rows with missing values
df = df.dropna(subset=['Cal_per_100g'])
print("\n✅ Cleaned data sample:")
print(df.head())

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df['Food'])

scaler = StandardScaler()
grams_scaled = scaler.fit_transform(df[['Grams']])


X_combined = hstack([X_text, grams_scaled])
X_dense = X_combined.toarray()
y = df['Cal_per_100g'].values

#Train
X_train, X_test, y_train, y_test = train_test_split(
    X_dense, y, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1)  # Regression output
])

#Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

loss, mae = model.evaluate(X_test, y_test)
print("\n📉 Mean Absolute Error:", round(mae, 2))

def predict_calories(food_name, grams):
    # Vectorize food name
    food_vector = vectorizer.transform([food_name])

    # Scale grams
    grams_scaled = scaler.transform(np.array([[grams]]))

    # Combine
    input_combined = hstack([food_vector, grams_scaled])
    input_dense = input_combined.toarray()

    # Predict
    prediction = model.predict(input_dense)
    return round(float(prediction[0]), 2)

#Sample Data
print("\n🍎 Apple (150g):", predict_calories("Apple", 150), "cal/100g")
print("🍞 Whole wheat bread (60g):", predict_calories("Whole wheat bread", 60), "cal/100g")
print("🍗 Fried chicken (200g):", predict_calories("Fried chicken", 200), "cal/100g")
print("🧁 Cupcake (120g):", predict_calories("Cupcake", 120), "cal/100g")

# This app is a Fitness & Diet Planner that allows users to log meals, track calories, 
# and get dietary suggestions. It includes a deep learning module to estimate calories 
# from user input for more accurate predictions.

import gradio as gr
import random


def predict_calories(food_name, grams):
    return f"Estimated calories in {food_name} ({grams}g): {round(grams * 1.24, 2)} kcal"

calorie_tips = [
    "🍎 Eat more whole foods.",
    "💧 Drink water before meals.",
    "🥦 High-fiber foods keep you full longer.",
    "🍫 Dark chocolate in moderation.",
    "🏃‍♂️ Exercise boosts calorie burn.",
    "🍽️ Smaller plates trick your brain.",
    "🥗 Include protein in every meal.",
    "🚶‍♀️ A 30-minute walk burns 100–150 calories."
]


def login_user(name, age, gender, weight, email):
    if not all([name, age, gender, weight, email]):
        return gr.update(visible=True), gr.update(visible=False), "⚠️ Fill all fields!"
    return gr.update(visible=False), gr.update(visible=True), f"👋 Hello {name}!"

def estimate(food, grams):
    if not food or not grams:
        return "⚠️ Enter both food name and grams!"
    return predict_calories(food, grams)

def random_tip():
    return random.choice(calorie_tips)

diet_log = {
    "Breakfast": "",
    "Lunch": "",
    "Snacks": "",
    "Dinner": ""
}

def log_diet(breakfast, lunch, snacks, dinner):
    diet_log["Breakfast"] = breakfast
    diet_log["Lunch"] = lunch
    diet_log["Snacks"] = snacks
    diet_log["Dinner"] = dinner
    return f"✅ Diet logged!\n\nBreakfast: {breakfast}\nLunch: {lunch}\nSnacks: {snacks}\nDinner: {dinner}"


with gr.Blocks() as app:
    gr.HTML("""
        <div style="text-align:center; padding:20px;
                    color:white; background-color:#4caf50;
                    border-radius:12px;">
            <h1>🍇 NutriTrack</h1>
            <p>Eat smart. Track better. Stay healthy.</p>
        </div>
    """)
    # ---------- LOGIN PAGE ----------
    with gr.Group(visible=True) as login_page:
        gr.Markdown("### 👤 Create Your Profile")
        name = gr.Textbox(label="Name")
        age = gr.Number(label="Age")
        gender = gr.Dropdown(["Male", "Female", "Other"], label="Gender")
        weight = gr.Number(label="Weight (kg)")
        email = gr.Textbox(label="Email")
        login_btn = gr.Button("🚀 Login / Register")
        login_msg = gr.Markdown("")

   
    with gr.Group(visible=False) as home_page:
        greeting = gr.Markdown("")
        tip_display = gr.Markdown(random.choice(calorie_tips), elem_id="tip-box")
        
        gr.Markdown("### 🍴 Choose an option:")
        cal_btn = gr.Button("🍽️ Calorie Estimator")
        tip_btn = gr.Button("💡 Random Tip")
        profile_btn = gr.Button("👤 Profile")
        diet_btn = gr.Button("📋 Diet Planner")
        logout_btn = gr.Button("🚪 Logout")

  
    with gr.Group(visible=False) as calorie_page:
        gr.Markdown("### 🥗 Calorie Estimator")
        food_inp = gr.Textbox(label="Food Name")
        grams_inp = gr.Number(label="Food Weight (g)")
        estimate_btn = gr.Button("Estimate")
        result_out = gr.Textbox(label="Result", interactive=False)
        back1 = gr.Button("⬅️ Back to Home")

    
    with gr.Group(visible=False) as profile_page:
        profile_info = gr.Markdown("")
        back2 = gr.Button("⬅️ Back to Home")

    # ---------- DIET PLANNER PAGE ----------
    with gr.Group(visible=False) as diet_page:
        gr.Markdown("### 🥘 Diet Planner")
        breakfast_inp = gr.Textbox(label="Breakfast")
        lunch_inp = gr.Textbox(label="Lunch")
        snacks_inp = gr.Textbox(label="Snacks")
        dinner_inp = gr.Textbox(label="Dinner")
        log_btn = gr.Button("Log My Diet")
        diet_display = gr.Markdown("")
        back3 = gr.Button("⬅️ Back to Home")

    
    login_btn.click(login_user, [name, age, gender, weight, email], [login_page, home_page, greeting])
    
  
    cal_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [home_page, calorie_page])
    tip_btn.click(random_tip, None, tip_display)
    profile_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True),
                               profile_info.update(f"Name: {name.value}\nAge: {age.value}\nGender: {gender.value}\nWeight: {weight.value}\nEmail: {email.value}")),
                      None, [home_page, profile_page])
    diet_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [home_page, diet_page])
    logout_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [home_page, login_page])

    
    estimate_btn.click(estimate, [food_inp, grams_inp], result_out)
    back1.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [calorie_page, home_page])

   
    back2.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [profile_page, home_page])

   
    log_btn.click(log_diet, [breakfast_inp, lunch_inp, snacks_inp, dinner_inp], diet_display)
    back3.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [diet_page, home_page])


app.launch()





