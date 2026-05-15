# I started by importing all the required libraries for this app.
# Flask is for building the web app, pandas and numpy help with data processing,
# pickle is used for loading saved models, and FastText gives word embeddings for the review text.
import csv
import os
from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import pickle
from gensim.models import FastText

# I initialized the Flask app here so it can handle incoming web requests.
app = Flask(__name__)

# I set the relative paths for the main dataset and the reviews file.
# Keeping them relative helps avoid issues across different machines.
CSV_PATH = "assignment3_II.csv"
REVIEWS_CSV_PATH = "data/reviews.csv"

# Loading the dataset
df = pd.read_csv(CSV_PATH).fillna("")

# I loaded the FastText model I had trained earlier — this will help turn review text into vectors.
ft_model = FastText.load("fasttext_model.bin")
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open("logistic_regression_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

# This is a helper function I defined to convert a review into a vector representation.
# It uses the FastText model and TF-IDF scores to create weighted word vectors,
# then averages them to produce a single feature vector for the whole review.
# If none of the words are in the FastText model, it returns a zero vector.
def vectorize_text(text, model, vectorizer):
    words = text.split()
    tfidf_weights = vectorizer.transform([text]).toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    weighted_vectors = []
    for word in words:
        if word in feature_names:
            try:
                weighted_vectors.append(model.wv[word] * tfidf_weights[feature_names.tolist().index(word)])
            except KeyError:
                pass
    return np.mean(weighted_vectors, axis=0) if weighted_vectors else np.zeros(model.vector_size)

# Homepage route – shows all clothing items as a list
@app.route('/')
def index():
    items = df.to_dict(orient='records')# Converting dataframe to list of dictionaries for the template
    return render_template("index.html", items=items)# Render homepage with all items

# Category-specific route – filters clothing based on category like tops, dresses, etc.
@app.route("/category/<category>")
def category_page(category):
    show_all = request.args.get('show') == 'all' # Checking if 'show=all' is in the URL to show full list
    
    # Defining which class names belong to which category
    category_map = {
        "tops": ['Blouses', 'Knits', 'Sweaters'],
        "bottoms": ['Pants', 'Jeans', 'Shorts', 'Skirts', 'Casual bottoms'],
        "dresses": ['Dresses'],
        "jackets": ['Jackets', 'Outerwear'],
        "intimates": ['Intimates', 'Sleep', 'Chemises'],
        "trend": ['Trend']
    }

    # If category is valid, filter items based on the mapped class names
    if category.lower() in category_map:
        valid_classes = category_map[category.lower()]
        filtered_items = df[df['Class Name'].isin(valid_classes)]

        # Keeping only one unique item per Clothing ID to avoid duplicates
        filtered_items = filtered_items.drop_duplicates(subset="Clothing ID", keep="first")
        
        # Showing only first 20 items by default unless 'show=all' is set

        if not show_all:
            filtered_items = filtered_items.head(20)
         # Converting to dictionary list for rendering
        filtered_items = filtered_items.to_dict(orient='records')
    else:
        filtered_items = [] # If invalid category, return empty list
        
    # Render category page with filtered items and category info
    return render_template("index.html", items=filtered_items, category=category.capitalize(), show_all=show_all)


# Route to display details and reviews for a specific clothing item
@app.route('/item/<int:item_id>')
def item(item_id):
    item = df[df['Clothing ID'] == item_id].iloc[0].to_dict() # Extracting the item’s details from the main dataset using the ID
    reviews = []

    # Including the original review from assignment3_II.csv if it exists
    if item['Review Text']:
        reviews.append({
            "Review Title": item.get("Title", "Original Review"),# Using item title or fallback
            "Review Text": item["Review Text"]
        })

    # Loading any additional reviews from the external reviews.csv file
    try:
        with open("data/reviews.csv", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Matching and collecting reviews for the current item
                if int(row["Clothing ID"]) == item_id:
                    reviews.append({
                        "Review Title": row["Review Title"],
                        "Review Text": row["Review Text"]
                    })
    except FileNotFoundError:
        pass # If reviews file is missing, skip it silently
    
    # Render the item page with all relevant reviews
    return render_template("item.html", item=item, reviews=reviews)

# Route to search for clothing items by ID, title, or description
@app.route("/search")
def search():
    query = request.args.get("query", "").lower()# Getting search query from URL
    matched_items = []
    
    # Going through each row and check if the query matches any key fields
    for _, row in df.iterrows():
        if query in str(row['Clothing ID']).lower() or \
           query in row['Title'].lower() or \
           query in row['Clothes Title'].lower() or \
           query in row['Clothes Description'].lower():
            matched_items.append(row.to_dict())
            
    # Render the index page with search results and total count
    return render_template("index.html", items=matched_items, query=query, count=len(matched_items))

# The route to handle both displaying the review form and submitting a review
@app.route("/review", methods=["GET", "POST"])
def review():
    global df  # We may update the main DataFrame, so declaring it global
    predicted_category_label = None # This will show whether the item is recommended or not
    success = False # Used to confirm if review submission was successful

    if request.method == "POST":
     # Collect form data submitted by the user
        review_title = request.form.get("review_title", "")
        clothing_id = request.form.get("clothing_id", "")
        clothing_name = request.form.get("clothing_name", "")
        clothing_desc = request.form.get("clothing_desc", "")
        clothing_cat = request.form.get("clothing_cat", "")
        review_text = request.form.get("review_text", "")
        is_new_id = request.form.get("new_id")  # Checkbox to identify if it's a new clothing item

        # Generating a text vector and make a prediction using the trained model
        vector = vectorize_text(review_text, ft_model, tfidf_vectorizer)
        prediction = lr_model.predict([vector])[0]
        predicted_category_label = "Recommended" if prediction == 1 else "Not Recommended"

        # Saving the submitted review to the reviews.csv file
        with open(REVIEWS_CSV_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # Writing header only if file is empty
            if file.tell() == 0:
                writer.writerow(["Review Title", "Clothing ID", "Clothing Name", "Description", "Category", "Review Text", "Label"])
            writer.writerow([review_title, clothing_id, clothing_name, clothing_desc, clothing_cat, review_text, prediction])

        # If this is a brand new clothing item, I am adding it to the main dataset and save
        if is_new_id == "on":
            new_row = {
                "Clothing ID": int(clothing_id),
                "Age": 0,
                "Title": review_title,
                "Review Text": review_text,
                "Rating": 0,
                "Recommended IND": prediction,
                "Positive Feedback Count": 0,
                "Division Name": "General",
                "Department Name": "New",
                "Class Name": clothing_cat,
                "Clothes Title": clothing_name,
                "Clothes Description": clothing_desc
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)# Append to existing DataFrame
            df.to_csv(CSV_PATH, index=False) # Saving updated dataset

        success = True # Here I set success flag to show confirmation message on the form
        
    # Render the review form and optionally display result
    return render_template("review.html", predicted_category=predicted_category_label, success=success)

# Main app runner
if __name__ == '__main__':
    app.run(debug=True) # Running the Flask app in debug mode
