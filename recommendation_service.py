#!/usr/bin/env python3
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

# Set up logging so that errors and debug info appear in the console.
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Initialize the model once at startup.
logging.info("Loading SentenceTransformer model...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    sys.exit(1)

def get_products_dataframe(posts):
    """
    Convert the list of posts (from JSON) into a DataFrame.
    If a post does not have an 'embedding', compute it using the model.
    """
    data = []
    for post in posts:
        try:
            post_id = post.get('_id') or post.get('id')
            description = post.get('description', '')
            embedding = post.get('embedding', None)
            if embedding is None:
                embedding = model.encode(description).tolist()
            else:
                if isinstance(embedding, str):
                    try:
                        embedding = eval(embedding)
                    except Exception as e:
                        logging.error(f"Error converting embedding for post {post_id}: {e}")
                        embedding = model.encode(description).tolist()
            data.append({
                "id": post_id,
                "description": description,
                "embedding": np.array(embedding)
            })
        except Exception as e:
            logging.error(f"Error processing post {post}: {e}")
            continue
    try:
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logging.error(f"Error creating DataFrame: {e}")
        raise

def recommend_products(user_liked_centers, user_disliked_centers, products_df, top_n=30, dislike_weight=1.0):
    """
    Compute a recommendation score for each product and return the top_n recommendations.
    """
    product_scores = []
    for idx, row in products_df.iterrows():
        try:
            product_embedding = row["embedding"].reshape(1, -1)
            liked_similarities = cosine_similarity(product_embedding, user_liked_centers)
            liked_score = liked_similarities.max()
            disliked_score = 0
            if user_disliked_centers is not None and len(user_disliked_centers) > 0:
                disliked_similarities = cosine_similarity(product_embedding, user_disliked_centers)
                disliked_score = disliked_similarities.max()
            final_score = liked_score - dislike_weight * disliked_score
            product_scores.append(final_score)
        except Exception as e:
            logging.error(f"Error scoring product at index {idx}: {e}")
            product_scores.append(-9999)
    products_df["final_score"] = product_scores
    recommended = products_df.sort_values("final_score", ascending=False)
    return recommended.head(top_n)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Log the request.
    logging.info("Received recommendation request.")
    
    # Get JSON data from the request.
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logging.error(f"Error parsing JSON input: {e}")
        return jsonify({"error": "Invalid JSON input", "details": str(e)}), 400

    liked_clusters = data.get("likedClusters", [])
    disliked_clusters = data.get("dislikedClusters", [])
    posts = data.get("posts", [])

    if not liked_clusters or not posts:
        error_msg = "Missing required data: likedClusters and posts are required."
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 400

    try:
        user_liked_centers = np.array(liked_clusters)
        user_disliked_centers = np.array(disliked_clusters) if disliked_clusters else None
    except Exception as e:
        logging.error(f"Error converting clusters to numpy arrays: {e}")
        return jsonify({"error": "Error processing clusters", "details": str(e)}), 500

    try:
        products_df = get_products_dataframe(posts)
    except Exception as e:
        logging.error(f"Error building products DataFrame: {e}")
        return jsonify({"error": "Error building products DataFrame", "details": str(e)}), 500

    try:
        recommendations = recommend_products(user_liked_centers, user_disliked_centers, products_df)
    except Exception as e:
        logging.error(f"Error computing recommendations: {e}")
        return jsonify({"error": "Error computing recommendations", "details": str(e)}), 500

    try:
        recommendations_list = recommendations[['id', 'description', 'final_score']].to_dict(orient='records')
    except Exception as e:
        logging.error(f"Error preparing output: {e}")
        return jsonify({"error": "Error preparing output", "details": str(e)}), 500

    logging.info(f"Returning {len(recommendations_list)} recommended posts.")
    return jsonify(recommendations_list), 200

if __name__ == '__main__':
    # Run Flask app. For production, you would typically use gunicorn.
    app.run(host='0.0.0.0', port=5000, debug=True)
