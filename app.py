'''#!/usr/bin/env python3
from flask import Flask, request, jsonify
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import logging

# Initialize Flask app.
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------
# Endpoint: /calculate_preferences
# ------------------------------
@app.route('/calculate_preferences', methods=['POST'])
def calculate_preferences_endpoint():
    try:
        # Parse JSON input.
        data = request.get_json(force=True)
    except Exception as e:
        app.logger.error(f"Error parsing JSON input: {e}")
        return jsonify({"error": "Invalid JSON input", "details": str(e)}), 400

    liked_descriptions = data.get("likedDescriptions", [])
    disliked_descriptions = data.get("dislikedDescriptions", [])

    # Function to load a SentenceTransformer model.
    def get_model(model_name="all-MiniLM-L6-v2"):
        try:
            app.logger.info("Loading SentenceTransformer model for calculate_preferences...")
            model = SentenceTransformer(model_name)
            return model
        except Exception as e:
            app.logger.error(f"Error loading model in calculate_preferences: {e}")
            raise

    # Function to cluster descriptions.
    def cluster_descriptions(descriptions, num_clusters=None, model_name="all-MiniLM-L6-v2"):
        if not descriptions:
            return []
        model = get_model(model_name)
        app.logger.info(f"Clustering {len(descriptions)} descriptions")
        embeddings = model.encode(descriptions)
        if num_clusters is None:
            num_clusters = min(3, len(embeddings))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(embeddings)
        return kmeans.cluster_centers_.tolist()

    try:
        liked_centers = cluster_descriptions(liked_descriptions) if liked_descriptions else []
        disliked_centers = cluster_descriptions(disliked_descriptions) if disliked_descriptions else []
    except Exception as e:
        app.logger.error(f"Error clustering descriptions: {e}")
        return jsonify({"error": "Error clustering descriptions", "details": str(e)}), 500

    output = {"likedClusters": liked_centers, "dislikedClusters": disliked_centers}
    app.logger.info(f"Calculated clusters: {output}")
    return jsonify(output), 200

# ------------------------------
# Endpoint: /recommend
# ------------------------------
@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        app.logger.error(f"Error parsing JSON input in recommend: {e}")
        return jsonify({"error": "Invalid JSON input", "details": str(e)}), 400

    liked_clusters = data.get("likedClusters", [])
    disliked_clusters = data.get("dislikedClusters", [])
    posts = data.get("posts", [])

    if not liked_clusters or not posts:
        error_msg = "Missing required data: likedClusters and posts are required."
        app.logger.error(error_msg)
        return jsonify({"error": error_msg}), 400

    try:
        user_liked_centers = np.array(liked_clusters)
        user_disliked_centers = np.array(disliked_clusters) if disliked_clusters else None
    except Exception as e:
        app.logger.error(f"Error converting clusters to numpy arrays: {e}")
        return jsonify({"error": "Error processing clusters", "details": str(e)}), 500

    app.logger.info("Loading SentenceTransformer model for recommendation...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        app.logger.error(f"Error loading model in recommend: {e}")
        return jsonify({"error": "Error loading model", "details": str(e)}), 500
    app.logger.info("Model loaded successfully for recommendation.")

    # Function to build a DataFrame from posts.
    def get_products_dataframe(posts):
        data_list = []
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
                            app.logger.error(f"Error converting embedding for post {post_id}: {e}")
                            embedding = model.encode(description).tolist()
                data_list.append({
                    "id": post_id,
                    "description": description,
                    "embedding": np.array(embedding)
                })
            except Exception as e:
                app.logger.error(f"Error processing post: {e}")
                continue
        try:
            return pd.DataFrame(data_list)
        except Exception as e:
            app.logger.error(f"Error creating DataFrame: {e}")
            raise

    try:
        products_df = get_products_dataframe(posts)
    except Exception as e:
        app.logger.error(f"Error building products DataFrame: {e}")
        return jsonify({"error": "Error building products DataFrame", "details": str(e)}), 500

    def recommend_products(user_liked_centers, user_disliked_centers, products_df, top_n=30, dislike_weight=1.0):
        scores = []
        for idx, row in products_df.iterrows():
            try:
                prod_emb = row["embedding"].reshape(1, -1)
                liked_sim = cosine_similarity(prod_emb, user_liked_centers)
                liked_score = liked_sim.max()
                disliked_score = 0
                if user_disliked_centers is not None and len(user_disliked_centers) > 0:
                    disliked_sim = cosine_similarity(prod_emb, user_disliked_centers)
                    disliked_score = disliked_sim.max()
                final_score = liked_score - dislike_weight * disliked_score
                scores.append(final_score)
            except Exception as e:
                app.logger.error(f"Error scoring product at index {idx}: {e}")
                scores.append(-9999)
        products_df["final_score"] = scores
        recommended = products_df.sort_values("final_score", ascending=False)
        return recommended.head(top_n)

    try:
        recommendations = recommend_products(user_liked_centers, user_disliked_centers, products_df)
    except Exception as e:
        app.logger.error(f"Error computing recommendations: {e}")
        return jsonify({"error": "Error computing recommendations", "details": str(e)}), 500

    try:
        rec_list = recommendations[['id', 'description', 'final_score']].to_dict(orient='records')
    except Exception as e:
        app.logger.error(f"Error preparing output: {e}")
        return jsonify({"error": "Error preparing output", "details": str(e)}), 500

    app.logger.info(f"Returning {len(rec_list)} recommended posts.")
    return jsonify(rec_list), 200

if __name__ == '__main__':
    # Run the app on the port specified by Render (default to 5000 if not provided).
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
'''
#!/usr/bin/env python3
from flask import Flask, request, jsonify
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import logging

# Initialize Flask app and logging.
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
app.logger.info("Starting recommendation service...")

@app.route('/')
def test():
    return 'Backend server is running'

# ------------------------------
# Helper: Build DataFrame from posts.
# ------------------------------
def get_products_dataframe(posts, model):
    app.logger.info(f"Building products DataFrame from {len(posts)} posts.")
    data_list = []
    for post in posts:
        try:
            post_id = post.get('_id') or post.get('id')
            description = post.get('description', '')
            app.logger.info(f"Processing post ID: {post_id} with description: {description[:60]}...")
            embedding = post.get('embedding', None)
            if embedding is None:
                app.logger.info(f"No embedding found for post {post_id}; computing embedding.")
                embedding = model.encode(description).tolist()
                app.logger.info(f"Computed embedding for post {post_id}; length: {len(embedding)}")
            else:
                if isinstance(embedding, str):
                    try:
                        embedding = eval(embedding)
                        app.logger.info(f"Converted embedding string to list for post {post_id}.")
                    except Exception as e:
                        app.logger.error(f"Error converting embedding for post {post_id}: {e}")
                        embedding = model.encode(description).tolist()
                        app.logger.info(f"Recomputed embedding for post {post_id}.")
            data_list.append({
                "id": post_id,
                "description": description,
                "embedding": np.array(embedding)
            })
        except Exception as e:
            app.logger.error(f"Error processing post {post}: {e}")
            continue
    try:
        df = pd.DataFrame(data_list)
        app.logger.info(f"Created DataFrame with shape: {df.shape}")
        return df
    except Exception as e:
        app.logger.error(f"Error creating DataFrame: {e}")
        raise

# ------------------------------
# Helper: Recommend products.
# ------------------------------
def recommend_products(user_liked_centers, user_disliked_centers, products_df, top_n=30, dislike_weight=1.0):
    app.logger.info("Computing recommendation scores for products...")
    scores = []
    for idx, row in products_df.iterrows():
        try:
            prod_emb = row["embedding"].reshape(1, -1)
            liked_sim = cosine_similarity(prod_emb, user_liked_centers)
            liked_score = liked_sim.max()
            disliked_score = 0
            if user_disliked_centers is not None and len(user_disliked_centers) > 0:
                disliked_sim = cosine_similarity(prod_emb, user_disliked_centers)
                disliked_score = disliked_sim.max()
            final_score = liked_score - dislike_weight * disliked_score
            scores.append(final_score)
            app.logger.info(f"Post {row['id']}: liked_score={liked_score:.4f}, disliked_score={disliked_score:.4f}, final_score={final_score:.4f}")
        except Exception as e:
            app.logger.error(f"Error scoring product at index {idx}: {e}")
            scores.append(-9999)
    products_df["final_score"] = scores
    recommended = products_df.sort_values("final_score", ascending=False)
    app.logger.info(f"Top {top_n} recommended posts selected.")
    return recommended.head(top_n)

# ------------------------------
# Endpoint: /calculate_preferences
# ------------------------------
@app.route('/calculate_preferences', methods=['POST'])
def calculate_preferences_endpoint():
    app.logger.info("Received /calculate_preferences request.")
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Input data: {data}")
    except Exception as e:
        app.logger.error(f"Error parsing JSON input: {e}")
        return jsonify({"error": "Invalid JSON input", "details": str(e)}), 400

    liked_descriptions = data.get("likedDescriptions", [])
    disliked_descriptions = data.get("dislikedDescriptions", [])
    app.logger.info(f"Number of liked descriptions: {len(liked_descriptions)}")
    app.logger.info(f"Number of disliked descriptions: {len(disliked_descriptions)}")

    def get_model(model_name="all-MiniLM-L6-v2"):
        try:
            app.logger.info("Loading SentenceTransformer model for calculate_preferences...")
            model = SentenceTransformer(model_name)
            app.logger.info("Model loaded successfully for calculate_preferences.")
            return model
        except Exception as e:
            app.logger.error(f"Error loading model: {e}")
            raise

    def cluster_descriptions(descriptions, num_clusters=None, model_name="all-MiniLM-L6-v2"):
        if not descriptions:
            app.logger.info("No descriptions provided for clustering.")
            return []
        model = get_model(model_name)
        app.logger.info(f"Clustering {len(descriptions)} descriptions...")
        try:
            embeddings = model.encode(descriptions)
            app.logger.info(f"Generated embeddings with shape: {np.array(embeddings).shape}")
        except Exception as e:
            app.logger.error(f"Error encoding descriptions: {e}")
            raise
        if num_clusters is None:
            num_clusters = min(3, len(embeddings))
            app.logger.info(f"Number of clusters set to: {num_clusters}")
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(embeddings)
            centers = kmeans.cluster_centers_.tolist()
            app.logger.info(f"Cluster centers computed: {centers}")
        except Exception as e:
            app.logger.error(f"Error during clustering: {e}")
            raise
        return centers

    try:
        liked_centers = cluster_descriptions(liked_descriptions) if liked_descriptions else []
        disliked_centers = cluster_descriptions(disliked_descriptions) if disliked_descriptions else []
    except Exception as e:
        app.logger.error(f"Error clustering descriptions: {e}")
        return jsonify({"error": "Error clustering descriptions", "details": str(e)}), 500

    output = {"likedClusters": liked_centers, "dislikedClusters": disliked_centers}
    app.logger.info(f"Calculated clusters output: {output}")
    return jsonify(output), 200

# ------------------------------
# Endpoint: /recommend
# ------------------------------
@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    app.logger.info("Received /recommend request.")
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Input recommendation data: {data}")
    except Exception as e:
        app.logger.error(f"Error parsing JSON input in /recommend: {e}")
        return jsonify({"error": "Invalid JSON input", "details": str(e)}), 400

    liked_clusters = data.get("likedClusters", [])
    disliked_clusters = data.get("dislikedClusters", [])
    posts = data.get("posts", [])

    app.logger.info(f"Liked clusters length: {len(liked_clusters)}")
    app.logger.info(f"Disliked clusters length: {len(disliked_clusters)}")
    app.logger.info(f"Number of posts received for recommendation: {len(posts)}")

    if not liked_clusters or not posts:
        error_msg = "Missing required data: likedClusters and posts are required."
        app.logger.error(error_msg)
        return jsonify({"error": error_msg}), 400

    try:
        user_liked_centers = np.array(liked_clusters)
        user_disliked_centers = np.array(disliked_clusters) if disliked_clusters else None
        app.logger.info(f"Converted likedClusters to numpy array with shape: {user_liked_centers.shape}")
        if user_disliked_centers is not None:
            app.logger.info(f"Converted dislikedClusters to numpy array with shape: {user_disliked_centers.shape}")
    except Exception as e:
        app.logger.error(f"Error converting clusters to numpy arrays: {e}")
        return jsonify({"error": "Error processing clusters", "details": str(e)}), 500

    app.logger.info("Loading SentenceTransformer model for recommendation...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        app.logger.info("Model loaded successfully for recommendation.")
    except Exception as e:
        app.logger.error(f"Error loading model in recommendation: {e}")
        return jsonify({"error": "Error loading model", "details": str(e)}), 500

    try:
        products_df = get_products_dataframe(posts, model)
        app.logger.info(f"Products DataFrame created with shape: {products_df.shape}")
    except Exception as e:
        app.logger.error(f"Error building products DataFrame: {e}")
        return jsonify({"error": "Error building products DataFrame", "details": str(e)}), 500

    try:
        recommendations = recommend_products(user_liked_centers, user_disliked_centers, products_df)
        app.logger.info(f"Computed recommendations for {len(recommendations)} products.")
    except Exception as e:
        app.logger.error(f"Error computing recommendations: {e}")
        return jsonify({"error": "Error computing recommendations", "details": str(e)}), 500

    try:
        rec_list = recommendations[['id', 'description', 'final_score']].to_dict(orient='records')
        app.logger.info(f"Final recommended posts: {rec_list}")
    except Exception as e:
        app.logger.error(f"Error preparing output: {e}")
        return jsonify({"error": "Error preparing output", "details": str(e)}), 500

    return jsonify(rec_list), 200

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)
