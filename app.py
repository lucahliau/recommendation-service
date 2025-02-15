from flask import Flask, request, jsonify
import sys
import json
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Global model variable to persist the model in memory.
_model = None

def get_model(model_name="all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        sys.stderr.write("Loading SentenceTransformer model...\n")
        sys.stderr.flush()
        _model = SentenceTransformer(model_name)
    return _model

def cluster_descriptions(descriptions, num_clusters=None, model_name="all-MiniLM-L6-v2"):
    if not descriptions:
        return []
    model = get_model(model_name)
    sys.stderr.write(f"Clustering {len(descriptions)} descriptions\n")
    sys.stderr.flush()
    try:
        embeddings = model.encode(descriptions)
    except Exception as e:
        sys.stderr.write(f"Error encoding descriptions: {e}\n")
        sys.stderr.flush()
        raise
    if num_clusters is None:
        num_clusters = min(3, len(embeddings))
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(embeddings)
        centers = kmeans.cluster_centers_.tolist()
    except Exception as e:
        sys.stderr.write(f"Error during clustering: {e}\n")
        sys.stderr.flush()
        raise
    return centers

def calculate_preferences(liked_descriptions, disliked_descriptions):
    liked_centers = cluster_descriptions(liked_descriptions) if liked_descriptions else []
    disliked_centers = cluster_descriptions(disliked_descriptions) if disliked_descriptions else []
    return {"likedClusters": liked_centers, "dislikedClusters": disliked_centers}

@app.route("/calculate_preferences", methods=["POST"])
def calc_preferences():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON input received"}), 400

        liked_descriptions = data.get("likedDescriptions", [])
        disliked_descriptions = data.get("dislikedDescriptions", [])
        
        clusters = calculate_preferences(liked_descriptions, disliked_descriptions)
        sys.stderr.write(f"Calculated clusters: {clusters}\n")
        sys.stderr.flush()
        return jsonify(clusters)
    except Exception as e:
        sys.stderr.write(f"Error in /calculate_preferences: {e}\n")
        sys.stderr.flush()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the app on a specified port (for example, 8000)
    app.run(host="0.0.0.0", port=8000)
