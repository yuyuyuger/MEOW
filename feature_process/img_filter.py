import os
import shutil
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data(filepath):
    """ Load data from a pickle file """
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_txt_descriptions(filepath):
    """ Load text descriptions from a .txt file """
    descriptions = {}
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                descriptions[parts[0]] = parts[1]
    return descriptions

# Load data
entity_descriptions = load_txt_descriptions("../data/WN9/_data_/gloss.txt")
image_captions = load_data("../captions/wn9_images.pkl")

vectorizer = TfidfVectorizer()

output_dir = "../data/WN9/selected_images"

# Ensure existing output directory is removed before recreating it
if os.path.exists(output_dir):
    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError as e:
        print(f"Failed to delete directory {output_dir}: {e}")

# Create a new output directory
os.makedirs(output_dir, exist_ok=True)

for entity_id, description in entity_descriptions.items():
    image_folder = os.path.join("../data/WN9/ima_data/wn9_images", entity_id)

    if not os.path.exists(image_folder):
        print(f"Image folder not found for {entity_id} at {image_folder}. Skipping...")
        continue

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.JPEG')]

    all_texts = [description]
    for img_name in image_files:
        captions_key = entity_id + "+" + img_name
        if captions_key in image_captions:
            all_texts.extend(image_captions[captions_key])  # Assume captions are stored as a list of strings

    if len(all_texts) == 1:  # Only the description was added
        print(f"No valid captions found for {entity_id}. Skipping...")
        continue

    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    similarity_scores = similarity_matrix[0]

    num_captions_per_image = 20
    average_similarity_scores = []
    for i in range(0, len(similarity_scores), num_captions_per_image):
        average_score = np.mean(similarity_scores[i:i + num_captions_per_image])
        average_similarity_scores.append(average_score)

    top_five_indices = np.argsort(average_similarity_scores)[-3:][::-1]  # Select top three images

    entity_output_dir = os.path.join(output_dir, entity_id)
    os.makedirs(entity_output_dir, exist_ok=True)

    for index in top_five_indices:
        source_path = os.path.join(image_folder, image_files[index])
        target_path = os.path.join(entity_output_dir, image_files[index])
        shutil.copy(source_path, target_path)
        print(f"Copied {source_path} to {target_path}")

print("Image selection and copying completed.")

