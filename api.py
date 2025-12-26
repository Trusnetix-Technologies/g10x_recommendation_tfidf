import os
import pickle
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# directories
MODEL_DIR = 'model_assets'
DATASETS_DIR = 'dataset' 

app = FastAPI()

# --- Global Variables ---
cosine_sim = None  # Similarity Matrix (Numpy Array)
df_recommender = None # Lean DataFrame for index lookup
df_full_metadata = None # Full Metadata DataFrame
title_to_index = None # Title to Index mapping

# --- Pydantic Models ---
class RecommendationRequest(BaseModel):
    title: str
    total_recommendations: int = 10

# --- Function to Load Assets ---
def load_model_assets():
    """Loads all necessary files (matrices, vectorizer, DataFrames) into memory."""
    global cosine_sim, df_recommender, df_full_metadata, title_to_index

    print("--- API Starting: Loading Assets ---")
    try:
        # 1. Load the Similarity Matrix (Numpy)
        sim_path = os.path.join(MODEL_DIR, 'cosine_sim.npy')
        cosine_sim = np.load(sim_path)
        print(f"Loaded Cosine Similarity Matrix: {cosine_sim.shape}")

        # 2. Load the clean data (For Index Lookup)
        df_recommender = pd.read_csv(os.path.join(DATASETS_DIR, 'books_cleaned.csv'))
        print(f"Loaded clean data. Total records: {len(df_recommender)}")

        # 3. Load the Full Metadata (For detailed info)
        df_full_metadata = pd.read_csv(os.path.join(DATASETS_DIR, 'books.csv'))
        df_full_metadata = df_full_metadata.drop_duplicates(subset=['isbn13']) # Clean duplicates
        print(f"Loaded Full Metadata: {len(df_full_metadata)} records")

        # 4. Pre-calculate the Title-to-Index map
        title_to_index = pd.Series(df_recommender.index, 
                                   index=df_recommender['title']).drop_duplicates()
        print("Pre-calculated Title-to-Index map.")
        
        print("--- Assets Loaded Successfully ---")
        print("--- API is ready to serve requests ---")
        return True
    
    except Exception as e:
        print(f"!!! ERROR LOADING ASSETS: {e}")
        # Return False to indicate the API cannot serve requests
        return False

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    load_model_assets()

# --- Recommendation ---

def generate_recommendations(input_title, N=10):
    """Core logic to find recommendations using the loaded matrix."""
    
    if input_title not in title_to_index:
        return {"error": f"Book title '{input_title}' not found in dataset."}
    
    # 1. Get the index of the input book
    idx = title_to_index[input_title]

    # 2. Get similarity scores, sort, and select top N
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:N+1] 
    
    # 3. Extract indices and scores
    book_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    # 4. Get ISBNs from the df_recommender
    recommendations_lean = df_recommender.iloc[book_indices][['isbn13']].copy()
    recommendations_lean['Similarity Score'] = similarity_scores
    
    # 5. Merge with full metadata
    final_recs = pd.merge(
        recommendations_lean,
        df_full_metadata,
        on='isbn13',
        how='left'
    )

    # 6. Select and clean up columns for final output
    output_cols = ['title', 'subtitle', 'authors', 'categories', 'published_year', 'average_rating', 'Similarity Score', 'thumbnail']
    final_recs = final_recs[output_cols]
    
    # Handle NaN values for JSON serialization
    final_recs['subtitle'] = final_recs['subtitle'].fillna('')
    final_recs['authors'] = final_recs['authors'].fillna('')
    final_recs['categories'] = final_recs['categories'].fillna('')
    final_recs['thumbnail'] = final_recs['thumbnail'].fillna('')
    final_recs['published_year'] = final_recs['published_year'].fillna(0).astype(int)
    final_recs['average_rating'] = final_recs['average_rating'].apply(lambda x: round(x, 2) if pd.notna(x) else None)
    
    # Convert DataFrame to a list of dictionaries for JSON
    return final_recs.to_dict('records')


# --- API Endpoint ---

@app.post('/recommend')
async def recommend_endpoint(request: RecommendationRequest):
    """Handles the POST request from the Next.js frontend."""
    
    # Ensure assets are loaded
    if cosine_sim is None:
        raise HTTPException(status_code=500, detail="Model assets not loaded. Check server logs.")
    
    results = generate_recommendations(request.title, N=request.total_recommendations)

    # If the result is an error message (from the logic function)
    if isinstance(results, dict) and 'error' in results:
        raise HTTPException(status_code=404, detail=results['error'])
    
    # If successful, return the list of recommendations
    return results


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)

# command to run uvicorn
# uvicorn api:app --host 0.0.0.0 --port 5000 --reload