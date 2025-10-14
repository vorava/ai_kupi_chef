import os
import openai
import kupiapi.scraper
import kupiapi.recipes
import os
import json
import numpy as np
from typing import List, Dict

"""
    OPENAI_API_KEY should be enviroment variable
    Can be set with export OPENAI_API_KEY=sk-...

    This script downloads data about recipes from kupi.cz, saves them and creates embeddings for the recipes
    Outputs of this script are further used in AI kupi chef app
"""

openai.api_key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI()

sc = kupiapi.scraper.KupiScraper()
rc = kupiapi.recipes.KupiRecipes()

categories = rc.get_categories()

def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Call OpenAI embeddings API for a list of strings and return numpy array (n, dim).
    Batches if necessary.
    """
    
    embeddings = []
    batch_size = 1
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        
        for item in resp.data:
            embeddings.append(np.array(item.embedding, dtype=np.float32))
    return np.vstack(embeddings)

def normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return embs / norms


def cosine_sim(query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
    
    return (corpus_embs @ query_emb.reshape(-1)).flatten()

def replace_nbsp(obj):
    """Recursively replace non-breaking spaces in all strings."""
    if isinstance(obj, str):
        return obj.replace(u'\xa0', u' ').replace(u'\n                ', u' ').replace(u'\n            ', u' ')
    elif isinstance(obj, list):
        return [replace_nbsp(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: replace_nbsp(v) for k, v in obj.items()}
    return obj


def load_json_docs(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    docs = [replace_nbsp(d) for d in docs]
                
    normalized = []
    for i, d in enumerate(docs):
        name = d.get("name")
        recipe = d.get("recipe") 
        
        for part in recipe.get("ingredients"):
            for ing in part.get("ingredients", []):
                ing.pop("ingredient_url", None)
        
        normalized.append({"name": name, "recipe": recipe})
    return normalized

def dict2str(dictionary: Dict) -> str:
    return json.dumps(dictionary, ensure_ascii=False)


for cat in categories[2:]:
    print("Downloading recipes category: ", cat)
    recipes = rc.get_recipes_by_category(cat, full=True)
    with open(f'recipes_texts/{cat}_text.json', 'w') as f:
        f.write(recipes)
        
    print("Successfully written to file.")
    
    input_json = load_json_docs(f'recipes_texts/{cat}_text.json')
    input_json = [dict2str(d) for d in input_json]
    
    print("Creating embeddings.")

    embeddings = get_embeddings(input_json)
    embeddings = normalize_embeddings(embeddings)
    
    np.save(f"recipes_embeddings/{cat}_embeddings.npy", embeddings)
    
    print("Successfull write of embeddings to file.")


