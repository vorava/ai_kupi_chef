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
"""
openai.api_key = os.getenv("OPENAI_API_KEY")
from openai import OpenAI


sc = kupiapi.scraper.KupiScraper()
rc = kupiapi.recipes.KupiRecipes()


class KupiBot:
    def __init__(self):
        self.ingredients = []
        self.discount_json = []
        self.client = OpenAI()
        self.MAX_TOKENS = 8000
        self.EMBED_MODEL = "text-embedding-3-small" # openAI model for embeddings
        self.CHAT_MODEL = "gpt-4.1" # openAI model for recipe recommendation
        self.TOP_K = 1
    
    def get_recipes_categories(self) -> List[str]:
        return rc.get_categories()
    
    def get_discounts(self, shop_name: str, max_pages: int) -> List[Dict]:
        discounts = sc.get_discounts_by_shop(shop_name, max_pages=max_pages)
        discounts_json = json.loads(discounts)

        for d in discounts_json:
            d.pop("shops", None)
            d.pop("validities", None)
            
        self.discount_json = discounts_json
            
        return discounts_json
    
    def __dict2str(self, dictionary: Dict) -> str:
        return json.dumps(dictionary, ensure_ascii=False)
    
    def __get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Creates embeddings for a list of recipes or ingredients."""
        embeddings = []
        batch_size = 1
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            #print(batch)
            resp = self.client.embeddings.create(model=self.EMBED_MODEL, input=batch)
            # resp.data is a list; each item has .embedding
            for item in resp.data:
                embeddings.append(np.array(item.embedding, dtype=np.float32))
        return np.vstack(embeddings)

        
    def __cosine_sim(self, query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
        return (corpus_embs @ query_emb.reshape(-1)).flatten()


    def __retrieve(self, ingredients: str, recipes: List[str], embeddings: np.ndarray, top_k: int = None):
        """
        Retrieve the top k recipes based on the cosine similarity of the query embeddings with the corpus embeddings.

        Parameters:
        ingredients (str): The ingredients in JSON format.
        recipes (List[str]): The list of recipes in JSON format.
        embeddings (np.ndarray): The embeddings of the corpus.
        top_k (int): The number of top retrieved recipes. If None, use self.top_K.

        Returns:
        List[Dict]: The top k retrieved recipes.
        """
        if top_k is None:
            top_k = self.TOP_K
        q_emb = self.__get_embeddings([ingredients])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        sims = self.__cosine_sim(q_emb, embeddings)
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            results.append({"score": float(sims[idx]), "text": recipes[idx]})
        return results
    
    def __load_embeddings(self, filename: str):
        with open(filename, "rb") as f:
            return np.load(f)
        
    def __load_recipes(self, filename: str):
        rec = self.__load_json_docs(filename)
        return [self.__dict2str(r) for r in rec]
        
    def __replace_nbsp(self, obj):
        """Recursively replace non-breaking spaces in all strings."""
        if isinstance(obj, str):
            return obj.replace(u'\xa0', u' ').replace(u'\n                ', u' ').replace(u'\n            ', u' ')
        elif isinstance(obj, list):
            return [self.__replace_nbsp(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self.__replace_nbsp(v) for k, v in obj.items()}
        return obj


    def __load_json_docs(self, path: str) -> List[Dict]:
        """
        Load JSON documents from a given path and replace non-breaking spaces in all strings.

        Parameters:
        path (str): Path to the JSON file.

        Returns:
        List[Dict]: List of normalized JSON documents.
        """
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        docs = [self.__replace_nbsp(d) for d in docs]
                    
        normalized = []
        for d in docs:
            name = d.get("name")
            recipe = d.get("recipe") 
            
            for part in recipe.get("ingredients"):
                for ing in part.get("ingredients", []):
                    ing.pop("ingredient_url", None)
            
            normalized.append({"name": name, "recipe": recipe})
        return normalized

        
    def __build_prompt(self, ingredients: str, retrieved: List[Dict]) -> List[Dict]:
        system = {
            "role": "system",
            "content": "You are a chef who knows how to prepare dishes. You will get a list of ingredients (in JSON format in Czech language with prices) in sale from specified shop and list of recipes also in JSON. Create or use existing recipe using provided list of goods and list of recipes, you can add ingredients not listed in the list, but add information about that. Use primarly ingredient from the list and show price. Use only Czech language in the output. Provide 2-3 recipes."
        }
        
        context_pieces = []
        for i, r in enumerate(retrieved, start=1):
            context_pieces.append(f"---\nRecipe {i}: {r['text']}\n")
        context_text = "\n\n".join(context_pieces)
        
        user_msg = {
            "role": "user",
            "content": f"List of ingredients that are in a sale in the grocery shop in JSON format: {ingredients}\n\n Recipes top match recipes in JSON format: {context_text}. Format output as HTML. Do not use markdown."
        }
        
        return [system, user_msg]
    
    def __answer_question(self, ingredients: str, recipes: List[str], embeddings: np.ndarray, top_k: int = None):
        """
        Answer the question given ingredients and recipes.

        Parameters:
        ingredients (str): Ingredients in a sale in JSON format
        recipes (List[str]): List of recipes in JSON format
        embeddings (np.ndarray): Embeddings of recipes
        top_k (int): Number of top retrieved recipes. If None, use self.top_K.

        Returns:
        str: The answer to the question in HTML format
        List[Dict]: Top retrieved recipes
        """
        if top_k is None:
            top_k = self.TOP_K
        retrieved = self.__retrieve(ingredients, recipes, embeddings, top_k=top_k)
        #print(f'Top retrieved (score, text): {[(r["score"], r["text"]) for r in retrieved]}')
        
        messages = self.__build_prompt(ingredients, retrieved)
        #print(messages)
        resp = self.client.chat.completions.create(
            model=self.CHAT_MODEL,
            messages=messages,
            max_tokens=self.MAX_TOKENS,
            temperature=0.5,
        )
        
        content = resp.choices[0].message
        return content, retrieved
        
    
    def get_chat_response(self, recipe_category: str, top_k: int = None) -> str:
        """
        Get the chat response based on the given recipe category and top k.

        Parameters:
        recipe_category (str): The category of recipes to retrieve.
        top_k (int): The number of recipes to retrieve.

        Returns:
        str: The chat response from the AI.
        """
        if top_k is None:
            top_k = self.TOP_K
        recipes_json = self.__load_recipes(f"recipes_texts/{recipe_category}_text.json")
        recipes_embeddings = self.__load_embeddings(f"recipes_embeddings/{recipe_category}_embeddings.npy")
        
        self.ingredients = [self.__dict2str(d) for d in self.discount_json]
        
        ans, retr = self.__answer_question(" ".join(self.ingredients), recipes_json, recipes_embeddings, top_k=top_k)
        return ans.content
    