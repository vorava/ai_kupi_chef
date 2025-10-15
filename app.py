from flask import Flask, render_template, request, jsonify, session, redirect
from markupsafe import escape
from kupibot import KupiBot
import math
from typing import List

app = Flask(__name__)
app.jinja_env.globals.update(ceil=math.ceil)


@app.route("/", methods=["GET"])
def index():
    session["selected_shop_name"] = SHOPS_URL[0]
    session["selected_category"] = 'hlavni-jidla'
    session["selected_pages"] = 5
    session["selected_topk"] = 3
    
    return redirect(session["selected_shop_name"])

## loads discounts from specified shop
@app.route("/<shop_name>", methods=["GET"])
def select_shop(shop_name):
    if request.path in ("/styles.css", "/favicon.ico"):
        return "", 204  # Ignore

    session["selected_shop_name"] = escape(shop_name)
    data = load_shop_data(shop_name)
    
    return render_template("index.html", **data)


@app.route("/set_pages", methods=["POST"])
def set_pages():
    data = request.get_json()
    pages = data.get("pages", 1)
    session["selected_pages"] = pages

    shop_name = session.get("selected_shop_name", "lidl")
    data = load_shop_data(shop_name)
    
    return jsonify(success=True, discounts=data["discounts"])


def load_shop_data(shop_name):
    """Shared logic for getting discounts and categories."""
    categories = kupibot.get_recipes_categories()
    discounts = kupibot.get_discounts(
        escape(shop_name),
        max_pages=session.get("selected_pages", 5)
    )

    print("Current TOP K", session.get("selected_topk", 3))
    print("Current PAGES", session.get("selected_pages", 5))
    print("Current CATEGORY", session.get("selected_category", "hlavni-jidla"))
    print("Current SHOP NAME", session.get("selected_shop_name", "lidl"))

    names = [d["name"] for d in discounts]
    prices = [d["prices"][0] for d in discounts]
    amounts = [d["amounts"][0] for d in discounts]

    return {
        "shop_name": escape(shop_name),
        "discounts": list(zip(names, prices, amounts)),
        "shops": list(zip(SHOPS, SHOPS_URL)),
        "categories": list(zip(categories, correct_category_spellings())),
        "pages_value": session.get("selected_pages", 5),
        "topk_value": session.get("selected_topk", 3),
    }
    
def correct_category_spellings() -> List[str]:
    return [
        "Dezerty a sladká jídla",
        "Hlavní jídla",
        "Nápoje",
        "Omáčky a guláše",
        "Polévky",
        "Předkrmy, chuťovky a svačiny",
        "Přílohy, pečivo",
        "Saláty",
        "Zavařování a nakládání"
    ]
    

@app.route('/set_category', methods=['POST'])
def set_category():
    data = request.get_json()
    category = data.get('category')
    session["selected_category"] = category
    print(f"Selected recipe category: {category}") 
    return jsonify({"message": f"Recipe category '{category}' received"})

@app.route('/run_chat', methods=['POST'])
def run_chat():
    print("Current TOP K", session.get("selected_topk", 3))
    print("Current PAGES", session.get("selected_pages", 5))
    print("Current CATEGORY", session.get("selected_category", "hlavni-jidla"))
    print("Current SHOP NAME", session.get("selected_shop_name", "lidl"))
    response = kupibot.get_chat_response(session.get("selected_category", "hlavni-jidla"), top_k=session.get("selected_topk", 3))
    
    return jsonify({"response": response})


@app.route("/set_topk", methods=["POST"])
def set_topk():
    data = request.get_json()
    topk = data.get("topk", 1)
    session["selected_topk"] = topk
    return jsonify(success=True)

if __name__ == "__main__":
    kupibot = KupiBot()

    SHOPS = ["Lidl", "Albert", "Billa", "Kaufland", "Tesco", "Globus", "Penny market"]
    SHOPS_URL = ["lidl", "albert", "billa", "kaufland", "tesco", "globus", "penny-market"]
    app.secret_key = "super_secret_key_in_dev" # for sessions in flask
    app.run(debug=True)
