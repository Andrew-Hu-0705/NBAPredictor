import os
from flask import Flask, render_template, request, jsonify
from predict import predict_game

app = Flask(__name__)

# ── Team list ──────────────────────────────────────────────────────────────────
NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

@app.route("/")
def index():
    return render_template("index.html", teams=NBA_TEAMS)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    home_team = data.get("home_team")
    away_team = data.get("away_team")
    
    if not home_team or not away_team:
        return jsonify({"error": "Both home_team and away_team are required"}), 400
    if home_team == away_team:
        return jsonify({"error": "Home and away teams must be different"}), 400
        
    try:
        result = predict_game(home_team, away_team)
        
        # Convert numpy arrays to python lists for JSON serialization
        if hasattr(result.get("shap_values"), "tolist"):
            result["shap_values"] = result["shap_values"].tolist()
        if hasattr(result.get("feature_values"), "tolist"):
            result["feature_values"] = result["feature_values"].tolist()
            
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)