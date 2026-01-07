from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn

# 1. Initialisation de l'application
app = FastAPI(title="AI Moderation API")

# 2. Chargement du modèle
# Assurez-vous que le fichier 'model_toxicity.pkl' est dans le même dossier
try:
    pipeline = joblib.load("model_toxicity.pkl")
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")

# 3. Structure des données attendues (Input)
class CommentRequest(BaseModel):
    text: str

# 4. Route de prédiction
@app.post("/predict")
async def predict_toxicity(request: CommentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")

    try:
        # La prédiction (retourne généralement le label de votre Excel)
        prediction = pipeline.predict([request.text])[0]
        
        # Logique de classification :
        # Modifiez la condition ci-dessous selon les labels de votre dataset
        # Exemple: si votre dataset utilise 'toxic' ou 1 pour les insultes
        is_insult = True if prediction in ["toxic", 1, "insult"] else False

        return {
            "text": request.text,
            "prediction": str(prediction),
            "is_insult": is_insult
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Route de test
@app.get("/")
def home():
    return {"status": "Online", "model": "LogisticRegression + Tfidf"}

if __name__ == "__main__":
    # Lancement du serveur sur le port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)