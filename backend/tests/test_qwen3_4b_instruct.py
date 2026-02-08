import json
import requests
import time

# --- Configuration ---
MODEL_NAME = "qwen3:4b-instruct-2507-fp16"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OUTPUT_FILE = "definition_queries_results.json"

# --- The 40 Definition Queries ---
queries = [
    "C'est quoi un revenu brut selon le règlement ?",
    "C'est quoi un revenu net selon le règlement ?",
    "Quelle est la différence exacte entre revenu brut et revenu net pour le calcul ?",
    "Si le règlement ne précise pas brut ou net, lequel doit-on utiliser par défaut ?",
    "C'est quoi une franchise sur l'activité lucrative ?",
    "C'est quoi le 'revenu à prendre en compte' ?",
    "Quelle est la différence mathématique entre la franchise et le revenu à prendre en compte ?",
    "C'est quoi un groupe familial ?",
    "Un mineur peut-il avoir sa propre limite de fortune indépendante ?",
    "C'est quoi la limite de fortune pour un enfant au sein d'une famille ?",
    "Un enfant est-il considéré comme une 'personne seule' dans le calcul de la fortune ?",
    "C'est quoi un établissement à des fins thérapeutiques ?",
    "Une clinique est-elle juridiquement considérée comme un établissement thérapeutique ?",
    "Un hôpital est-il juridiquement considéré comme un établissement thérapeutique ?",
    "C'est quoi le forfait pour dépenses personnelles en cas d'hospitalisation ?",
    "C'est quoi le forfait d'entretien de base ?",
    "Un étudiant en haute école a-t-il droit au forfait d'entretien standard à 100% ?",
    "C'est quoi la règle de réduction pour un étudiant en haute école ?",
    "Quelle est la différence de traitement entre un apprenti et un étudiant ?",
    "C'est quoi une franchise sur le salaire d'apprentissage ?",
    "C'est quoi le plafond global de franchise pour une famille ?",
    "Le revenu d'un apprenti est-il soumis au plafond global de la famille ?",
    "Est-ce que l'argent de poche d'un apprenti compte dans le plafond familial ?",
    "C'est quoi une activité lucrative indépendante selon le règlement ?",
    "C'est quoi une aide financière provisoire ?",
    "C'est quoi l'aide d'urgence (barème asile/NEM) ?",
    "Quelle est la différence entre l'aide sociale ordinaire et l'aide d'urgence ?",
    "C'est quoi le forfait d'intégration ?",
    "Un mineur a-t-il droit au forfait d'intégration ?",
    "C'est quoi une prestation circonstancielle ?",
    "C'est quoi des frais dentaires 'simples, économiques et adéquats' ?",
    "C'est quoi la prime moyenne cantonale d'assurance maladie ?",
    "C'est quoi la participation aux frais de garde ?",
    "C'est quoi une pension alimentaire perçue ?",
    "C'est quoi un gain extraordinaire (Loterie, Héritage) ?",
    "C'est quoi une allocation de rentrée scolaire ?",
    "C'est quoi une faute grave entraînant une réduction de prestation ?",
    "C'est quoi un jeune adulte sans formation ?",
    "C'est quoi une communauté de majeurs ?",
    "C'est quoi la cohabitation sans faire ménage commun ?"
]

def query_ollama(prompt, model):
    """Sends a prompt to the Ollama API and returns the response text."""
    # Context setup to force legal definitions
    system_instruction = (
        "Tu es un expert juridique du Règlement d'application de la loi sur l'aide sociale (RASLP) à Genève. "
        "Réponds de manière concise et précise en donnant la définition juridique exacte."
    )
    
    full_prompt = f"{system_instruction}\n\nQuestion: {prompt}\nRéponse:"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1  # Low temperature for more deterministic/factual answers
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return f"Error: {str(e)}"

def main():
    results_data = {"results": []}
    
    print(f"Starting execution of {len(queries)} queries on model '{MODEL_NAME}'...")
    
    for i, question in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Processing: {question}")
        
        start_time = time.time()
        answer = query_ollama(question, MODEL_NAME)
        duration = time.time() - start_time
        
        result_entry = {
            "id": i,
            "question": question,
            "generated_answer": answer,
            "duration_seconds": round(duration, 2)
        }
        
        results_data["results"].append(result_entry)
        
    # Save to JSON
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)
        print(f"\nSuccess! Results saved to '{OUTPUT_FILE}'")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()