# Deploiement chatbot: plan court

## 1. Preparation modele
- Construire le modele local via `Modelfile`.
- Verifier que le nom du modele correspond a `VETO_CHAT_MODEL` dans `rag/query.py`.

## 2. Indexation base de connaissances
- Ajouter/modifier les fichiers dans `kb/default/`.
- Lancer indexation RAG pour reconstruire la base vecteur.

## 3. Test terminal
- Tester 10 a 20 questions representant des cas faibles/moderes/eleves.
- Verifier coherence du triage et presence d'avertissement.

## 4. Integration page HTML
- Brancher un formulaire simple (champ question + bouton envoyer + zone reponse).
- Cote backend, appeler `rag/query.py` ou une API equivalente.
- Afficher la reponse et, idealement, les sources/extraits utilises.

## 5. Checklist qualite avant demo
- Pas de diagnostic definitif.
- Escalade correcte des urgences.
- Reponses claires et breves.
- Gestion correcte du manque d'informations.
