# Base de connaissances (KB)

## Structure
- `kb/default/`: corpus principal pour le chatbot VetIA.
- Chaque fichier `.md` couvre un theme unique.

## Bonnes pratiques de contenu
- Paragraphes courts et actionnables.
- Rappeler les limites (pas de diagnostic definitif).
- Ajouter des red flags clairs quand pertinent.
- Eviter les formulations ambigues.

## Cycle de mise a jour
1. Modifier ou ajouter des fichiers dans `kb/default/`.
2. Reindexer via `rag/index.py`.
3. Tester des scenarios dans `rag/query.py`.
