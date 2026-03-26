# C. elegans — Naissance d'une Conscience
## Simulation complète des phases embryonnaires

---

## Installation

```bash
pip install pygame numpy scipy
```

---

## Lancement

```bash
# Menu graphique (recommandé)
python run.py

# Phase directe
python run.py 1
python run.py 4
python run.py 1 2 3   # séquence

# Ou directement
python phase1_reaction_diffusion.py
python phase2_gene_network.py
python phase3_synaptogenesis.py
python phase4_activation.py
python phase5_memory.py
```

---

## Pipeline de données

```
phase1 → phase1_A.npy + phase1_B.npy
             ↓
phase2 → phase2_dom.npy + phase2_mex3.npy
             ↓
phase3 → phase3_network.npy
             ↓
phase4/5 → connectome C. elegans intégré (indépendants)
```

---

## Résumé des phases

| Phase | Mécanisme biologique | Mécanisme informatique | Sortie clé |
|-------|---------------------|----------------------|------------|
| 1 | Polarité PAR de l'œuf | Réaction-diffusion Gray-Scott | Gradient chimique A/B |
| 2 | Gènes maîtres pal-1/pie-1/skn-1/mex-3 | Réseau de régulation + inhibition latérale | Carte territoires génétiques |
| 3 | Chimiotaxie axonale + règle de Hebb | Agents + LTP discret | Matrice synaptique |
| 4 | Connectome C. elegans (302→12 nœuds) | LIF + décodage comportemental | Corps animé |
| 5 | Conditionnement olfactif (Rankin 1990) | LTP/LTD + protocole Pavlov | Courbe d'apprentissage |

---

## Comportements attendus par phase

### Phase 1
- **Spots** (f=0.037, k=0.060) : taches oranges isolées
- **Rayures** (f=0.022, k=0.051) : fronts horizontaux
- **Labyrinthes** (f=0.055, k=0.062) : réseau sinueux

### Phase 2
- Zone **bleue (mex-3)** à gauche/antérieur : 15-25% des cellules
- Frontières nettes si inhibition > 0.6
- Zone indéterminée (gris) si seuil trop élevé

### Phase 3
- Axones convergent vers zones mex-3 (chimiotaxie)
- Synapses jaunes → blanches par renforcement Hebb
- Élagage [P] : supprime ~30% des connexions faibles

### Phase 4
- [F] Nourriture → corps bleu, ondulation vers l'avant
- [D] Danger → corps orange, recul immédiat (< 5 steps)
- [T] Toucher → recul (même circuit que danger)
- Excitabilité > 0.8 → oscillations spontanées

### Phase 5
- Baseline → réponse odeur ~0.0-0.10
- Après 8 entraînements → réponse odeur > 0.40
- Extinction → retour progressif vers baseline
- LTP élevé (>0.12) → apprentissage en 3-4 essais

---

## Références biologiques

- **Sulston & Horvitz (1977)** : lignage cellulaire complet de C. elegans
- **White et al. (1986)** : connectome synaptique (première carte complète)
- **Turing (1952)** : "The Chemical Basis of Morphogenesis"
- **Rankin et al. (1990)** : mémoire associative olfactive chez C. elegans
- **Hebb (1949)** : "The Organization of Behavior" — règle de plasticité
- **Tononi (2004)** : théorie de l'information intégrée (Φ) — phase 6 future
