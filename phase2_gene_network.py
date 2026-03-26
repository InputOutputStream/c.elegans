"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 2 — RÉSEAU DE GÈNES RÉGULATEURS                                      ║
║  Différenciation cellulaire dans l'œuf de C. elegans                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ BIOLOGIE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Les gradients chimiques de la phase 1 (protéines PAR) sont lus par des
  GÈNES MAÎTRES. Chaque gène a un seuil d'activation spatial :

    pal-1  (gène Caudal)  → zone POSTÉRIEURE élevée → muscle + épiderme
    pie-1  (gène NANOS)   → zone CENTRALE          → lignée germinale P4
    skn-1  (gène Nrf2)    → zone VENTRALE           → intestin E
    mex-3  (gène KH-dom.) → zone ANTÉRIEURE         → ectoblastes → NEURONES

  L'INHIBITION LATÉRALE entre gènes crée des frontières nettes :
  si pal-1 est activé dans une cellule, il réprime mex-3 et skn-1.
  C'est le mécanisme de "winner-takes-all" cellulaire.

  À la fin, chaque cellule a UNE identité précise — la même pour chaque
  individu, à chaque génération. Déterminisme absolu à 959 cellules adultes.

━━━ INFORMATIQUE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Pipeline en 3 étapes :
  1. LECTURE DU GRADIENT : G_gene(x,y) = f(position) × g(B(x,y), seuil)
  2. DIFFUSION GÉNIQUE : moyenne pondérée avec les 4 voisins (dg itérations)
  3. INHIBITION LATÉRALE : le gène dominant dans chaque cellule réprime les
     autres de (inh × 0.3) à chaque étape de cascade.

  Critère de dominance : max(G_pal, G_pie, G_skn, G_mex) > min_level
  avec écart suffisant pour trancher (inh × 0.1).

━━━ COMPORTEMENTS ATTENDUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SEUIL élevé (>0.30) :
    → peu de cellules activées, territoires fragmentés
    Biologie : comme si les protéines PAR étaient diluées

  SEUIL bas (<0.08) :
    → presque tout activé, frontières floues
    Biologie : comme une sur-expression des facteurs de transcription

  INHIBITION forte (>0.80) :
    → territoires très nets, transitions abruptes
    Biologie : proche du développement réel (frontières cellulaires précises)

  INHIBITION faible (<0.20) :
    → zones mixtes, co-expression de plusieurs gènes
    Biologie : état indifférencié (cellule souche)

  DIFFUSION élevée (>0.15) :
    → les signaux géniques s'étalent → blobs larges
    Biologie : comme si les morphogènes diffusaient trop vite

  MEX-3 (bleu) ATTENDU : zone gauche/antérieure, ~15-25% des cellules
    C'est le précurseur neuronal → entrée directe en phase 3

━━━ CONTRÔLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ESPACE → pause/reprise   R → reset   S → sauvegarder
  ↑↓     → seuil           ←→ → inhibition   +/- → diffusion
  C      → étape cascade   Q/ESC → quitter

━━━ DÉPENDANCES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install pygame numpy scipy  (scipy optionnel pour zoom si N différent)
"""

import numpy as np
import pygame, sys, os

# ─── Paramètres ──────────────────────────────────────────────────────────────
N    = 128          # grille interne (sera interpolée si phase1 a taille diff.)
WIN  = 700
CELL = WIN // N

THRESHOLD  = 0.15   # seuil d'activation génique
INHIBITION = 0.60   # force inhibition latérale
DIFFUSION  = 0.08   # diffusion inter-cellulaire du signal génique
CASCADE    = 3      # étapes d'inhibition par frame

# Couleurs des 4 gènes + indéterminé (RGB)
GENE_COLORS = [
    (74,  158, 107),   # pal-1  vert     muscle/épiderme
    (123, 94,  167),   # pie-1  violet   lignée germinale
    (196, 125, 42),    # skn-1  ambre    intestin
    (58,  127, 193),   # mex-3  bleu     NEURONES ← clé
    (60,  60,  60),    # indét  gris
]
GENE_NAMES = ["pal-1", "pie-1", "skn-1", "mex-3", "indet"]
GENE_ROLES = ["muscle/epiderme", "lignee germinale", "intestin", "NEURONES", "indetermine"]

# ─── Chargement gradient phase 1 ─────────────────────────────────────────────
def load_gradient(n):
    """
    Charge les arrays A,B de la phase 1.
    Si absents, génère un gradient synthétique qui reproduit les axes
    antéro-postérieur et dorso-ventral de C. elegans.
    """
    if os.path.exists("phase1_A.npy") and os.path.exists("phase1_B.npy"):
        A = np.load("phase1_A.npy").astype(np.float32)
        B = np.load("phase1_B.npy").astype(np.float32)
        if A.shape[0] != n:
            from scipy.ndimage import zoom
            A = zoom(A, n/A.shape[0]).astype(np.float32)
            B = zoom(B, n/B.shape[0]).astype(np.float32)
        print("[Phase2] Gradient chargé depuis phase1_A/B.npy")
        return np.clip(A,0,1), np.clip(B,0,1)
    else:
        print("[Phase2] Phase 1 absente → gradient synthétique C. elegans")
        return synthetic_gradient(n)

def synthetic_gradient(n):
    """
    Gradient synthétique reproduisant la polarité réelle de C. elegans :
    - Axe AP (antéro-postérieur) : gradient horizontal de B
    - Axe DV (dorso-ventral) : gradient vertical de A
    - Bruit léger pour les graines de pattern
    """
    xs = np.linspace(0, 1, n, dtype=np.float32)
    ys = np.linspace(0, 1, n, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    rng = np.random.default_rng(42)
    B = np.clip(
        0.25 * np.sin(XX*np.pi*3) * np.cos(YY*np.pi*2) +
        0.20 * np.exp(-((XX-0.25)**2 + (YY-0.50)**2)/0.02) +
        0.15 * np.exp(-((XX-0.20)**2 + (YY-0.75)**2)/0.015) +
        rng.random((n,n)).astype(np.float32) * 0.05,
        0, 1)
    A = np.clip(1.0 - B*0.5, 0, 1)
    return A, B

# ─── Calcul des niveaux géniques ──────────────────────────────────────────────
def compute_genes(A, B, thr):
    """
    Calcule les 4 niveaux d'expression génique à partir du gradient spatial.

    Logique positionnelle (axes C. elegans) :
      pal-1 : fort à l'EST (postérieur) si B dépasse le seuil → muscle
      pie-1 : fort au CENTRE si B modéré → cellule P (germinale)
      skn-1 : fort au SUD (ventral) si A faible → intestin
      mex-3 : fort à l'OUEST (antérieur) si A fort → ectoblastes/neurones
    """
    n = A.shape[0]
    xs = np.linspace(0, 1, n, dtype=np.float32)
    ys = np.linspace(0, 1, n, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    dist_ctr = np.sqrt((XX-0.5)**2 + (YY-0.5)**2).astype(np.float32)
    ctr = np.clip(1.0 - dist_ctr * 2.0, 0, 1)

    G = [None]*4
    G[0] = np.maximum(0, XX   * np.where(B > thr,       B*1.5, 0))   # pal-1 post.
    G[1] = np.maximum(0, ctr  * np.where(B > thr*0.8,   B,     0))   # pie-1 centre
    G[2] = np.maximum(0, YY   * np.where(A < 0.60,   1.0-A,    0))   # skn-1 ventral
    G[3] = np.maximum(0, (1-XX)* np.where(A > 0.50,  A*0.80,   0))   # mex-3 ant.
    return G

def diffuse_genes(G, dg, n_iter=3):
    """
    Propagation du signal génique aux cellules voisines.
    Simule la diffusion des morphogènes et facteurs de transcription
    à travers les jonctions gap entre cellules adjacentes.
    """
    for _ in range(n_iter):
        for i in range(4):
            avg = (np.roll(G[i],1,0)+np.roll(G[i],-1,0)+
                   np.roll(G[i],1,1)+np.roll(G[i],-1,1)) / 4.0
            G[i] = G[i]*(1-dg) + avg*dg
    return G

def lateral_inhibition(G, inh, n_steps):
    """
    Inhibition latérale : winner-takes-all cellulaire.
    Le gène le plus exprimé dans une cellule réprime les autres.
    Mécanisme réel : via des protéines répresseurs comme MEX-5/6.
    """
    for _ in range(n_steps):
        stack   = np.stack(G, axis=0)           # (4, N, N)
        winner  = np.argmax(stack, axis=0)       # indice du gène dominant
        for gi in range(4):
            mask    = (winner != gi)
            G[gi]   = np.maximum(0, G[gi] - inh*0.3 * mask)
    return G

def dominant_map(G, min_lv=0.05):
    """
    Carte de dominance : pour chaque cellule, le gène dominant (0-3) ou -1.
    -1 = cellule indéterminée (pas encore différenciée).
    """
    stack  = np.stack(G, axis=0)
    maxval = np.max(stack, axis=0)
    dom    = np.argmax(stack, axis=0).astype(np.int8)
    dom[maxval < min_lv] = -1
    return dom

def dom_to_rgb(dom):
    """Convertit la carte de dominance en image RGB."""
    n = dom.shape[0]
    rgb = np.zeros((n, n, 3), dtype=np.uint8)
    for gi, col in enumerate(GENE_COLORS[:4]):
        mask = (dom == gi)
        rgb[mask] = col
    rgb[dom == -1] = GENE_COLORS[4]
    return rgb

def territory_stats(dom):
    total = dom.size
    return {GENE_NAMES[i]: round(np.mean(dom==i)*100, 1) for i in range(4)}

# ─── Rendu ────────────────────────────────────────────────────────────────────
def build_surfaces(A, B, dom):
    """
    Deux surfaces côte à côte :
    Gauche  : gradient chimique phase 1 (vert/orange)
    Droite  : territoires génétiques (4 couleurs)
    """
    half = WIN // 2
    cell = half // N

    # Gradient chimique
    mix = np.clip(B*3, 0, 1)
    R  = ((1-mix)*A*255).astype(np.uint8)
    G  = (mix*200 + (1-mix)*A*180).astype(np.uint8)
    Bc = ((1-mix)*A*100 + mix*60).astype(np.uint8)
    grad_rgb = np.repeat(np.repeat(np.stack([R,G,Bc],-1), cell, 0), cell, 1)
    surf_grad = pygame.surfarray.make_surface(grad_rgb.transpose(1,0,2))

    # Territoires génétiques
    dom_rgb = np.repeat(np.repeat(dom_to_rgb(dom), cell, 0), cell, 1)
    surf_dom = pygame.surfarray.make_surface(dom_rgb.transpose(1,0,2))

    return surf_grad, surf_dom, half

def draw_legend(screen, font, stats, half, hud_h):
    """Légende des gènes avec pourcentages de territoire."""
    x0 = half + 8
    for i, (name, role) in enumerate(zip(GENE_NAMES[:4], GENE_ROLES[:4])):
        col = GENE_COLORS[i]
        pct = stats.get(name, 0)
        pygame.draw.rect(screen, col, (x0, hud_h + 4 + i*22, 14, 14))
        txt = font.render(f"{name}  {role}  {pct}%", True, (200,200,200))
        screen.blit(txt, (x0+18, hud_h + 3 + i*22))

def draw_hud2(screen, font, thr, inh, dg, cas, cascade_step,
              paused, stats, win, hud_h):
    s = pygame.Surface((win, hud_h), pygame.SRCALPHA)
    s.fill((0,0,0,170)); screen.blit(s,(0,0))
    # Titres panneaux
    screen.blit(font.render("GRADIENT CHIMIQUE (phase 1)",True,(180,220,180)),(8,4))
    screen.blit(font.render("TERRITOIRES GÉNÉTIQUES",True,(180,180,220)),(win//2+8,4))
    lines = [
        (f"Seuil:{thr:.2f}  Inhibition:{inh:.2f}  Diffusion:{dg:.2f}  Cascade:{cas}  Step:{cascade_step}",
         (220,220,220)),
        (f"MEX-3(neurones):{stats.get('mex-3',0):.1f}%  pal-1:{stats.get('pal-1',0):.1f}%  "
         f"pie-1:{stats.get('pie-1',0):.1f}%  skn-1:{stats.get('skn-1',0):.1f}%",
         (180,200,255)),
        (f"[ESPACE]pause  [R]reset  [S]save  [↑↓]seuil  [←→]inhibition  [+/-]diffusion  [C]cascade  [Q]quit",
         (140,140,140)),
    ]
    for i,(txt,col) in enumerate(lines):
        screen.blit(font.render(txt,True,col),(8,22+i*20))

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    HUD_H = 84
    LEG_H = 100
    screen = pygame.display.set_mode((WIN, WIN//2 + HUD_H + LEG_H))
    pygame.display.set_caption("Phase 2 — Réseau de Gènes Régulateurs | C. elegans")
    font  = pygame.font.SysFont("monospace", 12)
    clock = pygame.time.Clock()

    A, B   = load_gradient(N)
    thr    = THRESHOLD
    inh    = INHIBITION
    dg     = DIFFUSION
    cas    = CASCADE
    cstep  = 0
    paused = False

    def recompute():
        G   = compute_genes(A, B, thr)
        G   = diffuse_genes(G, dg)
        G   = lateral_inhibition(G, inh, cas)
        dom = dominant_map(G)
        return G, dom

    G, dom = recompute()
    print(__doc__)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    A, B = load_gradient(N); G, dom = recompute(); cstep=0
                elif ev.key == pygame.K_s:
                    np.save("phase2_dom.npy", dom)
                    np.save("phase2_mex3.npy", G[3])
                    print(f"[Phase2] Sauvegardé phase2_dom.npy, phase2_mex3.npy")
                elif ev.key == pygame.K_UP:
                    thr = min(0.40, round(thr+0.01, 2)); G,dom=recompute()
                elif ev.key == pygame.K_DOWN:
                    thr = max(0.02, round(thr-0.01, 2)); G,dom=recompute()
                elif ev.key == pygame.K_RIGHT:
                    inh = min(1.00, round(inh+0.05, 2)); G,dom=recompute()
                elif ev.key == pygame.K_LEFT:
                    inh = max(0.05, round(inh-0.05, 2)); G,dom=recompute()
                elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                    dg = min(0.20, round(dg+0.01, 2)); G,dom=recompute()
                elif ev.key == pygame.K_MINUS:
                    dg = max(0.01, round(dg-0.01, 2)); G,dom=recompute()
                elif ev.key == pygame.K_c:
                    # étape manuelle de cascade
                    G = lateral_inhibition(G, inh, 1)
                    dom = dominant_map(G)
                    cstep += 1

        if not paused:
            G = lateral_inhibition(G, inh, 1)
            dom = dominant_map(G)
            cstep += 1

        stats = territory_stats(dom)
        surf_grad, surf_dom, half = build_surfaces(A, B, dom)
        screen.fill((10,12,20))
        screen.blit(surf_grad, (0, HUD_H))
        screen.blit(surf_dom,  (half, HUD_H))
        draw_legend(screen, font, stats, half, HUD_H + WIN//2)
        draw_hud2(screen, font, thr, inh, dg, cas, cstep, paused, stats, WIN, HUD_H)
        pygame.display.flip()
        clock.tick(20)

    np.save("phase2_dom.npy", dom)
    np.save("phase2_mex3.npy", G[3])
    print("[Phase2] Sauvegardé automatiquement")
    pygame.quit()

if __name__ == "__main__":
    main()
