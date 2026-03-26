"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 1 — RÉACTION-DIFFUSION DE TURING                                     ║
║  Segmentation chimique de l'œuf de C. elegans                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ BIOLOGIE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Dans l'œuf fécondé de C. elegans, deux familles de protéines s'affrontent :
    · PAR-3 / PAR-6 (activateurs) : se concentrent au pôle ANTÉRIEUR
    · PAR-1 / PAR-2 (inhibiteurs) : occupent le pôle POSTÉRIEUR, diffusent vite

  Ce déséquilibre crée spontanément une POLARITÉ ANTÉRO-POSTÉRIEURE — première
  information spatiale de l'organisme, sans aucune instruction externe.
  Turing a prédit ce mécanisme en 1952 avant la découverte des protéines PAR.

━━━ INFORMATIQUE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Modèle Gray-Scott sur grille N×N (conditions périodiques) :
    dA/dt = Da·∇²A  −  A·B²  +  f·(1−A)
    dB/dt = Db·∇²B  +  A·B²  −  (f+k)·B

  · A = activateur (PAR-3/6), B = inhibiteur (PAR-1/2)
  · A·B² = catalyse croisée non-linéaire → instabilité de Turing
  · Da < Db obligatoire pour l'émergence de patterns

━━━ COMPORTEMENTS ATTENDUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Preset [1] SPOTS       f=0.037 k=0.060 → taches isolées (type léopard)
    Biologie : précurseurs de clusters cellulaires distincts
    Signe visuel : B forme des îlots ronds orange sur fond vert

  Preset [2] RAYURES     f=0.022 k=0.051 → bandes parallèles
    Biologie : segmentation antérieure/postérieure de l'œuf
    Signe visuel : fronts orange horizontaux qui se stabilisent

  Preset [3] LABYRINTHES f=0.055 k=0.062 → canaux interconnectés (défaut)
    Biologie : patron observé dans le cortex cérébral
    Signe visuel : réseau orange sinueux continu

  VARIATION Da/Db :
    Da ≈ Db   → homogène, pas de pattern (instabilité absente)
    Db/Da > 2 → patterns plus fins, densité accrue
    Db/Da > 5 → spots très petits, dynamique rapide

━━━ CONTRÔLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ESPACE → pause/reprise   R → reset   S → sauvegarder
  1/2/3  → preset          Q/ESC → quitter (sauvegarde auto)
  ↑↓     → feed rate f     ←→ → kill rate k

━━━ DÉPENDANCES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install pygame numpy
"""

import numpy as np
import pygame, sys

# ─── Paramètres ──────────────────────────────────────────────────────────────
N    = 200          # grille N×N
WIN  = 700          # fenêtre en pixels
CELL = WIN // N
Da, Db = 0.20, 0.10
F,  K  = 0.055, 0.062
DT     = 1.0
SPF    = 8          # steps per frame

PRESETS = {
    pygame.K_1: ("spots",       0.037, 0.060,
                 "Taches isolees -> clusters cellulaires distincts"),
    pygame.K_2: ("rayures",     0.022, 0.051,
                 "Bandes paralleles -> segmentation A/P de l'oeuf"),
    pygame.K_3: ("labyrinthes", 0.055, 0.062,
                 "Canaux interconnectes -> patron cortical"),
}

# ─── Grille ───────────────────────────────────────────────────────────────────
def init_grid(n, seed=None):
    rng = np.random.default_rng(seed)
    A = np.ones((n, n), dtype=np.float32)
    B = np.zeros((n, n), dtype=np.float32)
    cx, cy, r = n//2, n//2, 8
    ys, xs = np.ogrid[-r:r+1, -r:r+1]
    mask = xs**2 + ys**2 < r**2
    A[cy-r:cy+r+1, cx-r:cx+r+1][mask] = 0.50
    B[cy-r:cy+r+1, cx-r:cx+r+1][mask] = 0.25
    for _ in range(6):
        sx, sy = rng.integers(8, n-8, size=2)
        sz = rng.integers(3, 6)
        A[sy-sz:sy+sz, sx-sz:sx+sz] = 0.50 + rng.random((2*sz, 2*sz)) * 0.12
        B[sy-sz:sy+sz, sx-sz:sx+sz] = 0.25 + rng.random((2*sz, 2*sz)) * 0.12
    return A, B

def laplacian(X):
    return (np.roll(X,1,0)+np.roll(X,-1,0)+np.roll(X,1,1)+np.roll(X,-1,1)-4*X)

def step_rd(A, B, da, db, f, k):
    react = A * B * B
    An = np.clip(A + DT*(da*laplacian(A) - react + f*(1-A)), 0, 1)
    Bn = np.clip(B + DT*(db*laplacian(B) + react - (f+k)*B), 0, 1)
    return An, Bn

# ─── Rendu ────────────────────────────────────────────────────────────────────
def build_surface(A, B):
    """Vert=activateur(ant.) / Orange=inhibiteur(post.)"""
    mix = np.clip(B * 3.0, 0, 1)
    R  = ((1-mix)*A*255).astype(np.uint8)
    G  = (mix*200 + (1-mix)*A*180).astype(np.uint8)
    Bc = ((1-mix)*A*100 + mix*60).astype(np.uint8)
    rgb = np.repeat(np.repeat(np.stack([R,G,Bc],-1), CELL, 0), CELL, 1)
    return pygame.surfarray.make_surface(rgb.transpose(1,0,2))

def draw_hud(screen, font, gen, f, k, da, db, paused, preset, zones, win):
    s = pygame.Surface((win, 95), pygame.SRCALPHA)
    s.fill((0,0,0,170)); screen.blit(s, (0,0))
    phase = ("init" if gen<100 else "activation" if gen<500 else
             "competition" if gen<1500 else "stabilisation" if gen<3000
             else "MATURITE → pret phase 2")
    lines = [
        (f"Gen:{gen:6d}  Phase:{phase}  {'[PAUSE]' if paused else '[RUN]  '}  Preset:{preset}",
         (220,220,220)),
        (f"f={f:.3f}  k={k:.3f}  Da={da:.2f}  Db={db:.2f}  |  Zones-B>15%:{zones:.1f}%",
         (180,200,255)),
        (f"[ESPACE]pause  [1/2/3]preset  [R]reset  [S]save  [↑↓]f  [←→]k  [Q]quit",
         (140,140,140)),
        (f"Comportement attendu: spots=taches | rayures=fronts | labyrinthes=reseau",
         (120,160,120)),
    ]
    for i,(txt,col) in enumerate(lines):
        screen.blit(font.render(txt, True, col), (8, 4+i*22))

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN, WIN+95))
    pygame.display.set_caption("Phase 1 — Réaction-Diffusion | C. elegans")
    font   = pygame.font.SysFont("monospace", 12)
    clock  = pygame.time.Clock()

    A, B = init_grid(N)
    f_val, k_val, da_val, db_val = F, K, Da, Db
    gen, paused, preset_name = 0, False, "labyrinthes"

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
                    A, B = init_grid(N); gen = 0
                elif ev.key == pygame.K_s:
                    np.save("phase1_A.npy", A); np.save("phase1_B.npy", B)
                    print(f"[Phase1] Sauvegardé gen={gen}")
                elif ev.key in PRESETS:
                    preset_name, f_val, k_val, desc = PRESETS[ev.key]
                    A, B = init_grid(N); gen = 0
                    print(f"[Phase1] Preset '{preset_name}': {desc}")
                elif ev.key == pygame.K_UP:
                    f_val = min(0.10, round(f_val+0.001, 3))
                elif ev.key == pygame.K_DOWN:
                    f_val = max(0.01, round(f_val-0.001, 3))
                elif ev.key == pygame.K_RIGHT:
                    k_val = min(0.08, round(k_val+0.001, 3))
                elif ev.key == pygame.K_LEFT:
                    k_val = max(0.04, round(k_val-0.001, 3))

        if not paused:
            for _ in range(SPF):
                A, B = step_rd(A, B, da_val, db_val, f_val, k_val)
            gen += SPF

        screen.blit(build_surface(A, B), (0, 95))
        draw_hud(screen, font, gen, f_val, k_val, da_val, db_val,
                 paused, preset_name, float(np.mean(B>0.15)*100), WIN)
        pygame.display.flip()
        clock.tick(30)

    np.save("phase1_A.npy", A); np.save("phase1_B.npy", B)
    print(f"[Phase1] Sauvegardé automatiquement gen={gen}")
    pygame.quit()

if __name__ == "__main__":
    main()
