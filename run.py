"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  C. ELEGANS — NAISSANCE D'UNE CONSCIENCE                                    ║
║  Lanceur unifié — Phases 1 à 5                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Ce programme lance le menu de sélection des phases.
  Chaque phase peut être lancée indépendamment ou dans l'ordre.

  ORDRE RECOMMANDÉ :
    Phase 1 → sauvegarde phase1_A.npy / phase1_B.npy
    Phase 2 → lit phase1_*.npy, sauvegarde phase2_dom.npy / phase2_mex3.npy
    Phase 3 → sauvegarde phase3_network.npy
    Phase 4 → utilise le connectome C. elegans intégré (indépendant)
    Phase 5 → utilise le connectome C. elegans intégré (indépendant)

  DÉPENDANCES : pip install pygame numpy scipy

  USAGE :
    python run.py          → menu graphique
    python run.py 1        → lancer directement la phase 1
    python run.py 1 2 3    → lancer les phases 1,2,3 en séquence
"""

import pygame, sys, os, subprocess

PHASES = [
    (1, "Réaction-Diffusion",      "Segmentation chimique de l'oeuf",
     "phase1_reaction_diffusion.py",
     [(0.20, 0.78, "Gradient chimique vert/orange"),
      (0.20, 0.88, "Vert = activateur (PAR-3/6, anterieur)"),
      (0.20, 0.98, "Orange = inhibiteur (PAR-1/2, posterieur)"),
      (0.20, 1.08, "Pattern emerge en ~500-3000 generations"),
      (0.20, 1.18, "Controles: [1/2/3]=presets  [S]=sauvegarder")]),

    (2, "Réseau de Gènes",         "Différenciation cellulaire",
     "phase2_gene_network.py",
     [(0.20, 0.78, "4 gènes maîtres colorés sur la carte droite"),
      (0.20, 0.88, "Bleu (mex-3) = précurseurs neuronaux"),
      (0.20, 0.98, "Vert (pal-1) = muscle/épiderme"),
      (0.20, 1.08, "Violet (pie-1) = lignée germinale"),
      (0.20, 1.18, "Controles: [↑↓]=seuil  [←→]=inhibition")]),

    (3, "Synaptogenèse",           "Formation des premières synapses",
     "phase3_synaptogenesis.py",
     [(0.20, 0.78, "Points colorés = neurones (bleu=mex-3)"),
      (0.20, 0.88, "Pointillés = axones explorant l'espace"),
      (0.20, 0.98, "Jaune = synapse naissante (<0.5)"),
      (0.20, 1.08, "Blanc = synapse établie (>0.5, Hebb)"),
      (0.20, 1.18, "Controles: [P]=élaguer  [↑↓]=chimiotaxie")]),

    (4, "Activation & Comportement","Sensation → traitement → mouvement",
     "phase4_activation.py",
     [(0.20, 0.78, "Gauche = réseau neuronal (15 neurones)"),
      (0.20, 0.88, "Droite = corps de C. elegans animé"),
      (0.20, 0.98, "[F]=nourriture → avance  [D]=danger → recul"),
      (0.20, 1.08, "Jaune = spike  Anneau = potentiel membranaire"),
      (0.20, 1.18, "Controles: [F/D/T/N]=stimulus  [↑↓]=excitabilite")]),

    (5, "Mémoire & Apprentissage", "Plasticité synaptique LTP/LTD",
     "phase5_memory.py",
     [(0.20, 0.78, "Gauche = réseau (vert=LTP, orange=LTD)"),
      (0.20, 0.88, "Centre = barres des synapses plastiques"),
      (0.20, 0.98, "Droite = courbe de réponse à l'odeur"),
      (0.20, 1.08, "Protocole: [1]baseline [2]entraîner [3]tester"),
      (0.20, 1.18, "Attendu: resp~0 avant, >0.4 après entraînement")]),
]

W, H = 820, 560
COL_BG    = (10,  12,  20)
COL_TITLE = (200, 200, 255)
COL_PHASE = (58,  127, 193)
COL_DESC  = (160, 160, 200)
COL_NOTE  = (120, 120, 140)
COL_HOV   = (30,  60,  100)
COL_SEL   = (20,  80,  50)
COL_BTN   = (40,  40,  70)
COL_BTN_H = (70,  70,  120)

def draw_menu(screen, font_big, font_med, font_sm, hover, selected):
    screen.fill(COL_BG)

    # Titre
    t = font_big.render("C. ELEGANS — NAISSANCE D'UNE CONSCIENCE", True, COL_TITLE)
    screen.blit(t, (W//2 - t.get_width()//2, 18))
    t2 = font_sm.render("Simulation des phases embryonnaires — de l'œuf à la mémoire", True, COL_NOTE)
    screen.blit(t2, (W//2 - t2.get_width()//2, 46))

    pygame.draw.line(screen,(40,40,70),(20,64),(W-20,64),1)

    # Phases
    ROW_H = 78; START_Y = 74; PAD = 16
    rects = []
    for i, (num,title,desc,fname,notes) in enumerate(PHASES):
        y = START_Y + i*ROW_H
        col_bg = COL_SEL if i in selected else (COL_HOV if hover==i else (18,20,32))
        rect = pygame.Rect(PAD, y, W-2*PAD, ROW_H-4)
        pygame.draw.rect(screen, col_bg, rect, border_radius=6)
        pygame.draw.rect(screen, (40,40,80), rect, 1, border_radius=6)
        rects.append(rect)

        # Numéro
        nc = (255,200,60) if i in selected else COL_PHASE
        n_surf = font_big.render(f"{num}", True, nc)
        screen.blit(n_surf, (PAD+14, y+12))

        # Titre + desc
        screen.blit(font_med.render(title, True, (220,220,255)), (PAD+52, y+8))
        screen.blit(font_sm.render(desc,   True, COL_DESC),      (PAD+52, y+30))

        # Fichier + status
        exists = os.path.exists(fname)
        st_col = (74,158,107) if exists else (196,80,80)
        st_txt = "✓ présent" if exists else "✗ introuvable"
        screen.blit(font_sm.render(f"{fname}  {st_txt}", True, st_col), (PAD+52, y+50))

        # Note visuelle (première ligne)
        if notes:
            screen.blit(font_sm.render(notes[0][2], True, (100,130,100)), (PAD+350, y+50))

    # Boutons
    btn_y = START_Y + len(PHASES)*ROW_H + 10
    btns = [
        ("Lancer phase sélectionnée",  (PAD,      btn_y, 260, 36)),
        ("Lancer toutes (séquence)",   (PAD+270,  btn_y, 220, 36)),
        ("Quitter",                    (PAD+500,  btn_y, 100, 36)),
    ]
    btn_rects = []
    for j,(label,r) in enumerate(btns):
        mx,my = pygame.mouse.get_pos()
        rect = pygame.Rect(*r)
        col = COL_BTN_H if rect.collidepoint(mx,my) else COL_BTN
        pygame.draw.rect(screen, col, rect, border_radius=5)
        pygame.draw.rect(screen, (70,70,120), rect, 1, border_radius=5)
        lt = font_sm.render(label, True, (220,220,255))
        screen.blit(lt, (rect.x + rect.w//2 - lt.get_width()//2,
                         rect.y + rect.h//2 - lt.get_height()//2))
        btn_rects.append(rect)

    # Instructions
    screen.blit(font_sm.render(
        "[Clic] sélectionner   [Entrée] lancer   [A] tout sélectionner   [Echap] quitter",
        True, COL_NOTE), (PAD, btn_y+44))

    return rects, btn_rects


def launch_phase(fname):
    """Lance un script phase dans un sous-processus."""
    if not os.path.exists(fname):
        print(f"[Lanceur] ERREUR : {fname} introuvable dans le dossier courant.")
        return
    print(f"[Lanceur] Lancement de {fname}...")
    subprocess.run([sys.executable, fname])
    print(f"[Lanceur] {fname} terminé.")


def main():
    # Mode ligne de commande
    if len(sys.argv) > 1:
        nums = []
        for arg in sys.argv[1:]:
            try: nums.append(int(arg)-1)
            except: pass
        for i in nums:
            if 0 <= i < len(PHASES):
                launch_phase(PHASES[i][3])
        return

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("C. elegans — Naissance d'une Conscience | Menu")
    font_big = pygame.font.SysFont("monospace", 17, bold=True)
    font_med = pygame.font.SysFont("monospace", 14, bold=True)
    font_sm  = pygame.font.SysFont("monospace", 11)
    clock    = pygame.time.Clock()

    selected = set()
    hover    = -1

    print(__doc__)

    running = True
    while running:
        mx, my = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_a:
                    selected = set(range(len(PHASES)))
                elif ev.key == pygame.K_RETURN:
                    if selected:
                        pygame.quit()
                        for i in sorted(selected):
                            launch_phase(PHASES[i][3])
                        return
                elif pygame.K_1 <= ev.key <= pygame.K_5:
                    i = ev.key - pygame.K_1
                    if i in selected: selected.discard(i)
                    else: selected.add(i)

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # Test sur les rects de phase
                ROW_H=78; START_Y=74; PAD=16
                for i in range(len(PHASES)):
                    r = pygame.Rect(PAD, START_Y+i*ROW_H, W-2*PAD, ROW_H-4)
                    if r.collidepoint(mx,my):
                        if i in selected: selected.discard(i)
                        else: selected.add(i)
                # Test boutons
                btn_y = START_Y + len(PHASES)*ROW_H + 10
                btns_rect = [
                    pygame.Rect(PAD,     btn_y, 260, 36),
                    pygame.Rect(PAD+270, btn_y, 220, 36),
                    pygame.Rect(PAD+500, btn_y, 100, 36),
                ]
                if btns_rect[0].collidepoint(mx,my) and selected:
                    pygame.quit()
                    for i in sorted(selected):
                        launch_phase(PHASES[i][3])
                    return
                elif btns_rect[1].collidepoint(mx,my):
                    pygame.quit()
                    for i in range(len(PHASES)):
                        launch_phase(PHASES[i][3])
                    return
                elif btns_rect[2].collidepoint(mx,my):
                    running = False

        # Hover
        ROW_H=78; START_Y=74; PAD=16
        hover = -1
        for i in range(len(PHASES)):
            r = pygame.Rect(PAD, START_Y+i*ROW_H, W-2*PAD, ROW_H-4)
            if r.collidepoint(mx,my): hover=i; break

        rects, btn_rects = draw_menu(screen, font_big, font_med, font_sm, hover, selected)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
