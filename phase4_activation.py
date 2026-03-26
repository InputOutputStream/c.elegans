"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 4 — ACTIVATION NEURONALE ET COMPORTEMENT                             ║
║  Connectome C. elegans : sensation → traitement → mouvement                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ BIOLOGIE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  C. elegans est le SEUL organisme dont le connectome complet est connu.
  302 neurones, 7341 synapses chimiques, 890 jonctions gap (électriques).

  Architecture fonctionnelle en 3 couches :

  COUCHE SENSORIELLE (avant du ver, pôle mex-3)
    ASE  : chémoréception des ions (Na+, Cl-) → nourriture
    AWC  : olfaction (benzaldéhyde, isoamyl alcool) → nourriture
    ASH  : nociception (osmolarité élevée, SDS, métaux lourds) → danger
    ALM/PLM : mécanoreception tactile (antérieure/postérieure)

  COUCHE D'INTERNEURONES (décision)
    AIY  : intègre les signaux de nourriture → active l'avance
    AIZ  : réprime AIY en absence de signal → contrôle de direction
    AVA  : commande motrice RECUL (dominante)
    AVB  : commande motrice AVANCE (concurrente d'AVA)
    RIA  : intégration multimodale + tournant
    AIB  : relais danger → recul

  COUCHE MOTRICE (muscles longitudinaux)
    VA/VB : muscles ventraux (A=recul, B=avance)
    DA/DB : muscles dorsaux (A=recul, B=avance)
    Contraction alternée D/V → ondulation sinusoïdale propulsive

  CIRCUITS COMPORTEMENTAUX CONNUS :
    Nourriture détectée → ASE/AWC → AIY → AVB → VB/DB → avance
    Danger/nociception  → ASH → AVA → VA → recul IMMÉDIAT (réflexe)
    Toucher antérieur   → ALM → AVA → recul
    Toucher postérieur  → PLM → AVB → accélération (fuite)
    Absence de signal   → AIZ inhibe AIY → pirouettes/changements de direction

━━━ INFORMATIQUE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Modèle LIF (Leaky Integrate-and-Fire) discret :
    V(t+1) = V(t) × 0.88  +  Σ(w_ij × sign_ij × fire_j) + bruit
    Si V ≥ θ : spike, V=0, timer_refractaire=5

  Corps simulé par une chaîne de 20 segments :
    y_segment(i,t) = A × sin(ω×t + φ×i/N) × sin(π×i/N)
    A = amplitude ∝ activité motrice    ω ∝ vitesse de contraction
    Direction de déplacement inversée si VA domine VB.

━━━ COMPORTEMENTS ATTENDUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  STIMULUS NOURRITURE :
    Attendu : activation ASE→AIY→AVB→VB/DB → corps bleu en ondulation
    Mesure : potentials[VB]+potentials[DB] > potentials[VA]
    Délai : 3-8 steps (propagation à travers 3 couches)

  STIMULUS DANGER :
    Attendu : activation ASH→AVA→VA → corps orange, déplacement inversé
    Mesure : potentials[VA] pic > 0.7 dans les 5 steps suivant ASH
    Caractéristique : réponse PLUS RAPIDE que nourriture (moins de relais)

  STIMULUS TOUCHER :
    Attendu : ALM → recul (même circuit que danger via AVA)
    Différence de nourriture : PLM peut activer AVB → accélération

  EXCITABILITÉ élevée (>0.8) :
    → spikes spontanés fréquents, comportement chaotique
    Biologie : mutant gain-of-function sur les canaux Na+ (NaV)

  EXCITABILITÉ faible (<0.2) :
    → réseau silencieux, pas de réponse même aux stimuli
    Biologie : mutant perd-de-fonction unc-2 (calcium channel)

  FORCE SYNAPTIQUE élevée (>1.2) :
    → oscillations, phénomènes épileptiformes
    Biologie : sur-expression de levnr-1 (levamisole receptor)

━━━ CONTRÔLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ESPACE → pause/reprise     R → reset
  F → stimulus nourriture    D → stimulus danger
  T → stimulus toucher       N → neutre (pas de stimulus)
  ↑↓ → excitabilité          ←→ → force synaptique
  +/- → vitesse corps        Q/ESC → quitter

━━━ DÉPENDANCES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install pygame numpy
"""

import numpy as np
import pygame, sys, math, collections

# ─── Connectome simplifié C. elegans ─────────────────────────────────────────
# (id, nom, type S/I/M, rôle, x_rel, y_rel)
NEURONS = [
    (0,  "ASE", "S", "chimio",    0.10, 0.28),
    (1,  "AWC", "S", "olfaction", 0.10, 0.45),
    (2,  "ASH", "S", "nocicept.", 0.10, 0.62),
    (3,  "ALM", "S", "toucher-A", 0.18, 0.18),
    (4,  "PLM", "S", "toucher-P", 0.18, 0.78),
    (5,  "AIY", "I", "avance",    0.38, 0.22),
    (6,  "AIZ", "I", "direction", 0.38, 0.40),
    (7,  "AVA", "I", "RECUL",     0.38, 0.60),
    (8,  "AVB", "I", "AVANCE",    0.38, 0.76),
    (9,  "RIA", "I", "tournant",  0.55, 0.33),
    (10, "AIB", "I", "fuite",     0.55, 0.55),
    (11, "VA",  "M", "v-recul",   0.80, 0.28),
    (12, "VB",  "M", "v-avance",  0.80, 0.45),
    (13, "DA",  "M", "d-recul",   0.80, 0.62),
    (14, "DB",  "M", "d-avance",  0.80, 0.78),
]
N_NEUR = len(NEURONS)

# (from, to, poids_base, signe +1/-1, label circuit)
CONNECTIONS = [
    # Sensoriel → interneurone
    (0, 5,  0.80,  1, "ASE→AIY nourriture"),
    (0, 6,  0.45,  1, "ASE→AIZ"),
    (0, 8,  0.40,  1, "ASE→AVB"),
    (1, 5,  0.70,  1, "AWC→AIY olfaction"),
    (1, 9,  0.50,  1, "AWC→RIA"),
    (2, 7,  0.90,  1, "ASH→AVA DANGER"),
    (2, 10, 0.75,  1, "ASH→AIB"),
    (2, 6,  0.30, -1, "ASH-|AIZ"),
    (3, 7,  0.65,  1, "ALM→AVA toucher"),
    (3, 10, 0.50,  1, "ALM→AIB"),
    (4, 7,  0.45,  1, "PLM→AVA"),
    (4, 8,  0.50,  1, "PLM→AVB acceleration"),
    # Interneurone → moteur
    (5, 12, 0.78,  1, "AIY→VB"),
    (5, 14, 0.60,  1, "AIY→DB"),
    (6, 12, 0.45,  1, "AIZ→VB"),
    (6, 11, 0.30, -1, "AIZ-|VA"),
    (7, 11, 0.92,  1, "AVA→VA RECUL"),
    (7, 13, 0.40, -1, "AVA-|DA"),
    (8, 12, 0.82,  1, "AVB→VB AVANCE"),
    (8, 14, 0.65,  1, "AVB→DB"),
    (9, 13, 0.70,  1, "RIA→DA tournant"),
    (9, 14, 0.65, -1, "RIA-|DB"),
    (10,11, 0.72,  1, "AIB→VA"),
    (10, 7, 0.50,  1, "AIB→AVA"),
    # Inhibitions mutuelles (circuit push-pull AVA/AVB)
    (7, 8,  0.60, -1, "AVA-|AVB"),
    (8, 7,  0.45, -1, "AVB-|AVA"),
    (5, 7,  0.30, -1, "AIY-|AVA"),
]

# Stimuli → activation forcée par neurone sensoriel
STIMULI = {
    "food":   {0: 0.65, 1: 0.55},            # ASE, AWC
    "danger": {2: 0.95, 3: 0.70},            # ASH, ALM
    "touch":  {3: 0.85, 4: 0.70},            # ALM, PLM
    "none":   {},
}

# ─── Paramètres ──────────────────────────────────────────────────────────────
WIN_NET  = 480    # largeur du panel réseau
WIN_BODY = 280    # largeur du panel corps
WIN_W    = WIN_NET + WIN_BODY
WIN_H    = 620
HUD_H    = 110
PANEL_H  = WIN_H - HUD_H

BODY_SEGS  = 20
BODY_LEN   = 200  # pixels

TYPE_COL = {
    "S": (196,125, 42),  # sensoriel  ambre
    "I": (123, 94,167),  # interneurone violet
    "M": ( 58,127,193),  # moteur    bleu
}

# ─── État simulation ──────────────────────────────────────────────────────────
def fresh_state():
    return {
        "V":          np.zeros(N_NEUR, dtype=np.float32),
        "firing":     np.zeros(N_NEUR, dtype=np.uint8),
        "ref_timer":  np.zeros(N_NEUR, dtype=np.int8),
        "spike_log":  collections.deque(maxlen=200),  # (tick, neuron_id)
        "tick":       0,
        "body_phase": 0.0,
        "body_x":     WIN_NET + WIN_BODY//2,
        "body_y":     HUD_H + PANEL_H//2,
        "body_dir":   0.0,
        "stimulus":   "none",
        "excit":      0.55,
        "noise":      0.03,
        "syn_str":    0.65,
        "body_spd":   1.0,
    }

def sim_step(st):
    exc   = st["excit"]
    noi   = st["noise"]
    synW  = st["syn_str"]
    stim  = STIMULI.get(st["stimulus"], {})
    V, firing, ref = st["V"], st["firing"], st["ref_timer"]

    new_firing = np.zeros(N_NEUR, dtype=np.uint8)
    for i in range(N_NEUR):
        if ref[i] > 0:
            ref[i] -= 1; continue
        inp = np.random.uniform(0, noi)
        if i in stim:
            inp += stim[i] * exc
        for (a,b,w,sign,_) in CONNECTIONS:
            if b == i and firing[a]:
                inp += w * sign * synW * 0.45
        V[i] = float(np.clip(V[i]*0.88 + inp, 0, 1))
        thr  = 0.55 - exc*0.12
        if V[i] >= thr:
            new_firing[i] = 1; V[i] = 0.0; ref[i] = 5
            st["spike_log"].append((st["tick"], i))

    st["firing"] = new_firing
    st["tick"]  += 1

    # Décodage comportement → mouvement
    fwd = float(V[12] + V[14])   # VB + DB
    rev = float(V[11])            # VA
    trn = abs(float(V[13]-V[14]))# DA - DB
    spd = st["body_spd"]
    bname = decode_behavior(fwd,rev,trn)

    if bname == "avance":
        st["body_x"] += math.cos(st["body_dir"])*1.2*spd
        st["body_y"] += math.sin(st["body_dir"])*1.2*spd
        st["body_phase"] += 0.07*spd
    elif bname == "recul":
        st["body_x"] -= math.cos(st["body_dir"])*0.9*spd
        st["body_y"] -= math.sin(st["body_dir"])*0.9*spd
        st["body_phase"] -= 0.05*spd
    elif bname == "tournant":
        st["body_dir"] += 0.035*spd*(1 if trn>0 else -1)
        st["body_phase"] += 0.04*spd

    # Contraindre dans le panel body
    bx_min = WIN_NET + 40; bx_max = WIN_W - 40
    by_min = HUD_H + 40;   by_max = WIN_H - 40
    st["body_x"] = float(np.clip(st["body_x"], bx_min, bx_max))
    st["body_y"] = float(np.clip(st["body_y"], by_min, by_max))


def decode_behavior(fwd,rev,trn):
    if fwd < 0.05 and rev < 0.05: return "idle"
    if rev > fwd and rev > 0.25:  return "recul"
    if trn > 0.40:                return "tournant"
    if fwd > 0.15:                return "avance"
    return "idle"


# ─── Rendu ────────────────────────────────────────────────────────────────────
def draw_network(screen, font, st, win_net, hud_h, panel_h):
    """Panel gauche : réseau avec activations en temps réel."""
    pygame.draw.rect(screen,(10,12,20),(0,hud_h,win_net,panel_h))

    # Connexions
    for (a,b,w,sign,label) in CONNECTIONS:
        na = NEURONS[a]; nb = NEURONS[b]
        ax = int(na[4]*win_net); ay = int(na[5]*panel_h)+hud_h
        bx = int(nb[4]*win_net); by = int(nb[5]*panel_h)+hud_h
        active = bool(st["firing"][a]) and st["V"][b]>0.1
        alpha  = 200 if active else 35
        col    = (100,180,255) if sign>0 else (255,100,100)
        lw     = max(1, int(w*2.5)) if active else 1
        s = pygame.Surface((win_net,panel_h),pygame.SRCALPHA)
        pygame.draw.line(s,(*col,alpha),(ax,ay-hud_h),(bx,by-hud_h),lw)
        screen.blit(s,(0,hud_h))

    # Neurones
    for (i,name,ntype,role,xr,yr) in NEURONS:
        x = int(xr*win_net); y = int(yr*panel_h)+hud_h
        fire = bool(st["firing"][i])
        col  = TYPE_COL[ntype]
        r    = 10 if fire else 7
        if fire:
            s = pygame.Surface((win_net,panel_h),pygame.SRCALPHA)
            pygame.draw.circle(s,(*col,45),(x,y-hud_h),18)
            screen.blit(s,(0,hud_h))
        pygame.draw.circle(screen,(255,220,60) if fire else col,(x,y),r)
        pygame.draw.circle(screen,(255,255,255),(x,y),r,1)
        # Anneau de potentiel
        pot_r = int(st["V"][i]*15)
        if pot_r > 2:
            s2 = pygame.Surface((win_net,panel_h),pygame.SRCALPHA)
            pygame.draw.circle(s2,(*col,80),(x,y-hud_h),r+pot_r,2)
            screen.blit(s2,(0,hud_h))
        lbl = font.render(f"{name}", True,
                          (255,220,60) if fire else (200,200,200))
        screen.blit(lbl,(x-12,y-r-14))

    # Légende types
    for j,(t,c,l) in enumerate([("S",(196,125,42),"sensoriel"),
                                  ("I",(123,94,167),"interneurone"),
                                  ("M",(58,127,193),"moteur")]):
        pygame.draw.circle(screen,c,(12,hud_h+panel_h-55+j*18),5)
        screen.blit(font.render(l,True,(180,180,180)),(22,hud_h+panel_h-58+j*18))


def draw_body(screen, font, st, win_net, win_body, hud_h, panel_h):
    """Panel droit : corps animé de C. elegans."""
    pygame.draw.rect(screen,(8,10,18),(win_net,hud_h,win_body,panel_h))

    fwd = float(st["V"][12]+st["V"][14])
    rev = float(st["V"][11])
    trn = abs(float(st["V"][13]-st["V"][14]))
    bname = decode_behavior(fwd,rev,trn)

    cx  = st["body_x"]
    cy  = st["body_y"]
    amp = panel_h * (0.08 if bname!="idle" else 0.01)
    ph  = st["body_phase"]
    ang = st["body_dir"]

    # Corps
    pts = []
    for seg in range(BODY_SEGS):
        t   = seg / BODY_SEGS
        lx  = -BODY_LEN/2 + t*BODY_LEN
        ly  = math.sin(ph + t*math.pi*2) * amp * math.sin(t*math.pi)
        rx  = cx + math.cos(ang)*lx - math.sin(ang)*ly
        ry  = cy + math.sin(ang)*lx + math.cos(ang)*ly
        pts.append((rx,ry))
        r   = max(2, int(7*(1-abs(t-0.5)*1.6)))
        col = ((196,125,42) if rev>fwd and rev>0.2
               else (58,127,193) if fwd>0.15
               else (80,140,80))
        if win_net < rx < win_net+win_body and hud_h < ry < hud_h+panel_h:
            pygame.draw.circle(screen, col, (int(rx),int(ry)), r)

    # Tête (cercle plus grand)
    hx, hy = pts[-1] if pts else (cx,cy)
    fire_head = any(st["firing"][i] for i in [0,1,2,3])
    hcol = (255,220,60) if fire_head else (200,200,200)
    if win_net < hx < win_net+win_body and hud_h < hy < hud_h+panel_h:
        pygame.draw.circle(screen, hcol, (int(hx),int(hy)), 9)

    # Flèche comportement
    col_beh = {"avance":(58,127,193),"recul":(196,125,42),
                "tournant":(123,94,167),"idle":(80,80,80)}
    c = col_beh.get(bname,(80,80,80))
    screen.blit(font.render(bname.upper(),True,c),(win_net+8,hud_h+panel_h-22))

    # Barres comportement
    labels = [("avance",fwd/0.6,(58,127,193)),
              ("recul", rev/0.6,(196,125,42)),
              ("tournant",min(1,trn/0.4),(123,94,167))]
    for j,(lbl,val,cl) in enumerate(labels):
        bw = int(min(1,max(0,val)) * (win_body-60))
        pygame.draw.rect(screen,(30,30,30),(win_net+8, hud_h+4+j*22, win_body-20,14))
        if bw > 0:
            pygame.draw.rect(screen,cl,(win_net+8, hud_h+4+j*22, bw, 14))
        screen.blit(font.render(lbl,True,(180,180,180)),(win_net+8+bw+4,hud_h+4+j*22))


def draw_hud4(screen, font, st, win_w, hud_h):
    s = pygame.Surface((win_w,hud_h),pygame.SRCALPHA)
    s.fill((0,0,0,175)); screen.blit(s,(0,0))
    stim_col = {"food":(74,158,107),"danger":(196,80,80),
                "touch":(196,125,42),"none":(120,120,120)}
    sc = stim_col.get(st["stimulus"],(120,120,120))
    n_spk = sum(1 for t,_ in st["spike_log"] if st["tick"]-t<50)

    lines = [
        (f"t={st['tick']:5d}  Stimulus:{st['stimulus'].upper():8s}  "
         f"Spikes/50t:{n_spk:3d}  "
         f"Actifs:{int(np.sum(st['firing'])):2d}/15", (220,220,220)),
        (f"Excitabilite:{st['excit']:.2f}  Bruit:{st['noise']:.3f}  "
         f"SynapseStr:{st['syn_str']:.2f}  VitesseCorps:{st['body_spd']:.1f}",
         (180,200,255)),
        (f"[F]nourriture  [D]danger  [T]toucher  [N]neutre  "
         f"[↑↓]excit  [←→]syn_str  [+/-]vitesse  [ESPACE]pause  [Q]quit",
         (140,140,140)),
        (f"Attendu: Food→avance(bleu)  Danger→recul_immediat(orange)  "
         f"Toucher→recul  Neutre→idle/oscillations_spontanees",
         (120,160,120)),
    ]
    for i,(txt,col) in enumerate(lines):
        screen.blit(font.render(txt,True,col),(8,4+i*24))
    # Indicateur stimulus
    pygame.draw.rect(screen,sc,(4,0,4,hud_h))


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Phase 4 — Activation & Comportement | C. elegans")
    font  = pygame.font.SysFont("monospace",12)
    clock = pygame.time.Clock()
    st    = fresh_state()
    paused = False

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
                    st = fresh_state()
                elif ev.key == pygame.K_f:
                    st["stimulus"]="food";   print("[Phase4] Stimulus: NOURRITURE → attendu: AVANCE")
                elif ev.key == pygame.K_d:
                    st["stimulus"]="danger"; print("[Phase4] Stimulus: DANGER → attendu: RECUL IMMÉDIAT")
                elif ev.key == pygame.K_t:
                    st["stimulus"]="touch";  print("[Phase4] Stimulus: TOUCHER → attendu: RECUL")
                elif ev.key == pygame.K_n:
                    st["stimulus"]="none";   print("[Phase4] Stimulus: NEUTRE")
                elif ev.key == pygame.K_UP:
                    st["excit"]=min(1.0,round(st["excit"]+0.05,2))
                elif ev.key == pygame.K_DOWN:
                    st["excit"]=max(0.05,round(st["excit"]-0.05,2))
                elif ev.key == pygame.K_RIGHT:
                    st["syn_str"]=min(1.5,round(st["syn_str"]+0.05,2))
                elif ev.key == pygame.K_LEFT:
                    st["syn_str"]=max(0.1,round(st["syn_str"]-0.05,2))
                elif ev.key in (pygame.K_PLUS,pygame.K_EQUALS):
                    st["body_spd"]=min(3.0,round(st["body_spd"]+0.1,1))
                elif ev.key == pygame.K_MINUS:
                    st["body_spd"]=max(0.2,round(st["body_spd"]-0.1,1))

        if not paused:
            sim_step(st)

        screen.fill((10,12,20))
        draw_network(screen,font,st,WIN_NET,HUD_H,PANEL_H)
        draw_body(screen,font,st,WIN_NET,WIN_BODY,HUD_H,PANEL_H)
        draw_hud4(screen,font,st,WIN_W,HUD_H)

        # Séparateur vertical
        pygame.draw.line(screen,(40,40,60),(WIN_NET,HUD_H),(WIN_NET,WIN_H),1)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
