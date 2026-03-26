"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 5 — MÉMOIRE ET APPRENTISSAGE                                         ║
║  Plasticité synaptique et conditionnement dans C. elegans                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ BIOLOGIE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  C. elegans possède une MÉMOIRE ASSOCIATIVE documentée expérimentalement :

  CONDITIONNEMENT OLFACTIF (Wen et al., 1997 ; Rankin et al., 1990)
    Protocole : présenter une odeur NEUTRE (benzaldéhyde via AWC) en même
    temps qu'un stimulus AVERSIF (privation de nourriture, ou choc osmotique).
    Résultat : après 4-8 couplages, l'odeur seule déclenche le comportement
    d'évitement. La mémoire persiste 16-24h à 20°C.
    Base moléculaire : modification des poids synaptiques AWC→AIY et AIZ→AVA.

  PLASTICITÉ SYNAPTIQUE — MÉCANISMES
    LTP (Long-Term Potentiation) :
      Co-activation pré+post → insertion de récepteurs AMPA (GLR-1 chez C.el.)
      → synapse plus forte de 20-40% après 5-10 co-activations
      Durée : heures à jours sans consolidation protéique

    LTD (Long-Term Depression) :
      Activation pré sans réponse post → endocytose des récepteurs AMPA
      → synapse affaiblie progressivement
      Mécanisme chez C. elegans : médié par LIN-10 et SOL-1

    Déclin passif (Forgetting) :
      Déstabilisation spontanée des synapses renforcées
      Taux biologiques : ~10% par heure pour la mémoire à court terme
      Ralenti par la synthèse protéique (consolidation à long terme)

  EXTINCTION :
    Présenter l'odeur sans danger de façon répétée → LTD progressif
    Résultat : retour vers la réponse de baseline
    Différence de l'oubli passif : actif, médié par les mêmes circuits

  PROTOCOLE EXPÉRIMENTAL SIMULÉ (calqué sur Rankin et al., 1990) :
    Phase 1 BASELINE : mesure réponse à l'odeur seule (devrait être ~0)
    Phase 2 ENTRAÎN. : n couplages odeur+danger → LTP sur AWC→AIZ et AIZ→AVA
    Phase 3 TEST     : odeur seule → mesure réponse (devrait être >0.4)
    Phase 4 EXTINCT. : odeur seule répétée → LTD progressif → retour baseline

━━━ INFORMATIQUE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  RÈGLE DE HEBB ÉTENDUE (BCM rule approximée) :
    ΔW = η × pre × post     si co-activation → LTP
    ΔW = −δ × pre × (1−post) si pré actif, post silencieux → LTD
    ΔW = −decay × (W − W0)  déclin passif vers la valeur initiale

  SYNAPSES PLASTIQUES (modifiables) :
    AWC→AIZ  : clé principale — lien odeur → circuit de recul
    AWC→RIA  : modulateur directionnel
    ASH→AIZ  : association danger → intégration
    AIZ→AVA  : relais décision → commande motrice recul
    AIZ→VA   : commande directe musculaire
    RIA→AVA  : modulation globale

  SYNAPSES FIXES (hardwired) :
    ASH→AVA, AVA→VA, AVB→VB, etc. — réflexes innés non-modifiables

━━━ COMPORTEMENTS ATTENDUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  BASELINE (avant apprentissage) :
    Odeur (AWC) → réponse recul ≈ 0.0-0.10
    Car AWC→AIZ est faible, AIZ→AVA est faible → VA ne s'active pas

  APRÈS ENTRAÎNEMENT (8+ essais) :
    Odeur seule → réponse recul > 0.40-0.60
    AWC→AIZ renforcé (LTP) → AIZ→AVA renforcé (LTP) → VA s'active
    Visible : barres synaptiques vertes (renforcées)

  VARIATION LTP élevé (>0.12) :
    → apprentissage en 3-4 essais seulement
    Biologie : sur-expression glr-1 ou expériences à haute température

  VARIATION LTD élevé (>0.08) :
    → les synapses renforcées déclinent vite
    → nécessite de réentraîner fréquemment (mémoire courte)
    Biologie : mutant lin-10 gain-of-function

  VARIATION DÉCLIN élevé (>0.012) :
    → extinction rapide, oubli spontané accéléré
    Biologie : mutant crh-1 (CREB), déficit de consolidation

  ESSAIS INSUFFISANTS (<4) :
    → LTP partiel, test montre réponse 0.15-0.30 (apprentissage incomplet)
    → Biologie : proto-mémoire, pas de consolidation stable

━━━ CONTRÔLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1 → Phase BASELINE   2 → Phase ENTRAÎNEMENT   3 → Phase TEST
  4 → Phase EXTINCTION  R → reset complet
  ↑↓ → taux LTP        ←→ → taux LTD    +/- → déclin mémoire
  N  → +1 essai manuel  Q/ESC → quitter

━━━ DÉPENDANCES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install pygame numpy
"""

import numpy as np
import pygame, sys, math

# ─── Connectome (même base que phase 4 + marquage plasticité) ────────────────
NEURONS = [
    (0,  "ASE","S","chimio",   0.08,0.28),
    (1,  "AWC","S","olfact.",  0.08,0.45),
    (2,  "ASH","S","nocicept.",0.08,0.62),
    (3,  "AIY","I","avance",   0.32,0.22),
    (4,  "AIZ","I","direction",0.32,0.42),
    (5,  "AVA","I","RECUL",    0.32,0.62),
    (6,  "AVB","I","AVANCE",   0.32,0.78),
    (7,  "RIA","I","tournant", 0.52,0.33),
    (8,  "AIB","I","fuite",    0.52,0.55),
    (9,  "VA", "M","v-recul",  0.80,0.30),
    (10, "VB", "M","v-avance", 0.80,0.48),
    (11, "DA", "M","d-recul",  0.80,0.65),
]
N_NEUR = len(NEURONS)

# (from, to, poids_init, signe, label, PLASTIQUE?)
CONN_DEF = [
    (0, 3,  0.75, 1,"ASE→AIY",   False),
    (0, 6,  0.50, 1,"ASE→AVB",   False),
    (1, 4,  0.28, 1,"AWC→AIZ",   True),   # ← clé apprentissage odeur
    (1, 7,  0.18, 1,"AWC→RIA",   True),   # ← modulateur
    (2, 5,  0.88, 1,"ASH→AVA",   False),  # réflexe inné fixe
    (2, 8,  0.72, 1,"ASH→AIB",   False),
    (2, 4,  0.38, 1,"ASH→AIZ",   True),   # ← association danger
    (3,10,  0.76, 1,"AIY→VB",    False),
    (4, 5,  0.32, 1,"AIZ→AVA",   True),   # ← clé apprentissage
    (4, 9,  0.25, 1,"AIZ→VA",    True),   # ← clé apprentissage
    (5, 9,  0.90, 1,"AVA→VA",    False),
    (6,10,  0.80, 1,"AVB→VB",    False),
    (7, 5,  0.32, 1,"RIA→AVA",   True),   # ← modulateur
    (8, 5,  0.68, 1,"AIB→AVA",   False),
    (5, 6,  0.48,-1,"AVA-|AVB",  False),
    (6, 5,  0.38,-1,"AVB-|AVA",  False),
]
N_CONN = len(CONN_DEF)
W0     = np.array([c[2] for c in CONN_DEF], dtype=np.float32)
PLAST  = [c[5] for c in CONN_DEF]

# ─── Paramètres ──────────────────────────────────────────────────────────────
WIN_W  = 900; WIN_H  = 680; HUD_H  = 105
PANEL_H= WIN_H - HUD_H
NET_W  = 380; SYN_W  = 280; TL_W   = WIN_W - NET_W - SYN_W

LTP_RATE  = 0.06
LTD_RATE  = 0.020
DECAY     = 0.003
N_TRAIN   = 8

TYPE_COL = {"S":(196,125,42),"I":(123,94,167),"M":(58,127,193)}

# ─── État ─────────────────────────────────────────────────────────────────────
def fresh_state():
    return {
        "W":         W0.copy(),
        "V":         np.zeros(N_NEUR,dtype=np.float32),
        "firing":    np.zeros(N_NEUR,dtype=np.uint8),
        "ref":       np.zeros(N_NEUR,dtype=np.int8),
        "phase":     "idle",     # idle/baseline/train/test/extinct
        "trial":     0,
        "timeline":  [],         # [(trial, response, phase)]
        "ltp_rate":  LTP_RATE,
        "ltd_rate":  LTD_RATE,
        "decay":     DECAY,
        "n_train":   N_TRAIN,
        "training":  False,      # entraînement automatique en cours
        "train_i":   0,
        "train_timer":0,
    }

# ─── Simulation d'un essai ─────────────────────────────────────────────────
def run_trial(st, stim_ids, steps=35):
    """
    Simule un essai complet et retourne la force de réponse de recul.
    stim_ids : dict {neuron_id: input_strength}
    Retourne : float ∈ [0,1] représentant V[VA] + V[AVA]*0.5
    """
    V   = np.zeros(N_NEUR,dtype=np.float32)
    fir = np.zeros(N_NEUR,dtype=np.uint8)
    ref = np.zeros(N_NEUR,dtype=np.int8)
    W   = st["W"]

    for _ in range(steps):
        nf = np.zeros(N_NEUR,dtype=np.uint8)
        for i in range(N_NEUR):
            if ref[i]>0: ref[i]-=1; continue
            inp = np.random.uniform(0,0.015)
            if i in stim_ids: inp += stim_ids[i]
            for ci,(a,b,w0_,sign,lbl,_) in enumerate(CONN_DEF):
                if b==i and fir[a]:
                    inp += W[ci]*sign*0.48
            V[i] = float(np.clip(V[i]*0.88+inp,0,1))
            if V[i]>=0.50: nf[i]=1; V[i]=0; ref[i]=5
        fir=nf

    return float(V[9] + V[5]*0.45)   # VA + AVA×0.45


def apply_plasticity(st, with_danger):
    """
    Applique LTP ou LTD selon la co-activation simulée.
    with_danger=True  → odeur+danger → LTP sur synapses plastiques AWC/AIZ
    with_danger=False → odeur seule  → LTD progressif (extinction)
    """
    ltp = st["ltp_rate"]; ltd = st["ltd_rate"]; dec = st["decay"]
    # Co-activations attendues selon le stimulus
    if with_danger:
        co = {1:0.75, 2:0.90, 4:0.85, 5:0.90}   # AWC,ASH,AIZ,AVA actifs
    else:
        co = {1:0.65, 4:0.55}                     # AWC, AIZ modérément actifs

    for ci,(a,b,w0_,sign,lbl,plast) in enumerate(CONN_DEF):
        if not plast: continue
        pre  = co.get(a, 0.0)
        post = co.get(b, 0.0)
        if pre>0.5 and post>0.5:
            st["W"][ci] = min(1.5, st["W"][ci] + ltp)   # LTP
        elif pre>0.5 and post<0.25:
            st["W"][ci] = max(0.01,st["W"][ci] - ltd)   # LTD
        # Déclin passif vers W0
        st["W"][ci] = st["W"][ci]*(1-dec) + W0[ci]*dec


# ─── Rendu ────────────────────────────────────────────────────────────────────
def draw_network(screen, font, st, hud_h, net_w, panel_h):
    pygame.draw.rect(screen,(10,12,20),(0,hud_h,net_w,panel_h))
    for ci,(a,b,w0_,sign,lbl,plast) in enumerate(CONN_DEF):
        na=NEURONS[a]; nb=NEURONS[b]
        ax=int(na[4]*net_w); ay=int(na[5]*panel_h)+hud_h
        bx=int(nb[4]*net_w); by=int(nb[5]*panel_h)+hud_h
        W_cur=st["W"][ci]
        changed = abs(W_cur-w0_)>0.04
        if changed:
            col=(74,158,107) if W_cur>w0_ else (196,125,42)  # vert=LTP, orange=LTD
        else:
            col=(100,180,255) if sign>0 else (255,100,100)
        alpha = 200 if changed else 40
        lw    = max(1, int(W_cur*2.8*(1.6 if changed else 1)))
        s=pygame.Surface((net_w,panel_h),pygame.SRCALPHA)
        pygame.draw.line(s,(*col,alpha),(ax,ay-hud_h),(bx,by-hud_h),lw)
        screen.blit(s,(0,hud_h))
        if plast:
            mx,my=(ax+bx)//2,(ay+by)//2
            pcol=(74,158,107) if W_cur>w0_+0.04 else (196,125,42) if W_cur<w0_-0.04 else (80,80,80)
            pygame.draw.circle(screen,pcol,(mx,my),3)

    for (i,name,ntype,role,xr,yr) in NEURONS:
        x=int(xr*net_w); y=int(yr*panel_h)+hud_h
        col=TYPE_COL[ntype]
        pygame.draw.circle(screen,col,(x,y),7)
        pygame.draw.circle(screen,(255,255,255),(x,y),7,1)
        screen.blit(font.render(name,True,(200,200,200)),(x-10,y-18))

    # Légende
    for j,(c,l) in enumerate([(74,158,107,"LTP renforcé"),
                               (196,125,42,"LTD affaibli"),
                               (80,80,80,"stable/fixe")]):
        pygame.draw.rect(screen,(c,l[0] if isinstance(l,int) else 0,0) if False else
                         [(74,158,107),(196,125,42),(80,80,80)][j],
                         (6,hud_h+panel_h-50+j*16,12,4))
        screen.blit(font.render(["LTP renforcé","LTD affaibli","stable/fixe"][j],
                                True,(170,170,170)),(22,hud_h+panel_h-52+j*16))


def draw_synapse_bars(screen, font, st, hud_h, net_w, syn_w, panel_h):
    pygame.draw.rect(screen,(8,10,18),(net_w,hud_h,syn_w,panel_h))
    pygame.draw.line(screen,(40,40,60),(net_w,hud_h),(net_w,WIN_H),1)
    screen.blit(font.render("POIDS SYNAPTIQUES PLASTIQUES",True,(160,160,200)),
                (net_w+6,hud_h+4))
    plastic = [(ci,c) for ci,c in enumerate(CONN_DEF) if c[5]]
    for j,(ci,c) in enumerate(plastic):
        a,b,w0_,sign,lbl,_ = c
        W_cur = st["W"][ci]
        delta = W_cur - w0_
        # Barre de fond
        bar_x = net_w+8; bar_y = hud_h+22+j*26
        bar_max = syn_w-20
        pygame.draw.rect(screen,(25,28,35),(bar_x,bar_y,bar_max,12))
        # Valeur actuelle
        bar_w = int(min(1.5,W_cur)/1.5 * bar_max)
        col = (74,158,107) if delta>0.04 else (196,125,42) if delta<-0.04 else (58,127,193)
        if bar_w>0:
            pygame.draw.rect(screen,col,(bar_x,bar_y,bar_w,12))
        # Marqueur W0
        w0_x = int(w0_/1.5*bar_max)
        pygame.draw.line(screen,(255,255,255),(bar_x+w0_x,bar_y),(bar_x+w0_x,bar_y+12),1)
        # Labels
        d_col=(74,158,107) if delta>0.02 else (196,125,42) if delta<-0.02 else (120,120,120)
        screen.blit(font.render(f"{lbl[:12]:12s} {W_cur:.2f} ({delta:+.2f})",True,d_col),
                    (bar_x,bar_y+13))


def draw_timeline(screen, font, st, hud_h, net_w, syn_w, tl_w, panel_h):
    x0=net_w+syn_w
    pygame.draw.rect(screen,(10,10,16),(x0,hud_h,tl_w,panel_h))
    pygame.draw.line(screen,(40,40,60),(x0,hud_h),(x0,WIN_H),1)
    screen.blit(font.render("RÉPONSE À L'ODEUR",True,(160,200,160)),(x0+6,hud_h+4))
    screen.blit(font.render("(avant/après apprentissage)",True,(120,140,120)),(x0+6,hud_h+16))

    tl = st["timeline"]
    if len(tl)<2:
        screen.blit(font.render("En attente...",True,(80,80,80)),(x0+10,hud_h+panel_h//2))
        return

    pad=18; h_plot=panel_h-60
    max_t = max(p[0] for p in tl)+1
    phase_col={"baseline":(120,120,120),"train":(255,200,60),
               "test":(58,127,193),"extinct":(196,125,42),"idle":(60,60,60)}

    # Ligne zéro
    y0=hud_h+pad+h_plot
    pygame.draw.line(screen,(40,40,40),(x0+pad,y0),(x0+tl_w-pad,y0),1)

    for i in range(1,len(tl)):
        t1,r1,p1 = tl[i-1]; t2,r2,p2 = tl[i]
        x1=int(x0+pad+(t1/max_t)*(tl_w-2*pad))
        y1=int(y0 - (r1/1.0)*h_plot*0.85)
        x2=int(x0+pad+(t2/max_t)*(tl_w-2*pad))
        y2=int(y0 - (r2/1.0)*h_plot*0.85)
        col=phase_col.get(p2,(120,120,120))
        pygame.draw.line(screen,col,(x1,y1),(x2,y2),2)
        pygame.draw.circle(screen,col,(x2,y2),3)

    # Ligne seuil apprentissage (0.4)
    ythr=int(y0 - 0.4*h_plot*0.85)
    pygame.draw.line(screen,(80,80,80),(x0+pad,ythr),(x0+tl_w-pad,ythr),1)
    screen.blit(font.render("seuil 0.4",True,(80,80,80)),(x0+pad,ythr-12))

    # Légende phases
    for j,(ph,col) in enumerate(phase_col.items()):
        if ph=="idle": continue
        pygame.draw.rect(screen,col,(x0+6,hud_h+panel_h-55+j*11,10,3))
        screen.blit(font.render(ph,True,col),(x0+20,hud_h+panel_h-57+j*11))


def draw_hud5(screen, font, st, win_w, hud_h):
    s=pygame.Surface((win_w,hud_h),pygame.SRCALPHA)
    s.fill((0,0,0,175)); screen.blit(s,(0,0))
    ph_col={"idle":(80,80,80),"baseline":(120,120,120),"train":(255,200,60),
            "test":(58,127,193),"extinct":(196,125,42)}
    pc=ph_col.get(st["phase"],(120,120,120))
    resp = st["timeline"][-1][1] if st["timeline"] else 0
    ltp_n=sum(1 for ci,c in enumerate(CONN_DEF) if c[5] and st["W"][ci]>W0[ci]+0.04)
    ltd_n=sum(1 for ci,c in enumerate(CONN_DEF) if c[5] and st["W"][ci]<W0[ci]-0.04)

    lines=[
        (f"Phase:{st['phase'].upper():10s}  Essais:{st['trial']:3d}  "
         f"Dernière réponse:{resp:.2f}  LTP:{ltp_n} synapses  LTD:{ltd_n} synapses",
         pc),
        (f"LTP:{st['ltp_rate']:.3f}  LTD:{st['ltd_rate']:.3f}  "
         f"Déclin:{st['decay']:.3f}  Essais entr.:{st['n_train']}",
         (180,200,255)),
        (f"[1]baseline  [2]entrainer  [3]tester  [4]extinction  [R]reset  "
         f"[N]essai manuel  [↑↓]LTP  [←→]LTD  [+/-]declin",
         (140,140,140)),
        (f"Attendu: baseline→resp~0  après entr.→resp>0.4  test→recul  extinction→retour baseline",
         (120,160,120)),
    ]
    for i,(txt,col) in enumerate(lines):
        screen.blit(font.render(txt,True,col),(8,4+i*24))
    pygame.draw.rect(screen,pc,(4,0,4,hud_h))


# ─── Protocole ────────────────────────────────────────────────────────────────
def do_baseline(st):
    st["phase"]="baseline"
    for _ in range(3):
        r=run_trial(st,{1:0.55})   # AWC seul
        st["trial"]+=1
        st["timeline"].append((st["trial"],r,"baseline"))
    print(f"[Phase5] Baseline : réponse={r:.3f} (attendu ~0.0-0.10)")

def do_train_step(st):
    """Un essai d'entraînement : odeur + danger → LTP."""
    r=run_trial(st,{1:0.55, 2:0.92})   # AWC + ASH
    apply_plasticity(st, with_danger=True)
    st["trial"]+=1
    st["timeline"].append((st["trial"],r,"train"))
    print(f"[Phase5] Entr. essai {st['train_i']+1}/{st['n_train']}: réponse={r:.3f}")

def do_test(st):
    st["phase"]="test"
    for _ in range(4):
        r=run_trial(st,{1:0.60})   # AWC seul — le réseau a-t-il appris ?
        st["trial"]+=1
        st["timeline"].append((st["trial"],r,"test"))
    msg = "APPRENTISSAGE OK" if r>0.40 else "Apprentissage insuffisant (augmenter LTP ou essais)"
    print(f"[Phase5] Test : réponse={r:.3f} → {msg}")

def do_extinct_step(st):
    """Un essai d'extinction : odeur sans danger → LTD."""
    run_trial(st,{1:0.45})
    apply_plasticity(st, with_danger=False)
    r=run_trial(st,{1:0.55})
    st["trial"]+=1
    st["timeline"].append((st["trial"],r,"extinct"))


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen=pygame.display.set_mode((WIN_W,WIN_H))
    pygame.display.set_caption("Phase 5 — Mémoire & Apprentissage | C. elegans")
    font  =pygame.font.SysFont("monospace",11)
    clock =pygame.time.Clock()
    st    =fresh_state()

    print(__doc__)
    print("Protocole : [1] baseline → [2] entraîner → [3] tester → [4] extinction")

    running=True
    auto_action=None  # ("train",n_total) ou ("extinct",n_total)
    auto_i=0; auto_timer=0

    while running:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: running=False
            if ev.type==pygame.KEYDOWN:
                if ev.key in (pygame.K_q,pygame.K_ESCAPE): running=False
                elif ev.key==pygame.K_r:
                    st=fresh_state(); auto_action=None
                elif ev.key==pygame.K_1:
                    do_baseline(st)
                elif ev.key==pygame.K_2:
                    st["phase"]="train"; auto_action=("train",st["n_train"]); auto_i=0; auto_timer=0
                elif ev.key==pygame.K_3:
                    do_test(st)
                elif ev.key==pygame.K_4:
                    st["phase"]="extinct"; auto_action=("extinct",8); auto_i=0; auto_timer=0
                elif ev.key==pygame.K_n:
                    if st["phase"]=="train": do_train_step(st)
                    elif st["phase"]=="extinct": do_extinct_step(st)
                    else: do_train_step(st)
                elif ev.key==pygame.K_UP:
                    st["ltp_rate"]=min(0.20,round(st["ltp_rate"]+0.005,3))
                elif ev.key==pygame.K_DOWN:
                    st["ltp_rate"]=max(0.005,round(st["ltp_rate"]-0.005,3))
                elif ev.key==pygame.K_RIGHT:
                    st["ltd_rate"]=min(0.10,round(st["ltd_rate"]+0.005,3))
                elif ev.key==pygame.K_LEFT:
                    st["ltd_rate"]=max(0.002,round(st["ltd_rate"]-0.005,3))
                elif ev.key in (pygame.K_PLUS,pygame.K_EQUALS):
                    st["n_train"]=min(20,st["n_train"]+1)
                elif ev.key==pygame.K_MINUS:
                    st["n_train"]=max(2,st["n_train"]-1)
                elif ev.key==pygame.K_d:
                    st["decay"]=min(0.020,round(st["decay"]+0.001,3))

        # Auto-entraînement / extinction progressive
        if auto_action is not None:
            auto_timer+=1
            if auto_timer>=18:   # ~0.6s par essai à 30fps
                auto_timer=0
                action,n_total=auto_action
                if action=="train":
                    do_train_step(st); st["train_i"]=auto_i
                    auto_i+=1
                    if auto_i>=n_total: auto_action=None
                elif action=="extinct":
                    do_extinct_step(st)
                    auto_i+=1
                    if auto_i>=n_total: auto_action=None

        screen.fill((10,12,20))
        draw_network(screen,font,st,HUD_H,NET_W,PANEL_H)
        draw_synapse_bars(screen,font,st,HUD_H,NET_W,SYN_W,PANEL_H)
        draw_timeline(screen,font,st,HUD_H,NET_W,SYN_W,TL_W,PANEL_H)
        draw_hud5(screen,font,st,WIN_W,HUD_H)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__=="__main__":
    main()
