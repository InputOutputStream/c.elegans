"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 3 — SYNAPTOGENÈSE                                                    ║
║  Formation des premières connexions synaptiques dans C. elegans             ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ BIOLOGIE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Les cellules mex-3 (zone antérieure, phase 2) ont reçu leur identité
  neuronale. Elles entrent maintenant en SYNAPTOGENÈSE :

  1. CÔNE DE CROISSANCE (Growth Cone)
     Chaque neurone émet un axone terminé par un cône exploratoire.
     Le cône avance par cycles d'extension/rétraction pilotés par l'actine.
     Durée réelle dans C. elegans : quelques heures à la température ambiante.

  2. CHIMIOTAXIE AXONALE
     Des molécules attractives (nétrine/UNC-6, sémaphorine/MAB-20) créent
     des gradients que le cône suit vers ses cibles génétiquement déterminées.
     Règle : les neurones mex-3 préfèrent se connecter entre eux (homo-typique)
     puis aux interneurones (AIY, AIZ) identifiés dans le connectome réel.

  3. RÈGLE DE HEBB (1949)
     "Cells that fire together wire together"
     Deux neurones co-actifs répétitivement renforcent leur synapse.
     Mécanisme moléculaire : insertion de récepteurs AMPA post-synaptiques.
     Dans C. elegans : médié par GLR-1 (glutamate receptor).

  4. ÉLAGAGE SYNAPTIQUE (Synaptic Pruning)
     Les synapses inactives sont éliminées par les cellules gliales.
     Dans C. elegans : ~30% des contacts initiaux sont élaguées.
     Résultat : le connectome adulte de 7341 synapses est stable et reproductible.

━━━ INFORMATIQUE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Chaque neurone est un agent avec :
    · position (x, y) dans le canvas
    · potentiel membranaire V ∈ [0,1]
    · seuil de décharge θ ∈ [0.5, 0.7] (hétérogénéité biologique)
    · axone : angle θ_ax, longueur L_ax ≤ reach
    · état firing (bool) + timer réfractaire

  Règle de Hebb discrète :
    w += η  si  neurone_A.firing AND neurone_B.firing
    w -= δ  sinon (déclin passif)
  avec η = hebb_rate, δ = 0.002 (oubli lent)

  Détection de contact : ||tip_A − soma_B|| < contact_radius → synapse créée

━━━ COMPORTEMENTS ATTENDUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CHIMIOTAXIE forte (>0.7) :
    → axones convergent directement vers les cibles mex-3
    → réseau dense et centré sur la zone antérieure (bleu)
    Biologie : sur-expression de UNC-6 (nétrine)

  CHIMIOTAXIE faible (<0.2) :
    → exploration aléatoire, axones divergents
    → contacts synaptiques rares et distribués aléatoirement
    Biologie : mutant unc-6 de C. elegans (désorientation axonale connue)

  BRUIT élevé (>0.8) :
    → axones erratiques, spirales, rebroussements
    Biologie : perturbation du cytosquelette du cône de croissance

  PORTÉE courte (<40px) :
    → peu de synapses, réseau fragmenté
    Biologie : déficit d'élongation axonale (mutant unc-76)

  HEBB élevé (>0.10) :
    → consolidation rapide, synapses blanches dès les premières co-activations
    Biologie : sur-expression de GLR-1 (glutamate receptor)

  ÉLAGAGE :
    → supprime les synapses < seuil_pruning
    → le réseau se clarifie, les hubs mex-3 restent, les connexions faibles disparaissent
    Attendu : ~30% de réduction du nombre de synapses

━━━ CONTRÔLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ESPACE → pause/reprise   R → reset   S → sauvegarder
  P      → élagage synaptique (pruning)
  ↑↓     → chimiotaxie     ←→ → portée axone
  +/-    → hebb rate       N  → +5 neurones
  Q/ESC  → quitter

━━━ DÉPENDANCES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install pygame numpy
"""

import numpy as np
import pygame, sys, os

# ─── Paramètres ──────────────────────────────────────────────────────────────
WIN          = 760
HUD_H        = 100
N_NEURONS    = 24
CHEMO        = 0.55
REACH        = 90
HEBB_RATE    = 0.04
AXO_SPEED    = 1.6
NOISE        = 0.30
FIRE_PROB    = 0.025
PRUNE_THR    = 0.25
CONTACT_R    = 18.0

GENE_COL = {
    "mex": (58,  127, 193),   # bleu
    "pie": (123, 94,  167),   # violet
    "pal": (74,  158, 107),   # vert
    "skn": (196, 125, 42),    # ambre
}
# Zones mex-3 (antérieures) — issues de la phase 2
MEX3_ZONES = [
    (0.22, 0.33, 0.17),  # (cx, cy, r) en fraction de WIN
    (0.18, 0.65, 0.13),
    (0.32, 0.50, 0.11),
]

# ─── Classes ──────────────────────────────────────────────────────────────────
class Neuron:
    """
    Agent neuronal avec dynamique de potentiel d'action simplifiée.
    Le modèle Leaky-Integrate-and-Fire (LIF) discret est utilisé :
      V(t+1) = V(t) × decay + input
      Si V ≥ θ → spike, V=0, timer_refractaire=5 steps
    """
    _id_counter = 0
    def __init__(self, x, y, gene_type):
        self.id         = Neuron._id_counter; Neuron._id_counter += 1
        self.x, self.y  = x, y
        self.gtype      = gene_type
        self.V          = np.random.uniform(0.0, 0.25)
        self.theta      = np.random.uniform(0.50, 0.70)
        self.firing     = False
        self.ref_timer  = 0
        self.ax_angle   = np.random.uniform(0, 2*np.pi)
        self.ax_len     = 0.0
        self.connected  = False

    @property
    def tip(self):
        return (self.x + np.cos(self.ax_angle)*self.ax_len,
                self.y + np.sin(self.ax_angle)*self.ax_len)

    def update(self, others, chemo, reach, noise):
        # Réfractaire
        if self.ref_timer > 0:
            self.ref_timer -= 1; self.firing = False; return
        # Bruit thermique + activation spontanée
        self.V = np.clip(self.V*0.92 + np.random.uniform(0, 0.06), 0, 1)
        if np.random.random() < FIRE_PROB:
            self.V = min(1.0, self.V + 0.15)
        if self.V >= self.theta:
            self.firing = True; self.V = 0.0; self.ref_timer = 5
        else:
            self.firing = False
        # Chimiotaxie axonale
        if self.ax_len < reach:
            best_score, best_angle = -np.inf, self.ax_angle
            for t in others:
                if t.id == self.id: continue
                d = np.hypot(t.x-self.x, t.y-self.y)
                if 0 < d < reach*1.5:
                    ang   = np.arctan2(t.y-self.y, t.x-self.x)
                    score = chemo/d * 100
                    if t.gtype == "mex" and self.gtype == "mex":
                        score += 25  # préférence homo-typique mex-3
                    if score > best_score:
                        best_score, best_angle = score, ang
            self.ax_angle = self.ax_angle*0.82 + best_angle*0.18
            self.ax_angle += np.random.uniform(-noise/2, noise/2)
            self.ax_len    = min(reach, self.ax_len + AXO_SPEED)


class Synapse:
    """
    Connexion synaptique avec plasticité Hebbienne.
    force ∈ [0,1] : 0=inexistante, <0.25=faible(jaune), >0.5=établie(blanc)
    """
    def __init__(self, a_id, b_id):
        self.a       = a_id
        self.b       = b_id
        self.force   = 0.10
        self.age     = 0

    def update(self, na, nb, hebb):
        self.age += 1
        if na.firing and nb.firing:
            self.force = min(1.0, self.force + hebb)   # LTP
        else:
            self.force = max(0.0, self.force - 0.002)  # déclin passif


# ─── Placement des neurones ────────────────────────────────────────────────────
def place_neurons(n_total, win, rng=None):
    if rng is None: rng = np.random.default_rng()
    Neuron._id_counter = 0
    neurons = []
    n_mex = int(n_total * 0.60)

    # Neurones mex-3 dans les zones antérieures
    while len(neurons) < n_mex:
        z = MEX3_ZONES[rng.integers(len(MEX3_ZONES))]
        a = rng.uniform(0, 2*np.pi)
        d = rng.uniform(0, z[2]*win)
        x = z[0]*win + np.cos(a)*d
        y = z[1]*win + np.sin(a)*d
        if 20 < x < win-20 and 20 < y < win-20:
            neurons.append(Neuron(x, y, "mex"))

    # Autres types (cibles synaptiques)
    types = ["pie","pal","skn"]
    i = 0
    while len(neurons) < n_total:
        t = types[i % 3]
        x = win*0.45 + rng.uniform(0, win*0.50)
        y = 20 + rng.uniform(0, win-40)
        if 20 < x < win-20:
            neurons.append(Neuron(x, y, t))
        i += 1
    return neurons


# ─── Synapses ─────────────────────────────────────────────────────────────────
def check_new_synapses(neurons, synapses):
    existing = {(s.a,s.b) for s in synapses} | {(s.b,s.a) for s in synapses}
    for na in neurons:
        tx, ty = na.tip
        for nb in neurons:
            if nb.id <= na.id: continue
            if np.hypot(nb.x-tx, nb.y-ty) < CONTACT_R:
                if (na.id, nb.id) not in existing:
                    synapses.append(Synapse(na.id, nb.id))
                    existing.add((na.id, nb.id))
                    na.connected = nb.connected = True

def prune_synapses(synapses, thr=PRUNE_THR):
    before = len(synapses)
    synapses[:] = [s for s in synapses if s.force >= thr]
    return before - len(synapses)


# ─── Rendu ────────────────────────────────────────────────────────────────────
def draw_scene(screen, neurons, synapses, win, hud_h):
    # Fond sombre (environnement de l'embryon)
    pygame.draw.rect(screen, (10,14,22), (0, hud_h, win, win))

    # Zones mex-3 en fond (ectoplasme antérieur)
    for (cx,cy,r) in MEX3_ZONES:
        s = pygame.Surface((win,win), pygame.SRCALPHA)
        pygame.draw.circle(s,(58,127,193,18),(int(cx*win),int(cy*win)),int(r*win))
        screen.blit(s,(0,hud_h))

    # Synapses
    for syn in synapses:
        na = next((n for n in neurons if n.id==syn.a), None)
        nb = next((n for n in neurons if n.id==syn.b), None)
        if na is None or nb is None: continue
        alpha = min(255, int(syn.force*1.5*255))
        col   = (255,255,255) if syn.force>0.5 else (255,200,60)
        lw    = max(1, int(syn.force * (2.5 if syn.force>0.5 else 1.5)))
        s     = pygame.Surface((win,win), pygame.SRCALPHA)
        pygame.draw.line(s, (*col, alpha),
                         (int(na.x), int(na.y)),
                         (int(nb.x), int(nb.y)), lw)
        screen.blit(s, (0,hud_h))
        if syn.force > 0.5:
            mx,my = int((na.x+nb.x)/2), int((na.y+nb.y)/2)
            pygame.draw.circle(screen,(255,255,255),(mx,my+hud_h),2)

    # Axones
    for n in neurons:
        if n.ax_len < 5: continue
        tx,ty = n.tip
        col   = GENE_COL.get(n.gtype,(128,128,128))
        s     = pygame.Surface((win,win), pygame.SRCALPHA)
        pygame.draw.line(s,(*col,90),(int(n.x),int(n.y)),(int(tx),int(ty)),1)
        screen.blit(s,(0,hud_h))
        pygame.draw.circle(screen,col,(int(tx),int(ty)+hud_h),3)

    # Corps neuronaux
    for n in neurons:
        col = GENE_COL.get(n.gtype,(128,128,128))
        r   = 9 if n.firing else 6
        if n.firing:
            s = pygame.Surface((win,win),pygame.SRCALPHA)
            pygame.draw.circle(s,(*col,50),(int(n.x),int(n.y)),16)
            screen.blit(s,(0,hud_h))
        pygame.draw.circle(screen,col,(int(n.x),int(n.y)+hud_h),r)
        pygame.draw.circle(screen,(255,255,255,80),(int(n.x),int(n.y)+hud_h),r,1)


def draw_hud3(screen, font, neurons, synapses, chemo, reach, hebb,
              paused, tick, win, hud_h):
    s = pygame.Surface((win,hud_h),pygame.SRCALPHA)
    s.fill((0,0,0,170)); screen.blit(s,(0,0))
    n_syn  = len(synapses)
    n_axo  = sum(1 for n in neurons if n.ax_len > 10)
    avg_f  = (sum(s.force for s in synapses)/n_syn) if n_syn else 0
    n_fire = sum(1 for n in neurons if n.firing)

    phase = ("exploration" if tick<30 else
             "premiers contacts" if n_syn<5 else
             "renforcement Hebb" if avg_f<0.30 else
             "reseau en formation" if avg_f<0.60 else
             "RESEAU STABLE → pret phase 4")

    lines = [
        (f"t={tick}  Phase:{phase}  {'[PAUSE]' if paused else '[RUN]  '}",
         (220,220,220)),
        (f"Neurones:{len(neurons)}  Axones actifs:{n_axo}  "
         f"Synapses:{n_syn}  Force moy:{avg_f:.2f}  Actifs:{n_fire}",
         (180,200,255)),
        (f"Chimio:{chemo:.2f}  Portee:{reach:.0f}px  Hebb:{hebb:.3f}  "
         f"[P]elaguer  [N]+neurones  [↑↓]chimio  [←→]portee  [+/-]hebb",
         (140,140,140)),
        (f"mex(bleu)={sum(1 for n in neurons if n.gtype=='mex')}  "
         f"pie(violet)={sum(1 for n in neurons if n.gtype=='pie')}  "
         f"pal(vert)={sum(1 for n in neurons if n.gtype=='pal')}  "
         f"skn(ambre)={sum(1 for n in neurons if n.gtype=='skn')}",
         (120,160,120)),
    ]
    for i,(txt,col) in enumerate(lines):
        screen.blit(font.render(txt,True,col),(8,4+i*22))

    # Mini barre synaptique
    if n_syn > 0:
        bar_w = min(win-20, n_syn*8)
        pygame.draw.rect(screen,(255,200,60),(10,hud_h-8,bar_w,5))


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN, WIN+HUD_H))
    pygame.display.set_caption("Phase 3 — Synaptogenèse | C. elegans")
    font  = pygame.font.SysFont("monospace",12)
    clock = pygame.time.Clock()

    rng      = np.random.default_rng()
    neurons  = place_neurons(N_NEURONS, WIN, rng)
    synapses = []
    tick     = 0
    paused   = False
    chemo    = CHEMO
    reach    = float(REACH)
    hebb     = HEBB_RATE

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
                    neurons = place_neurons(N_NEURONS,WIN,rng)
                    synapses=[]; tick=0
                elif ev.key == pygame.K_s:
                    n = len(neurons)
                    adj = np.zeros((n,n),dtype=np.float32)
                    for s in synapses:
                        if s.a<n and s.b<n:
                            adj[s.a,s.b]=adj[s.b,s.a]=s.force
                    np.save("phase3_network.npy",adj)
                    print(f"[Phase3] Sauvegardé phase3_network.npy ({n} neurones, {len(synapses)} synapses)")
                elif ev.key == pygame.K_p:
                    n = prune_synapses(synapses)
                    print(f"[Phase3] Élagage : {n} synapses supprimées, {len(synapses)} restantes")
                elif ev.key == pygame.K_n:
                    neurons += place_neurons(5,WIN,rng)[N_NEURONS:]
                elif ev.key == pygame.K_UP:
                    chemo = min(1.0, round(chemo+0.05,2))
                elif ev.key == pygame.K_DOWN:
                    chemo = max(0.05,round(chemo-0.05,2))
                elif ev.key == pygame.K_RIGHT:
                    reach = min(180,reach+5)
                elif ev.key == pygame.K_LEFT:
                    reach = max(20,reach-5)
                elif ev.key in (pygame.K_PLUS,pygame.K_EQUALS):
                    hebb = min(0.15,round(hebb+0.005,3))
                elif ev.key == pygame.K_MINUS:
                    hebb = max(0.005,round(hebb-0.005,3))

        if not paused:
            id_map = {n.id: n for n in neurons}
            for n in neurons:
                n.update(neurons, chemo, reach, NOISE)
            check_new_synapses(neurons, synapses)
            for s in synapses:
                na = id_map.get(s.a); nb = id_map.get(s.b)
                if na and nb: s.update(na, nb, hebb)
            tick += 1

        screen.fill((10,14,22))
        draw_scene(screen, neurons, synapses, WIN, HUD_H)
        draw_hud3(screen, font, neurons, synapses, chemo, reach, hebb,
                  paused, tick, WIN, HUD_H)
        pygame.display.flip()
        clock.tick(30)

    n = len(neurons)
    adj = np.zeros((n,n),dtype=np.float32)
    for s in synapses:
        if s.a<n and s.b<n:
            adj[s.a,s.b]=adj[s.b,s.a]=s.force
    np.save("phase3_network.npy",adj)
    print(f"[Phase3] Sauvegardé automatiquement")
    pygame.quit()

if __name__ == "__main__":
    main()
