import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# === Paramètres numériques pour la simulation ===
dt = 1E-7            # Pas de temps
dx = 0.001           # Pas d'espace
nx = int(1 / dx) * 2 # Nombre de points spatiaux
nt = 90000           # Nombre total de pas de temps simulés
nd = int(nt / 1000) + 1  # Nombre d'images sauvegardées pour l’animation
n_frame = nd             # Alias pour le nombre de frames
s = dt / (dx**2)         # Coefficient utilisé dans le schéma numérique

# === Paramètres du paquet d’onde initial ===
xc = 0.6                 # Position initiale du centre du paquet
sigma = 0.05             # Largeur du paquet
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))  # Normalisation
v0 = -4000               # Profondeur du puits de potentiel (négatif = attractif)
e = 5                    # Rapport E / V0
E = e * v0               # Énergie du paquet d’onde
k = math.sqrt(2 * abs(E))  # Nombre d’onde associé à cette énergie

# === Discrétisation de l’espace ===
o = np.linspace(0, (nx - 1) * dx, nx)  # Grille spatiale de 0 à ~2

# === Création du potentiel : puits rectangulaire entre x = 0.8 et x = 0.9 ===
V = np.zeros(nx)  # Potentiel nul partout
V[(o >= 0.8) & (o <= 0.9)] = v0  # Potentiel = v0 dans la zone du puits

# === Initialisation de l’onde complexe ψ(x, 0) = paquet gaussien modulé ===
cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * sigma ** 2))

# === Préparation des tableaux pour stocker la densité de probabilité ===
densite = np.zeros((nt, nx))             # Densité de probabilité à chaque instant
densite[0, :] = np.abs(cpt[:]) ** 2      # Densité initiale |ψ|²
final_densite = np.zeros((n_frame, nx))  # Sous-échantillonnage pour animation

# === Séparation de ψ en parties réelle et imaginaire ===
re = np.real(cpt)    # Partie réelle
im = np.imag(cpt)    # Partie imaginaire
b = np.zeros(nx)     # Stock temporaire utilisé dans le calcul

# === Boucle temporelle : propagation du paquet d’onde ===
it = 0
for i in range(1, nt):
    if i % 2 != 0:
        # Mise à jour de la partie imaginaire
        b[1:-1] = im[1:-1]
        im[1:-1] += s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i, 1:-1] = re[1:-1]**2 + im[1:-1] * b[1:-1]
    else:
        # Mise à jour de la partie réelle
        re[1:-1] -= s * (im[2:] + im[:-2]) - 2 * im[1:-1] * (s + V[1:-1] * dt)

    # Sauvegarde d’un snapshot toutes les 1000 itérations
    if (i - 1) % 1000 == 0:
        final_densite[it, :] = densite[i, :]
        it += 1

# === Fonctions pour l’animation matplotlib ===
def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j, :])  # Met à jour la courbe à la frame j
    return line,

# === Création de la figure d’animation ===
plot_title = f"Propagation du paquet d'onde avec E/V₀ = {e}"
fig = plt.figure()
line, = plt.plot([], [])  # Ligne initialement vide
plt.ylim(0, 13)            # Axe Y : densité max visible
plt.xlim(0, 2)             # Axe X : position
plt.plot(o, V, 'k--', label="Potentiel")  # Affichage du profil de potentiel
plt.title(plot_title)
plt.xlabel("x")
plt.ylabel("Densité de probabilité de présence")
plt.legend()

# === Lancement de l’animation ===
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=n_frame, blit=False, interval=100, repeat=False
)
plt.show()
