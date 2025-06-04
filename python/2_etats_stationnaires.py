import numpy as np
import matplotlib.pyplot as plt

# === Paramètres physiques (unités réduites) ===
hbar = 1.0  # constante de Planck réduite
m = 1.0     # masse de la particule

# === Discrétisation spatiale ===
xmin, xmax = -50, 50      # bornes de la boîte spatiale
N = 500                   # nombre de points pour discrétiser l'espace
x = np.linspace(xmin, xmax, N)  # grille d'espace
dx = x[1] - x[0]                # pas d'espace

# === Définition du potentiel : puits carré fini ===
V0 = 50.0    # profondeur du puits (positive mais utilisée comme négative)
a = 20.0     # largeur du puits
V = np.zeros(N)  # potentiel initialisé à 0 partout
V[np.abs(x) <= a / 2] = -V0  # à l’intérieur du puits, V(x) = -V0

# === Construction du Laplacien (approximation par différences finies) ===
# Laplacien 1D : -2 au centre, +1 sur les côtés
main_diag = np.full(N, -2.0) / dx**2
off_diag = np.full(N - 1, 1.0) / dx**2
laplacian = (
    np.diag(main_diag) +
    np.diag(off_diag, k=1) +
    np.diag(off_diag, k=-1)
)

# === Construction de l'Hamiltonien H = T + V ===
# T = opérateur d'énergie cinétique, V = potentiel
H = -(hbar**2) / (2 * m) * laplacian + np.diag(V)

# === Diagonalisation de H : résout Hψ = Eψ ===
energies, wavefuncs = np.linalg.eigh(H)  # valeurs et vecteurs propres

# === Paramètres d’affichage ===
num_states = 10  # nombre d’états stationnaires à afficher
scale = 5        # facteur d’échelle pour mieux voir ψ(x)

# Calcul des bornes verticales du graphe pour bien cadrer les courbes
e_min = energies[0]
e_max = energies[num_states - 1]
v_margin = 4  # marge au-dessus et au-dessous des courbes

# === Affichage graphique des 10 premiers états stationnaires ===
plt.figure(figsize=(12, 7))
for i in range(num_states):
    psi = wavefuncs[:, i]
    # Normalisation discrète : somme(|ψ|^2) dx = 1
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
    # Affichage : ψ(x) décalée verticalement de E_n
    plt.plot(x, scale * psi + energies[i], label=f"n={i}, E={energies[i]:.2f}")

# Ajout du potentiel en pointillés noirs
plt.plot(x, V, 'k--', label='Potentiel')

# === Mise en forme du graphique ===
plt.xlabel('Position x')
plt.ylabel('Énergie + ψ(x) (échelle visuelle)')
plt.title('États stationnaires n = 0 à 9 dans un puits carré fini')
plt.ylim(e_min - v_margin, e_max + v_margin)  # fenêtre verticale propre
plt.legend()
plt.grid(True)
plt.show()
