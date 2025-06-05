import numpy as np
import matplotlib.pyplot as plt

# Constantes physiques réalistes
hbar = 1.055e-34  # J.s
m = 9.11e-31      # masse de l'électron en kg
eV = 1.602e-19    # 1 eV en joules

# Paramètres du puits
V0 = 10 * eV      # profondeur du puits en joules
a = 1e-10         # demi-largeur du puits en mètres (1 Å)

# Énergie choisie
E = 20 * eV        # énergie de la particule en joules

# Constantes d'onde
k = np.sqrt(2 * m * E) / hbar
q = np.sqrt(2 * m * (E + V0)) / hbar

# On pose A1 = 1
A1 = 1

# Expressions complexes trouvées à la main :
terme = (2j * q) / (1j * k + 1j * q)
B2 = A1 * (terme - 1) * np.exp(2j * q * a)
A2 = -2j * k * np.exp(-1j * k * a) / (
    1j * q * (-np.exp(-1j * q * a) + np.exp(3j * q * a) * (terme - 1))
    - 1j * k * (np.exp(-1j * q * a) + np.exp(3j * q * a) * (terme - 1))
)
A3 = terme * A2 * np.exp(1j * q * a - 1j * k * a)
B1 = A1 - A2 * np.exp(-2j * k * a) - B2 * np.exp(-2j * k * a)

# Grille spatiale
x = np.linspace(-3*a, 3*a, 1000)
phi_real = np.zeros_like(x, dtype=np.float64)
V_x = np.zeros_like(x)

# Construction de la solution réelle par morceaux + V(x)
for i, xi in enumerate(x):
    if xi < -a:
        phi = A1 * np.exp(1j * k * xi) + B1 * np.exp(-1j * k * xi)
        V_x[i] = 0
    elif xi <= a:
        phi = A2 * np.exp(1j * q * xi) + B2 * np.exp(-1j * q * xi)
        V_x[i] = -V0
    else:
        phi = A3 * np.exp(1j * k * xi)
        V_x[i] = 0
    phi_real[i] = np.real(phi)

# Tracé
plt.figure(figsize=(10, 6))
plt.plot(x, phi_real, label=f"Re[φ(x)] pour E = {E/eV:.1f} eV", color="blue")

# Puits de potentiel (mis à l’échelle pour la visibilité)
plt.plot(x, V_x / V0, label='Puits de potentiel (échelle réduite)', color='red', linestyle='--')

# Ligne de niveau de l’énergie E
plt.axhline(y=0.5, color='green', linestyle=':', label=f'Énergie E = {E/eV:.1f} eV (visuel)')

# Marques pour les bords
plt.axvline(-a, color='gray', linestyle='--', label='Bords du puits')
plt.axvline(a, color='gray', linestyle='--')

plt.title("Fonction d’onde stationnaire réelle et puits de potentiel")
plt.xlabel("Position x")
plt.ylabel("Amplitude ψ(x) (échelle arbitraire)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

