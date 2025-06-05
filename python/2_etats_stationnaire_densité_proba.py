import numpy as np
import matplotlib.pyplot as plt

# Constantes physiques réalistes
hbar = 1.055e-34  # J.s
m = 9.11e-31      # masse de l'électron (kg)
eV = 1.602e-19    # 1 eV en joules

# Paramètres du puits
V0 = 10 * eV      # profondeur du puits
a = 1e-10         # demi-largeur du puits (1 Å)

# Énergie de la particule
E = 50 * eV       # énergie de la particule (en J)

# Constantes d'onde
k = np.sqrt(2 * m * E) / hbar
q = np.sqrt(2 * m * (E + V0)) / hbar

# Amplitudes complexes
A1 = 1
terme = (2j * q) / (1j * k + 1j * q)
B2 = A1 * (terme - 1) * np.exp(2j * q * a)
A2 = -2j * k * np.exp(-1j * k * a) / (
    1j * q * (-np.exp(-1j * q * a) + np.exp(3j * q * a) * (terme - 1))
    - 1j * k * (np.exp(-1j * q * a) + np.exp(3j * q * a) * (terme - 1))
)
A3 = terme * A2 * np.exp(1j * q * a - 1j * k * a)
B1 = A1 - A2 * np.exp(-2j * k * a) - B2 * np.exp(-2j * k * a)

# Grille spatiale
x = np.linspace(-5*a, 5*a, 2000)  # en mètres
psi_squared = np.zeros_like(x)
V_x = np.zeros_like(x)

# Construction de la densité de probabilité
for i, xi in enumerate(x):
    if xi < -a:
        psi = A1 * np.exp(1j * k * xi) + B1 * np.exp(-1j * k * xi)
        V_x[i] = 0
    elif xi <= a:
        psi = A2 * np.exp(1j * q * xi) + B2 * np.exp(-1j * q * xi)
        V_x[i] = -V0
    else:
        psi = A3 * np.exp(1j * k * xi)
        V_x[i] = 0
    psi_squared[i] = np.abs(psi)**2

# Normalisation de la densité de probabilité (facultatif)
dx = x[1] - x[0]
psi_squared /= np.sum(psi_squared * dx)

# Mise à l’échelle du potentiel pour l'affichage
V_plot = V_x / V0 * np.max(psi_squared)

# Conversion des positions en Ångström
x_angstrom = x * 1e10

# Tracé
plt.figure(figsize=(12, 6))
plt.plot(x_angstrom, psi_squared, label=r"$|\psi(x)|^2$ (densité de probabilité normalisée)", color="blue")
plt.plot(x_angstrom, V_plot, label='Puits de potentiel (forme visuelle)', color='red', linestyle='--')
plt.axhline(y=np.max(psi_squared) * 0.5, color='green', linestyle=':', label=f'Énergie E = {E/eV:.1f} eV (indicative)')

# Marques des bords du puits
plt.axvline(-a * 1e10, color='gray', linestyle='--', label='Bords du puits')
plt.axvline(a * 1e10, color='gray', linestyle='--')

plt.title("Densité de probabilité normalisée et puits de potentiel (E > 0)")
plt.xlabel("Position x (Å)")
plt.ylabel("Densité de probabilité (normée)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
