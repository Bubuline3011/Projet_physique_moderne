import numpy as np
import matplotlib.pyplot as plt

# Constantes physiques (unités naturelles : hbar = m = 1)
hbar = 1.0
m = 1.0
V0 = 10.0   # profondeur du puits
a = 1.0     # demi-largeur du puits

# Domaine d’énergie
E_values = np.linspace(0.01, 20, 500)
T_values = []

for E in E_values:
    k = np.sqrt(2 * m * E) / hbar
    q = np.sqrt(2 * m * (E + V0)) / hbar
    i = 1j

    # Calcul des exponentielles
    exp_mika = np.exp(-i * k * a)
    exp_miqa = np.exp(-i * q * a)
    exp_3iqa = np.exp(3 * i * q * a)
    exp_phase = np.exp(i * (q - k) * a)

    # Terme commun à B2
    terme = (2 * i * q) / (i * k + i * q) - 1

    # Dénominateur de l'expression de A2
    denom = i * q * (exp_miqa - exp_3iqa * terme) + i * k * (exp_miqa + exp_3iqa * terme)

    # Formule complète de T(E)
    T = np.abs((2 * i * k * exp_mika / denom) * ((2 * i * q) / (i * k + i * q)) * exp_phase) ** 2

    T_values.append(T)

# Tracé
plt.figure(figsize=(10, 6))
plt.plot(E_values, T_values, label='T(E)', color='blue')
plt.xlabel("Énergie E")
plt.ylabel("Transmission T(E)")
plt.title("Transmission T(E) — Formule analytique complète")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

