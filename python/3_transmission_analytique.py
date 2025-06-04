import numpy as np
import matplotlib.pyplot as plt

# Constantes physiques (unités naturelles hbar = m = 1)
hbar = 1.0
m = 1.0

# Paramètres du puits
V0 = 10.0  # profondeur du puits (positive)
a = 1.0    # demi-largeur du puits

# Domaine d’énergie
E_values = np.linspace(0.01, 20, 500)
T_values = []

for E in E_values:
    k = np.sqrt(2 * m * E) / hbar
    q = np.sqrt(2 * m * (E + V0)) / hbar

    # Éviter les divisions par zéro si k ou q est nul (très basse énergie)
    if k == 0 or q == 0:
        T_values.append(0)
        continue

    # Expression analytique de A3 trouvée à partir des conditions de continuité

    # Pour simplifier on fixe A1 = 1
    A1 = 1
    exp_iqa = np.exp(1j * q * a)
    exp_ika = np.exp(1j * k * a)

    # Calcul de A2 à partir du système qui a été fait a l'ecrit
    num_A2 = -2j * k * np.exp(-1j * k * a)

    terme = (2j * q) / (1j * k + 1j * q) - 1
    exp_miqa = np.exp(-1j * q * a)
    exp_3iqa = np.exp(3j * q * a)

    den_A2 = 1j * q * (-exp_miqa + exp_3iqa * terme) - 1j * k * (exp_miqa + exp_3iqa * terme)

    A2 = num_A2 / den_A2

    # Expression correcte de A3
    A3 = (2j * q / (1j * k + 1j * q)) * A2 * np.exp(1j * q * a - 1j * k * a)


    # Transmission : T = |A3|²
    T = np.abs(A3)**2
    T_values.append(T)

# Tracé de T(E)
plt.figure(figsize=(10, 6))
plt.plot(E_values, T_values, label='T(E)', color='blue')
plt.xlabel("Énergie E")
plt.ylabel("Transmission T(E)")
plt.title("Coefficient de transmission en fonction de l'énergie")
plt.grid(True)
plt.legend()
plt.show()
