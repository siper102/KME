import numpy as np
from framework.Embedding.HSCIC import HSCIC
import matplotlib.pyplot as plt

def eps():
    return np.random.normal(size=500) * 0.3
# Erzeugen der Daten
Z  = np.random.normal(size=500)
X = np.exp(-0.5*Z**2) * np.sin(2*Z) + eps()
Y_noise = np.random.normal(size = 500) * 0.3

Z_2  = np.random.normal(size=500)
X_2 = np.exp(-0.5*Z_2**2) * np.sin(2*Z_2) + eps()
Y_dep = np.exp(-0.5*Z_2**2) * np.sin(2*Z_2) + eps() + 0.2*X_2

Z_3  = np.random.normal(size=500)
X_3 = np.exp(-0.5*Z_3**2) * np.sin(2*Z_3) + eps()
Yp_dep = np.exp(-0.5*Z_3**2) * np.sin(2*Z_3) + eps() + 0.4*X_3
# Visualisieren der Daten
plt.scatter(Z, X)
plt.scatter(Z, Y_noise)
plt.scatter(Z_2, Y_dep)
plt.scatter(Z_3, Yp_dep)
plt.legend(["X", "$Y_{noise}$", "$Y^{\prime}_{dep}$", "$Y_{dep}$"], fontsize = 10)
plt.show()

# Initialisieren der Schätzer
hscic = HSCIC()
hscic_2 = HSCIC()
hscic_3 = HSCIC()
# Anpassen der Schätzer
hscic.fit(X, Y_noise, Z)
hscic_2.fit(X_2, Y_dep, Z_2)
hscic_3.fit(X_3, Yp_dep, Z_3)
# Resultat visualisieren
xp = np.linspace(-2, 2, 200)
plt.plot(xp, hscic(xp))
plt.plot(xp, hscic_2(xp))
plt.plot(xp, hscic_3(xp))
plt.legend(["HSCIC(X, $Y_{noise}$, Z)", "HSCIC(X, $Y_{dep}$, Z)", "$HSCIC(X, Y^{\prime}_{dep}, Z)$"], fontsize = 10)
plt.xlabel("Z")
plt.show()
