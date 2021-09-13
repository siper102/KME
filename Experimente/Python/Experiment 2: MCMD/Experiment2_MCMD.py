import numpy as np
import matplotlib.pyplot as plt
from framework.Embedding.MCMD import MCMD

np.random.seed(22)
n = 500

# 1. Realisierungen von Y und X
Y_1 = np.random.normal(size = 500)
X_1 = np.exp(-0.5 * Y_1**2) * np.sin(2 * Y_1) + 0.3 * np.random.normal(size = 500)
# 2. Realsierungen von Y und X_same
Y_2 =  np.random.normal(size = 500)
X_same = np.exp(-0.5 * Y_2 ** 2) * np.sin(2 * Y_2) + 0.3 * np.random.normal(size = 500)
# 3. Realisierungen von Y und X_diff
Y_3 = np.random.normal(size = 500)
X_diff = Y_3 + 0.3 * np.random.normal(size = 500)

# Daten visualisieren
plt.scatter(Y_1, X_1)
plt.scatter(Y_2, X_same, c ="r", marker="+", )
plt.scatter(Y_3, X_diff, c ="k", marker ="*")
plt.legend(["X", "$X_{same}$", "$X_{diff}$"])
plt.xlabel("Y")
plt.xlabel("X")
plt.savefig("/Users/simonperschel/Dropbox/Bachelorarbeit Simon/Arbeit/Abgabe/Images/Scatter_MCMD.pdf")
plt.show()

# Schätzer erstellen
mcmd_1 = MCMD()
mcmd_2 = MCMD()

# Schätzer anpassen
mcmd_1.fit(X_1, X_same, Y_1, Y_2)
mcmd_2.fit(X_1, X_diff, Y_1, Y_3)

# Resultat visualisieren
xp = np.linspace(-3, 3, 200)
plt.plot(xp, mcmd_1(xp))
plt.plot(xp, mcmd_2(xp))

plt.legend(["$\widehat{MCMD}^{2}(P^{X|Y},P^{X_{same}|Y}, H_{\mathcal{X}})$",
            "$\widehat{MCMD}^{2}(P^{X|Y}, P^{X_{diff}|Y}, H_{\mathcal{X}})$"], loc = "upper center",
           fontsize = 15)
plt.xlabel("Z")
plt.savefig("/Users/simonperschel/Dropbox/Bachelorarbeit Simon/Arbeit/Abgabe/Images/Experiment_MCMD.pdf")
plt.show()