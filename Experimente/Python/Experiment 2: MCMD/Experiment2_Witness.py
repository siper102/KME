from framework.Embedding.CME import CME
import numpy as np
import matplotlib.pyplot as plt

n = 500
np.random.seed(22)

# 1. Realisierungen von Y und X
Y_1 = np.random.normal(size=n)
X_1 = np.exp(-1/2*Y_1**2) * np.sin(2*Y_1) + 0.3 * np.random.normal(size=n)
# 2. Realisierungen von Y und X_diff
Y_2 = np.random.normal(size=n)
Xp_diff = Y_2 + 0.3 * np.random.normal(size=n)
# 3. Realisierungen von Y und X
Y_3 = np.random.normal(size=n)
X_3 = np.exp(-1/2*Y_3**2) * np.sin(2*Y_3) + 0.3 * np.random.normal(size=n)
# 4. Realisierungen von Y und X_same
Y_4 = np.random.normal(size=n)
Xp_same = np.exp(-1 / 2 * Y_4 ** 2) * np.sin(2 * Y_4) + 0.3 * np.random.normal(size=n)

# Schätzer erstellen
f_1 = CME()
f_2 = CME()
f_3 = CME()
f_4 = CME()
# Schätzer anpassen
f_1.fit(X_1, Y_1)
f_2.fit(Xp_diff, Y_2)
f_3.fit(X_3, Y_3)
f_4.fit(Xp_same, Y_4)

# Für alle (x, y) aus [-3, 3] x [-3, 3] den Wert F_1(x,y) - F_2(x,y) ausrechnen
xp = np.arange(-3, 3, 0.1)
yp = np.arange(-3, 3, 0.1)
wit_diff = f_1(xp, yp) - f_2(xp, yp)
wit_same = f_3(xp, yp) - f_4(xp, yp)


# Resultat plotten:
maxi = max(np.max(wit_same), np.max(wit_diff))
mini = min(np.min(wit_same), np.min(wit_diff))

im_diff = plt.imshow(wit_diff, cmap='viridis', interpolation='nearest',
                     extent=[yp[0], yp[-1], xp[0], xp[-1]], vmin=mini, vmax=maxi,)
plt.title("X|Y vs. $X_{diff}$|Y",fontsize = 15)
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()

im_same = plt.imshow(wit_same, cmap='viridis', interpolation='nearest',
                     extent=[yp[0], yp[-1], xp[0], xp[-1]], vmin=mini, vmax=maxi,)
plt.title("X|Y vs. $X_{same}$|Y", fontsize = 20)
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()
