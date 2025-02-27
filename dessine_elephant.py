import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# ------------- PARTIE 1 : Extraction des contours -------------
# Charger l'image en niveaux de gris
image = cv2.imread("elephant.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise IOError("L'image 'elephant.png' n'a pas été trouvée.")

# Seuillage pour binariser l'image
_, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Détection des contours avec Canny
edges = cv2.Canny(thresh, 100, 200)

# Trouver les contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extraire tous les points de tous les contours
coordinates = []
for contour in contours:
    for point in contour:
        x, y = point[0]
        coordinates.append((x, y))

# Affichage du contour dans un repère classique :
# On transforme y en y' = image.shape[0] - y pour que y augmente vers le haut
x_coords, y_coords = zip(*coordinates)
y_coords_corr = [image.shape[0] - y for y in y_coords]
plt.figure(figsize=(8, 8))
plt.plot(x_coords, y_coords_corr, marker='.', linestyle='-', color='b', markersize=1)
plt.title("Contours de l'éléphant (axes classiques)")
plt.xlabel("Coordonnée X")
plt.ylabel("Coordonnée Y")
plt.show()


# ------------- PARTIE 2 : Calcul de la série de Fourier complexe -------------
# Conversion des points en nombres complexes en utilisant y' = image.shape[0] - y
z = np.array([x + 1j*(image.shape[0] - y) for x, y in coordinates])
N = len(z)  # Nombre total de points

# Calcul de la Transformée de Fourier discrète normalisée
c = np.fft.fft(z) / N
# Recentre les coefficients (fréquences négatives à gauche, positives à droite)
c_shifted = np.fft.fftshift(c)

# Définir les indices de fréquences
n = np.arange(-N//2, N//2)
# Regrouper les fréquences et les coefficients dans l'ordre naturel
epicycles = list(zip(n, c_shifted))


# ------------- PARTIE 3 : Animation par épicycles avec affichage de l'angle -------------
def get_epicycles_position(t, epicycles):
    """
    Calcule la position finale en sommant les contributions de chaque épicycle pour un instant t.
    Retourne la liste des positions (centres successifs).
    """
    pos = 0 + 0j
    positions = [pos]
    for freq, coeff in epicycles:
        pos += coeff * np.exp(1j * 2 * np.pi * freq * t)
        positions.append(pos)
    return positions

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
# Définition des limites basées sur la taille de l'image
margin = 20
ax.set_xlim(0 - margin, image.shape[1] + margin)
ax.set_ylim(0 - margin, image.shape[0] + margin)
# Ici, on n'inverse pas l'axe y car nous avons déjà corrigé les coordonnées

# Tracés pour la trajectoire et les segments reliant les centres
line_path, = ax.plot([], [], 'r-', lw=2, label='Trajectoire')
line_segments, = ax.plot([], [], 'k-', lw=1)

# Création d'un cercle pour chaque épicycle
circles = [Circle((0, 0), 0, fill=False, color='gray', lw=1) for _ in epicycles]
for circle in circles:
    ax.add_patch(circle)

# Texte pour afficher l'angle
angle_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, color='purple',
                     verticalalignment='top')

path_points = []  # Pour mémoriser la trajectoire du point final
frames = 300     # Nombre total de frames de l'animation

def init():
    line_path.set_data([], [])
    line_segments.set_data([], [])
    angle_text.set_text("")
    return circles + [line_path, line_segments, angle_text]

def update(frame):
    t = frame / frames  # t varie de 0 à 1
    positions = get_epicycles_position(t, epicycles)
    
    centers_x = []
    centers_y = []
    # Mise à jour de chaque cercle et récupération des centres
    for i, (freq, coeff) in enumerate(epicycles):
        center = positions[i]
        radius = abs(coeff)
        circles[i].center = (center.real, center.imag)
        circles[i].radius = radius
        centers_x.append(center.real)
        centers_y.append(center.imag)
    # Ajout du point final de la chaîne
    centers_x.append(positions[-1].real)
    centers_y.append(positions[-1].imag)
    line_segments.set_data(centers_x, centers_y)
    
    # Mémorisation et affichage de la trajectoire du point final
    path_points.append(positions[-1])
    path_x = [p.real for p in path_points]
    path_y = [p.imag for p in path_points]
    line_path.set_data(path_x, path_y)
    
    # Calcul et affichage de l'angle (en degrés) correspondant à t
    angle_degrees = 360 * t
    angle_text.set_text(f"Angle : {angle_degrees:.1f}°")
    
    return circles + [line_path, line_segments, angle_text]

anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                     blit=True, interval=50, repeat=True)

anim.save("epicycles_animation.gif", writer='pillow', fps=30)

plt.title("Animation par épicycles de la courbe de l'éléphant")
plt.legend()
plt.show()

