import cv2
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
pixels = image.reshape(-1, 3)


model = KMeans()