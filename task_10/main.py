import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.utils import resample
from scipy.spatial.distance import cdist


def draw_picture(image_func, title, bgr=True):
    if bgr:
        b, g, r = cv2.split(image_func)
        image_func = cv2.merge([r, g, b])

    plt.figure(figsize=(7, 5))
    plt.axis('off')
    plt.imshow(image_func)
    plt.title(title)
    plt.show()


image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image.reshape(-1, 3)


for n in (2, 5, 10, 20):
    model = KMeans(n_clusters=n, random_state=1, n_init=10)

    labels = model.fit_predict(X=pixels)
    new_pixels = model.cluster_centers_[labels]

    new_image = new_pixels.reshape(image.shape).astype('uint8')
    draw_picture(image_func=new_image, title=f'Kmeans{n}', bgr=False)


model = DBSCAN(eps=3, min_samples=4)

labels = model.fit_predict(X=pixels)
clusters_id = np.unique(labels[labels != -1])

new_pixels = pixels.copy()

for cluster_id in clusters_id:
    pixel_cluster = pixels[labels == cluster_id]
    new_pixels[labels == cluster_id] = np.mean(pixel_cluster, axis=0)
new_pixels[labels == -1] = [0, 0, 0]

new_image = new_pixels.reshape(image.shape).astype('uint8')
draw_picture(image_func=new_image, title='DBSCAN', bgr=False)


sample = resample(pixels, n_samples=10000, random_state=0, replace=False) if pixels.shape[0] > 10000 else pixels

for n in (2, 5, 10, 20):

    model = AgglomerativeClustering(n_clusters=n, linkage='ward')
    sample_labels = model.fit_predict(sample)

    centroids = []
    for cluster_id in range(n):
        cluster_pixels = sample[sample_labels == cluster_id]
        centroids.append(cluster_pixels.mean(axis=0))
    centroids = np.array(centroids).astype('uint8')

    distances = cdist(pixels, centroids)
    full_labels = np.argmin(distances, axis=1)

    new_pixels = centroids[full_labels]

    new_image = new_pixels.reshape(image.shape)
    draw_picture(image_func=new_image, title=f'Agglomerative {n}', bgr=False)
