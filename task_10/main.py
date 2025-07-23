import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


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
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
pixels = image.reshape(-1, 3)


for n in (2, 5, 10, 20):
    model = KMeans(n_clusters=n, random_state=1, n_init=10)

    labels = model.fit_predict(X=pixels)
    new_pixels = model.cluster_centers_[labels]

    new_image = new_pixels.reshape(image.shape).astype('uint8')

    draw_picture(image_func=new_image, title=f'Kmeans{n}', bgr=False)
