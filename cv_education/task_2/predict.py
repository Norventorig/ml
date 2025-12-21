from tensorflow.keras.models import load_model
import cv2
from pathlib import Path
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import pandas as pd

test_dataset_path = r"C:\Users\123\Downloads\datasets\test"


def prepare(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, 0)

    return img


model = load_model("model.keras", compile=False)
data = {'id': [], 'label': []}

paths = sorted(
    Path(test_dataset_path).iterdir(),
    key=lambda x: int(x.stem)
)
for n_iter, i_path in enumerate(paths):
    if n_iter % 100 == 0:
        pd.DataFrame(data=data).to_csv('result.csv', index=False)
        print(n_iter)

    value = prepare(path=i_path)

    value_id = str(i_path.name)[:-4]
    predicted_label= int(model.predict(value)[0][0])

    data['label'].append(predicted_label)
    data['id'].append(value_id)

pd.DataFrame(data=data).to_csv('result.csv', index=False)
