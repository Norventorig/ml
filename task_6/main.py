import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('dataset.csv')

print(f'Кол-во пропусков в каждом признаке: \n{dataset.isnull().sum()}')
for col in dataset.columns.to_list():
    plt.figure(figsize=(8, 6))
    plt.boxplot(dataset[col], vert=False)
    plt.title(f'Boxplot для признака "{col}"')
    plt.xlabel('Значения')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# Пропуски не были обнаружены
# Коробки с усами показали множественные выбросы к которым чувствительна нормализация
# Оставшиеся пригодными для нормализации признаки (в основном широта и возраст дома) в ней не нуждаются
