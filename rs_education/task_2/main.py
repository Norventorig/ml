from surprise import Dataset, Reader

from surprise.model_selection import train_test_split
from surprise import KNNWithMeans
from surprise.model_selection import GridSearchCV

from surprise.accuracy import rmse


data = Dataset.load_builtin("ml-100k")

param_grid = {
    'k': list(range(10, 51, 10)),
    'sim_options': {
        'name': ['cosine', 'pearson'],
        'user_based': [True]
    }
}
grid = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'], cv=3)
grid.fit(data)

trainset, testset = train_test_split(data, test_size=0.2)
model = KNNWithMeans(**grid.best_params['rmse'])

model.fit(trainset)

predictions = model.test(testset)
print(rmse(predictions))