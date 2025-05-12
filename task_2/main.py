import pandas
import pandas as pd
import sklearn
import numpy


from abc import ABC


class LogisticRecursion(ABC):
    @staticmethod
    def fit(x_param: pandas.DataFrame, y_param: pandas.Series, learning_rate: float = 0.2, iters: int = 100):
        x_param['intercept'] = 1
        point = numpy.zeros(len(x_param.columns))

        for _ in range(iters):
            res = LogisticRecursion.gradient(x_param=x_param, y_param=y_param, weights=point)
            point = [point[i_weight] - learning_rate * i_gradient for i_weight, i_gradient in enumerate(res)]

        return {'koefs': {i: point[j] for j, i in enumerate(x_param.columns)},
                'likelihood': LogisticRecursion.likelihood(x_param=x_param, y_param=y_param, weights=point),
                'log_loss': LogisticRecursion.log_loss(x_param=x_param, y_param=y_param, weights=point)}

    @staticmethod
    def likelihood(x_param: pandas.DataFrame, y_param: pandas.Series, weights: list):
        x_np = x_param.to_numpy()
        y_np = y_param.to_numpy()
        weights = numpy.array(weights)

        scores = numpy.dot(x_np, weights)
        probabilities = 1 / (1 + numpy.exp(-scores))
        res = numpy.sum(y_np * numpy.log(probabilities) + (1 - y_np) * numpy.log(1 - probabilities))

        return res / len(x_param.index)

    @staticmethod
    def log_loss(x_param: pandas.DataFrame, y_param: pandas.Series, weights: list):
        x_np = x_param.to_numpy()
        y_np = y_param.to_numpy()
        weights = numpy.array(weights)

        scores = numpy.dot(x_np, weights)
        probabilities = 1 / (1 + numpy.exp(-scores))
        res = numpy.sum(-1 * (y_np * numpy.log(probabilities) + (1 - y_np) * numpy.log(1 - probabilities)))

        return res / len(x_param.index)

    @staticmethod
    def gradient(x_param: pandas.DataFrame, y_param: pandas.Series, weights: list):
        x_np = x_param.to_numpy()
        y_np = y_param.to_numpy()
        weights = numpy.array(weights)

        scores = numpy.dot(x_np, weights)
        probabilities = 1 / (1 + numpy.exp(-scores))
        res = numpy.dot((probabilities - y_np), x_np)

        return pd.Series(res / len(x_param.index), index=x_param.columns)


dataset = sklearn.datasets.load_iris(as_frame=True).frame
dataset = dataset.loc[dataset['target'] != 0]

x = dataset.iloc[:, :4]
y = dataset['target'].replace({1: 0, 2: 1})

result = LogisticRecursion.fit(x, y)

for i_key in result.keys():
    print(result[i_key], '\n')
