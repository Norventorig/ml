import pandas
import pandas as pd
import sklearn
import numpy


from abc import ABC


class LogisticRecursion(ABC):
    @staticmethod
    def fit(x_param: pandas.DataFrame, y_param: pandas.Series, learning_rate: float = 0.2, iters: int = 1000):
        x_param = LogisticRecursion.normalization(x_param=x_param)
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
        probabilities = numpy.clip(probabilities, 1e-12, 1 - 1e-12)
        res = numpy.sum(y_np * numpy.log(probabilities) + (1 - y_np) * numpy.log(1 - probabilities))

        return res / len(x_param.index)

    @staticmethod
    def log_loss(x_param: pandas.DataFrame, y_param: pandas.Series, weights: list):
        return -LogisticRecursion.likelihood(x_param=x_param, y_param=y_param, weights=weights)

    @staticmethod
    def gradient(x_param: pandas.DataFrame, y_param: pandas.Series, weights: list):
        x_np = x_param.to_numpy()
        y_np = y_param.to_numpy()
        weights = numpy.array(weights)

        scores = numpy.dot(x_np, weights)
        probabilities = 1 / (1 + numpy.exp(-scores))
        probabilities = numpy.clip(probabilities, 1e-12, 1 - 1e-12)
        res = numpy.dot((probabilities - y_np), x_np)

        return pd.Series(res / len(x_param.index), index=x_param.columns)

    @staticmethod
    def normalization(x_param: pandas.DataFrame):
        exp_val = {i: 0 for i in x_param.columns}
        std_dev = {i: 0 for i in x_param.columns}

        for i_column in x_param.columns:
            column_values = x_param[i_column].to_list()

            for i_value in column_values:
                exp_val[i_column] += i_value * (column_values.count(i_value) / len(column_values))

        for i_column in x_param.columns:
            column_values = x_param[i_column].to_list()

            for i_value in column_values:
                std_dev[i_column] += (i_value - exp_val[i_column]) ** 2 \
                                     * (column_values.count(i_value) / len(column_values))
            std_dev[i_column] = numpy.sqrt(std_dev[i_column])

        x_param = (x_param - exp_val) / std_dev
        return x_param


dataset = sklearn.datasets.load_iris(as_frame=True).frame
dataset = dataset.loc[dataset['target'] != 0]

x = dataset.iloc[:, :4]
y = dataset['target'].replace({1: 0, 2: 1})

result = LogisticRecursion.fit(x, y)

for i_key in result.keys():
    print(result[i_key], '\n')
