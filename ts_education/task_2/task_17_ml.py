import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error


non_stationary_series = pd.read_csv("monthly-sales-of-company-x-jan-6.csv").iloc[:, 1]
stationary_series = pd.read_csv("mean-monthly-air-temperature-deg.csv").iloc[:, 1]


class MA_Handler:
    @staticmethod
    def predict(series: pd.Series, k: int = 5, n_iter: int = 10) -> pd.Series:
        data = series.iloc[-1 * k:].values.tolist()

        for _ in range(n_iter):
            predicted_value = np.mean(data[-1: -1 * (k + 1): -1])
            data.append(predicted_value)

        data = data[k - 1:]
        index = [i_ind for i_ind in range(series.index[-1], series.index[-1] + len(data))]

        return pd.Series(data=data, index=index)

    @staticmethod
    def smooth(series: pd.Series, k: int = 5) -> pd.Series:
        transformed_series = series.copy()
        transformed_series = transformed_series.rolling(k).mean()
        transformed_series.dropna()

        return transformed_series


class EMA_Handler:
    @staticmethod
    def smooth(series: pd.Series, alpha: float = 0.2) -> pd.Series:
        transformed_series = np.zeros(len(series))
        transformed_series[0] = series.iloc[0]

        for i_index in range(1, len(series)):
            transformed_series[i_index] = series.iloc[i_index] * alpha + (1 - alpha) * transformed_series[i_index - 1]

        return pd.Series(data=transformed_series)

    @staticmethod
    def predict(series: pd.Series, alpha: float = 0.2, n_iter: int = 10) -> pd.Series:
        data = [series.iloc[-1]]
        smooth = [EMA_Handler.smooth(series=series, alpha=alpha).iloc[-1]]

        for _ in range(n_iter):
            res = data[-1] * alpha + (1 - alpha) * smooth[-1]

            smooth.append(res)
            data.append(res)

        index = [i_ind for i_ind in range(series.index[-1], series.index[-1] + len(data))]

        return pd.Series(data=data, index=index)


class DEMA_Handler:
    @staticmethod
    def smooth_predict(series: pd.Series,
                       alpha: float = 0.2, beta: float = 0.2,
                       n_iter: int = None) -> pd.Series:
        trends = []
        levels = []
        transformed_series = []

        levels.append(series.iloc[0])
        trends.append(series.iloc[1] - series.iloc[0])
        transformed_series.append(series.iloc[0])

        for i_index in range(1, len(series)):
            curr_level = alpha * series.iloc[i_index] + (1 - alpha) * (levels[-1] + trends[-1])
            curr_trend = beta * (curr_level - levels[-1]) + (1 - beta) * trends[-1]

            transformed_series.append(curr_level + curr_trend)

            levels.append(curr_level)
            trends.append(curr_trend)

        if n_iter:
            for _ in range(n_iter):
                curr_level = alpha * transformed_series[-1] + (1 - alpha) * (levels[-1] + trends[-1])
                curr_trend = beta * (curr_level - levels[-1]) + (1 - beta) * trends[-1]

                transformed_series.append(curr_level + curr_trend)

                levels.append(curr_level)
                trends.append(curr_trend)

        return pd.Series(data=transformed_series)


def triple_exponential_smoothing(series: pd.Series,
                                 alpha: float = 0.2,
                                 beta: float = 0.2,
                                 gamma: float = 0.2,
                                 seasonality: int = 12,
                                 n_preds: int = 10) -> pd.Series:
    def initial_trend(series, season_len):
        return sum([(series[i + season_len] - series[i]) / season_len
                    for i in range(season_len)]) / season_len

    def initial_seasonal_components(series, season_len):
        seasonals = {}
        n_seasons = int(len(series) / season_len)
        season_averages = [sum(series[season_len*j:season_len*j+season_len])/season_len
                           for j in range(n_seasons)]
        for i in range(season_len):
            sum_over_avg = sum(series[season_len*j + i] - season_averages[j]
                               for j in range(n_seasons))
            seasonals[i] = sum_over_avg / n_seasons
        return seasonals

    smooth = series.iloc[0]
    trend = initial_trend(series, seasonality)
    seasonals = initial_seasonal_components(series, seasonality)

    result = []

    for i in range(len(series)):
        val = series.iloc[i]
        if i == 0:
            result.append(val)
            continue

        last_smooth = smooth
        smooth = alpha * (val - seasonals[i % seasonality]) + (1 - alpha) * (smooth + trend)
        trend = beta * (smooth - last_smooth) + (1 - beta) * trend
        seasonals[i % seasonality] = gamma * (val - smooth) + (1 - gamma) * seasonals[i % seasonality]

        result.append(smooth + trend + seasonals[i % seasonality])

    for m in range(1, n_preds + 1):
        result.append(smooth + m*trend + seasonals[(len(series) + m - 1) % seasonality])

    return pd.Series(result)


pipeline = {
    'non stationary time series': {
        'ma': (MA_Handler.predict, [{'k': i_k} for i_k in range(1, 16)], non_stationary_series,),
        'ema': (EMA_Handler.predict, [{'alpha': i_k / 10} for i_k in range(1, 10)], non_stationary_series),
        'dema': (DEMA_Handler.smooth_predict, [{'alpha': alpha / 10, 'beta': beta / 10}
                                               for alpha in range(1, 10)
                                               for beta in range(1, 10)], non_stationary_series,),
        'tema': (triple_exponential_smoothing, [{'alpha': alpha / 10, 'beta': beta / 10, 'gamma': gamma / 10}
                                                for alpha in range(1, 10)
                                                for beta in range(1, 10)
                                                for gamma in range(1, 10)], non_stationary_series,)},

    'stationary time series': {
        'ma': (MA_Handler.predict, [{'k': i_k} for i_k in range(1, 16)], stationary_series),
        'ema': (EMA_Handler.predict, [{'alpha': i_k / 10} for i_k in range(1, 10)], stationary_series),
        'dema': (DEMA_Handler.smooth_predict, [{'alpha': alpha / 10, 'beta': beta / 10}
                                               for alpha in range(1, 10)
                                               for beta in range(1, 10)], stationary_series),
        'tema': (triple_exponential_smoothing, [{'alpha': alpha / 10, 'beta': beta / 10, 'gamma': gamma / 10}
                                                for alpha in range(1, 10)
                                                for beta in range(1, 10)
                                                for gamma in range(1, 10)], stationary_series)}}

research_data = {'non stationary time series': {'ma': [], 'ema': [], 'dema': [], 'tema': []},
                 'stationary time series': {'ma': [], 'ema': [], 'dema': [], 'tema': []}}

for ts in pipeline.keys():
    for method in pipeline[ts].keys():
        data = pipeline[ts][method]

        func = data[0]
        params = data[1]
        curr_ts = data[2]

        for i_param in params:
            predictions = func(curr_ts[:-10], **i_param)[-10:]
            true_values = curr_ts[-10:]

            metrics = root_mean_squared_error(y_true=true_values, y_pred=predictions)

            research_data[ts][method].append((metrics, i_param))

        research_data[ts][method] = min(research_data[ts][method])

print(research_data)

plt.figure(figsize=(24, 24))

plt.subplot(2, 2, 1)
plt.plot(non_stationary_series)
plt.plot(MA_Handler.predict(series=non_stationary_series, k=13))

plt.subplot(2, 2, 2)
plt.plot(non_stationary_series)
plt.plot(EMA_Handler.predict(series=non_stationary_series, alpha=0.1))

plt.subplot(2, 2, 3)
plt.plot(DEMA_Handler.smooth_predict(series=non_stationary_series, alpha=0.3, beta=0.5, n_iter=10))

plt.subplot(2, 2, 4)
plt.plot(triple_exponential_smoothing(series=non_stationary_series, alpha=0.2, beta=0.5, gamma=0.6))

plt.show()


plt.figure(figsize=(24, 24))

plt.subplot(2, 2, 1)
plt.plot(stationary_series)
plt.plot(MA_Handler.predict(series=stationary_series, k=14))

plt.subplot(2, 2, 2)
plt.plot(stationary_series)
plt.plot(EMA_Handler.predict(series=stationary_series, alpha=0.1))

plt.subplot(2, 2, 3)
plt.plot(DEMA_Handler.smooth_predict(series=stationary_series, alpha=0.3, beta=0.3, n_iter=10))

plt.subplot(2, 2, 4)
plt.plot(triple_exponential_smoothing(series=stationary_series, alpha=0.2, beta=0.3, gamma=0.5))

plt.show()
