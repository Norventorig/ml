import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MA_Handler:
    @staticmethod
    def predict(series: pd.Series, k: int = 5, n_iter: int = 5) -> pd.Series:
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
    def predict(series: pd.Series, alpha: float = 0.2, n_iter: int = 5) -> pd.Series:
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
    def smooth_predict(series: pd.Series, alpha: float = 0.2, beta: float = 0.2, n_iter: int = None) -> pd.Series:
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
                                 n_preds: int = 0) -> pd.Series:
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


non_stationary_series = pd.read_csv("monthly-sales-of-company-x-jan-6.csv").iloc[:, 1]
stationary_series = pd.read_csv("mean-monthly-air-temperature-deg.csv").iloc[:, 1]
