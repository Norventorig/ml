# IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


from statsmodels.tsa.stattools import adfuller


def multi_linear_visualisation(data):
    """
    multi_linear_visualisation creates linear graphics of series in data and names them

    :param data: dict(str: pandas.Series)
    :return:
    """
    y = len(data) % 2 + len(data) // 2

    plt.figure(figsize=(24, 12))
    for i_index, i_key in enumerate(data.keys()):
        plt.subplot(y, 2, i_index + 1)
        plt.plot(data[i_key])
        plt.title(i_key)

    plt.show()


def search_lags(data, lags_amount=60):
    """
    search_lags returns dictionary which contains values in such format: {series_name: lag}

    :param data: dict(str: pandas.Series)
    :param lags_amount: int
    :return: dict(str: int)
    """
    result = {}

    for i_key, i_value in data.items():
        acf_values = acf(i_value, nlags=lags_amount)
        max_lag = np.argmax(acf_values[1:]) + 1

        result[i_key] = max_lag

    return result


# Init datasets
sales_of_company_x = pd.read_csv("monthly-sales-of-company-x-jan-6.csv")
robberies_in_boston = pd.read_csv("monthly-boston-armed-robberies-j.csv")
airlines_passengers = pd.read_csv("international-airline-passengers.csv")
mean_monthly_temp = pd.read_csv("mean-monthly-air-temperature-deg.csv")
dowjones_closing = pd.read_csv("weekly-closings-of-the-dowjones-.csv")
female_births = pd.read_csv("daily-total-female-births-in-cal.csv")

# creation of convenient datasets holder
all_series = {
    "Monthly sales of company X": sales_of_company_x.iloc[:, 1],
    "Monthly Boston armed robberies": robberies_in_boston.iloc[:, 1],
    "International airline passengers: monthly totals in thousands": airlines_passengers.iloc[:, 1],
    "Mean monthly air temperature (Deg. F) Nottingham Castle": mean_monthly_temp.iloc[:, 1],
    "Weekly closings of the Dow-Jones industrial average": dowjones_closing.iloc[:, 1],
    "Daily total female births in California": female_births.iloc[:, 1]
}

# creation of convenient datasets holder processed
all_series_processed = {
    "Monthly sales of company X": np.diff(np.log(sales_of_company_x.iloc[:, 1])),
    "Monthly Boston armed robberies": np.diff(np.log(robberies_in_boston.iloc[:, 1])),
    "International airline passengers: monthly totals in thousands": np.diff(np.log(airlines_passengers.iloc[:, 1])),
    "Mean monthly air temperature (Deg. F) Nottingham Castle": np.diff(mean_monthly_temp.iloc[:, 1]),
    "Weekly closings of the Dow-Jones industrial average": np.diff(dowjones_closing.iloc[:, 1]),
    "Daily total female births in California": np.diff(female_births.iloc[:, 1])
}


# Each time series visualisation
multi_linear_visualisation(data=all_series)


# Visualisation of each processed time series
fig, axes = plt.subplots(3, 2, figsize=(24, 12))
for index, i_key in enumerate(all_series_processed.keys()):
    x = index % 2
    y = int((index - x) / 2)

    plot_acf(all_series_processed[i_key], ax=axes[y, x], lags=60)
    axes[y, x].set_title(f"{i_key}")

plt.show()


# lags for each time series
lags = search_lags(all_series_processed)


# Improves data preparation. Makes diff with proper lag
gen = ((i_key, np.log(i_value)) for i_key, i_value in all_series.items())
all_series_processed = {gen_val[0]:
                            gen_val[1].diff(lags[gen_val[0]]).dropna()
                        for gen_val in gen}


# Visualisation of each processed time series diffed with proper lag
multi_linear_visualisation(data=all_series_processed)


# Checking for stationary time series
for i_series in all_series_processed.keys():
    print(f'{i_series} - p_value: {adfuller(all_series_processed[i_series])[1]}')

# Result:
# Every time series except for
# "Daily total female births in California" and "Mean monthly air temperature (Deg. F) Nottingham Castle"
# were successfully turned into stationary time series
