import pandas as pd
import statsmodels.api as sm

def calc_ssd_using_rolling_window_and_threshold(data: pd.Series, window: int = 10, threshold: int = 3):


    # Apply moving average
    moving_avg = data.rolling(window=10).mean()

    # Detect periods with low variability
    variance = data.rolling(window=10).var()
    steady_states = variance < threshold  # Define your threshold

    print("Periods of steady state:", steady_states)
    return steady_states