import os
import time

def seconds_per_n_years(num_years):
    return 24 * 60 * 60 * 365 * num_years

def n_year_index(num_years):
    return round(time.time()) % seconds_per_n_years(num_years)
