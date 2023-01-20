###############################################################################
#
# This file contains definitions for database query functions used by multiple other scripts.
# NOTE Currently not in use (functions are defined in each script).
# 
###############################################################################

import numpy as np
import pandas as pd
import requests

# Queries a url and returns the result as a pandas dataframe
def get_dataframe(session, url, param=None):
    try:
        table = session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    return pd.DataFrame(table.json())

# Queries a url and returns the requested column of the result as a numpy array
def get_column(session, url, param=None, field='id'):
    try:
        table = session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    try:
        entries = [row[field] for row in table.json()]
    except KeyError as err:
        print(err)
        print(f"The table '{url}' has no column '{field}'.")
    return np.array(entries)
