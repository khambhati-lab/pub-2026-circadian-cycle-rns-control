import pandas as pd
import numpy as np

def catalogdf_edgecases(patient_code, catdf):

    except_flag = False
    if patient_code == 'NPE0012':  # happens only in the first few cases
        catdf.loc[catdf['Ch 1 name'].isna(), 'Ch 1 name'] = 'Missing_name_Ch1'
        catdf.loc[catdf['Ch 2 name'].isna(), 'Ch 2 name'] = 'Missing_name_Ch2'
        except_flag = True

    else: # no modifications of catalog df
        pass

    if except_flag:
        print(f"Handled exception in catalog for: {patient_code}")

    return catdf
