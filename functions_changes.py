import pandas as pd
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from numpy.lib.stride_tricks import sliding_window_view
from tqdm.notebook import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def execute_sql_query(table_name):

    sql_query = f"""
    SELECT 
        *
    FROM CAB_EPIC.dbo.{table_name} 
    """
    return sql_query

def execute_diagnosis_fact(diagnosis_numbers):
    diagnosis_values = ", ".join(str(diagnosis) for diagnosis in diagnosis_numbers)
    
    sql_query = f"""
    SELECT 
        DiagnosisKey,
        PatientDurableKey,
        EncounterKey,
        StartDateKey
    FROM CAB_EPIC.dbo.DiagnosisEventFact
    WHERE DiagnosisKey IN ({diagnosis_values})
    """
    return sql_query

def execute_diagnosis_ev(pts, encs):
    patients = ", ".join(str(pt) for pt in pts)
    encounters = ", ".join(str(enc) for enc in encs)
    
    sql_query = f"""
    SELECT 
        DiagnosisKey,
        PatientDurableKey,
        EncounterKey,
        Type,
        StartDateKey,
        EndDateKey,
        IsPrimary,
        PresentOnAdmission,
        HospitalDiagnosis,
        EmergencyDepartmentDiagnosis 
    FROM CAB_EPIC.dbo.DiagnosisEventFact
    WHERE PatientDurableKey IN ({patients}) and EncounterKey IN ({encounters})
    """
    return sql_query

def execute_flowsheet_fact_enc(encs, flows):
    flow_keys = ", ".join(str(key) for key in flows)
    filtered_encs = ", ".join(str(key) for key in encs)
    
    sql_query = f"""
    SELECT 
        FlowsheetValueKey,
        FlowsheetRowKey,
        EncounterKey,
        PatientDurableKey,
        DateKey,
        TimeOfDayKey,
        Value
    FROM CAB_EPIC.dbo.FlowsheetValueFact
    WHERE EncounterKey IN ({filtered_encs}) AND FlowsheetRowKey IN ({flow_keys})
    """
    return sql_query

def execute_flowsheet_sepsisInfo():

    sql_query = f"""
    SELECT *
    FROM CAB_EPIC_Oxford.dbo.SepsisFlowInfo
    """
    return sql_query

def execute_lab_sepsisInfo():
    
    sql_query = f"""
    SELECT *
    FROM CAB_EPIC_Oxford.dbo.SepsisLabInfo
     """
    return sql_query

def execute_lab_enc(enckeys ,lab_keys):
    filtered = ", ".join(str(key) for key in enckeys)
    filtered_lab = ", ".join(str(key) for key in lab_keys)
    
    sql_query = f"""
    SELECT 
        LabComponentKey,
        EncounterKey,
        PatientDurableKey,
        OrderedDateKey,
        OrderedTimeOfDayKey,
        ResultDateKey,
        ResultTimeOfDayKey,
        CollectionDateKey,
        CollectionTimeOfDayKey,
        Value
    FROM CAB_EPIC.dbo.LabComponentResultFact
    WHERE EncounterKey IN ({filtered}) AND LabComponentKey IN ({filtered_lab})
     """
    return sql_query

def execute_diagnosis_dim():
    sql_query = f"""
    SELECT 
        DiagnosisKey,
        Name
    FROM CAB_EPIC.dbo.DiagnosisDim  
    """
    return sql_query

def execute_flowsheet_fact():

    sql_query = f"""
    SELECT *
    FROM CAB_EPIC_Oxford.dbo.SepsisInfo
    """
    return sql_query

def execute_flowsheet_dim(keys):
    filtro = ", ".join(str(key) for key in keys)
    
    sql_query = f"""
        SELECT 
          FlowsheetRowKey,
          DisplayName
        FROM CAB_EPIC.dbo.FlowsheetRowDim
        WHERE FlowsheetRowKey IN ({filtro})
    """
    return sql_query
    
def execute_lab_fact():
    sql_query = f"""
    SELECT *
    FROM CAB_EPIC_Oxford.dbo.SepsisLabInfo
    """
    return sql_query

def execute_lab_dim():
    sql_query = f"""
    SELECT 
        *
    FROM CAB_EPIC.dbo.LabComponentDim
    """
    return sql_query
    
def execute_lab_factwithkeys(LabComponentKeys):
    ComponentKeys = ", ".join(str(key) for key in LabComponentKeys)

    sql_query = f"""
    SELECT *
    FROM CAB_EPIC_Oxford.dbo.SepsisLabInfo
    WHERE LabComponentKey IN ({ComponentKeys})
    """
    return sql_query
    
def execute_lab_dimwithkeys(LabComponentKeys):
    ComponentKeys = ", ".join(str(key) for key in LabComponentKeys)

    sql_query = f"""
        SELECT 
            LabComponentKey,
            Name
        FROM CAB_EPIC.dbo.LabComponentDim
        WHERE LabComponentKey IN ({ComponentKeys})
    """
    return sql_query

def extract_number(value):
    match = re.search(r'\d+', value)  # Regular expression to find a number
    return int(match.group()) if match else None  # Return the number or None if no match

def extract_features(Table1, keys):
    ft = Table1.loc[Table1['FlowsheetRowKey'].isin(keys), 
                    ['EncounterKey', 'PatientDurableKey', 'DateKey', 'TimeOfDayKey', 'DisplayName', 'Value']]

    mask = ft['Value'].str.contains('[a-zA-Z]', regex=True) | ft['Value'].isin(['', ' '])
    ft = ft[~mask]
    ft = ft.drop_duplicates().sort_values(by=['PatientDurableKey', 'DateKey'])
    if (ft['DisplayName'] != 'BP').all():
        ft['Value'] = pd.to_numeric(ft['Value'], errors='coerce')
    ft.dropna(subset=['Value'], inplace=True)
    return ft

def extract_lab_features(feat_keys, LabResults):
    lab_feature = LabResults.loc[LabResults['LabComponentKey'].isin(feat_keys), 
                                 ['EncounterKey', 'PatientDurableKey', 'OrderedDateKey', 'OrderedTimeOfDayKey', 'DisplayName', 'Value']]
    lab_feature['Value'].replace({'>25': 25, '0-5': 2.5}, inplace=True)
    mask = lab_feature['Value'].str.contains('[a-zA-Z]', regex=True, na=False) | lab_feature['Value'].isin(['', ' '])
    lab_feature = lab_feature[~mask]
    lab_feature['Value'] = pd.to_numeric(lab_feature['Value'], errors='coerce')
    lab_feature.dropna(subset=['Value'], inplace=True)
    feature = lab_feature.sort_values(by=['PatientDurableKey', 'OrderedDateKey'])
    feature = feature.rename(columns={'OrderedTimeOfDayKey':'TimeOfDayKey', 'OrderedDateKey':'DateKey'})
    return feature
        
def map_values(df1, df2, column1, column2, column3, column4=None):
    diagnosis_mapping = dict(zip(df1[column1], df1[column2]))

    df2[column3] = df2[column1].map(diagnosis_mapping)
    
    if column4 != None:
        df3 = df2[df2[column4] != -1]
        return df3
    return df2
    
def find_keys(table, column):
    table = table[table[column] != -1]
    return table[column].drop_duplicates()
        
def find_sepsis(table):
    return table[table.iloc[:, 1].str.contains('Sepsis', na=False)]
        
def filter_for_patient(table, patient, column1, column2):
    table1 = table[table[column1] == patient]
    return table2
       
def convert_temp(table):
    table['Value'] = pd.to_numeric(table['Value'], errors='coerce')
    celsius_filter = table['DisplayName'].str.contains('Celsius')
    table.loc[celsius_filter, 'Value'] = (table.loc[celsius_filter, 'Value'] * 9/5) + 32
    table = table.dropna()
    return table
    
def EncounterPatient(n1, n2, keys):
    filtro = ", ".join(str(key) for key in keys)
    
    if n1 is None and n2 is None:
        sql_query = f"""
        SELECT *
        FROM (
            SELECT 
              AdmissionInstant,
              DischargeInstant,
              EncounterKey,
              PatientClass,
              PatientDurableKey,
              AdmissionType,
              AdmissionDateKey,
              DischargeDateKey,
              DateKey,
              EndDateKey,
              PrimaryDiagnosisKey,
              ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS row_num
            FROM CAB_EPIC.dbo.EncounterFact
            WHERE PatientDurableKey IN ({filtro})
        ) AS sub
        """
    else:
        sql_query = f"""
        SELECT *
        FROM (
            SELECT 
              AdmissionInstant,
              DischargeInstant,
              EncounterKey,
              PatientClass,
              PatientDurableKey,
              AdmissionType,
              AdmissionDateKey,
              DischargeDateKey,
              DateKey,
              EndDateKey,
              PrimaryDiagnosisKey,
              ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS row_num
            FROM CAB_EPIC.dbo.EncounterFact
            WHERE PatientDurableKey IN ({filtro})
        ) AS sub
        WHERE sub.row_num BETWEEN {n1} AND {n2}
        """
    return sql_query
    
    
def DiagnosisPatient(n1, n2, keys):
    filtro = ", ".join(str(key) for key in keys)
    
    if n1 is None and n2 is None:
        sql_query = f"""
        SELECT *
        FROM (
            SELECT 
              PatientDurableKey,
              DiagnosisKey,
              EncounterKey,
              StartDateKey,
              EndDateKey,
              ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS row_num
            FROM CAB_EPIC.dbo.DiagnosisEventFact
            WHERE PatientDurableKey IN ({filtro})
        ) AS sub
        """
    else:
        sql_query = f"""
        SELECT *
        FROM (
            SELECT 
              PatientDurableKey,
              DiagnosisKey,
              EncounterKey,
              StartDateKey,
              EndDateKey,
              ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS row_num
            FROM CAB_EPIC.dbo.DiagnosisEventFact
            WHERE PatientDurableKey IN ({filtro})
        ) AS sub
        WHERE sub.row_num BETWEEN {n1} AND {n2}
        """
    return sql_query 
    
def get_LabCriteriakeys(param):
    wbc_keys = [6220]
    bun_keys = [4969]
    ast_keys = [4967]
    alt_keys = [4960]
    alp_keys = [4959]
    bil_keys = [5011]
    plat_keys = [6192]
    la_keys = [5725]
    ph_keys = [18021]
    pf_keys = [4465, 17881]
    cr_keys = [4974]
    lympcount_keys = [5086]
    neutrcount_keys = [9640]
    cult_keys = [7396,41178,16314,7401,9876,7579,12081,35488,7398,7576,35493,16312,16320,11885,16733,16326,15595,7594,15537,13766,7490,
            7404,32856,18108,35494,4588,7591,23970,22035,16322,16629,15553,44782,11958,7581,2947,18347,18117,15571,16315,11337,
            11905,45332,11912,6177,10325,11907,18115,8281,8302,7575,6176,2948,11065,15583,7403,18107,10721,7580,11803,11911,13791,
            32855,11742,18113,4672,13835,17104,1276,17846,23978,16321,1271,17826,14890,7585,13650,11778,16323,18106,19533,23971,
            34596,32477,34657,10595,13585,13923,15597,15579,24000,6174,24195,10648,25008,15301,17754,11823,18733,13938,13123,13895,
            11960,15556,16734,11336,16709]
    po2_keys = [18686, 10170]
    procal = [14969, 19683] 
    sodium = [5015] 
    potas = [5013] 
    chlor = [5016] 
    co2 = [4970]
    cprotein = [5823]
    alb = [8071,5026,10626, 5310]

    lft_keys = ast_keys+alp_keys+alt_keys+bil_keys
    LabComponentKeys = wbc_keys+ast_keys+alt_keys+alp_keys+bil_keys+plat_keys+la_keys+pf_keys+cr_keys+cult_keys+po2_keys+procal+bun_keys+sodium+potas+chlor+co2+neutrcount_keys+lympcount_keys + bun_keys+cprotein+alb

    if param == 'wbc':
        return wbc_keys
    elif param == 'alb':
        return alb
    elif param == 'cprotein':
        return cprotein
    elif param == 'po2':
        return po2_keys
    elif param == 'ast':
        return ast_keys
    elif param == 'alt':
        return alt_keys
    elif param == 'alp':
        return alp_keys
    elif param == 'bil':
        return bil_keys
    elif param == 'plat':
        return plat_keys
    elif param == 'la':
        return la_keys
    elif param == 'lymp':
        return lympcount_keys
    elif param == 'pf':
        return pf_keys
    elif param == 'cr':
        return cr_keys
    elif param == 'ph':
        return ph_keys
    elif param == 'cult':
        return cult_keys
    elif param == 'all':
        return LabComponentKeys
    elif param == 'lft_all':
        return lft_keys
    elif param == 'co2':
        return co2
    elif param == 'chlor':
        return chlor
    elif param == 'potas':
        return potas
    elif param == 'sodium':
        return sodium
    elif param == 'bun':
        return bun_keys
    elif param == 'procal':
        return procal
    elif param == 'neutr':
        return neutrcount_keys
    return []

def get_FlowCriteriaKeys(param):
    temps = [19908, 31083,  873, 874]
    resp= [42917]
    sbp = [29405]
    hr = [42308]
    gcs_keys = [37993, 28569, 28570, 28571, 28572]
    nibp_keys =[777]
    ur_keys= [31189]
    fio2_keys = [28789, 23059, 34517, 24967]
    dem = [5608, 669]
    spO2 = [2]
    flowsheetsRowKeys = temps + resp + sbp + hr + gcs_keys + nibp_keys + dem + fio2_keys + ur_keys + spO2
    if param == 'temp':
        return temps
    elif param == 'resp':
        return resp
    elif param == 'sbp':
        return sbp
    elif param == 'ur':
        return ur_keys
    elif param == 'hr':
        return hr
    elif param == 'all':
        return flowsheetsRowKeys
    elif param == 'gcs':
        return gcs_keys
    elif param == 'nibp':
        return nibp_keys
    elif param == 'fio2':
        return fio2_keys
    elif param == 'spo2':
        return spO2
    return []

def extract_right_part(value):
    if '/' in value:
        return pd.to_numeric(value.split('/')[1]).astype('float64')
    else:
        return pd.to_numeric(value).astype('float64')

def extract_left_part(value):
    if '/' in value:
        return pd.to_numeric(value.split('/')[0]).astype('float64')
    else:
        return pd.to_numeric(value).astype('float64')
    
def execute_image_fact():
    sql_query = f"""
    SELECT *
    FROM CAB_EPIC_Oxford.dbo.ImageFact
    """
    return sql_query
    
def execute_image_text():
    sql_query = f"""
    SELECT
        ImagingKey,
        Impression,
        Narrative
    FROM CAB_EPIC_Oxford.dbo.ImageText
     """
    return sql_query

def execute_non_sepsis_flow(patkeys, flow_keys):
    filtered = ", ".join(str(key) for key in patkeys)
    filtered_flow = ", ".join(str(key) for key in flow_keys)
    
    sql_query = f"""
    SELECT 
        FlowsheetValueKey,
        FlowsheetRowKey,
        Value,
        TimeOfDayKey,
        DateKey,
        EncounterKey,
        PatientDurableKey
    FROM CAB_EPIC.dbo.FlowsheetValueFact
    WHERE PatientKey IN ({filtered}) AND FlowsheetRowKey IN ({filtered_flow})
     """
    return sql_query


def execute_non_sepsis_lab(patkeys, lab_keys):
    filtered = ", ".join(str(key) for key in patkeys)
    filtered_lab = ", ".join(str(key) for key in lab_keys)
    
    sql_query = f"""
    SELECT 
        LabComponentKey,
        EncounterKey,
        PatientDurableKey,
        OrderedDateKey,
        OrderedTimeOfDayKey,
        ResultDateKey,
        ResultTimeOfDayKey,
        CollectionDateKey,
        CollectionTimeOfDayKey,
        Value
    FROM CAB_EPIC.dbo.LabComponentResultFact
    WHERE PatientDurableKey IN ({filtered}) AND LabComponentKey IN ({filtered_lab})
     """
    return sql_query

def DateGenerator(year_inicial, year_final):
    valores = []
    for year in range(year_inicial, year_final + 1):
        fecha = date(year, 1, 1)
        un_dia = timedelta(days=1)
        while fecha.year == year:
            for parte_del_dia in range(1, 4):
                valor = fecha.strftime('%Y%m%d') + str(parte_del_dia)
                valores.append(valor)
            fecha += un_dia
    return valores

def asignar_valor(hour):
    hour = int(hour)  # Convertir la hora del día de string a entero
    if hour < 800:
        return 1
    elif 800 <= hour <= 1600:
        return 2
    elif 1601 <= hour <= 2400:
        return 3
    
def date_to_index(total_df, image=None):
    DateArray = DateGenerator(2015, 2025)
    total_df2 = total_df.copy()
    if image==None:
        total_df2['TimeValue'] = total_df2['TimeOfDayKey'].apply(asignar_valor)
        total_df2['DateKey'] = total_df2['DateKey'].astype(str)
        total_df2['DateTimeKey'] = total_df2['DateKey'] + total_df2['TimeValue'].astype(str)
        total_df2['Value'] = pd.to_numeric(total_df2['Value'], errors='coerce')
    else:
        total_df2['DateKey'] = total_df2['DateKey'].astype(str)
        total_df2['DateTimeKey'] = total_df2.apply(lambda row: [row['DateKey'] + suffix for suffix in ['1', '2', '3']], axis=1)
        total_df2 = total_df2.explode('DateTimeKey')
        total_df2 = total_df2.drop('DateKey', axis=1)

    xt = total_df2[['PatientDurableKey', 'DateTimeKey', 'Value', 'DisplayName', 'EncounterKey']]
    xt = xt.sort_values(by='DateTimeKey')

    return xt

def sepsis_dates(pivot_df, label):
    cols_to_convert = ['mean_Pulse', 'mean_Resp', 'mean_Temp', 'mean_WBC', 'min_Pulse', 'min_Resp', 'min_Temp', 'min_WBC',
                       'max_Pulse', 'max_Resp', 'max_Temp', 'max_WBC','max_image','max_culture']# 'image', 'culture']

    # Convert necessary columns to float only once
    pivot_df[cols_to_convert] = pivot_df[cols_to_convert].astype(float)

    # Precompute boolean masks to avoid repeated comparisons
    temp_mask = (pivot_df['mean_Temp'] > 100.4) | (pivot_df['min_Temp'] > 100.4) | (pivot_df['max_Temp'] > 100.4) | (pivot_df['mean_Temp'] < 96.8) | (pivot_df['min_Temp'] < 96.8) | (pivot_df['max_Temp'] < 96.8)

    pulse_mask = (pivot_df['mean_Pulse'] > 90) | (pivot_df['min_Pulse'] > 90) | (pivot_df['max_Pulse'] > 90)

    resp_mask = (pivot_df['mean_Resp'] > 20) | (pivot_df['min_Resp'] > 20) | (pivot_df['max_Resp'] > 20)

    wbc_mask = (pivot_df['mean_WBC'] > 12) | (pivot_df['min_WBC'] > 12) | (pivot_df['max_WBC'] > 12) | (pivot_df['mean_WBC'] < 4) | (pivot_df['min_WBC'] < 4) | (pivot_df['max_WBC'] < 4)

    culture_mask = pivot_df['max_culture'] == 1
    image_mask = pivot_df['max_image'] == 1

    # Calculate counts for each row
    count1 = temp_mask.astype(int) + pulse_mask.astype(int) + resp_mask.astype(int) + wbc_mask.astype(int)
    count2 = culture_mask.astype(int)  + image_mask.astype(int)

    # Assign the label based on conditions
    pivot_df[label] = np.where((count1 >= 2) & (count2 >= 1), 1, 0)
    #pivot_df['Label_sirs'] = np.where((count1 >= 2) & (count2 >= 1), 1, 0)


    return pivot_df
    
def execute_non_sepsis():
    sql_query = f"""
        SELECT *
        FROM CAB_EPIC_Oxford.dbo.NonSepsisKeys
    """
    return sql_query

def impute_if_missing_below_threshold(df, threshold=0.4):
    missing_percentage = df.isna().mean().mean()
    return missing_percentage <= threshold

def find_values_for_patient(filtered_dfs, exp_columns, sum_column):
    
    total_df = pd.concat(filtered_dfs)
    sum_df = total_df[total_df['DisplayName'] == sum_column]
    other_df = total_df[total_df['DisplayName'] != sum_column]
 
    pivot_sum = sum_df.pivot_table(
        index=['DateTimeKey', 'PatientDurableKey', 'EncounterKey'],
        columns='DisplayName',
        values='Value',
        aggfunc='sum'
    )
    
    pivot_other = other_df.pivot_table(
        index=['DateTimeKey', 'PatientDurableKey', 'EncounterKey'],
        columns='DisplayName',
        values='Value',
        aggfunc=['mean', 'min', 'max']
    )
    
    pivot_sum = pivot_sum.reset_index()
    pivot_other = pivot_other.reset_index()
    pivot_other.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in pivot_other.columns]
    pivot_sum = pivot_sum.rename(columns={sum_column: f'sum_{sum_column}'})
    
    merged_df = pd.merge(
        pivot_other,
        pivot_sum,
        on=['DateTimeKey', 'PatientDurableKey', 'EncounterKey'],
        how='outer'
    )
    
    temp = merged_df#.copy()
    for column in exp_columns:
        if column not in merged_df.columns:
            last_col_index = len(merged_df.columns) - 1
            temp.insert(last_col_index, column, np.nan)
    
    temp['max_culture'] = temp['max_culture'].fillna(0)
    temp['max_image'] = temp['max_image'].fillna(0)
    temp.columns.name = None
    pivot_df = temp[exp_columns]
    pivot_df = pivot_df.interpolate()
    pivot_df['NaN_Percentage'] = pivot_df.isna().mean().mean()
    return pivot_df


def find_values_for_patient3(filtered_dfs, exp_columns, sum_column):
    # Concatenate all dataframes at once instead of iteratively
    total_df = pd.concat(filtered_dfs, ignore_index=True)
    
    # Use boolean indexing for filtering - more efficient than copy()
    mask_sum = total_df['DisplayName'] == sum_column
    sum_df = total_df[mask_sum]
    other_df = total_df[~mask_sum]
    
    # Create both pivot tables with optimized settings
    pivot_tables = {
        'sum': pd.pivot_table(
            sum_df,
            index=['DateTimeKey', 'PatientDurableKey', 'EncounterKey'],
            columns='DisplayName',
            values='Value',
            aggfunc='sum'
        ),
        'other': pd.pivot_table(
            other_df,
            index=['DateTimeKey', 'PatientDurableKey', 'EncounterKey'],
            columns='DisplayName',
            values='Value',
            aggfunc=['mean', 'min', 'max']
        )
    }
    
    # Process pivot tables efficiently
    for name, pivot in pivot_tables.items():
        pivot.reset_index(inplace=True)
        if name == 'other':
            # Optimize column renaming with list comprehension
            pivot.columns = [f'{i}_{j}' if j else f'{i}' for i, j in pivot.columns]
        else:
            # Rename sum column directly
            pivot.rename(columns={sum_column: f'sum_{sum_column}'}, inplace=True)
    
    # Merge dataframes efficiently
    result_df = pd.merge(
        pivot_tables['other'],
        pivot_tables['sum'],
        on=['DateTimeKey', 'PatientDurableKey', 'EncounterKey'],
        how='outer'
    )
    
    # Add missing columns efficiently
    missing_columns = set(exp_columns) - set(result_df.columns)
    if missing_columns:
        for column in missing_columns:
            result_df[column] = np.nan
    
    # Optimize specific column operations
    result_df[['max_culture', 'max_image']] = result_df[['max_culture', 'max_image']].fillna(0)
    
    # Select and process final output efficiently
    final_df = result_df[exp_columns]
    final_df.interpolate(inplace=True)
    
    # Calculate NaN percentage efficiently
    final_df['NaN_Percentage'] = final_df.isna().mean().mean()
    
    return final_df
    
def find_values_for_patient2(filtered_dfs, exp_columns):
    total_df = pd.concat(filtered_dfs)
    pivot_df = total_df.pivot_table(index=['DateTimeKey','PatientDurableKey','EncounterKey'], columns='DisplayName', values='Value', aggfunc=['mean','min','max']).reset_index()
    pivot_df.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in pivot_df.columns]
    temp = pivot_df
    for column in exp_columns:
        if column not in pivot_df.columns:
            last_col_index = len(pivot_df.columns) - 1
            temp.insert(last_col_index, column, np.nan)
    pivot_df['max_culture'] = pivot_df['max_culture'].fillna(0)
    pivot_df['max_image'] = pivot_df['max_image'].fillna(0)
    pivot_df.columns.name = None
    pivot_df = temp[exp_columns] 
    pivot_df = pivot_df.interpolate()
    pivot_df['NaN_Percentage'] = pivot_df.isna().mean().mean()
    return pivot_df

def get_images(encounter_keys):
    encounters =  ", ".join(str(key) for key in encounter_keys)

    sql_query = f"""
    SELECT
        ImagingKey,
        PatientDurableKey,
        OrderingEncounterKey,
        OrderingDateKey,
        ExamEndDateKey
    FROM  CAB_EPIC.dbo.ImagingFact
    WHERE OrderingEncounterKey IN ({encounters})
    """
    return sql_query

def non_sepsis_images(patient_keys):
    patients =  ", ".join(str(key) for key in patient_keys)

    sql_query = f"""
    SELECT
        ImagingKey,
        PatientDurableKey,
        OrderingEncounterKey,
        OrderingDateKey,
        ExamEndDateKey
    FROM  CAB_EPIC.dbo.ImagingFact
    WHERE PatientDurableKey IN ({patients})
    """
    return sql_query

def non_sepsis_image_text(img_keys):
    images =  ", ".join(str(key) for key in img_keys)

    sql_query = f"""
    SELECT
        ImagingKey,
        Impression,
        Narrative
    FROM  CAB_EPIC.dbo.ImagingTextFact
    WHERE ImagingKey IN ({images})
    """
    return sql_query
       
def execute_sql_query_dem(patients):
    
    sql_query = f"""
        SELECT 
            DurableKey,
            PatientKey,
            Sex,
            AgeInYears,
            Ethnicity,
            MultiRacial,
            FirstRace,
            SmokingStatus
        FROM CAB_EPIC.dbo.PatientDim
        WHERE DurableKey IN ({patients})
        """
    return sql_query

def get_EdInfo(encounter_keys):
    filtro = ", ".join(str(key) for key in encounter_keys)
    
    sql_query = f""" select EncounterKey, IsEdVisit
                    from CAB_EPIC.dbo.EncounterFact where EncounterKey in ({filtro})"""
    return sql_query

def combine_unique_dates(dfs):
    datetime_columns = []
    
    # Process dfs_flow dataframes
    for df in dfs:
        if 'DateKey' in df.columns and 'TimeOfDayKey' in df.columns:
            try:
                # Ensure TimeOfDayKey is in correct HHMM format
                time_of_day = df['TimeOfDayKey'].astype(str).str.zfill(4)  # Pad with leading zeros if necessary
                datetime_column = pd.to_datetime(df['DateKey'].astype(str) + time_of_day, format='%Y%m%d%H%M')
                datetime_columns.append(datetime_column)
            except ValueError as e:
                print(f"Error processing dataframe in dfs_flow: {e}")
    
    # Concatenate all datetime columns into one series
    all_datetimes_series = pd.concat(datetime_columns, ignore_index=True)
    
    # Drop duplicates to get unique date/times
    unique_datetimes = all_datetimes_series.drop_duplicates()
    
    # Create a new dataframe with unique date/times
    unique_datetimes_df = pd.DataFrame(unique_datetimes, columns=['datetime'])
    
    return unique_datetimes_df

def categorize_time(hour):
    if hour < 8:
        return '1'
    elif 8 <= hour < 16:
        return '2'
    else:
        return '3'

def transform_datetime_format(df, datetime_column):
    # Extraer la fecha y hora del datetime
    df['Date'] = df[datetime_column].dt.strftime('%Y%m%d')
    df['Hour'] = df[datetime_column].dt.hour
    df['TimeCategory'] = df['Hour'].apply(categorize_time)
    df['DateTimeKey'] = df['Date']  + df['TimeCategory']
    df.drop(['Date', 'Hour', 'TimeCategory'], axis=1, inplace=True)
    return df

def date_format(total_df, image=None):
    DateArray = DateGenerator(2015, 2025)
    #OBTAIN TIME CODE 1, 2, 3
    total_df2 = total_df.copy()
    total_df2['DateKey'] = total_df2['DateKey'].astype(str)
    if image == None:
        total_df2['TimeValue'] = total_df2['TimeOfDayKey'].apply(asignar_valor)
        total_df2['DateTimeKey'] = total_df2['DateKey'] + total_df2['TimeValue'].astype(str)
        total_df2['Value'] = pd.to_numeric(total_df2['Value'], errors='coerce')
        xt = total_df2[['DateTimeKey', 'Value']]
    else:
        if not total_df.empty:
            total_df2['DateTimeKey'] = total_df2.apply(lambda row: [row['DateKey'] + suffix for suffix in ['1', '2', '3']], axis=1)
            total_df2 = total_df2.explode('DateTimeKey')
            total_df2 = total_df2.drop('DateKey', axis=1)
            xt = total_df2[['DateTimeKey', 'Value']]
        else:
            xt = total_df2[['DateKey', 'Value']]
            xt = xt.rename(columns={'DateKey':'DateTimeKey'})    
    xt = xt.sort_values(by='DateTimeKey')
    return xt

def find_sepsis_dates(unique_datetimes_df_transformed, xt, xhr, xrr, xwb, xCU, xIM):
    dates = []
    for fecha in unique_datetimes_df_transformed['DateTimeKey']:
        # Counter to count how many criteria are met for this date.
        count1 = 0
        count2 = 0
        # Check the temperature for this date
        temperature_series = xt.loc[xt['DateTimeKey'] == fecha, 'Value'].astype(float)
        temp_values_satisfying_condition = temperature_series[(temperature_series > 100.4) | (temperature_series < 96.8)]
        if not temp_values_satisfying_condition.empty:
            count1 += 1

        # Check heart rate for this date
        heart_rate_series = xhr.loc[xhr['DateTimeKey'] == fecha, 'Value'].astype(float)
        heart_rate_values_satisfying_condition = heart_rate_series[heart_rate_series > 95]
        if not heart_rate_values_satisfying_condition.empty:
            count1 += 1

        # Check respiratory rate for this date
        respiratory_rate_series = xrr.loc[xrr['DateTimeKey'] == fecha, 'Value'].astype(float)
        respiratory_rate_values_satisfying_condition = respiratory_rate_series[respiratory_rate_series > 23]
        if not respiratory_rate_values_satisfying_condition.empty:
            count1 += 1

        # Check respiratory WBC for this date
        WhiteBloodCells_series = xwb.loc[xwb['DateTimeKey'] == fecha, 'Value'].astype(float)
        WhiteBloodCells_values_satisfying_condition = WhiteBloodCells_series[(WhiteBloodCells_series > 12) | (WhiteBloodCells_series < 4)]
        if not WhiteBloodCells_values_satisfying_condition.empty:
            count1 += 1

        Cult_series = xCU.loc[xCU['DateTimeKey'] == fecha, 'Value']
        if not Cult_series.empty:
            count2 += 1

        Img_series = xIM.loc[xIM['DateTimeKey'] == fecha, 'Value']
        if not Img_series.empty:
            count2 += 1

        # If at least two criteria are met for this date, add the date to the list.
        if count1 >= 2 and count2 >= 1:
            dates.append(fecha)

    Dates_basic = np.unique(dates)
    criteria = pd.DataFrame(Dates_basic,columns=['DateTimeKey'])
    return criteria

def impute_value_og(x, lower_limit, upper_limit, bot_margin, top_margin):
    # If the value is missing or below lower limit, generate a random value above the lower limit
    if pd.isnull(x) or x < lower_limit:
        return np.random.uniform(lower_limit, lower_limit + bot_margin)
    
    elif x > upper_limit:
        return np.random.uniform(upper_limit, upper_limit + top_margin)
    
    return x

def impute_value(x, lower_limit, upper_limit, bot_margin, top_margin):
    if pd.isnull(x):
        return np.nan
    if x < lower_limit:
        return np.random.uniform(lower_limit, lower_limit + bot_margin)
    elif x > upper_limit:
        return np.random.uniform(upper_limit, upper_limit + top_margin)
    return x

def handle_outliers(df, lower_limit, upper_limit, col, bot_margin, top_margin):
    
    df[col] = df[col].apply(lambda x: impute_value(x, lower_limit, upper_limit, bot_margin, top_margin))
    
    return df

def convert_to_datetime_with_segment(date_time_key):
    date_str = str(date_time_key)[:-1]  # Extract date part
    segment = int(str(date_time_key)[-1])  # Extract segment part
    date = datetime.strptime(date_str, "%Y%m%d")
    
    if segment == 1:
        date = date.replace(hour=0, minute=0)
    elif segment == 2:
        date = date.replace(hour=8, minute=0)
    elif segment == 3:
        date = date.replace(hour=16, minute=0)
    
    return date

def date_f(total_df, column1, column2, column3):
    DateArray = DateGenerator(2015, 2025)
    #OBTAIN TIME CODE 1, 2, 3
    total_df2 = total_df.copy()
    total_df2['DateKey'] = total_df2[column1].astype(str)
    total_df2['TimeValue'] = total_df2[column2].apply(asignar_valor)
    #OBTAIN FINAL COLUMN YYYY-MM-DD-T
    total_df2[column3] = total_df2['DateKey'] + total_df2['TimeValue'].astype(str)
    xt = total_df2.drop(columns=['DateKey', 'TimeValue'])

    #ONLY SAVE VALUE AND DATE IN CODE YYYY-MM-DD-T
    xt = xt.sort_values(by=column3)
    return xt

def process_demographics_data(demogr_patients):
    # Drop duplicates
    demogr_patients = demogr_patients.drop_duplicates(
        subset=['DurableKey', 'Sex', 'AgeInYears', 'Ethnicity', 'MultiRacial', 'FirstRace'], 
        keep='last'
    )
    # Mapping dictionaries
    mapping = {
        'Former': 'Former Smoker',
        'Never': 'Never Smoker',
        'Never Smoker ' : 'Never Smoker',
        'Every Day': 'Current Smoker',
        'Current Every Day Smoker': 'Current Smoker',
        'Some Days': 'Occasional Smoker',
        'Current Some Day Smoker': 'Occasional Smoker',
        'Light Smoker': 'Occasional Smoker',
        'Unknown If Ever Smoked': 'Unknown',
        '*Unknown': 'Unknown',
        'Never Assessed': 'Unknown',
        'Passive Smoke Exposure - Never Smoker': 'Passive Smoke Exposure',
        'Smoker, Current Status Unknown': 'Unknown',
        'Heavy Tobacco Smoker': 'Heavy Smoker',
        'Light Tobacco Smoker': 'Occasional Smoker'
    }
    mapping2 = {
        'Patient Refused': 'Unknown',
        '': 'Unknown'
    }
    mapping3 = {
        'Patient Refused': 'Unknown',
        '*Unspecified': 'Unknown'
    }
    # Replace values in specified columns
    demogr_patients['SmokingStatus'] = demogr_patients['SmokingStatus'].replace(mapping)
    demogr_patients['FirstRace'] = demogr_patients['FirstRace'].replace(mapping2)
    demogr_patients['Ethnicity'] = demogr_patients['Ethnicity'].replace(mapping3)

    # Fill missing values in AgeInYears with the mean
    # mean_A = demogr_patients['AgeInYears'].mean()
    # demogr_patients['AgeInYears'].fillna(mean_A, inplace=True)
    demogr_patients['MultiRacial'].fillna(0, inplace=True)
    # Factorize categorical columns
    columns_to_factorize = ['Sex', 'Ethnicity', 'FirstRace', 'SmokingStatus']
    for column in columns_to_factorize:
        demogr_patients[column], _ = pd.factorize(demogr_patients[column])

    # Drop 'PatientKey' column
    demogr_patients.drop(columns=['PatientKey'], inplace=True)

    return demogr_patients


def replace_year(row, column_name):
    instant = row[column_name]
    if pd.isnull(instant):
        return None
    elif instant.year == 9999:
        return None
    else:
        return instant
     
def fill_na_with_date(row, column_name, hour, minute, second):
    if pd.isnull(row[column_name]):
        date_key = str(row['DateKey'])
        year = int(date_key[:4])
        month = int(date_key[4:6])
        day = int(date_key[6:])
        return pd.Timestamp(year, month, day, hour, minute, second)
    else:
        return row[column_name]

def get_period(hour):
    if hour < 8:
        return 1
    elif hour < 16:
        return 2
    else:
        return 3

def corregir_end_date(row):
    if row['EndDateKey'] == -2:
        return row['StartDateKey']
    else:
        return row['EndDateKey']

def convertir_formato_fecha(fecha_str):
    year = fecha_str[:4]
    month = fecha_str[4:6]
    day = fecha_str[6:8]
    numero = fecha_str[-1]
    fecha_nuevo_formato = f"{year}-{month}-{day}-{numero}"
    return fecha_nuevo_formato

def plot_data_with_limits(hr_max_values,hr_min_values,hr90,rr20,criteria,sb90,sb_min_values, sb_max_values,sblabelsposition,sblabels,xsb,ysb,la20,la_min_values, la_max_values,lalabelsposition,lalabels,xla,yla,rr_min_values, rr_max_values,wbc_min_values, wbc_max_values,Temp_min_values,Temp_max_values,t1004,t968,wbc12,wbc4,xtemp, ytemp, xwbc, ywbc, xRR, yRR, xHR, yHR, merged_df, templabelsposition, wblabelsposition, rrlabelsposition, hrlabelsposition, templabels, wblabels, rrlabels, hrlabels, DateArray,merged_diagnosis, xlim, ax=None):
    diaglabel=['D','i','a','g','n','o','s','i','s']
    diagglabelposition=np.linspace(40.6,35.6,9)
    position = list(templabelsposition) + list(wblabelsposition) + list(rrlabelsposition) + list(hrlabelsposition)+ list(lalabelsposition)+ list(sblabelsposition)+list(diagglabelposition)
    labels = list(templabels) + list(wblabels) + list(rrlabels) + list(hrlabels)+ list(lalabels)+ list(sblabels)+list(diaglabel)
    xcrit=criteria['endIndex']
    for x in xcrit:
        ax.axvline(x=x, color="#7c1158", linestyle='--', linewidth=1)

    min_value = merged_df[['startIndex', 'endIndex']].min().min()
    max_value = merged_df[['startIndex', 'endIndex']].max().max()
    floor = [min_value, max_value]

    if ax is None:
        fig, ax = plt.subplots(figsize=(40.6, 10))
    else:
        fig = ax.figure
        
    ax.plot(xtemp, ytemp, marker='.', linestyle='', label='Temperature', color="#7c1158", markersize=10)
    ax.plot(xwbc, ywbc, marker='.', linestyle='', label='WBC', color="#7c1158", markersize=10)
    ax.plot(xRR, yRR, marker='.', linestyle='', label='Respiratory Rate', color="#7c1158", markersize=10)
    ax.plot(xHR, yHR, marker='.', linestyle='', label='Heart Rate', color="#7c1158", markersize=10)
    ax.plot(xla, yla, marker='.', linestyle='', label='LA', color="#7c1158", markersize=10)
    ax.plot(xsb, ysb, marker='.', linestyle='', label='SBP', color="#7c1158", markersize=10)
    ax.set_yticks(position)
    ax.set_yticklabels(labels)
    ax.set_yticklabels(labels, fontsize=8)
    indmin = int(min(min(xtemp), min(xwbc), min(xRR), min(xRR)))
    indmax = int(max(max(xtemp), max(xwbc), min(xHR), min(xHR)))
    labelposition = list(range(indmin, indmax + 1))
    paso_muestreo = 1
    xlabelposition_muestreo = labelposition[::paso_muestreo]
    xlabels_muestreo = [DateArray[index] for index in xlabelposition_muestreo]
    xlabels_muestreo = [convertir_formato_fecha(fecha_str) for fecha_str in xlabels_muestreo]
    ax.set_xticks(xlabelposition_muestreo)
    ax.set_xticklabels(xlabels_muestreo, rotation=90)
    ax.fill_between(floor, 0, 5, color="#9b19f5", alpha=0.3)
    ax.fill_between(floor, 5 + 0.8, 10 + 0.8, color="#00bfa0", alpha=0.3)
    ax.fill_between(floor, 10 + 0.8 * 2, 15 + 0.8 * 2, color="#e60049", alpha=0.3)
    ax.fill_between(floor, 15 + 0.8 * 3, 20 + 0.8 * 3, color="#ffa300", alpha=0.3)
    ax.fill_between(floor, 20 + 0.8 * 4, 25 + 0.8 * 4, color="#b2e061", alpha=0.3)
    ax.fill_between(floor, 25 + 0.8 * 5, 30 + 0.8 * 5, color="#b3d4ff", alpha=0.3)

    for index, row in merged_df.iterrows():
        start = row['startIndex']
        end = row['endIndex']
        segment_xtemp = xtemp[(xtemp >= start) & (xtemp <= end)]
        segment_ytemp = ytemp[(xtemp >= start) & (xtemp <= end)]
        ax.plot(segment_xtemp, segment_ytemp, linestyle='-', linewidth=0.5, color="#7c1158")
        segment_xrr = xRR[(xRR >= start) & (xRR <= end)]
        segment_yrr = yRR[(xRR >= start) & (xRR <= end)]
        ax.plot(segment_xrr, segment_yrr, linestyle='-', linewidth=0.5, color="#7c1158")
        segment_xhr = xHR[(xHR >= start) & (xHR <= end)]
        segment_yhr = yHR[(xHR >= start) & (xHR <= end)]
        ax.plot(segment_xhr, segment_yhr, linestyle='-', linewidth=0.5, color="#7c1158")
        segment_xwbc = xwbc[(xwbc >= start) & (xwbc <= end)]
        segment_ywbc = ywbc[(xwbc >= start) & (xwbc <= end)]
        ax.plot(segment_xwbc, segment_ywbc, linestyle='-', linewidth=0.5, color="#7c1158")
        segment_xla = xla[(xla >= start) & (xla <= end)]
        segment_yla = yla[(xla >= start) & (xla <= end)]
        ax.plot(segment_xla, segment_yla, linestyle='-', linewidth=0.5, color="#7c1158")
        segment_xsb = xsb[(xsb >= start) & (xsb <= end)]
        segment_ysb = ysb[(xsb >= start) & (xsb <= end)]
        ax.plot(segment_xsb, segment_ysb, linestyle='-', linewidth=0.5, color="#7c1158")
        
    x1, x2 = floor
    rect = plt.Rectangle((x1, t1004), x2 - x1, 5  - t1004, fill=False, hatch='//', edgecolor='red', linewidth=0.1)
    ax.add_patch(rect)
    rect = plt.Rectangle((x1, 0), x2 - x1, t968 , fill=False, hatch='//', edgecolor='red', linewidth=0.1)
    ax.add_patch(rect)
    rect = plt.Rectangle((x1, wbc12), x2 - x1, 10 + 0.8 - wbc12, fill=False, hatch='//', edgecolor='red', linewidth=0.1)
    ax.add_patch(rect)
    rect = plt.Rectangle((x1, 5 + 0.8), x2 - x1, wbc4 - (5 + 0.8), fill=False, hatch='//', edgecolor='red', linewidth=0.1)
    ax.add_patch(rect)
    rect = plt.Rectangle((x1, rr20), x2 - x1, 15 + 0.8*2 - rr20, fill=False, hatch='//', edgecolor='red', linewidth=0.1)
    ax.add_patch(rect)
    rect = plt.Rectangle((x1, hr90), x2 - x1, 20 + 0.8*3 - hr90, fill=False, hatch='//', edgecolor='red', linewidth=0.1)
    ax.add_patch(rect)
    rect = plt.Rectangle((x1, la20), x2 - x1, 25 + 0.8*4 - la20, fill=False, hatch='//', edgecolor='red', linewidth=0.1)
    ax.add_patch(rect)
    rect = plt.Rectangle((x1, 5*5+0.8*5), x2 - x1, sb90 - (5*5+0.8*5), fill=False, hatch='//', edgecolor='red', linewidth=0.1)
    ax.add_patch(rect)

    difsepsis= merged_diagnosis['SepKey'].nunique()
    heith=5/difsepsis
    
    custom_colors = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]
    custom_colors = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
    custom_colors = ["#e60049", "#b2e061", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
    for index, row in merged_diagnosis.iterrows():
        color_index = (row['SepKey'] - 1) % len(custom_colors)
        color = custom_colors[color_index]
        ax.fill_betweenx(y=[row['SepKey_normalized'], row['SepKey_normalized'] + heith],
                         x1=row['startIndex'],
                         x2=row['endIndex'],
                         color=color,
                         alpha=0.7,
                         edgecolor='#7c1158',  # Color del borde
                         linewidth=2.5)

    ax.errorbar(xtemp, ytemp, yerr=[ytemp - Temp_min_values, Temp_max_values - ytemp], fmt='.', color="#7c1158", ecolor="#7c1158", capsize=4, elinewidth=2)
    ax.errorbar(xwbc, ywbc, yerr=[ywbc - wbc_min_values, wbc_max_values - ywbc], fmt='.', color="#7c1158", ecolor="#7c1158", capsize=4, elinewidth=2)
    ax.errorbar(xRR, yRR, yerr=[yRR - rr_min_values, rr_max_values - yRR], fmt='.', color="#7c1158", ecolor="#7c1158", capsize=4, elinewidth=2)
    ax.errorbar(xHR, yHR, yerr=[yHR - hr_min_values, hr_max_values - yHR], fmt='.', color="#7c1158", ecolor="#7c1158", capsize=4, elinewidth=2)
    ax.errorbar(xla, yla, yerr=[yla - la_min_values, la_max_values - yla], fmt='.', color="#7c1158", ecolor="#7c1158", capsize=4, elinewidth=2)
    ax.errorbar(xsb, ysb, yerr=[ysb - sb_min_values, sb_max_values - ysb], fmt='.', color="#7c1158", ecolor="#7c1158", capsize=4, elinewidth=2)
    ax.set_xlim(xlim)
    return fig

def explain_datetime_key_html(datetime_key):
    # Extraer el año, mes, día y t
    year = datetime_key[:4]
    month = datetime_key[4:6]
    day = datetime_key[6:8]
    time_index = datetime_key[8]
    
    # Definir el intervalo de tiempo según t
    if time_index == '1':
        time_period = "before 8:00 AM"
    elif time_index == '2':
        time_period = "between 8:00 AM and 4:00 PM"
    elif time_index == '3':
        time_period = "after 4:00 PM"
    else:
        time_period = "unknown time period"
    
    # Crear la explicación con etiquetas HTML para negrita y tamaño
    explanation_html = f"<b><span style='font-size:20px;'>The SEPSIS starts: {year}-{month}-{day}, and the time is {time_period}.</span></b>"
    return explanation_html


def evaluate_model(model, X_test, y_test, threshold=0.5):
    # Obtener las probabilidades predichas
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calcular las predicciones usando el umbral especificado
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    # ROC Curve y AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
    
    # Confusion Matrix, Sensitivity, Specificity
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print(f'Sensitivity: {sensitivity:.2f}')
    print(f'Specificity: {specificity:.2f}')
    
    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
    
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()
    
def intervalos_se_superponen(start1, end1, start2, end2):
    """Función para verificar si dos intervalos [start1, end1] y [start2, end2] se superponen."""
    return start1 <= end2 and start2 <= end1

def one_hot_encoding_chapter(diagnosis_key, df):
    total_chapters = 21
    chapter_number = df.loc[df['DiagnosisKey'] == diagnosis_key, 'chapter_number'].values
    
    if len(chapter_number) == 0:
        raise ValueError("DiagnosisKey no encontrado.")
    
    chapter_number = int(chapter_number[0])
    one_hot_vector = np.zeros(total_chapters)
    one_hot_vector[chapter_number - 1] = 1
    
    return one_hot_vector

def aggregate_one_hot_encodings(diagnosis_keys, df):
    total_chapters = 21
    aggregated_vector = np.zeros(total_chapters)
    
    for key in diagnosis_keys:
        one_hot_vector = one_hot_encoding_chapter(key, df)
        aggregated_vector += one_hot_vector
    
    # Asegurarse de que el vector sea binario (0 o 1)
    aggregated_vector = (aggregated_vector > 0).astype(int)
    
    return aggregated_vector

def concatenate_with_current(past_diagnosis_keys, current_diagnosis_key, df):
    past_vector = aggregate_one_hot_encodings(past_diagnosis_keys, df)
    current_vector = one_hot_encoding_chapter(current_diagnosis_key, df)
    concatenated_vector = np.concatenate([past_vector, current_vector])
    
    return concatenated_vector

def repeat_concatenated_vector(concatenated_vector, existing_matrix):
    num_rows = existing_matrix.shape[0]
    repeated_matrix = pd.DataFrame(np.tile(concatenated_vector, (num_rows, 1)))
    return repeated_matrix

def execute_sql_query_oxf(table_name):
    
    sql_query = f"""
    SELECT *
    FROM CAB_EPIC_Oxford.dbo.{table_name}
"""
    return sql_query

def calculate_sofa(row):
    score = 0
    
    # Respiratory: PaO2/FiO2
    if row['PaO2_FiO2'] < 100:
        score += 4
    elif row['PaO2_FiO2'] < 200:
        score += 3
    elif row['PaO2_FiO2'] < 300:
        score += 2
    elif row['PaO2_FiO2'] < 400:
        score += 1

    # Coagulation: Platelet count
    if row['Mean Platelet Count'] < 20:
        score += 4
    elif row['Mean Platelet Count'] < 50:
        score += 3
    elif row['Mean Platelet Count'] < 100:
        score += 2
    elif row['Mean Platelet Count'] < 150:
        score += 1

    # Liver: Bilirubin
    if row['Mean Bilirubin'] > 12.0:
        score += 4
    elif row['Mean Bilirubin'] >= 6.0:
        score += 3
    elif row['Mean Bilirubin'] >= 2.0:
        score += 2
    elif row['Mean Bilirubin'] >= 1.2:
        score += 1

    # Cardiovascular: Hypotension and vasopressors
    if row['Mean Mean Arterial Pressure'] < 70:
        if row['mean_DOPAMINE'] > 15.0 or row['mean_EPINEPHRINE'] > 0.1 or row['mean_NOREPINEPHRINE'] > 0.1:
            score += 4
        elif row['mean_DOPAMINE'] > 5.0 or (0.0 < row['mean_EPINEPHRINE'] <= 0.1) or (0.0 < row['mean_NOREPINEPHRINE'] <= 0.1):
            score += 3
        elif (0 < row['mean_DOPAMINE'] <= 5.0) or row['mean_DOBUTAMINE'] > 0.0:
            score += 2
        else:
            score += 1

    # Central Nervous System: Glasgow Coma Scale
    if row['Mean Glasgow Coma Scale Score'] < 6:
        score += 4
    elif row['Mean Glasgow Coma Scale Score'] <= 9:
        score += 3
    elif row['Mean Glasgow Coma Scale Score'] <= 12:
        score += 2
    elif row['Mean Glasgow Coma Scale Score'] <= 14:
        score += 1

    # Renal: Creatinine and urine output
    if row['Mean Creatinine'] >= 5.0 or row['Total Urine Output'] * 3 < 200:
        score += 4
    elif row['Mean Creatinine'] >= 3.5 or row['Total Urine Output'] * 3 < 500:
        score += 3
    elif 2.0 <= row['Mean Creatinine'] < 3.5:
        score += 2
    elif 1.2 <= row['Mean Creatinine'] < 2.0:
        score += 1

    return score


def drop_column_efficient(dataframes, column_name):
    dataframes_with_column = [df for df in dataframes if column_name in df.columns]
    for df in dataframes_with_column:
        df.drop(columns=[column_name], inplace=True)
    return dataframes 

def analyze_patient_texts(texts):
    # Crear una lista para almacenar los resultados
    results = []
    
    # Analizar cada texto
    for text in texts:
        # Obtener el puntaje de sentimiento usando VADER
        sentiment_score = analyzer.polarity_scores(text)
        for keyword in keywords:
            # Inicializar la valoración como "not found"
            context = "not found"
            # Verificar si la palabra clave está en el texto
            if keyword in text:
                # Evaluar el contexto de la palabra clave usando el puntaje de VADER
                context = "positive" if sentiment_score['compound'] > 0 else "negative"
            # Añadir el resultado para la palabra clave
            results.append({"Keyword": keyword, "Valuation": context})
    
    # Convertir los resultados en un DataFrame
    results_df = pd.DataFrame(results)
    
    # Condición 1: al menos una de las primeras 5 palabras tiene valoración 'positive'
    cond1 = (results_df.iloc[:5]['Valuation'] == 'positive').any()
   # cond1 = results_df.iloc[:5]['Valuation'].isin(['positive', 'not found']).any()
    cond2 = (results_df['Valuation'] == 'positive').sum() >= 4

    # Condición 2: al menos 4 de las últimas 6 palabras tienen valoración 'positive'
    #cond2 = (results_df.iloc[-6:]['Valuation'] == 'positive').sum() >= 4
    
    # Etiquetar al paciente
    #patient_label = 1 if cond else 0
    patient_label = 1 if cond1 or cond2 else 0
    return patient_label

import re

# Función para extraer la fecha en formato yyyymmdd
def extraer_fecha(texto):
    # Expresión regular para capturar fechas en formato "Admission Date: m/d/yy" o "m/d/yyyy" en la misma línea
    #fecha_match = re.search(r"Admission Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})", texto)
    #fecha_match = re.search(r"(?:Admission|ADM) Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})", texto)
    #fecha_match = re.search(r"(?:Admission|ADM|^|\s)Date:?\s*(\d{1,2}/\d{1,2}/\d{2,4})", texto)
    fecha_match = re.search(r"(?:Admission|ADM|^|\s)Date:?\s*(\d{1,2}/\d{1,2}/\d{2,4})|(?:DATE\s*OF\s*SERVICE\s*:?)\s*(\d{1,2}/\d{1,2}/\d{2,4})", texto, re.IGNORECASE)

    if fecha_match:
        # Capturar el mes/día/año
        #fecha_str = fecha_match.group(1)
        fecha_str = fecha_match.group(1) if fecha_match.group(1) else fecha_match.group(2)
        # Si el año tiene 2 dígitos, buscar el año en la siguiente línea
        if len(fecha_str.split('/')[-1]) == 2:
            # Intentar encontrar el año en la siguiente línea
            lineas = texto.splitlines()
            if len(lineas) > 1:
                siguiente_linea = lineas[1].strip()  # Línea siguiente, eliminando espacios en blanco
                if siguiente_linea.isdigit():  # Verificar si la línea contiene solo el año
                    fecha_str = f"{fecha_str[:-2]}{siguiente_linea}"  # Reemplazar el año

        # Convertir la fecha encontrada en el formato adecuado
        fecha = pd.to_datetime(fecha_str, format='%m/%d/%Y', errors='coerce')
        if pd.notnull(fecha):
            return fecha.strftime('%Y%m%d')  # Convertir a formato yyyymmdd
    
    return None  # Si no encuentra la fecha
# Crear un DataFrame con el texto y la fecha
def crear_dataframe(text_series):
    # Generar una lista de diccionarios con el texto y la fecha extraída
    data = [{'text': texto, 'date': extraer_fecha(texto)} for texto in text_series]
    return pd.DataFrame(data)

# Función para obtener el texto y la fecha más cercanos a la fecha de entrada
def obtener_texto_y_fecha(durable_key_value, df, input_date):
    filtered_row = df[df['PatientDurableKey'] == durable_key_value]
    
    if filtered_row.empty:
        return None  # Si no hay filas filtradas
    
    text_series = filtered_row['Text']
    resultado_df = crear_dataframe(text_series)
    
    # Convertir la columna 'date' a tipo datetime para comparación
    resultado_df['date'] = pd.to_datetime(resultado_df['date'], format='%Y%m%d', errors='coerce')

    # Filtrar la fila con la fecha más cercana a input_date
    if not resultado_df.empty:
        input_date_dt = pd.to_datetime(input_date, format='%Y%m%d')
        resultado_df['date_diff'] = (resultado_df['date'] - input_date_dt).abs()  # Calcular la diferencia absoluta
        closest_row = resultado_df.loc[resultado_df['date_diff'].idxmin()]  # Encontrar la fila con la fecha más cercana
        return closest_row[['text']]  # Devolver solo las columnas 'text' y 'date'
    
    return None  # Si no hay resultados

def process_patient_data(data_df, sizes, date_dict, min_consecutive):
    filtered_data = []
    start = 0
    #count = 0
    for size in tqdm(sizes, desc="Processing Data", unit="record", miniters=1000):
        pr = data_df.iloc[start:start + size]
        start += size
        pr['Index'] = pr['DateTimeKey'].map(date_dict)
        pr['consecutive_group'] = (pr['Index'] != pr['Index'].shift() + 1).cumsum()
        group_counts = pr.groupby('consecutive_group')['Index'].transform('count')
        pr_filtered = pr[group_counts >= min_consecutive]
        columns = [col for col in pr_filtered.columns if col != 'Label'] + ['Label']
        pr_filtered = pr_filtered[columns]
        for group in pr_filtered['consecutive_group'].unique():
            group_df = pr_filtered[pr_filtered['consecutive_group'] == group].drop(columns=['consecutive_group'])
            if len(group_df) >= min_consecutive:
               # group_df['Num']=count
                #count = count + 1
                filtered_data.append(group_df)

    return filtered_data


def preprocess_dataset_og(input_df, encoded_df, demographic_df, encounter_df, label_sirs_column=None):
    input_df['Label'] = 0
    if label_sirs_column:
        input_df[label_sirs_column] = 0
    
    # Impute missing values
    imputer = IterativeImputer()
    data_imputed = imputer.fit_transform(input_df)
    data_imputed_df = pd.DataFrame(data_imputed, columns=input_df.columns)
    
    if label_sirs_column:
        data_imputed_df = sepsis_dates(data_imputed_df, label_sirs_column)
    else:
        # Apply date manipulations
        data_imputed_df = sepsis_dates(data_imputed_df, 'Label')
    
    data_imputed_df['PatientDurableKey'] = data_imputed_df['PatientDurableKey'].astype('int64')
    data_imputed_df['DateTimeKey'] = data_imputed_df['DateTimeKey'].astype('int64')
    data_imputed_df = data_imputed_df.merge(demographic_df, how='left', left_on='PatientDurableKey', right_on='DurableKey')
    data_imputed_df = data_imputed_df.merge(encounter_df, how='left', on='EncounterKey')
    data_imputed_df = data_imputed_df.merge(encoded_df, how='left', on='EncounterKey')
    
    #data_imputed_df = data_imputed_df.dropna()
    
    return data_imputed_df


def preprocess_dataset(input_df, encoded_df, demographic_df, encounter_df, label_sirs_column=None):
    processed_df = input_df.copy()
    processed_df['Label'] = 0
    if label_sirs_column:
        processed_df[label_sirs_column] = 0
        processed_df = sepsis_dates(processed_df, label_sirs_column)
    else:
        processed_df = sepsis_dates(processed_df, 'Label')

    processed_df['PatientDurableKey'] = processed_df['PatientDurableKey'].astype('int64')
    processed_df['DateTimeKey'] = processed_df['DateTimeKey'].astype('int64')
    processed_df = processed_df.merge(demographic_df, how='left', left_on='PatientDurableKey', right_on='DurableKey')
    processed_df = processed_df.merge(encounter_df, how='left', on='EncounterKey')
    processed_df = processed_df.merge(encoded_df, how='left', on='EncounterKey')
    return processed_df

def obtain_text_one_encounter(durable_key_value, df):
    filtered_row = df[df['PatientDurableKey'] == durable_key_value]
    if filtered_row.empty:
        return None  # Return None if no rows are found
    return filtered_row[['Text']].iloc[0] 
   
def obtain_text_based_encounter(encounterkey, df):
    filtered_row = df[df['EncounterKey'] == encounterkey]
    if filtered_row.empty:
        return None  # Return None if no rows are found
    return filtered_row[['Text']].iloc[0] 

def extract_diagnosis_section(text):
    # Pattern to match any of the possible starting phrases
    start_patterns = [
        r"Discharge Diagnosis",
        r"Primary Discharge Diagnosis",
        r"Primary Diagnosis",
        r"Diagnosis Discharge",
        r"Principal Problem",
        r"Final Diagnosis",
        r"HISTORY OF PRESENT ILLNESS"
    ]
    
    # Combine patterns with OR operator
    start_pattern = "|".join(start_patterns)
    
    # Pattern to match the end phrase
    end_patterns = [
        r"Discharge Disposition",
        r"Procedures",
        r"Discharge Medications",
        r"Consultations",
        r" Past Medical History",
        r"mg",
        r"DETAILS OF HOSPITAL STAY",
        r"Diagnostic Studies"
    ]
    end_pattern = "|".join(end_patterns)
    
    # Create the full pattern
    pattern = f"({start_pattern})(.+?)(?={end_pattern})"
    
    # Find the match with re.IGNORECASE flag
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        # Return the content between the start and end patterns
        return (match.group(1) + match.group(2)).strip()
    return ""



def extract_diagnosis_section2(text):
    # Pattern to match any of the possible starting phrases
    start_patterns = [
        r"Hospital Course",
        r"Hospital  Course",
        r"Details of Hospital Stay",
        r"Details  of Hospital Stay",
        r"COURSE IN THE HOSPITAL",
        r"HISTORY:",
        r"Discharge Diagnoses",
        r"Discharge diagnosis",
        r"DETAILED ICU COURSE:",
        r"Diagnoses At Time of Death:",
        r"HISTORY OF PRESENT ILLNESS"
    ]
    
    # Combine patterns with OR operator
    start_pattern = "|".join(start_patterns)
    
    # Pattern to match the end phrase
    end_patterns = [
        r"Inpatient Treatments:",
        r"Consults:",
        r"Procedures:",
        r"Consultations",
        r"Diagnostic Studies",
        r"Discharge Medications",
        r"Results from last 7 days",
        r"Discharge Condition",
        r"Physical Exam:",
        r"Last CBC  Lab Results",
        r"Last Lab Results",
        r"ml"
    ]
    end_pattern = "|".join(end_patterns)
    
    # Create the full pattern
    pattern = f"({start_pattern})(.+?)(?={end_pattern})"
    
    # Find the match with re.IGNORECASE flag
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        # Return the content between the start and end patterns
        return (match.group(1) + match.group(2)).strip()
    
    start_only_pattern = f"({start_pattern})(.+)"
    start_only_match = re.search(start_only_pattern, text, re.DOTALL | re.IGNORECASE)
    if start_only_match:
        return (start_only_match.group(1) + start_only_match.group(2)).strip()

    # Case 3: End pattern exists, but no start pattern
    end_only_pattern = f"(.+?)(?={end_pattern})"
    end_only_match = re.search(end_only_pattern, text, re.DOTALL | re.IGNORECASE)
    if end_only_match:
        return end_only_match.group(1).strip()
  
    return ""


import re

def extract_diagnosis_section3(text):
    # Define patterns in order of priority
    prioritized_patterns = [
        r"Hospital Course",
        r"Hospital  Course",
        r"Details of Hospital Stay",
        r"Details  of Hospital Stay",
        r"COURSE IN THE HOSPITAL",
        r"Presenting Problem/History of Present Illness",
        r"HISTORY:",
        r"Discharge Diagnoses",
        r"Discharge diagnosis",
        r"DETAILED ICU COURSE:",
        r"Diagnoses At Time of Death:",
        r"HISTORY OF PRESENT ILLNESS"
    ]
    
    # Define end patterns
    end_patterns = [
        r"Inpatient Treatments:",
        r"Consults:",
        r"Procedures:",
        r"Consultations",
        r"Diagnostic Studies",
        r"Discharge Medications",
        r"Results from last 7 days",
        r"Discharge Condition",
        r"Physical Exam:",
        r"Last CBC  Lab Results",
        r"Last Lab Results",
        r"ml"
    ]
    
    # Escape special regex characters in patterns
    prioritized_patterns = [re.escape(pattern) for pattern in prioritized_patterns]
    end_patterns = [re.escape(pattern) for pattern in end_patterns]
    end_pattern = "|".join(end_patterns)
    
    # Sequentially search for each prioritized pattern
    for start_pattern in prioritized_patterns:
        # Add word boundaries to ensure exact phrase matching
        pattern = f"({start_pattern})(.+?)(?={end_pattern})"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # Return the first matched section based on priority
            return (match.group(1) + match.group(2)).strip()
    
    # If no match is found, try alternative searches
    # Case 1: Start pattern only
    for start_pattern in prioritized_patterns:
        start_only_pattern = f"({start_pattern})(.+)"
        start_only_match = re.search(start_only_pattern, text, re.DOTALL | re.IGNORECASE)
        if start_only_match:
            return (start_only_match.group(1) + start_only_match.group(2)).strip()
    
    # Case 2: End pattern exists but no start pattern
    end_only_pattern = f"(.+?)(?={end_pattern})"
    end_only_match = re.search(end_only_pattern, text, re.DOTALL | re.IGNORECASE)
    if end_only_match:
        return end_only_match.group(1).strip()
    
    return ""

def extract_diagnosis_section5(text):
    # Define patterns in order of priority
    prioritized_patterns = [
        r"Hospital Course",
        r"Hospital\s+Course",
        r"COURSE IN THE HOSPITAL",
        #r"HISTORY:",
        r"Details of Hospital Stay\s*",
        r"Details\s+of Hospital Stay\s*",
        r"Presenting Problem/History of Present Illness",
        r"Presenting Problem/History of\s+Present Illness",
        r"Presenting Problem",
        r"History of\s+Present Illness",
        r"History of Present Illness",
        r"HISTORY OF PRESENT ILLNESS",
        r"Discharge Diagnoses",
        r"Discharge diagnosis",
        r"DETAILED ICU COURSE:",
        r"Diagnoses At Time of Death:"
    ]
    
    # Define end patterns in order of priority
    end_patterns = [
        r"Consults:",
        r"Consults:\s*",
        r"Inpatient Treatments:",
        r"Procedures:",
        r"Consultations",
        r"Diagnostic Studies",
        r"Discharge Medications",
        r"Results from last 7 days",
        r"Discharge Condition",
        r"Physical Exam:",
        r"Last CBC\s+Lab Results",
        r"Last Lab Results",
        r"ml"
    ]
    
    # Check each start pattern with each end pattern in order
    for start_pattern in prioritized_patterns:
        for end_pattern in end_patterns:
            pattern = f"({start_pattern})\s*(.+?)(?={end_pattern})"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return (match.group(1) + match.group(2)).strip()
    
    # If no match found with both start and end, try with just start patterns
    for start_pattern in prioritized_patterns:
        start_only_pattern = f"({start_pattern})\s*(.+)"
        start_only_match = re.search(start_only_pattern, text, re.DOTALL | re.IGNORECASE)
        if start_only_match:
            return (start_only_match.group(1) + start_only_match.group(2)).strip()
    
    # If still no match, try finding anything before the first end pattern
    for end_pattern in end_patterns:
        end_only_pattern = f"(.+?)(?={end_pattern})"
        end_only_match = re.search(end_only_pattern, text, re.DOTALL | re.IGNORECASE)
        if end_only_match:
            return end_only_match.group(1).strip()
    
    return ""

import re
def extract_diagnosis_section4(text):
    # Define patterns in order of priority
    prioritized_patterns = [
        r"Details of Hospital Stay\s*",
        r"Hospital Course",
        r"Hospital\s+Course",
        r"COURSE IN THE HOSPITAL",
        r"Details\s+of Hospital Stay\s*",
        r"Presenting Problem/History of Present Illness",
        r"Presenting Problem/History of\s+Present Illness",
        r"Presenting Problem",
        r"History of\s+Present Illness",
        r"History of Present Illness",
        r"HISTORY OF PRESENT ILLNESS",
        r"Discharge Diagnoses",
        r"Discharge diagnosis",
        r"DETAILED ICU COURSE:",
        r"Diagnoses At Time of Death:",
        r"Assessment & Plan",
        r"Impression:"]
    
    # Define end patterns in order of priority
    end_patterns = [
        r"Consults:",
        r"Consults:\s*",
        r"Inpatient Treatments:",
        r"Procedures:",
        r"Consultations",
        r"Results from last 7 days",
        r"Discharge Disposition",
        r"Diagnostic Studies",
        r"Discharge Medications",
        r"Discharge Condition",
        r"Physical Exam:",
        r"Last CBC\s+Lab Results",
        r"Last Lab Results",
        r" Discharge Orders",
        r"Administered Medications",
        r"ml"
    ]
    
    # Check each start pattern with each end pattern in order
    for start_pattern in prioritized_patterns:
        for end_pattern in end_patterns:
            pattern = f"({start_pattern})\s*(.+?)(?={end_pattern})"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return (match.group(1) + match.group(2)).strip()
    
    # If no match found with both start and end, try with just start patterns
    for start_pattern in prioritized_patterns:
        start_only_pattern = f"({start_pattern})\s*(.+)"
        start_only_match = re.search(start_only_pattern, text, re.DOTALL | re.IGNORECASE)
        if start_only_match:
            return (start_only_match.group(1) + start_only_match.group(2)).strip()
    
    # If still no match, try finding anything before the first end pattern
    for end_pattern in end_patterns:
        end_only_pattern = f"(.+?)(?={end_pattern})"
        end_only_match = re.search(end_only_pattern, text, re.DOTALL | re.IGNORECASE)
        if end_only_match:
            return end_only_match.group(1).strip()
    
    return ""

import re
def extract_diagnosis_section6(text):
    prioritized_patterns = [
        r"Hospital Course",
        r"Details of Hospital Stay\s*",
        r"Hospital\s+Course",
        r"COURSE IN THE HOSPITAL",
        r"Details\s+of Hospital Stay\s*",
        r"Presenting Problem/History of Present Illness",
        r"Presenting Problem/History of\s+Present Illness",
        r"Presenting Problem",
        r"History of\s+Present Illness",
        r"History of Present Illness",
        r"HISTORY OF PRESENT ILLNESS",
        r"Discharge Diagnoses",
        r"Discharge diagnosis",
        r"DETAILED ICU COURSE:",
        r"Diagnoses At Time of Death:",
        r"Assessment & Plan",
        r"Impression:"]
    
    # Define end patterns in order of priority
    end_patterns = [
        r"Consults:",
        r"Consults:\s*",
        r"Inpatient Treatments:",
        r"Procedures:",
        r"Consultations",
        r"Administered\s+Medications",
        r"Results from last 7 days",
        r"Discharge Disposition",
        r"Diagnostic Studies",
        r"Discharge Medications",
        r"Discharge Condition",
        r"Physical Exam:",
        r"Last CBC\s+Lab Results",
        r"Last Lab Results",
        r" Discharge Orders",
        #r"Administered\s+Medications",
        r"ml"
    ]

    candidates = []

    # Collect all start+end matches
    for priority, start_pattern in enumerate(prioritized_patterns):
        for end_pattern in end_patterns:
            pattern = f"({start_pattern})\\s*(.+?)(?={end_pattern})"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1) + match.group(2)
                candidates.append((priority, match.start(), content.strip()))

    # Collect all start-only matches
    for priority, start_pattern in enumerate(prioritized_patterns):
        pattern = f"({start_pattern})\\s*(.+)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1) + match.group(2)
            candidates.append((priority, match.start(), content.strip()))

    # Return the one with the highest priority (lowest priority index)
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))  # sort by priority, then by position
        return candidates[0][2]

    # Last resort: find anything before an end pattern
    for end_pattern in end_patterns:
        pattern = f"(.+?)(?={end_pattern})"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return text.strip()

def create_sliding_windows(patient_df, window_size):
    n_windows = len(patient_df) - window_size + 1
    feature_data = patient_df.drop(columns=['EncounterKey', 'Label']).values
    full_windows = sliding_window_view(feature_data, (window_size, feature_data.shape[1]), axis=(0, 1))
    
    windows = full_windows[:n_windows]
    X = windows.reshape(n_windows, -1)
    
    y = patient_df['Label'].values[window_size - 1 : window_size - 1 + n_windows]
    encounter_keys = patient_df['EncounterKey'].values[:n_windows].tolist()
    datetime_keys = patient_df['DateTimeKey'].iloc[window_size - 1 : window_size - 1 + n_windows].tolist()

    return X, y, encounter_keys, datetime_keys


def process_population_data(population_data, window_size=6):
    patients_dict = defaultdict(list)
    for df in population_data:
        patient_id = df['PatientDurableKey'].iloc[0]
        patients_dict[patient_id].append(df)
    
    unique_patient_ids = list(patients_dict.keys())
    train_patient_ids, test_patient_ids = train_test_split(unique_patient_ids, test_size=0.3, random_state=42)
    
    patients_train = [df.copy() for pid in train_patient_ids for df in patients_dict[pid]]
    patients_test = [df.copy() for pid in test_patient_ids for df in patients_dict[pid]]
    
    patients_train = drop_column_efficient(patients_train, 'Label_sirs')
    patients_test = drop_column_efficient(patients_test, 'Label_sirs')
    for df in patients_train:
        df.drop(columns=['PatientDurableKey', 'Index', 'DurableKey', 'max_culture', 'max_image', 'Num', 'NaN_Percentage'], inplace=True)

    for df in patients_test:
        df.drop(columns=['PatientDurableKey', 'Index', 'DurableKey', 'max_culture', 'max_image', 'Num', 'NaN_Percentage'], inplace=True)

    X_train_all, y_train_all, train_encounter_keys, train_datetime_keys = [], [], [], []
    for patient_df in tqdm(patients_train, desc="Processing Training Patients"):
        X_patient, y_patient, encounter_keys_patient, datetime_keys_patient = create_sliding_windows(patient_df, window_size)
        X_train_all.append(X_patient)
        y_train_all.append(y_patient)
        train_encounter_keys.extend(encounter_keys_patient)
        train_datetime_keys.extend(datetime_keys_patient)
    
    X_train_combined = np.concatenate(X_train_all, axis=0)
    y_train_combined = np.concatenate(y_train_all, axis=0)
    
    X_test_all, y_test_all, test_encounter_keys, test_datetime_keys = [], [], [], []
    for patient_df in tqdm(patients_test, desc="Processing Testing Patients"):
        X_patient, y_patient, encounter_keys_patient, datetime_keys_patient = create_sliding_windows(patient_df, window_size)
        X_test_all.append(X_patient)
        y_test_all.append(y_patient)
        test_encounter_keys.extend(encounter_keys_patient)
        test_datetime_keys.extend(datetime_keys_patient)
    
    X_test_combined = np.concatenate(X_test_all, axis=0)
    y_test_combined = np.concatenate(y_test_all, axis=0)
    
    scaler = StandardScaler()
    X_train_combined = scaler.fit_transform(X_train_combined)
    X_test_combined = scaler.transform(X_test_combined)
    
    return {
        'X_train': X_train_combined,
        'y_train': y_train_combined,
        'X_test': X_test_combined,
        'y_test': y_test_combined,
        'test_encounter_keys': test_encounter_keys,
        'test_datetime_keys': test_datetime_keys
    }


def plot_roc_curves(populations, models):
    plt.figure(figsize=(15, 5))
    
    for i, (pop_name, pop_data) in enumerate(populations.items(), 1):
        plt.subplot(1, len(populations), i)
        
        for model_name, y_pred_proba in pop_data['y_pred_proba'].items():
            fpr, tpr, _ = roc_curve(pop_data['y_true'], y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, 
                     label=f'{model_name} (AUC = {roc_auc:.2f})', 
                     linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {pop_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def rename_columns(df):
    new_column_names = {
    'mean_Procalcitonin': 'Mean Procalcitonin',
    'mean_C-ReactiveProtein': 'Mean C-Reactive Protein',
    'mean_Albumin': 'Mean Albumin',
    'mean_CO2': 'Mean CO2',
    'mean_Chloride': 'Mean Chloride',
    'mean_Potassium': 'Mean Potassium',
    'mean_Sodium': 'Mean Sodium',
    'mean_BUN': 'Mean Blood Urea Nitrogen',
    'mean_NeutrCount': 'Mean Neutrophil Count',
    'mean_LympCount': 'Mean Lymphocyte Count',
    'mean_ALT': 'Mean Alanine Aminotransferase',
    'mean_ALP': 'Mean Alkaline Phosphatase',
    'mean_AST': 'Mean Aspartate Aminotransferase',
    'mean_SpO2': 'Mean SpO2',
    'mean_SystolicBP': 'Mean Systolic Blood Pressure',
    'mean_DiastolicBP': 'Mean Diastolic Blood Pressure',
    'mean_Bilirubin_Total': 'Mean Bilirubin',
    'mean_Creatinine': 'Mean Creatinine',
    'mean_Fio2': 'Mean FiO2',
    'mean_Glasgow_Coma_Scale_Score': 'Mean Glasgow Coma Scale Score',
    'mean_Height': 'Mean Height',
    'mean_Lactic_Acid': 'Mean Lactic Acid',
    'mean_NIBP_MAP': 'Mean Mean Arterial Pressure',
    'mean_Platelet_Count': 'Mean Platelet Count',
    'mean_Pulse': 'Mean Pulse',
    'mean_Resp': 'Mean Respiratory Rate',
    'mean_Temp': 'Mean Temperature',
    'mean_WBC': 'Mean White Blood Cell Count',
    'mean_Weight': 'Mean Weight',
    'mean_pO2_Arterial': 'Mean PaO2',
    'min_Procalcitonin': 'Minimum Procalcitonin',
    'min_C-ReactiveProtein': 'Minimum C-Reactive Protein',
    'min_Albumin': 'Minimum Albumin',
    'min_CO2': 'Minimum CO2',
    'min_Chloride': 'Minimum Chloride',
    'min_Potassium': 'Minimum Potassium',
    'min_Sodium': 'Minimum Sodium',
    'min_BUN': 'Minimum Blood Urea Nitrogen',
    'min_NeutrCount': 'Minimum Neutrophil Count',
    'min_LympCount': 'Minimum Lymphocyte Count',
    'min_ALT': 'Minimum Alanine Aminotransferase',
    'min_ALP': 'Minimum Alkaline Phosphatase',
    'min_AST': 'Minimum Aspartate Aminotransferase',
    'min_SpO2': 'Minimum SpO2',
    'min_SystolicBP': 'Minimum Systolic Blood Pressure',
    'min_DiastolicBP': 'Minimum Diastolic Blood Pressure',
    'min_Bilirubin_Total': 'Minimum Bilirubin',
    'min_Creatinine': 'Minimum Creatinine',
    'min_Fio2': 'Minimum FiO2',
    'min_Glasgow_Coma_Scale_Score': 'Minimum Glasgow Coma Scale Score',
    'min_Lactic_Acid': 'Minimum Lactic Acid',
    'min_NIBP_MAP': 'Minimum Mean Arterial Pressure',
    'min_Platelet_Count': 'Minimum Platelet Count',
    'min_Pulse': 'Minimum Pulse',
    'min_Resp': 'Minimum Respiratory Rate',
    'min_Temp': 'Minimum Temperature',
    'min_WBC': 'Minimum White Blood Cell Count',
    'min_pO2_Arterial': 'Minimum PaO2',
    'max_Procalcitonin': 'Maximum Procalcitonin',
    'max_C-ReactiveProtein': 'Maximum C-Reactive Protein',
    'max_Albumin': 'Maximum Albumin',
    'max_CO2': 'Maximum CO2',
    'max_Chloride': 'Maximum Chloride',
    'max_Potassium': 'Maximum Potassium',
    'max_Sodium': 'Maximum Sodium',
    'max_BUN': 'Maximum Blood Urea Nitrogen',
    'max_NeutrCount': 'Maximum Neutrophil Count',
    'max_LympCount': 'Maximum Lymphocyte Count',
    'max_ALT': 'Maximum Alanine Aminotransferase',
    'max_ALP': 'Maximum Alkaline Phosphatase',
    'max_AST': 'Maximum Aspartate Aminotransferase',
    'max_SpO2': 'Maximum SpO2',
    'max_SystolicBP': 'Maximum Systolic Blood Pressure',
    'max_DiastolicBP': 'Maximum Diastolic Blood Pressure',
    'max_Bilirubin_Total': 'Maximum Bilirubin',
    'max_Creatinine': 'Maximum Creatinine',
    'max_Fio2': 'Maximum FiO2',
    'max_Glasgow_Coma_Scale_Score': 'Maximum Glasgow Coma Scale Score',
    'max_Lactic_Acid': 'Maximum Lactic Acid',
    'max_NIBP_MAP': 'Maximum Mean Arterial Pressure',
    'max_Platelet_Count': 'Maximum Platelet Count',
    'max_Pulse': 'Maximum Pulse',
    'max_Resp': 'Maximum Respiratory Rate',
    'max_Temp': 'Maximum Temperature',
    'max_WBC': 'Maximum White Blood Cell Count',
    'max_pO2_Arterial': 'Maximum PaO2',
    'sum_Urine': 'Total Urine Output',
    'max_culture': 'Culture Exam',
    'max_image': 'Image Exam'
    }

    # Rename the columns
    df.rename(columns=new_column_names, inplace=True)
    
    return df


def process_epic_scores(epic_total, pop_data):
    # Copy the data
#     epic_total = pd.concat([scores, scores_non])
#     pop_data = pd.concat(filtered_data)

#     # Apply transformations
#     epic_total['TimeValue'] = epic_total['ScoreTimeOfDayKey'].apply(asignar_valor)
#     epic_total['ScoreDateKey'] = epic_total['ScoreDateKey'].astype(str)
#     epic_total['DateTimeKey'] = epic_total['ScoreDateKey'] + epic_total['TimeValue'].astype(str)
#     epic_total['ScoreDateKey'] = epic_total['ScoreDateKey'].astype('float64')
#     epic_total['DateTimeKey'] = epic_total['DateTimeKey'].astype('int64')
#     epic_total['EncounterKey'] = epic_total['EncounterKey'].astype('float64')

#     # Group and sort
#     epic_total = epic_total.groupby(['EncounterKey', 'DateTimeKey'], as_index=False)['TotalScore'].max()
#     epic_total = epic_total.sort_values(by=['EncounterKey', 'DateTimeKey'])
#     epic_total = epic_total.rename(columns={'TotalScore': 'EpicScore'})
    
#     scaler = MinMaxScaler()
#     epic_total['NormalizedEpicScore'] = scaler.fit_transform(epic_total[['EpicScore']])

    # Merge with population data and clean
    # epic_lb = epic_total.merge(pop_data, on=['EncounterKey', 'DateTimeKey'], how='left')
    # epic_lb = epic_lb.dropna()
    # epic_lb = epic_lb[['EncounterKey', 'DateTimeKey', 'EpicScore', 'NormalizedEpicScore', 'Label']]

    keys = pop_data[['EncounterKey', 'DateTimeKey', 'Label']].drop_duplicates(['EncounterKey', 'DateTimeKey'])
    epic_lb = epic_total.merge(keys, on=['EncounterKey', 'DateTimeKey'], how='inner')
    epic_lb = epic_lb.dropna(subset=['NormalizedEpicScore', 'Label'])
    epic_lb = epic_lb[['EncounterKey', 'DateTimeKey', 'EpicScore', 'NormalizedEpicScore', 'Label']]

    return epic_lb


def convert_to_mcg_per_kg_min(row):
    if row['DoseUnit'] == 'mcg/min':
        return row['Dose'] / row['Mean Weight'], 'mcg/kg/min'
    else:
        return row['Dose'], 'mcg/kg/min'


from sklearn.preprocessing import MinMaxScaler
def process_population_sofa(population_sofa, Meds=None, full_data=None):
    # Concatenate filtered data
    if full_data is not None:
        population_sofa = population_sofa.merge(
            full_data[['EncounterKey', 'DateTimeKey', 'Mean FiO2', 'Mean PaO2', 'Mean Weight', 'Total Urine Output', 'Mean Glasgow Coma Scale Score', 'Mean Lactic Acid', 'Mean Mean Arterial Pressure']],
            on=['EncounterKey', 'DateTimeKey'],
            how='left'
        )
#     population_sofa = pd.concat(filtered_data, axis=0, ignore_index=True)
    population_sofa = population_sofa[['DateTimeKey', 'EncounterKey',
        'Mean Bilirubin', 'Mean Creatinine', 'Mean FiO2',
        'Mean Glasgow Coma Scale Score', 'Mean Lactic Acid',
        'Mean Mean Arterial Pressure', 'Mean Platelet Count',
        'Total Urine Output', 'Mean Weight', 'Mean PaO2', 'Label']]

    # Compute PaO2/FiO2 ratio
    population_sofa['Mean FiO2'] = population_sofa['Mean FiO2'] / 100
    population_sofa['PaO2_FiO2'] = population_sofa['Mean PaO2'] / population_sofa['Mean FiO2']

    if Meds is not None:
        # Get medications data
        Meds = Meds[Meds['EncounterKey'].isin(population_sofa['EncounterKey'])]

        # MedicationKey mappings
        medication_keys_to_update = [
            133298, 135514, 135823, 135824, 135825,
            135826, 135853, 135895, 135896, 135897,
            135898, 135904, 136980, 140212, 140213,
            140579, 140580, 71537, 159230, 159231, 70104, 71269, 71327, 71358
        ]
        medication_keys_to_update2 = [
            145625,159229,127577, 73887, 71468, 71469, 70920, 32356
        ]
        medication_keys_to_update3 = [
            40543,40544, 52778, 71472, 73824, 159594
        ]
        medication_keys_to_update4 = [
            41801,41802,71470,73852,104744,123624
        ]
        Meds.loc[Meds['MedicationKey'].isin(medication_keys_to_update), 'Name'] = 'EPINEPHRINE'
        Meds.loc[Meds['MedicationKey'].isin(medication_keys_to_update2), 'Name'] = 'NOREPINEPHRINE'
        Meds.loc[Meds['MedicationKey'].isin(medication_keys_to_update3), 'Name'] = 'DOPAMINE'
        Meds.loc[Meds['MedicationKey'].isin(medication_keys_to_update4), 'Name'] = 'DOBUTAMINE'

        # DateTimeKey transformation
        Meds['TimeValue'] = Meds['AdministrationTimeOfDayKey'].apply(asignar_valor)
        Meds['AdministrationDateKey'] = Meds['AdministrationDateKey'].astype(str)
        Meds['DateTimeKey'] = Meds['AdministrationDateKey'] + Meds['TimeValue'].astype(str)
        Meds['AdministrationDateKey'] = Meds['AdministrationDateKey'].astype('float64')
        Meds['DateTimeKey'] = Meds['DateTimeKey'].astype('int64')
        Meds = Meds.merge(population_sofa[['EncounterKey', 'DateTimeKey', 'Mean Weight']], how='left', on=['EncounterKey', 'DateTimeKey'])
        Meds['Mean Weight'] = Meds['Mean Weight'] * 0.0283495231  # ounces to kg
        Meds = Meds.dropna()
        Meds = Meds[Meds['Dose'] != '']
        Meds['Dose'] = Meds['Dose'].astype(float)

        # Filter and convert dose units
        Meds = Meds[Meds['DoseUnit'].isin(['mcg/min', 'mcg/kg/min'])]
        Meds[['Dose', 'DoseUnit']] = Meds.apply(lambda row: convert_to_mcg_per_kg_min(row), axis=1, result_type='expand')

        # Pivot medications
        Meds = Meds[['EncounterKey', 'DateTimeKey','Dose', 'Name']]
        mds = Meds.pivot_table(index=['EncounterKey', 'DateTimeKey'], columns='Name', values='Dose', aggfunc=['mean']).reset_index()
        mds.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in mds.columns]
        mds = mds.fillna(0)

        # Final merge and cleanup
        #population_sofa = population_sofa.drop(columns=['Mean PaO2', 'Mean FiO2', 'Mean Weight'])
#         population_sofa = population_sofa.drop(columns=['Mean FiO2', 'Mean PaO2', 'Mean Weight', 'Total Urine Output', 'Mean Glasgow Coma Scale Score', 'Mean Lactic Acid'])
        population_sofa = population_sofa.merge(mds, how='left', on=['EncounterKey', 'DateTimeKey'])
        population_sofa = population_sofa.fillna(0)
    else:
        for col in ['mean_DOPAMINE', 'mean_EPINEPHRINE', 'mean_NOREPINEPHRINE', 'mean_DOBUTAMINE']:
            if col not in population_sofa.columns:
                population_sofa[col] = 0.0
            
    population_sofa['SOFA_Score'] = population_sofa.apply(calculate_sofa, axis=1)
    scaler = MinMaxScaler()
    population_sofa['NormalizedSofaScore'] = scaler.fit_transform(population_sofa[['SOFA_Score']])

    return population_sofa

def meet_sirs(pivot_df):
    df = pivot_df.copy()

    cols_to_convert = [
        'Mean Pulse', 'Mean Respiratory Rate', 'Mean Temperature', 'Mean White Blood Cell Count',
        'Minimum Pulse', 'Minimum Respiratory Rate', 'Minimum Temperature', 'Minimum White Blood Cell Count',
        'Maximum Pulse', 'Maximum Respiratory Rate', 'Maximum Temperature', 'Maximum White Blood Cell Count'
    ]
    df[cols_to_convert] = df[cols_to_convert].astype(float)

    temp_mask = (df['Mean Temperature'] > 100.4) | (df['Minimum Temperature'] > 100.4) | (df['Maximum Temperature'] > 100.4) | \
                (df['Mean Temperature'] < 96.8) | (df['Minimum Temperature'] < 96.8) | (df['Maximum Temperature'] < 96.8)

    pulse_mask = (df['Mean Pulse'] > 90) | (df['Minimum Pulse'] > 90) | (df['Maximum Pulse'] > 90)

    resp_mask = (df['Mean Respiratory Rate'] > 20) | (df['Minimum Respiratory Rate'] > 20) | (df['Maximum Respiratory Rate'] > 20)

    wbc_mask = (df['Mean White Blood Cell Count'] > 12) | (df['Minimum White Blood Cell Count'] > 12) | (df['Maximum White Blood Cell Count'] > 12) | \
               (df['Mean White Blood Cell Count'] < 4) | (df['Minimum White Blood Cell Count'] < 4) | (df['Maximum White Blood Cell Count'] < 4)

    count1 = temp_mask.astype(int) + pulse_mask.astype(int) + resp_mask.astype(int) + wbc_mask.astype(int)

    df['Sirs'] = np.where(count1 >= 2, 1, 0)

    return df


def fill_sepsis_intervals(sepsisLabels):
    sepsisLabels = sepsisLabels.copy()  # Avoid modifying the original list
    n = len(sepsisLabels)
    i = 0

    while i < n:
        # Look for the next 1 to identify the start
        if sepsisLabels[i] == 1:
            start = i

            # Search for the end: a 1 followed by at least 9 zeros
            end = start
            j = start + 1
            while j < n:
                if sepsisLabels[j] == 1:
                    # Check the next 9 values
                    lookahead = sepsisLabels[j+1:j+10]
                    if 1 not in lookahead:
                        end = j
                        break
                    else:
                        j += 1
                else:
                    j += 1

            # If no 'end' was found with the lookahead logic, use the last 1 seen
            if j >= n:
                end = n - 1
                while end > start and sepsisLabels[end] != 1:
                    end -= 1

            # Fill all values between start and end with 1
            for k in range(start, end + 1):
                sepsisLabels[k] = 1

            # Continue from the position after the end
            i = end + 1
        else:
            i += 1

    return sepsisLabels


def apply_outlier_handling(df):

    # Predefined rules for outlier handling
    outlier_rules = {
        "Mean Alkaline Phosphatase": (0, 300, 0, 40),
        "Minimum Alkaline Phosphatase": (0, 300, 0, 40),
        "Maximum Alkaline Phosphatase": (0, 300, 0, 40),
        "Mean Aspartate Aminotransferase": (0, 120, 0, 40),
        "Minimum Aspartate Aminotransferase": (0, 120, 0, 40),
        "Maximum Aspartate Aminotransferase": (0, 120, 0, 40),
        "Mean Alanine Aminotransferase": (0, 110, 0, 40),
        "Minimum Alanine Aminotransferase": (0, 110, 0, 40),
        "Maximum Alanine Aminotransferase": (0, 110, 0, 40),
    }

    # Apply rules
    for col, (lb, ub, min_val, max_val) in outlier_rules.items():
        if col in df.columns:
            df = handle_outliers(df, lb, ub, col, min_val, max_val)

    return df

def execute_score(encss):
    encs = ", ".join(str(s) for s in encss)
    
    sql_query = f"""
    select 
        EarlyDetectionOfSepsisRiskKey,
        EncounterKey,
        PatientDurableKey,
        ScoreDateKey,
        ScoreTimeOfDayKey,
        TotalScore
    from  CAB_EPIC.dbo.EarlyDetectionOfSepsisRiskFact 
    WHERE EncounterKey IN ({encs})
    """
    return sql_query

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def query_epic_scores(encounter_keys, batch_size=40000):
    encounter_key_batches = chunk_list(encounter_keys, batch_size)
    df_list = []

    for batch in encounter_key_batches:
        sql_query = execute_score(batch)
        df_batch = do_hhc_sql(sql_query)
        df_list.append(df_batch)

    return pd.concat(df_list, ignore_index=True)

cols_to_drop = [
    'Mean Neutrophil Count', 'Minimum Neutrophil Count', 'Maximum Neutrophil Count',
    'Mean C-Reactive Protein', 'Minimum C-Reactive Protein', 'Maximum C-Reactive Protein',
    'Mean Procalcitonin', 'Minimum Procalcitonin', 'Maximum Procalcitonin',
    'Mean FiO2', 'Minimum FiO2', 'Maximum FiO2',
    'Mean PaO2', 'Minimum PaO2', 'Maximum PaO2',
    'Mean Glasgow Coma Scale Score', 'Minimum Glasgow Coma Scale Score', 'Maximum Glasgow Coma Scale Score',
    'Mean Lactic Acid', 'Minimum Lactic Acid', 'Maximum Lactic Acid',
    'Mean Mean Arterial Pressure', 'Minimum Mean Arterial Pressure', 'Maximum Mean Arterial Pressure',
    'Mean Lymphocyte Count', 'Minimum Lymphocyte Count', 'Maximum Lymphocyte Count',
    'Total Urine Output', 'Mean Height', 'Mean Weight'
]

def get_drop_columns(df):
    pattern_cols = [col for col in df.columns if col.startswith('Dept_')]
    return list(set(cols_to_drop + pattern_cols))

def filter_dataframes_with_label_one(dataframes, name):
    filtered_dataframes = [df for df in dataframes if (df[name] == 1).any()]
    return filtered_dataframes

def filter_dataframes_with_label_zero(dataframes, name):
    filtered_dataframes = [df for df in dataframes if (df[name] == 0).all()]
    return filtered_dataframes

def metrics_from_threshold(y_true, y_score, target_sensitivity=0.80):
    unique_scores = np.unique(y_score)

    # Use manual threshold search for discrete scores
    if len(unique_scores) < 20:
        best_threshold = None
        best_metrics = (np.nan, np.nan, np.nan, np.nan)

        for t in sorted(unique_scores):
            y_pred = (y_score >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
            if sensitivity >= target_sensitivity:
                specificity = tn / (tn + fp) if (tn + fp) else np.nan
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                best_metrics = (sensitivity, specificity, accuracy, t)
                break

        return best_metrics

    # Use precision-recall curve for continuous scores
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if len(thresholds) == 0 or np.all(recall < target_sensitivity):
        return np.nan, np.nan, np.nan, np.nan

    valid_indices = np.where(recall[:-1] >= target_sensitivity)[0]
    if len(valid_indices) == 0:
        return np.nan, np.nan, np.nan, np.nan

    best_idx = valid_indices[-1]
    best_threshold = thresholds[best_idx]

    y_pred = (y_score >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return sensitivity, specificity, accuracy, best_threshold

# Utility function
def summarize_metric(name, values):
    clean_values = [v for v in values if not np.isnan(v)]
    if len(clean_values) == 0:
        print(f"{name}: No valid scores")
        return
    mean = np.mean(clean_values)
    ci = np.percentile(clean_values, [2.5, 97.5])
    print(f"{name}: {mean:.4f} | 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

def evaluate_all_thresholds(y_true_list, y_score_list, score_name, name):
    results = {}
    for threshold in [0.80, 0.85, 0.90]:
        sens_vals, spec_vals, acc_vals, thresh_vals = [], [], [], []
        
        for y_true, y_score in zip(y_true_list, y_score_list):
            sens, spec, acc, thresh = name(y_true, y_score, target_sensitivity=threshold)
            sens_vals.append(sens)
            spec_vals.append(spec)
            acc_vals.append(acc)
            thresh_vals.append(thresh)
        
        results[threshold] = {
            'sensitivity': sens_vals,
            'specificity': spec_vals,
            'accuracy': acc_vals,
            'threshold': thresh_vals
        }
    
    return results

def interpolate_roc_curve(y_trues, y_scores, n_points=100):
    mean_fpr = np.linspace(0, 1, n_points)
    tprs = []

    for y_true, y_score in zip(y_trues, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # ensure starting at 0
        tprs.append(interp_tpr)

    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)

    lower_tpr = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
    upper_tpr = np.minimum(mean_tpr + 1.96 * std_tpr, 1)

    return mean_fpr, mean_tpr, lower_tpr, upper_tpr

def plot_roc_with_ci(ax, mean_fpr, mean_tpr, lower_tpr, upper_tpr, label, color, linestyle):
    ax.plot(mean_fpr, mean_tpr, label=label, color=color, linestyle=linestyle, linewidth=1.2)
    ax.fill_between(mean_fpr, lower_tpr, upper_tpr, color=color, alpha=0.2)
    
def plot_all_roc(curve_data, time_window):
    colors = {
        'model': '#2980b9',    
        'sirs': '#e74c3c',       
        'sofa': '#f39c12',     
        'epic': '#27ae60'      
    }

    line_styles = {
        'model': '-',         
        'sirs': '--',          
        'sofa': '-.',         
        'epic': ':'           
    }

    legend_labels = {
        'model': 'XGBoost',
        'sirs': 'SIRS',
        'sofa': 'SOFA',
        'epic': 'EPIC'
    }

    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)

    for ax, dataset in zip(axs, ['test', 'val']):
        cohort = "validation" if dataset == "val" else "testing set of derivation"
        title = f'ROC curve for {time_window} hour time window\n({cohort} cohort)'
        ax.set_title(title)

        for method in ['model', 'sirs', 'sofa', 'epic']:
            y_trues = curve_data[dataset][method]['y_true']
            y_scores = curve_data[dataset][method]['y_score']

            mean_fpr, mean_tpr, lower_tpr, upper_tpr = interpolate_roc_curve(y_trues, y_scores)
            plot_roc_with_ci(ax, mean_fpr, mean_tpr, lower_tpr, upper_tpr, 
                             label=legend_labels[method], 
                             color=colors[method], 
                             linestyle=line_styles[method])

        ax.set_xlabel('False positive rate\n(1 - specificity)')
        ax.set_ylabel('True positive rate\n(sensitivity)')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.3)

        ax.grid(False)
        ax.legend(loc='lower right', frameon=True, fontsize=8)

    axs[0].text(-0.15, 1.15, 'a', transform=axs[0].transAxes,
                fontsize=12, fontweight='bold', va='top')  
    axs[0].text(1.05, 1.15, 'b', transform=axs[0].transAxes,
                fontsize=12, fontweight='bold', va='top')

    plt.tight_layout()
    #plt.savefig(f"auc_{time_window}tw_fig.png", bbox_inches='tight')
    plt.show()


def bootstrap_calibration_curve(y_trues, y_scores, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    interpolated_curves = []

    for y_true, y_score in zip(y_trues, y_scores):
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy='uniform')
            interp = np.interp(bin_centers, prob_pred, prob_true, left=np.nan, right=np.nan)
            interpolated_curves.append(interp)
        except:
            continue

    curves = np.array(interpolated_curves)
    mean = np.nanmean(curves, axis=0)
    lower = np.nanpercentile(curves, 2.5, axis=0)
    upper = np.nanpercentile(curves, 97.5, axis=0)
    return bin_centers, mean, lower, upper

def plot_calibrations_combined(curve_data, n_bins=10, time_window=24):
    methods = ['model', 'sofa', 'epic']
    method_labels = {
        'model': 'STRIDE-',
        'sofa': 'SOFA',
        'epic': 'EPIC'
    }
    colors = {
        'model': '#2980b9',   
        'sofa': '#f39c12',   
        'epic': '#27ae60'     
    }
    line_styles = {
        'model': '-',         
        'sofa': '-.',         
        'epic': ':'            
    }

    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)

    for ax, group in zip(axs, ['test', 'val']):
        cohort = "validation" if group == "val" else "testing set of derivation"
        ax.set_title(f'Calibration curve for {time_window} hour time window\n({cohort} cohort)')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.3, color='k')

        for method in methods:
            y_trues = curve_data[group][method]['y_true']
            y_scores = curve_data[group][method]['y_score']
            bin_centers, mean, lower, upper = bootstrap_calibration_curve(y_trues, y_scores, n_bins)

            ax.plot(bin_centers, mean, label=method_labels[method],
                    color=colors[method],
                    linestyle=line_styles[method],
                    linewidth=1.2)
            ax.fill_between(bin_centers, lower, upper, color=colors[method], alpha=0.2)

        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed frequency')
        ax.grid(False)
        ax.legend(loc='upper left', fontsize=8, frameon=True)
        
    axs[0].text(-0.15, 1.15, 'a', transform=axs[0].transAxes,
                fontsize=12, fontweight='bold', va='top')  
    axs[0].text(1.05, 1.15, 'b', transform=axs[0].transAxes,
                fontsize=12, fontweight='bold', va='top')

    plt.tight_layout()
    # plt.savefig(f"calib_{time_window}tw_fig.png", bbox_inches='tight')
    plt.show()

    
def bootstrap_pr_curve(y_trues, y_scores, n_points=100):
    recall_range = np.linspace(0, 1, n_points)
    interpolated_precisions = []

    for y_true, y_score in zip(y_trues, y_scores):
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            interp_precision = np.interp(recall_range, recall[::-1], precision[::-1], left=np.nan, right=np.nan)
            interpolated_precisions.append(interp_precision)
        except:
            continue

    curves = np.array(interpolated_precisions)
    mean = np.nanmean(curves, axis=0)
    lower = np.nanpercentile(curves, 2.5, axis=0)
    upper = np.nanpercentile(curves, 97.5, axis=0)
    return recall_range, mean, lower, upper

# --- Side-by-side PR plot panels ---
def plot_pr_curves_combined(curve_data, n_points=100, time_window=24):
    methods = ['model', 'sofa', 'epic', 'sirs']
    method_labels = {
        'model': 'STRIDE',
        'sofa': 'SOFA',
        'epic': 'EPIC',
        'sirs': 'SIRS'
    }
    colors = {
        'model': '#2980b9',    
        'sirs': '#e74c3c',       
        'sofa': '#f39c12',     
        'epic': '#27ae60'      
    }

    line_styles = {
        'model': '-',          
        'sirs': '--',          
        'sofa': '-.',          
        'epic': ':'            
    }

    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)

    for ax, group in zip(axs, ['test', 'val']):
        cohort = "Validation" if group == "val" else "Testing set of derivation"
        title = f'Precision-recall curve for {time_window} Hour Time window\n({cohort} cohort)'
        ax.set_title(title)

        for method in methods:
            y_trues = curve_data[group][method]['y_true']
            y_scores = curve_data[group][method]['y_score']
            recall, mean, lower, upper = bootstrap_pr_curve(y_trues, y_scores, n_points)

            ax.plot(recall, mean,
                    label=method_labels[method],
                    color=colors[method],
                    linestyle=line_styles[method],
                    linewidth=1.2)
            ax.fill_between(recall, lower, upper, color=colors[method], alpha=0.2)

            try:
                aucs = [auc(*precision_recall_curve(y_true, y_score)[1::-1]) 
                        for y_true, y_score in zip(y_trues, y_scores)]
                avg_auc = np.mean(aucs)
                # print(f"{group.upper()} {method_labels[method]} PR AUC: {avg_auc:.4f}")
            except:
                pass

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.grid(False)
        ax.legend(loc='upper right', fontsize=8, frameon=True)
        
    axs[0].text(-0.15, 1.15, 'a', transform=axs[0].transAxes,
                fontsize=12, fontweight='bold', va='top')  
    axs[0].text(1.05, 1.15, 'b', transform=axs[0].transAxes,
                fontsize=12, fontweight='bold', va='top')

    plt.tight_layout()
    # plt.savefig(f"pr_{time_window}tw_fig.png", bbox_inches='tight')
    plt.show()







"""
PR-AUC, Brier score, and ECE computed from your existing curve_data structure:

    curve_data[group][method]['y_true']   # list of per-fold arrays
    curve_data[group][method]['y_score']  # list of per-fold arrays
    group  in {'test', 'val'}
    method in {'model', 'sofa', 'epic', 'sirs'}

Aggregation uses 2.5 / 97.5 percentiles across folds (matches your bootstrap_* funcs).
SIRS is a binary rule, so Brier/ECE are returned as NaN for it.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

METHOD_LABELS = {'model': 'STRIDE', 'sofa': 'SOFA', 'epic': 'Epic', 'sirs': 'SIRS'}
COHORT_LABELS = {'test': 'Testing set of derivation', 'val': 'Validation'}
CALIB_METHODS = {'model', 'sofa', 'epic'}   # ECE/Brier not meaningful for binary SIRS


def calibration_error(y_true, y_prob, n_bins=10, strategy='uniform'):
    """
    Return (ece, mce), computed on the SAME (pred, obs) points that
    sklearn.calibration.calibration_curve produces for the figures.
    Use strategy='uniform', n_bins=10 to match the plotted calibration curves.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # identical binning to the figure's calibration_curve call
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy)

    # recover per-bin counts using sklearn's exact bin assignment for weighting
    if strategy == 'quantile':
        bins = np.percentile(y_prob, np.linspace(0, 1, n_bins + 1) * 100)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], y_prob)
    counts = np.bincount(binids, minlength=len(bins))
    counts = counts[counts != 0]          # aligns with calibration_curve's kept bins

    gaps = np.abs(prob_true - prob_pred)
    if gaps.size == 0:
        return np.nan, np.nan
    w = counts / counts.sum()
    ece = float(np.sum(w * gaps))         # population-weighted mean gap
    mce = float(gaps.max())               # worst-bin gap
    return ece, mce


def _clean(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    keep = ~(np.isnan(y_true) | np.isnan(y_score))
    return y_true[keep], y_score[keep]


def _agg(values):
    """Mean and 2.5 / 97.5 percentiles across folds."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    return v.mean(), np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


def compute_metrics_table(curve_data, time_window, n_bins=10, strategy='uniform'):
    """
    Returns a DataFrame with one row per (window, cohort, method) holding
    mean + 95% CI for PR-AUC, Brier, and ECE.
    """
    rows = []
    for group in ('test', 'val'):
        if group not in curve_data:
            continue
        for method, label in METHOD_LABELS.items():
            if method not in curve_data[group]:
                continue
            folds = zip(curve_data[group][method]['y_true'],
                        curve_data[group][method]['y_score'])
            pr, br, ec = [], [], []
            for y_true, y_score in folds:
                y_true, y_score = _clean(y_true, y_score)
                if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                    continue
                pr.append(average_precision_score(y_true, y_score))
                if method in CALIB_METHODS:
                    br.append(brier_score_loss(y_true, y_score))
                    ece, _ = calibration_error(y_true, y_score, n_bins, strategy)
                    ec.append(ece)
            pr_m = _agg(pr)
            br_m = _agg(br) if method in CALIB_METHODS else (np.nan,) * 3
            ec_m = _agg(ec) if method in CALIB_METHODS else (np.nan,) * 3
            rows.append({
                'window': f'{time_window}h', 'cohort': COHORT_LABELS[group],
                'method': label,
                'pr_auc': pr_m[0], 'pr_lo': pr_m[1], 'pr_hi': pr_m[2],
                'brier': br_m[0], 'brier_lo': br_m[1], 'brier_hi': br_m[2],
                'ece': ec_m[0], 'ece_lo': ec_m[1], 'ece_hi': ec_m[2],
            })
    return pd.DataFrame(rows)
