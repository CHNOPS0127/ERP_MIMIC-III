"""
MIMIC-III Data Extraction Script
Extracts and processes per-subject data from MIMIC-III CSV files.
"""

import argparse  
import csv
import numpy as np
import os
import pandas as pd
from tqdm import tqdm



# Data Load


def dataframe_from_csv(path, header=0, index_col=0):
    """Load CSV file into pandas DataFrame with specified header and index column."""
    return pd.read_csv(path, header=header, index_col=index_col)


def read_patients_table(mimic3_path):
    """Read and preprocess PATIENTS table with demographics data."""
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    # Convert date columns to datetime format
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats


def read_admissions_table(mimic3_path):
    """Read and preprocess ADMISSIONS table with hospital admission data."""
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
    # Convert timestamp columns to datetime format
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits


def read_icustays_table(mimic3_path):
    """Read and preprocess ICU stays table."""
    stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    # Convert ICU in/out times to datetime format
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays


def read_icd_diagnoses_table(mimic3_path):
    """Read ICD diagnosis codes and merge with diagnosis descriptions."""
    # Read diagnosis code definitions
    codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    
    # Read patient diagnoses and merge with code definitions
    diagnoses = dataframe_from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return diagnoses


def read_events_table_by_row(mimic3_path, table):
    """Generator function to read event tables row by row to handle large files."""
    # Expected number of rows for progress tracking
    nb_rows = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
    reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv'), 'r'))
    
    for i, row in enumerate(reader):
        # Ensure ICUSTAY_ID column exists (some tables may not have it)
        if 'ICUSTAY_ID' not in row:
            row['ICUSTAY_ID'] = ''
        yield row, i, nb_rows[table.lower()]



# Data Filtering


def count_icd_codes(diagnoses, output_path=None):
    """Count frequency of each ICD diagnosis code."""
    codes = diagnoses[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']].drop_duplicates().set_index('ICD9_CODE')
    codes['COUNT'] = diagnoses.groupby('ICD9_CODE')['ICUSTAY_ID'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    codes = codes[codes.COUNT > 0]
    
    # Save to file if output path provided
    if output_path:
        codes.to_csv(output_path, index_label='ICD9_CODE')
    return codes.sort_values('COUNT', ascending=False).reset_index()


def remove_icustays_with_transfers(stays):
    """Remove ICU stays where patient was transferred between units/wards."""
    stays = stays[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]


def merge_on_subject(table1, table2):
    """Inner join two tables on SUBJECT_ID."""
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])


def merge_on_subject_admission(table1, table2):
    """Inner join two tables on SUBJECT_ID and HADM_ID."""
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


def add_age_to_icustays(stays):
    """Calculate patient age at ICU admission in years."""
    stays['AGE'] = stays.apply(lambda e: (e['INTIME'].to_pydatetime()
                                          - e['DOB'].to_pydatetime()).total_seconds() / 3600.0 / 24.0 / 365.0,
                               axis=1)
    # MIMIC-III shifts ages >89 to 300+, reset these to 90
    stays.loc[stays.AGE < 0, 'AGE'] = 90
    return stays


def add_inhospital_mortality_to_icustays(stays):
    """Add binary flag for in-hospital mortality."""
    # Check if death occurred during hospital stay
    mortality = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    stays['MORTALITY'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays


def add_inunit_mortality_to_icustays(stays):
    """Add binary flag for in-ICU-unit mortality."""
    # Check if death occurred during ICU stay
    mortality = stays.DOD.notnull() & ((stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)))
    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays


def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    """Filter to keep only admissions with specified number of ICU stays."""
    to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    to_keep = to_keep[(to_keep.ICUSTAY_ID >= min_nb_stays) & (to_keep.ICUSTAY_ID <= max_nb_stays)][['HADM_ID']]
    stays = stays.merge(to_keep, how='inner', left_on='HADM_ID', right_on='HADM_ID')
    return stays


def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    """Filter ICU stays by patient age range."""
    stays = stays[(stays.AGE >= min_age) & (stays.AGE <= max_age)]
    return stays


def filter_diagnoses_on_stays(diagnoses, stays):
    """Filter diagnoses to only include those from selected ICU stays."""
    return diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(), how='inner',
                           left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


# Subject Extraction


def break_up_stays_by_subject(stays, output_path, subjects=None):
    """Create per-subject directories and save stays data."""
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        # Create subject directory
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        # Save stays for this subject, sorted by admission time
        stays[stays.SUBJECT_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'),
                                                                              index=False)


def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    """Create per-subject directories and save diagnosis data."""
    subjects = diagnoses.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        # Create subject directory
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        # Save diagnoses for this subject, sorted by ICU stay and sequence
        diagnoses[diagnoses.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID', 'SEQ_NUM'])\
                                                     .to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)


def read_events_table_and_break_up_by_subject(mimic3_path, table, output_path,
                                              items_to_keep=None, subjects_to_keep=None):
    """Process large event tables and split by subject to manage memory usage."""
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
    
    # Convert filter sets to strings for comparison
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        """Helper class to track current subject and observations."""
        def __init__(self):
            self.curr_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        """Write accumulated observations for current subject to file."""
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
            
        fn = os.path.join(dn, 'events.csv')
        # Create file with header if it doesn't exist
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
            
        # Append observations to file
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    # Get expected number of rows for progress bar
    nb_rows_dict = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
    nb_rows = nb_rows_dict[table.lower()]

    # Process each row of the events table
    for row, row_no, _ in tqdm(read_events_table_by_row(mimic3_path, table), total=nb_rows,
                                                        desc='Processing {} table'.format(table)):

        # Apply subject filter
        if (subjects_to_keep is not None) and (row['SUBJECT_ID'] not in subjects_to_keep):
            continue
        # Apply item ID filter
        if (items_to_keep is not None) and (row['ITEMID'] not in items_to_keep):
            continue

        # Extract relevant columns
        row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                   'HADM_ID': row['HADM_ID'],
                   'ICUSTAY_ID': '' if 'ICUSTAY_ID' not in row else row['ICUSTAY_ID'],
                   'CHARTTIME': row['CHARTTIME'],
                   'ITEMID': row['ITEMID'],
                   'VALUE': row['VALUE'],
                   'VALUEUOM': row['VALUEUOM']}
        
        # Write observations when switching to new subject
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
            write_current_observations()
            
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['SUBJECT_ID']

    # Write final subject's observations
    if data_stats.curr_subject_id != '':
        write_current_observations()


# Main Execution

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
    parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
    parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
    parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                        default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'])
    parser.add_argument('--itemids_file', '-i', type=str, help='CSV containing list of ITEMIDs to keep.')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
    parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details')
    parser.set_defaults(verbose=True)
    parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1000000 events.')
    
    args, _ = parser.parse_known_args()

    # Create output directory
    try:
        os.makedirs(args.output_path)
    except:
        pass

   
    # Load and merge core tables
  
    print("Loading core tables...")
    patients = read_patients_table(args.mimic3_path)
    admits = read_admissions_table(args.mimic3_path)
    stays = read_icustays_table(args.mimic3_path)
    
    if args.verbose:
        print('START:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(
            stays.ICUSTAY_ID.unique().shape[0],
            stays.HADM_ID.unique().shape[0], 
            stays.SUBJECT_ID.unique().shape[0]))


    # Apply filters and data quality checks

    print("Applying filters...")
    
    # Remove ICU stays with transfers between units
    stays = remove_icustays_with_transfers(stays)
    if args.verbose:
        print('REMOVE ICU TRANSFERS:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(
            stays.ICUSTAY_ID.unique().shape[0],
            stays.HADM_ID.unique().shape[0], 
            stays.SUBJECT_ID.unique().shape[0]))

    # Merge with admissions and patient data
    stays = merge_on_subject_admission(stays, admits)
    stays = merge_on_subject(stays, patients)
    
    # Filter to single ICU stay per hospital admission
    stays = filter_admissions_on_nb_icustays(stays)
    if args.verbose:
        print('REMOVE MULTIPLE STAYS PER ADMIT:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(
            stays.ICUSTAY_ID.unique().shape[0],
            stays.HADM_ID.unique().shape[0], 
            stays.SUBJECT_ID.unique().shape[0]))

    # Add calculated fields
    stays = add_age_to_icustays(stays)
    stays = add_inunit_mortality_to_icustays(stays)
    stays = add_inhospital_mortality_to_icustays(stays)
    
    # Filter by age (adults only)
    stays = filter_icustays_on_age(stays)
    if args.verbose:
        print('REMOVE PATIENTS AGE < 18:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(
            stays.ICUSTAY_ID.unique().shape[0],
            stays.HADM_ID.unique().shape[0], 
            stays.SUBJECT_ID.unique().shape[0]))

    
    # Process diagnoses
    
    print("Processing diagnoses...")
    
    # Save filtered stays
    stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)
    
    # Load and filter diagnoses
    diagnoses = read_icd_diagnoses_table(args.mimic3_path)
    diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
    diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
    
    # Generate diagnosis code counts
    count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))

    
    # Handle test mode (optional data reduction)
    
    if args.test:
        print("Running in test mode - reducing data size...")
        # Randomly sample 1000 patients for testing
        pat_idx = np.random.choice(patients.shape[0], size=1000)
        patients = patients.iloc[pat_idx]
        stays = stays.merge(patients[['SUBJECT_ID']], left_on='SUBJECT_ID', right_on='SUBJECT_ID')
        # Use only first event table
        args.event_tables = [args.event_tables[0]]
        print('Using only', stays.shape[0], 'stays and only', args.event_tables[0], 'table')

    
    # Break up data by subject and process events
    
    print("Creating per-subject files...")
    
    # Get final subject list
    subjects = stays.SUBJECT_ID.unique()
    
    # Create per-subject stay files
    break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
    
    # Create per-subject diagnosis files
    break_up_diagnoses_by_subject(diagnoses, args.output_path, subjects=subjects)
    
    # Load item ID filter if provided
    items_to_keep = set(
        [int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['ITEMID'].unique()]) if args.itemids_file else None
    
    # Process each event table
    for table in args.event_tables:
        print(f"Processing {table} table...")
        read_events_table_and_break_up_by_subject(args.mimic3_path, table, args.output_path, 
                                                  items_to_keep=items_to_keep,
                                                  subjects_to_keep=subjects)
    
    print("Data extraction completed successfully!")

