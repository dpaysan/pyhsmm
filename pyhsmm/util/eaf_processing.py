import numpy as np
import pandas as pd
import time
import xml.etree.ElementTree as ET
import os

from xml.dom import minidom

class Record:
    def __init__(self, df=None, path=None):
        self._df = df
        self._path = path

    @property
    def df(self):
        """Gets the record as a dataframe.

        If the _df attribute is None, the method reads the dataframe
        from the csv file and returns that; otherwise it returns
        self._df.
        """
        if self._df is None:
            if self._path is not None:
                try:
                    df = pd.read_csv(self._path, delimiter=';')
                except FileNotFoundError:
                    print("{} does not exist.".format(self._path))
            else:
                print("Please provide the record as a dataframe\
                       or a path to the respective .record file.")
        else:
            df = self._df

        return df

    @df.setter
    def df(self, df):
        self._df = df

    @df.deleter
    def df(self):
        self._df = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path



def get_changepoints_as_list(record, column_name,
                             time_column_name='TimeElapsed',
                             time_factor=1000,
                             time_out_as_integer=True):
    """Create a list of changepoints from a column of a record file.

    Args:
        record (pd.DataFrame): record file as pd.DataFrame
        column_name (str): Name of the column to be processed
        time_column_name (str):
        time_factor:
        time_out_as_integer (bool):

    Returns:
        changepoint_list (list): Elements are dictionaries
            corresponding to 'timepoints' at which the value in the
            record's corresponding column changes; these dictionaries
            have keys 'column', 'value', 'time', 'ts_id'. The 'ts_id'
            entries are initialized to emptry strings. (This list can
            be used as an auxiliary object for creating a tier
            corresponding to that column (besides other tiers
            corresponding to other columns) in an .eaf file.)

    """

    assert column_name, time_column_name in record.columns

    current_value = record.iloc[0][column_name]
    changepoint_list = [{'column': column_name, 'value':
                         current_value, 'time': 0}]
    k = 1
    for i in range(1, len(record)):
        if record.iloc[i][column_name] != current_value:
            current_value = record.iloc[i][column_name]
            time = time_factor * record.iloc[i][time_column_name]
            if time_out_as_integer:
                time = int(time)
            changepoint_list.append({'column': column_name, 'value':
                                     current_value, 'time': time})
            k += 1

    end_time = time_factor * record.iloc[-1][time_column_name]
    if time_out_as_integer:
        end_time = int(end_time)
    changepoint_list.append({'column': column_name, 'value': '',
                             'time': end_time})

    return changepoint_list


def get_changepoints(record, column_names,
                     time_column_name='TimeElapsed',
                     time_factor=1000,
                     time_out_as_integer=True):
    """Create auxiliary dataframe for transferring simulator info to eaf.

    Create a dataframe in which all the information from record
    that should be transferred to an eaf file is collected.

    Args:
        record (pd.DataFrame): record file as pd.DataFrame
        column_names (list(str)): List containing the names of the
           record columns for which changepoint information should be
           computed
        time_column_name:
        time_factor:
        time_out_as_integer:
    """

    for column_name in column_names:
        assert column_name in record.columns

    df = pd.DataFrame(columns=['column', 'value', 'time'])
    for column_name in column_names:
        df_new = pd.DataFrame(get_changepoints_as_list(record,
                                                       column_name,
                                                       time_column_name,
                                                       time_factor,
                                                       time_out_as_integer))
        df = pd.concat([df, df_new], sort=True)

    df.sort_values(by=['time'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # for i in range(len(df)):
    #     df.loc[i, 'ts_id'] = int(i)

    return df


def write_to_eaf(changepoints, eaf_fname_in, eaf_fname_out):
    """Write information about changepoints to an eaf file.

    Args:
      changepoints (pd.DataFrame): Should have columns {'column',
          'value', 'time'}; every row correponds to a changepoint in the
          column 'column' of a record file at time 'time', and with new
          value 'value'.
      eaf_fname_in (str): Name of the eaf file that will be parsed,
          resulting in an ET.ElementTree to which changepoint
          information will be added.
      eaf_fname_out (str): Name of the eaf file to which this
          ET.ElementTree will be written.

    CHECK for correctness
    TODO Refactor, make modular
    """
    assert set(changepoints.columns) == {'column', 'value', 'time'}
    column_names = changepoints['column'].unique()

    # Get a timestamp that will be used in order to create annotation
    # and time_slot IDs that are different from those that potentially
    # exist already in the eaf_fname_in.
    timestamp = str(time.time()).split('.')[0]

    # Create a 'timeslot_id' column of the auxiliary dataframe
    changepoints = changepoints.copy()
    changepoints['timeslot_id'] = changepoints.index
    changepoints['timeslot_id'] = changepoints['timeslot_id'].apply(
        lambda n: str(n) + '-' +
                  timestamp)
    # Parse the input eaf file
    tree = ET.parse(eaf_fname_in)
    root = tree.getroot()

    # Create an ET.Element for every row in the changepoints DataFrame
    time_order = root.findall('TIME_ORDER')[0]
    for index, row in changepoints.iterrows():
        time_slot = ET.Element('TIME_SLOT',
                               {'TIME_SLOT_ID': str(row['timeslot_id']),
                                'TIME_VALUE': str(row['time'])})
        time_order.insert(len(time_order), time_slot)

    # For each column_name appearing in changepoints['column'], find
    # the 'TIER' node with that id (create it if it doesn't
    # exist). Then create an annotation for every corresponding row in
    # the changepoints DataFrame.
    for column_name in column_names:
        tier_id = column_name
        if len(root.findall("TIER[@TIER_ID=\'" + tier_id + "\']")) > 0:
            tier = root.findall("TIER[@TIER_ID=\'" + tier_id + "\']")[0]
        else:
            tier = ET.Element('TIER', {'DEFAULT_LOCALE': 'en',
                                       'LINGUISTIC_TYPE_REF': 'Simulator',
                                       'TIER_ID': tier_id})

        df = changepoints[changepoints['column'] == column_name].sort_values(
            by='time')
        row0 = df.iloc[0]
        for index, row in df.iloc[1:].iterrows():
            row1 = row
            annotation = ET.Element('ANNOTATION')
            alignable_annotation = ET.Element('ALIGNABLE_ANNOTATION',
                                              {
                                                  'ANNOTATION_ID': 'a-'
                                                                   + tier_id + '-' +
                                                                   str(row0[
                                                                           'timeslot_id']),
                                                  'TIME_SLOT_REF1': str(
                                                      row0['timeslot_id']),
                                                  'TIME_SLOT_REF2': str(
                                                      row1['timeslot_id'])})
            annotation_value = ET.Element('ANNOTATION_VALUE')
            annotation_value.text = str(row0['value'])

            alignable_annotation.insert(len(alignable_annotation),
                                        annotation_value)
            annotation.insert(len(annotation), alignable_annotation)
            tier.insert(len(tier), annotation)

            row0 = row1

        root.insert(len(root), tier)

    # Create nicely indented string and write to output file
    # TODO: So far the new elements are strangely formatted (newline missing)
    tree_str = minidom.parseString(
        ET.tostring(root, method='xml')).toprettyxml(indent="   ",
                                                     newl="")
    with open(eaf_fname_out, 'w') as file:
        file.write(tree_str)
    # tree.write(eaf_fname_out, encoding='UTF-8', xml_declaration=True)

    return tree, root


def get_eaf_tier_as_df(eaf_fname, tier_id):
    """Create a pd.DataFrame holding information from eaf_file.

    TODO Extend so it can do multiple tiers at a time

    TODO handle case when there is only 1 time boundary point

    Returns:
        tier_as_df (pd.DataFrame): A pd.DataFrame whose columns
            tier_id, 'ts_ref1' and 'ts_ref2'.

    """
    tier_as_df = pd.DataFrame(columns=[tier_id, 'ts_ref1', 'ts_ref2'])

    tree = ET.parse(eaf_fname)
    root = tree.getroot()
    time_order = root.findall('TIME_ORDER')[0]

    # Create a dictionary for time slots (keys are time slots ids,
    # values are time values).
    time_slots = {}
    for time_slot in time_order:
        try:
            time_slots[time_slot.get('TIME_SLOT_ID')] = int(
                time_slot.get('TIME_VALUE'))
        except TypeError:
            continue

    # Create a pd.DataFrame containing all annotations in the tier.
    try:
        for annotation in root.findall("TIER[@TIER_ID=\'" + tier_id + "\']") \
                [0]:
            try:
                ts_ref1 = annotation.findall("ALIGNABLE_ANNOTATION")[0].get(
                    'TIME_SLOT_REF1')
                ts_ref2 = annotation.findall("ALIGNABLE_ANNOTATION")[0].get(
                    'TIME_SLOT_REF2')
            except TypeError:
                continue

            annotation_value = annotation.findall("./*ANNOTATION_VALUE")[
                0].text
            new_row = pd.DataFrame([{tier_id: annotation_value,
                                     'ts_ref1': time_slots[ts_ref1],
                                     'ts_ref2': time_slots[ts_ref2]}])
            tier_as_df = pd.concat([tier_as_df, new_row])
    except IndexError:
        print(
            'The file {} does not seem to have a tier whose ID is \'{}\'.'
            .format(eaf_fname, tier_id))

    tier_as_df = tier_as_df.reset_index(drop=True)

    return tier_as_df


def transfer_eaf_tier_to_record(eaf_fname, tier_id, record,
                                time_column_name='TimeElapsed'):
    """CHECK for correctness

    TODOs:

        - time column in eafs and record typically don't have same
          units (ms vs. s). Might include checking that

    """
    tier_as_df = get_eaf_tier_as_df(eaf_fname, tier_id)

    # Convert time info from milliseconds to seconds
    tier_as_df["ts_ref1"] /= 1000
    tier_as_df["ts_ref2"] /= 1000

    # Initialize new column
    # record[tier_id] = np.nan
    # record[tier_id + '_changepoint'] = False
    record.sort_values(by=time_column_name, inplace=True)

    # Reset index
    record = record.reset_index(drop=True)

    for row in tier_as_df.index:
        # print("len: ", len(record))
        annotation_value = tier_as_df.loc[row, tier_id]
        ts_ref1 = tier_as_df.loc[row, 'ts_ref1']
        ts_ref2 = tier_as_df.loc[row, 'ts_ref2']

        # Create a mask for selecting all rows of the record whose time
        # value is between ts_ref1 and ts_ref2
        mask = (ts_ref1 <= record[time_column_name]) & (
                record[time_column_name] <= ts_ref2)
        start_index = record[mask].index[0]
        end_index = record[mask].index[-1]

        # print(ts_ref1, ts_ref2, record.loc[start_index, time_column_name],
        #      record.loc[end_index, time_column_name],
        #      start_index, end_index)

        record.loc[start_index:end_index, tier_id] = annotation_value

    record.sort_values(by='TimeElapsed', inplace=True)
    record = record.reset_index(drop=True)
    return record, tier_as_df


def transfer_eaf_tier_to_record_dtype_num(eaf_fname, tier_id, record,
                                          time_column_name='TimeElapsed'):
    """CHECK for correctness

    TODOs:

        - time column in eafs and record typically don't have same
          units (ms vs. s). Might include checking that

    """
    tier_as_df = get_eaf_tier_as_df(eaf_fname, tier_id)

    # Convert time info from milliseconds to seconds
    tier_as_df["ts_ref1"] /= 1000
    tier_as_df["ts_ref2"] /= 1000

    # Initialize new column
    # record[tier_id] = np.nan
    # record[tier_id + '_changepoint'] = False
    record.sort_values(by=time_column_name, inplace=True)

    # Reset index
    record = record.reset_index(drop=True)

    for row in tier_as_df.index:
        # print("len: ", len(record))
        annotation_value = float(tier_as_df.loc[row, tier_id])
        ts_ref1 = tier_as_df.loc[row, 'ts_ref1']
        ts_ref2 = tier_as_df.loc[row, 'ts_ref2']

        # Create a mask for selecting all rows of the record whose time
        # value is between ts_ref1 and ts_ref2
        mask = (ts_ref1 <= record[time_column_name]) & (
                record[time_column_name] <= ts_ref2)
        start_index = record[mask].index[0]
        end_index = record[mask].index[-1]

        record.loc[start_index:end_index, tier_id] = annotation_value

    record.sort_values(by='TimeElapsed', inplace=True)
    record = record.reset_index(drop=True)
    return record, tier_as_df


def write_to_file(data, column_name, base_path=None, FPS=40):
    """Write to file function.

    Write a dict of arbitrary values to separate files. For each key
    (file name) the values are written to the corresponding eaf file.
    The function returns a dict with None values to adhere to the data pipeline
    convention.

    Args:
        data (dict): Dictionary containing the extracted features. Keys are
            the video file names, values are the ones to be added to the eaf
            file.
        base_path (str): Path to the base folder of the video files.

    Returns:
        dictionary: Keys are the video file names, values are None.

    """
    return_data = {}
    for file in data:
        value = data[file][0]
        video = file.split('/')[0]
        return_data[file] = None
        eaf_file_found = False

        if os.path.isdir(base_path + video):
            for f in os.listdir(base_path + video):
                if f.endswith('.eaf'):
                    eaf_file_found = True
                    v = np.array(value)
                    v_compressed = np.concatenate([v[0].reshape(-1),
                                                   v[np.where(v[:-1] != v[1:])[
                                                         0] + 1],
                                                   v[-1].reshape(-1)], axis=0)
                    ind_compressed = np.concatenate([np.zeros(1), np.where(
                        v[:-1] != v[1:])[0] + 1, np.array(v.shape[0])
                                                    .reshape(-1)], axis=0)
                    t_compressed = ind_compressed / FPS * 1000
                    name = np.repeat(np.array([column_name]),
                                     ind_compressed.shape[0])
                    changepoints = pd.DataFrame(data={"column": name,
                                                      "value": v_compressed,
                                                      "time": t_compressed
                                                .astype(int)})
                    write_to_eaf(changepoints, base_path + video + "/" +
                                 video + ".eaf", base_path + video + "/" +
                                 video + "_new.eaf")
            if not eaf_file_found:
                raise FileNotFoundError("No .eaf file found in {0}!".format(
                    base_path + video))
                return
        else:
            raise FileNotFoundError("Directory {0} does not exist!".format(
                base_path + video))
            return

    return return_data


def to_eaf(state_seq,  decode_df,  states, eaf_file, output_dir="."):
    """Converts decoded sequence to eaf files.

    Returns:
        EAF files with decoded sequence imported
    """

    states_dict = {}
    for i, state in enumerate(states):
        states_dict[i] = state


    # Replace state indices by state names
    decode_df["Decoded"] = state_seq
    decode_df.replace({"Decoded": states_dict}, inplace=True)
    state_seq_decoded = [states[s] for s in state_seq]

    changepoints = get_changepoints(decode_df,
                                        column_names=['Decoded'])
    eaf_path_out = output_dir + "/" + \
                       eaf_file.split("/")[-1][:-4] + \
                       "-decoded.eaf"
    # eaf_path_out = self.eaf_paths[i][:-4] + "_NEW"+str(time.time())+".eaf"
    write_to_eaf(changepoints, eaf_fname_in=eaf_file,
                     eaf_fname_out=eaf_path_out)
