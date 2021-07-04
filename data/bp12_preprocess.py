import os
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd
from datetime import datetime

def many_to_one_prep(journey):
    """
    Setting up a many-to-one scheme.
    Function Input is a complete journey.
    Returns an RNN input of a history of activities, output is the following activity
    """
    inp = [journey[:i] for i in range(1,len(journey))]
    out = journey[1:]
    return (inp,out)

def many_to_many_prep(journey):
    """
    Setting up a many-to-many scheme shifted by 1.  
    Function input is a complete journey.
    Returns an RNN input of a history of activities omitting the final activity, output is the same history shifted by 1 timeslot (omitting the first activity)
    """
    inp = journey[:-1]
    out = journey[1:]
    return(inp, out)

def many_to_many_make_data(id_indexes, arrays_df):
    """
    Creates training and testing sets for an RNN model.  
    Function input are indices of a selected subset of data, a DataFrame consisting of aggregated list-like journey data, and a data preparation method.
    Returns a tuple of training and testing data for journeys and inter-arrival times.
    """
    X_j = []
    Y_j = []
    X_t = []
    Y_t = []

    selected = arrays_df[arrays_df["case:concept:name"].isin(id_indexes)]

    for index, row in selected.iterrows():
        j_inp, j_out = many_to_many_prep(row['concept:encoded'])
        t_inp, t_out = many_to_many_prep(row['time:interarrival_min'])
        X_j.append(j_inp)
        X_t.append(t_inp)
        Y_j.append(j_out)
        Y_t.append(t_out)
    X_j = keras.preprocessing.sequence.pad_sequences(X_j, padding='pre', maxlen=60)
    X_j = to_categorical(X_j)
    X_t = keras.preprocessing.sequence.pad_sequences(X_t, padding='pre', maxlen=60)
    Y_j = keras.preprocessing.sequence.pad_sequences(Y_j, padding='pre', maxlen=60)
    Y_j = to_categorical(Y_j)
    Y_t = keras.preprocessing.sequence.pad_sequences(Y_t, padding='pre', maxlen=60)
    return (X_j, X_t, Y_j, Y_t)

def many_to_one_make_data(id_indexes, arrays_df):
    """
    Creates training and testing sets for an RNN model.  
    Function input are indices of a selected subset of data, a DataFrame consisting of aggregated list-like journey data, and a data preparation method.
    Returns a tuple of training and testing data for journeys and inter-arrival times.
    """
    X_j = []
    Y_j = []
    X_t = []
    Y_t = []

    selected = arrays_df[arrays_df["case:concept:name"].isin(id_indexes)]

    for index, row in selected.iterrows():
        j_inp, j_out = many_to_one_prep(row['concept:encoded'])
        t_inp, t_out = many_to_one_prep(row['time:interarrival_min'])
        X_j.extend(j_inp)
        X_t.extend(t_inp)
        Y_j.extend(j_out)
        Y_t.extend(t_out)
    X_j = keras.preprocessing.sequence.pad_sequences(X_j, padding='pre', maxlen=60)
    X_j = to_categorical(X_j)
    X_t = keras.preprocessing.sequence.pad_sequences(X_t, padding='pre', maxlen=60)
    Y_j = np.asarray(Y_j).astype("float32")
    Y_j = to_categorical(Y_j)
    Y_t = np.asarray(Y_t).astype("float32")
    return (X_j, X_t, Y_j, Y_t)


if __name__ == '__main__':
	# Array form of activities + interarrival times

	curr_path = os.path.abspath('')
	filepath = os.path.join(curr_path, 'BPI_Challenge_2012.xes')
	log = pm4py.read_xes(filepath)
	bp12 = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
	array_form = bp12.groupby(['case:concept:name']).agg(list)
	array_form.reset_index(inplace=True)

	# Array form in minutes
	array_form['time:interarrival'] = array_form['time:timestamp'].apply(lambda x: [0] + [((x[i+1] - x[i]).total_seconds() / 60) for i in range(len(x)-1)])

	array_form.to_csv('bp12_arrays.csv', index=False)


