import os
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd
from datetime import datetime

def mto_lstm_prep(journey):
    inp = [journey[:i] for i in range(1,len(journey))]
    out = journey[1:]
    return (inp,out)

def make_data(id_indexes, arrays_df):
    X_j = []
    Y_j = []
    X_t = []
    Y_t = []

    selected = arrays_df[arrays_df["case:concept:name"].isin(id_indexes)]

    for index, row in selected.iterrows():
        j_inp, j_out = mto_lstm_prep(row['concept:encoded'])
        t_inp, t_out = mto_lstm_prep(row['time:interarrival_min'])
        X_j.extend(j_inp)
        X_t.extend(t_inp)
        Y_j.extend(j_out)
        Y_t.extend(t_out)
    X_j = keras.preprocessing.sequence.pad_sequences(X_j, padding='pre')
    X_t = keras.preprocessing.sequence.pad_sequences(X_t, padding='pre')
    Y_j = np.asarray(Y_j).astype("float32")
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


