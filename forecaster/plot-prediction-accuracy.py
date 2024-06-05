import numpy as np
import os
import sys
import fnmatch
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict

GRAPH_DIR = "../plot/"  # Directory to output graphs to

# Get the data in the prediction results directory as a dictionary
def GetDataDict(input_dir):
    data_dict = {}
    
    # Iterate through all files in directory, add to data dictionary
    for root, dirnames, filenames in os.walk(input_dir):
        for filename in sorted(fnmatch.filter(filenames, '*.csv')):
            print(filename)
            file_path = os.path.join(root, filename)
            # Add the actual and predicted values for the cluster data stored in file path
            cluster_idx = int(file_path[file_path.rfind('\\')+1:].split('.')[0])
            dates, actual, predicted = LoadData(file_path)
            file_path = file_path[:file_path.rfind('\\')]
            if file_path in data_dict.keys():
                data_dict[file_path]["clusters"][cluster_idx] = {"actual": actual, "predicted": predicted}
            else:
                data_dict[file_path] = {"dates": dates, "clusters": {cluster_idx: {"actual": actual, "predicted": predicted}}}

    return data_dict

# Load data from csv file
def LoadData(input_path):
    dates = []
    actual = []
    predicted = []
    with open(input_path) as input_file:
        reader = csv.reader(input_file)
        for line in reader:
            dates.append(datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S'))
            actual.append(float(line[1]))
            predicted.append(float(line[2]))
    return dates, actual, predicted

# Calculate the MSE from lists of actual and predicted values
def GetMSE(actual, predict):
    y = np.array(actual)
    y_hat = np.array(predict)

    data_min = 2 - np.min([np.min(y), np.min(y_hat)])
    se = (np.log(y + data_min) - np.log(y_hat + data_min)) ** 2

    return np.mean(se)

start_date = datetime(year=2017, month=1, day=1)        # Start date for the plot
end_date = datetime(year=2017, month=1, day=5, hour=3)  # End date for the plot

MSE_dict = {}   # Dictionary to store MSE values

# Plot the predicted and actual values stored in the specified file path
def PlotData(file_path, data):
    global MSE_dict

    # Get the parameter values from the file path
    path = file_path.split('\\')
    interval = int(path[1].split('-')[1])
    horizon = int(path[2].split('-')[1])
    method = path[3]
    
    # Sum the arrival rate values from each cluster
    dates = data["dates"]
    actual = np.zeros(len(dates))
    predicted = np.zeros(len(dates))
    for cluster_idx, cluster_data in data["clusters"].items():
        # Uncomment below to show results for selected cluster
        # if not cluster_idx == 43: continue
        actual += cluster_data["actual"]
        predicted += cluster_data["predicted"]

    # Calculate and store the mean squared error for this model and prediction horizon
    MSE = GetMSE(actual, predicted)
    if not method in MSE_dict.keys(): MSE_dict[method] = SortedDict()
    MSE_dict[method][horizon] = MSE

    # print(f'MSE for {method.upper()} (Interval = {interval} minutes, Horizon = {horizon} minutes): {MSE}')

    # Filter the data by the specified start and end dates
    actual = [actual[i] for i, date in enumerate(dates) if start_date <= date <= end_date]
    predicted = [predicted[i] for i, date in enumerate(dates) if start_date <= date <= end_date]
    dates = [date for date in dates if start_date <= date <= end_date]

    # Plot the predicted and actual values
    # plt.figure(figsize=(12, 3))
    # plt.plot(dates, actual, linestyle='-', color='dodgerblue', label='Actual')
    # plt.plot(dates, predicted, linestyle='-', color='tomato', label='Prediction')
    # if interval == 1: plt.ylabel('Queries per minute')
    # else: plt.ylabel(f'Queries per {interval} minutes')
    # plt.title(f'Predicted vs. Actual Query Arrival Rates for {method.upper()} (Interval = {interval} minutes, Horizon = {horizon} minutes)')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(GRAPH_DIR + f'{method}_{interval}_{horizon}.png')
    # plt.show()

# Plot a comparison of the MSEs for each model / prediction horizon
def PlotMSE():
    # Get the MSE data and horizon values
    data = {key: value for key, value in MSE_dict.items() if key in ('svr', 'brr', 'hybrid')}
    horizons = data[next(iter(data))].keys()

    x = np.arange(len(horizons))  # the label locations
    width = 0.25                  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(6,4))

    # Plot the bars
    maxMSE = 0
    for method, mses in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, mses.values(), width, label=method)
        ax.bar_label(rects, fmt='%.2f', fontsize=8, padding=3)
        multiplier += 1
        maxMSE = max(max(mses.values()), maxMSE)   # Record max MSE value

    # Add labels, title, custom x-axis tick labels, etc.
    ax.set_xticks(x + width, horizons)
    ax.legend(loc='upper left', ncols=3)
    ax.set_xlabel('Prediction Horizon (minutes)')
    ax.set_ylabel('MSE (log space)')
    ax.set_ylim(0, maxMSE * 1.2)
    
    plt.show()

def Main(input_dir):
    if not os.path.exists(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)

    data_dict = GetDataDict(input_dir)

    for file_path, data in data_dict.items():
        PlotData(file_path, data)
    
    PlotMSE()

# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    """
    Generate MSE result plots

    Args:
        arg1 : the result dir
    """
    Main(sys.argv[1])
