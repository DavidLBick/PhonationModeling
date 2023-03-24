import pandas as pd 
import numpy as np
import os 
import pickle
from tqdm import tqdm
from sklearn.metrics import r2_score
import scipy.stats as stats
from pyinform import mutual_info, transfer_entropy
import pyinform.utils as utils
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon
import argparse 
import json 

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-cf", "--configure_file", required=True, help="configure file for experiment")
args = parser.parse_args()

configure_file = args.configure_file
try:
    with open(configure_file) as f:
        configs = json.load(f)
except OSError as e:
    print(f"OS error: {e}")

label_df = pd.read_csv(configs["label_df_path"])
data_dir = configs['data_dir']

indices_d = {
    'velocities': [2, 4],
    'displacements': [1, 3],
    'vel_vs_dis_r': [1, 2], 
    'vel_vs_dis_l': [3, 4]
}
right_displ_i = [1]
left_displ_i = [3]
right_vel_i = [2]
left_vel_i = [4]

def get_sampled_files(files, step_sizes):
    vfo = dict()
    for f_root in tqdm(files):
        vfo[f_root] = []
        for step_size in step_sizes:
            with open(os.path.join(data_dir, f"{f_root}_{step_size}.pkl"), 'rb') as f:
                data = pickle.load(f)
                data = data['sol'][0]
                vfo[f_root].append(data)
    return vfo 

def shoelace_formula(x, y):
    pgon = Polygon(zip(x, y))
    return pgon.area

def get_stats(vfo, tme_srs_r_idx, tme_srs_l_idx):
    # since we already select the right step_size index in get_sampled_files 
    # this just gets rid of the dimension of the list
    step_size_i = 0
    # amplitude is defined as 1/2(max - min)
    # get the amplitude of displacement for the right vocal fold
    files = [f for f in vfo]
    amp_displ_r = [np.ptp(vfo[f][step_size_i][:, tme_srs_r_idx]) / 2 for f in vfo]
    amp_displ_l = [np.ptp(vfo[f][step_size_i][:, tme_srs_l_idx]) / 2 for f in vfo]
    # get the mean displacement of the right vocal fold
    mean_displ_r = [np.mean(vfo[f][step_size_i][:, tme_srs_r_idx]) for f in vfo]
    mean_displ_l = [np.mean(vfo[f][step_size_i][:, tme_srs_l_idx]) for f in vfo]
    # get the range of displacement for the right vocal fold
    range_displ_r = [np.ptp(vfo[f][step_size_i][:, tme_srs_r_idx]) for f in vfo]
    range_displ_l = [np.ptp(vfo[f][step_size_i][:, tme_srs_l_idx]) for f in vfo]
    # get the standard deviation of displacement for the right vocal fold
    std_displ_r = [np.std(vfo[f][step_size_i][:, tme_srs_r_idx]) for f in vfo]
    std_displ_l = [np.std(vfo[f][step_size_i][:, tme_srs_l_idx]) for f in vfo]
    # get the correlation coefficient between the right and left vocal folds
    #corr_displ = [np.corrcoef(vfo[f][step_size_i][:, tme_srs_r_idx], vfo[f][step_size_i][:, tme_srs_l_idx])[0, 1] for f in vfo]
    # get the area of the enclosed region formed by the displacement data points with the Shoelace formula
    area = [shoelace_formula(vfo[f][step_size_i][:, tme_srs_r_idx], vfo[f][step_size_i][:, tme_srs_l_idx]) for f in vfo]
    # get the slope of the regression line fitted to the displacement data points
    regressions = [
        np.polyfit(
            vfo[f][step_size_i][:, tme_srs_r_idx].flatten(), 
            vfo[f][step_size_i][:, tme_srs_l_idx].flatten(), 
            1) 
        for f in vfo
        ]
    slope_displ = [r[0] for r in regressions]
    # get the intercept of the regression line fitted to the displacement data points
    intercept_displ = [r[1] for r in regressions]
    # get the residuals (vertical distances from each data point to the regression line)
    # polyval returns the value of the polynomial at each of the right displ values, which would be 
    # the estimated left displ values. then subtract the actual left displ values to get the residuals
    y_hat = [np.polyval(
                regressions[i],
                vfo[f][step_size_i][:, tme_srs_r_idx].flatten()
            )
        for i, f in enumerate(vfo)
        ]
    residuals = [
        y_hat[i] - vfo[f][step_size_i][:, tme_srs_l_idx].flatten()
        for i, f in enumerate(vfo)
        ]
    # coefficient of determination (R^2) for the regression line fitted to the displacement data points
    r2_displ = [r2_score(
            vfo[f][step_size_i][:, tme_srs_l_idx].flatten(), 
            y_hat[i]
            )
        for i, f in enumerate(vfo)
        ]
    # get the amplitude ratio of the right to left vocal folds
    amp_ratio = [np.divide(amp_displ_r[i], amp_displ_l[i]) for i in range(len(amp_displ_r))]
    # create dictionary of stats
    out_dict = {
        "amp_r": amp_displ_r,
        "amp_l": amp_displ_l,
        "mean_r": mean_displ_r,
        "mean_l": mean_displ_l,
        "range_r": range_displ_r,
        "range_l": range_displ_l,
        "std_r": std_displ_r,
        "std_l": std_displ_l,
        "area": area,
        "slope": slope_displ,
        "intercept": intercept_displ,
        "r2": r2_displ,
        "amp_ratio": amp_ratio
        }
    return out_dict

def get_dist(arr):
    # to get likelihood of each bin, need to count the number of times each bin occurs
    unique, counts = np.unique(arr, return_counts=True)
    # get sum of counts
    sum_counts = np.sum(counts)
    # divide each count by the sum of counts to get the likelihood of each bin
    dist = counts / sum_counts
    return dist 

def get_dist_smoothed(arr, alpha):
    dist = get_dist(arr)
    # smooth the distribution by taking the average of the distribution and the uniform distribution
    smoothed_dist = (1 - alpha) * dist + alpha * np.ones(len(dist)) / len(dist)
    return smoothed_dist

def bin_data(vfo, step_size_i, tme_srs_r_idx, tme_srs_l_idx, smoothing=True):
    num_bins = 1000
    all_x = [vfo[f][step_size_i][:, tme_srs_r_idx].flatten() for f in vfo]
    x_lens = [len(x) for x in all_x]
    concat_x = np.concatenate(all_x)
    all_y = [vfo[f][step_size_i][:, tme_srs_l_idx].flatten() for f in vfo]
    y_lens = [len(y) for y in all_y]
    concat_y = np.concatenate(all_y)
    # bin the data
    concat_all = np.concatenate((concat_x, concat_y))
    if smoothing:
        # split concat_x and concat_y into 60% training and 40% testing
        x_train, x_test, y_train, y_test = train_test_split(
            concat_x,
            concat_y,
            test_size=0.4,
            random_state=42
        )
        # concat x_train and y_train to get the training data
        train_data = np.concatenate((x_train, y_train))
        test_data = np.concatenate((x_test, y_test))
        # bin the training data
        binned_train, _, bin_size = utils.bin_series(train_data, b=num_bins)
        # use the binning from the training data to bin the test data 
        # get min of train data
        min_train = np.min(train_data)
        # create threshold array starting from min stepping by bin_size
        thresholds = np.arange(min_train + bin_size, min_train + bin_size * (num_bins), bin_size)
        # re-bin the train data using the thresholds
        # rebin_train, _, _ = utils.bin_series(train_data, bounds=thresholds) this worked to match binned_train
        binned_test, _, _ = utils.bin_series(test_data, bounds=thresholds)

        # loop over alpha values and calculate the log likelihood of the test data
        alpha_vals = np.linspace(0.1, 1, 20)
        log_likelihoods = []
        for alpha in tqdm(alpha_vals):
            dist = get_dist_smoothed(binned_test, alpha)
            log_likelihood = np.sum(np.log(dist))
            log_likelihoods.append(log_likelihood)
        # get the alpha value that maximizes the log likelihood
        max_log_likelihood = np.max(log_likelihoods)
        max_log_likelihood_idx = np.argmax(log_likelihoods)
        alpha = alpha_vals[max_log_likelihood_idx]
        print("alpha: ", alpha)
        return None, None
    else:
        binned_all, _, bin_size = utils.bin_series(concat_all, b=num_bins)
        binned_x = binned_all[:len(concat_x)]
        # recreate each waveform by splitting binned_x into the original lengths of each vfo
        binned_x = np.split(binned_x, np.cumsum(x_lens)[:-1])
        binned_y = binned_all[len(concat_y):]
        binned_y = np.split(binned_y, np.cumsum(y_lens)[:-1])
        return binned_x, binned_y

def get_info_theory_stats(vfo, tme_srs_r_idx, tme_srs_l_idx):
    step_size_i = 0
    binned_x, binned_y = bin_data(vfo, step_size_i, tme_srs_r_idx, tme_srs_l_idx)
    # calculate the entropy of the left and right vocal fold displacement data
    entropy_r = [stats.entropy(
            binned_x[i],
            base=2
            )
        for i in range(len(binned_x))
        ]
    entropy_l = [stats.entropy(
            binned_y[i],
            base=2
            )
        for i in range(len(binned_y))
        ]
    # calculate the mutual information between the left and right vocal fold displacement data
    mi = [mutual_info(
            binned_x[i],
            binned_y[i]
            )
        for i in range(len(binned_x))
        ]
    # calculate the KL divergence between the left and right vocal fold displacement data
    kl_div = [stats.entropy(
            binned_x[i],
            binned_y[i]
            )
        for i in range(len(binned_x))
        ]
    # create dictionary of stats
    out_dict = {
        "entropy_r": entropy_r,
        "entropy_l": entropy_l,
        "mutual_info": mi,
        "kl_div": kl_div,
        }
    return out_dict

def t_test_outputs(pos_stats, neg_stats, pos_def, series_type, step_size, sample_size, measure):
    for key in pos_stats:
        try:
            test_result = stats.ttest_ind(pos_stats[key], neg_stats[key], equal_var=True)
            if test_result.pvalue < 0.05:
                p_value = round(test_result.pvalue, 3)
                with open(f"results/cough_results_{sample_size}_{pos_def}_{measure}.txt", "a") as f:
                    f.write(f"{step_size}, {series_type}, {key}, {p_value}\n")
        except:
            print(f"Error in computing the test on the following key: {key}")

DATA_PROP = configs['data_proportion']
MEASURES = configs['measures']  
WRITE_TO_FILE = True

if DATA_PROP == 1:
    sample_df = label_df 
    sample_size = 'full'
else:
    sample_size = int(label_df.shape[0] * DATA_PROP)
    sample_df = label_df.sample(n=sample_size, random_state=42)
    sample_df = sample_df.reset_index(drop=True)

# create a directory "results" if it doesn't exist
if not os.path.exists("results"):
    os.mkdir("results")

pos_definitions = configs['pos_definitions']
for pos_def in pos_definitions:
    print(f"Positive-definition: {pos_def}")
    if WRITE_TO_FILE:
        with open(f"results_{sample_size}_{pos_def}_{MEASURES}.txt", "a") as f:
            f.write(f"Positive-definition: {pos_def}\n\n")
    pos_val, neg_val = configs["pos_neg_values"][pos_def]
    pos_df = sample_df[sample_df[pos_def] == pos_val]
    neg_df = sample_df[sample_df[pos_def] == neg_val]

    # get the file names for the positive and negative samples
    pos_files = pos_df['path'].tolist()
    neg_files = neg_df['path'].tolist()

    for step_size in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for series in ['displ', 'vel']:
            print(f"Time series: {series}")
            if series == 'displ':
                right_series_i = right_displ_i
                left_series_i = left_displ_i
            elif series == 'vel':
                right_series_i = right_vel_i
                left_series_i = left_vel_i

            pos_vfo = get_sampled_files(pos_files, [step_size])
            neg_vfo = get_sampled_files(neg_files, [step_size])

            if MEASURES == 'stats' or MEASURES == 'all':
                pos_stats = get_stats(pos_vfo, right_series_i, left_series_i)
                neg_stats = get_stats(neg_vfo, right_series_i, left_series_i)
                t_test_outputs(pos_stats, neg_stats, pos_def, series, step_size, sample_size, MEASURES)
            
            elif MEASURES == 'info-theory' or MEASURES == 'all': 
                pos_info_theory_stats = get_info_theory_stats(pos_vfo, right_series_i, left_series_i)
                neg_info_theory_stats = get_info_theory_stats(neg_vfo, right_series_i, left_series_i)
                t_test_outputs(pos_info_theory_stats, neg_info_theory_stats, pos_def, series, step_size, sample_size, MEASURES)
            
            else:
                print('Invalid value for MEASURES')
                break