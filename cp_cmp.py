from pathlib import Path
import numpy as np
import scipy.io

import pandas as pd



'''
# result has keys: ['__
# gallery_f - features of gallery images
# gallery_label - location labels of gallery images
# gallery_path - paths of gallery images
# query_f - features of query images
# query_label - location labels of query images
# query_path - paths of query images

Each query location has 3 images, one for each altitude (80,90,100 meters)

Each gallery location has 6 images,
    2 images per altitude (80,90,100 meters)
        1 recent
        1 old

The query_path has the image names, ordered per location
    altitude order 100,80,90

The gallery_path has the image names, ordered per location and then
    H100, H100_old, H80, H80_old, H90, H90_old
'''




'''
Let's do conformal prediction based retrieval.

First,
1. filter the query features to only the H100 image in each location (the first feature vector in each location)
2. filter the gallery features to only the H100 and H100_old images in each location (the first and second feature vectors in each location)

Then,
1. compute the euclidean distance between each query image's features (H100, first image from each location)
and all gallery images (H100 and H100_old, first and second images from each location)
2. sort the distances in ascending order
3. for each query image, find the rank of the correct location's H100 OR H100_old image 
4. take the distance from the closest correct location
5. compute the alpha percentile of the correct distances
'''


def apply_cp(dists_, threshold):
    in_prediction_set = dists_<= threshold
    return np.where(in_prediction_set)


def apply_cp_aps(sims_, threshold):
    sorted_inds = np.argsort(sims_, axis=1)
    sorted_inds = sorted_inds[:,::-1] # ascending to descending order

    sorted_sims = sims_[np.arange(len(sims_)).reshape(-1,1), sorted_inds]

    sim_cumsum = np.cumsum(sorted_sims, axis=1)
    in_prediction_set = sim_cumsum <= threshold

    # get original indeces for each row
    ax0, ax1 = np.where(in_prediction_set)
    ax1_original = sorted_inds[ax0, ax1]

    return ax0, ax1_original

def compute_rank_of_true(query_labels, gallery_labels, distances=None, similarities=None):
    """
    Compute the rank of the true label for each query image, i.e.
        after sorting the distances/similarities, find the index of the closest true label in the sorted array

    Args:
        query_labels: np.array, N array, the labels of the query images
        gallery_labels: np.array, M array, the labels of the gallery images
        distances: np.array, NxM array, the distances between N query and M gallery images
        similarities: np.array, NxM array, the similarities between N query and M gallery images

    Returns:
        rank_of_true: np.array, N array, the rank of the true label for each query image
    
    """
    if distances is not None and similarities is not None:
        raise ValueError("Only one of distances or similarities should be provided")
    
    if distances is not None:
        sorted_indices = np.argsort(distances, axis=1)
    elif similarities is not None:
        sorted_indices = np.argsort(similarities, axis=1)
        sorted_indices = sorted_indices[:,::-1]
    else:
        raise ValueError("Either distances or similarities should be provided")

    rank_of_true = np.argmax((gallery_labels[sorted_indices] == query_labels[:, np.newaxis]),axis=1)
    return rank_of_true


def create_report(ax0_inds, ax1_inds, n, m, rank_of_true):
    in_prediction_set = np.zeros((n,m), dtype=bool)
    in_prediction_set[ax0_inds, ax1_inds] = True

    prediction_set_size = np.sum(in_prediction_set, axis=1)

    sets_without_true = (prediction_set_size < rank_of_true).sum()
    # print("Sets without true:", sets_without_true / len(prediction_set_size) * 100, "%")

    report = {
        "n empty prediction sets": np.sum(prediction_set_size == 0),
        "min prediction set size": np.min(prediction_set_size),
        "25th percentile prediction set size": np.percentile(prediction_set_size, 25),
        "50th percentile prediction set size": np.percentile(prediction_set_size, 50),
        "75th percentile prediction set size": np.percentile(prediction_set_size, 75),
        "max prediction set size": np.max(prediction_set_size),
        "sets without true [pct]": sets_without_true / len(prediction_set_size) * 100
    }
    return report


def conformal_prediction(alpha, dists, cal_labels, gallery_labels):
    sorted_indices = np.argsort(dists, axis=1)


    rank_of_true = np.argmax((gallery_labels[sorted_indices] == cal_labels[:, np.newaxis]),axis=1)
    
    dist_of_true = dists[np.arange(len(rank_of_true)), sorted_indices[np.arange(len(rank_of_true)),rank_of_true]]

    # Compute the alpha percentile of the correct distances
    n = len(cal_labels)
    tau = np.ceil(((n+1)*(1-alpha))) / n
    tau

    alpha_percentile_thresh = np.percentile(dist_of_true, tau*100)


    # print("Alpha percentile distance:", alpha_percentile_thresh)

    # compute the prediction set size on the calibration set
    prediction_set_size = np.sum(dists <= alpha_percentile_thresh, axis=1)

    #print("Prediction set size:", prediction_set_size)
    # print("# empty prediction sets:", np.sum(prediction_set_size == 0))

    # print("Min prediction set size:", np.min(prediction_set_size))
    # print("25th percentile prediction set size:", np.percentile(prediction_set_size, 25))
    # print("50th percentile prediction set size:", np.percentile(prediction_set_size, 50))
    # print("75th percentile prediction set size:", np.percentile(prediction_set_size, 75))
    # print("Max prediction set size:", np.max(prediction_set_size))


    sets_without_true = (prediction_set_size < rank_of_true).sum()
    # print("Sets without true:", sets_without_true / len(prediction_set_size) * 100, "%")

    report = {
        "alpha": alpha,
        "alpha percentile threshold": alpha_percentile_thresh,
        "n empty prediction sets": np.sum(prediction_set_size == 0),
        "min prediction set size": np.min(prediction_set_size),
        "25th percentile prediction set size": np.percentile(prediction_set_size, 25),
        "50th percentile prediction set size": np.percentile(prediction_set_size, 50),
        "75th percentile prediction set size": np.percentile(prediction_set_size, 75),
        "max prediction set size": np.max(prediction_set_size),
        "sets without true [pct]": sets_without_true / len(prediction_set_size) * 100
    }

    return report, alpha_percentile_thresh, prediction_set_size




def conformal_prediction_aps(alpha, sims, cal_labels, gallery_labels):
    '''
    Compute the conformal prediction set size using the Adaptive Prediction Sets (APS) method.
    The APS method uses similarity scores instead of distances to compute the prediction set size.

    Args:
    alpha: float, the significance level
    sims: np.array, NxM array, the similarity scores between N query and M gallery images
    cal_labels: np.array, N array, the labels of the query images
    gallery_labels: np.array, M array, the labels of the gallery images

    Returns:
    report: dict, a dictionary containing the results
    alpha_percentile_thresh: float, the alpha percentile threshold
    prediction_set_size: np.array, N array, the prediction set size for each query image
    '''

    sorted_indices = np.argsort(sims, axis=1)
    sorted_indices = sorted_indices[:,::-1] # ascending to descending order

    # get rank of closest true label    
    rank_of_true = np.argmax((gallery_labels[sorted_indices] == cal_labels[:, np.newaxis]),axis=1)

    # get similarity of closest true label
    sims_of_true = sims[np.arange(len(rank_of_true)), sorted_indices[np.arange(len(rank_of_true)),rank_of_true]]

    sorted_similarities = np.sort(sims, axis=1)
    sorted_similarities = sorted_similarities[:,::-1] # ascending to descending order

    # zero out all similarities after correct label
    sorted_similarities[sorted_similarities < sims_of_true[:, np.newaxis]] = 0

    # cumulative sum of similarities until the true label
    cumsim_of_true = np.sum(sorted_similarities, axis=1) 



    # Compute the alpha percentile of the correct distances
    n = len(cal_labels)
    tau = np.ceil(((n+1)*(1-alpha))) / n
    print(alpha, tau)

    alpha_percentile_thresh = np.percentile(cumsim_of_true, tau*100)

    ax0_inds, ax1_inds = apply_cp_aps(sims, alpha_percentile_thresh)

    in_prediction_set = np.zeros_like(sims, dtype=bool)
    in_prediction_set[ax0_inds, ax1_inds] = True

    prediction_set_size = np.sum(in_prediction_set, axis=1)

    sets_without_true = (prediction_set_size < rank_of_true).sum()
    # print("Sets without true:", sets_without_true / len(prediction_set_size) * 100, "%")

    report = {
        "alpha": alpha,
        "alpha percentile threshold": alpha_percentile_thresh,
        "n empty prediction sets": np.sum(prediction_set_size == 0),
        "min prediction set size": np.min(prediction_set_size),
        "25th percentile prediction set size": np.percentile(prediction_set_size, 25),
        "50th percentile prediction set size": np.percentile(prediction_set_size, 50),
        "75th percentile prediction set size": np.percentile(prediction_set_size, 75),
        "max prediction set size": np.max(prediction_set_size),
        "sets without true [pct]": sets_without_true / len(prediction_set_size) * 100
    }

    return report, alpha_percentile_thresh, prediction_set_size















# Define the path to the checkpoints directory
checkpoints_dir = Path('checkpoints')
mat_files = list(checkpoints_dir.rglob('*.mat'))


'''
In the same directory of each file of mat_files, there is a file named inference_time.txt
this file has a string like "Test complete in 0m 41s"
Extract all such files and parse the time in seconds


there is also a "results.txt" file in the same directory as the mat file
this file has a string like "Recall@1:55.30 Recall@5:75.72 Recall@10:83.91 Recall@top1:98.24 AP:42.60"
let's collect all metrics from all files
'''

results = {}
for mat_file in mat_files:
    # Read the results file
    results_file = mat_file.parent / 'results.txt'
    with open(results_file, 'r') as f:
        results_str = f.read().strip()

    # Parse the results string
    results_parts = results_str.split()
    recall_at_1 = float(results_parts[0].split(':')[1])
    recall_at_5 = float(results_parts[1].split(':')[1])
    recall_at_10 = float(results_parts[2].split(':')[1])
    recall_at_top1 = float(results_parts[3].split(':')[1])
    ap = float(results_parts[4].split(':')[1])



    # Read the inference time file
    time_file = mat_file.parent / 'inference_time.txt'
    with open(time_file, 'r') as f:
        time_str = f.read().strip()

    # Parse the time string to seconds
    time_parts = time_str.split()
    minutes = int(time_parts[3][:-1])
    seconds = int(time_parts[4][:-1])
    total_seconds = minutes * 60 + seconds

    # Store the results
    results[mat_file.parent.name] = {
        'recall@1': recall_at_1,
        'recall@5': recall_at_5,
        'recall@10': recall_at_10,
        'recall@top1': recall_at_top1,
        'ap': ap,
        'inference_time [s]': total_seconds,
    }

# Convert the results to a DataFrame for easier analysis
results_df = pd.DataFrame(results).T

# Display the results
print(results_df)



results2 = {}

mat_file = mat_files[0]

alpha = 0.05

for mat_file in mat_files:

    result = scipy.io.loadmat(mat_file)

    # Extract necessary data
    query_f = result["query_f"]
    query_labels = result["query_label"]
    gallery_f = result["gallery_f"]
    gallery_labels = result["gallery_label"]

    # Filter to only H100
    filtered_query_inds = np.arange(len(query_f))[::3]
    filtered_gallery_inds = np.concatenate((
        np.arange(len(result["gallery_label"][0]))[::6],
        np.arange(len(result["gallery_label"][0]))[1::6]
    ))

    filtered_query_f = query_f[filtered_query_inds, :]
    filtered_gallery_features = gallery_f[filtered_gallery_inds, :]
    filtered_query_labels = query_labels[0, filtered_query_inds]
    filtered_gallery_labels = gallery_labels[0, filtered_gallery_inds]

    # Compute distances and similarities
    distances = np.linalg.norm(filtered_query_f[:, np.newaxis] - filtered_gallery_features, axis=2)
    similarities = 1 / (1 + distances)
    sim_distances = 1 - similarities

    # CALIBRATION SET
    num_locations = 350
    np.random.seed(42)  # For reproducibility
    cal_inds = np.random.choice(np.arange(len(filtered_query_labels)), num_locations, replace=False)
    cal_distances = distances[cal_inds]
    cal_sim_distances = sim_distances[cal_inds]
    cal_similarities = similarities[cal_inds]
    cal_labels = filtered_query_labels[cal_inds]

    test_inds = np.setdiff1d(np.arange(len(filtered_query_labels)), cal_inds)
    test_distances = distances[test_inds]
    test_sim_distances = sim_distances[test_inds]
    test_similarities = similarities[test_inds]
    test_labels = filtered_query_labels[test_inds]

    _, alpha_percentile_thresh_dist, _ = conformal_prediction(alpha, cal_distances, cal_labels, filtered_gallery_labels)
    _, alpha_percentile_thresh_simdist, _ = conformal_prediction(alpha, cal_sim_distances, cal_labels, filtered_gallery_labels)
    _, alpha_percentile_thresh_sim, _ = conformal_prediction_aps(alpha, cal_similarities, cal_labels, filtered_gallery_labels)

    # Euclidean distances
    cal_ax0, cal_ax1 = apply_cp(cal_distances, alpha_percentile_thresh_dist)
    test_ax0, test_ax1 = apply_cp(test_distances, alpha_percentile_thresh_dist)
    cal_true_rank = compute_rank_of_true(cal_labels, filtered_gallery_labels, distances=cal_distances)
    test_true_rank = compute_rank_of_true(test_labels, filtered_gallery_labels, distances=test_distances)
    cal_report = create_report(cal_ax0, cal_ax1, len(cal_labels), len(filtered_gallery_labels), cal_true_rank)
    test_report = create_report(test_ax0, test_ax1, len(test_labels), len(filtered_gallery_labels), test_true_rank)

    # Euclidean (similarity) distances
    cal_ax0, cal_ax1 = apply_cp(cal_sim_distances, alpha_percentile_thresh_simdist)
    test_ax0, test_ax1 = apply_cp(test_sim_distances, alpha_percentile_thresh_simdist)
    cal_true_rank = compute_rank_of_true(cal_labels, filtered_gallery_labels, distances=cal_sim_distances)
    test_true_rank = compute_rank_of_true(test_labels, filtered_gallery_labels, distances=test_sim_distances)
    cal_report_simdist = create_report(cal_ax0, cal_ax1, len(cal_labels), len(filtered_gallery_labels), cal_true_rank)
    test_report_simdist = create_report(test_ax0, test_ax1, len(test_labels), len(filtered_gallery_labels), test_true_rank)

    # APS Similarity
    cal_ax0, cal_ax1 = apply_cp_aps(cal_similarities, alpha_percentile_thresh_sim)
    test_ax0, test_ax1 = apply_cp_aps(test_similarities, alpha_percentile_thresh_sim)
    cal_true_rank = compute_rank_of_true(cal_labels, filtered_gallery_labels, similarities=cal_similarities)
    test_true_rank = compute_rank_of_true(test_labels, filtered_gallery_labels, similarities=test_similarities)
    cal_report_sim = create_report(cal_ax0, cal_ax1, len(cal_labels), len(filtered_gallery_labels), cal_true_rank)
    test_report_sim = create_report(test_ax0, test_ax1, len(test_labels), len(filtered_gallery_labels), test_true_rank)


    report_data = {
    #    'cal': cal_report,
        'CP': test_report,
    #    'cal_report_simdist': cal_report_simdist,
    #    'test_report_simdist': test_report_simdist,
    #    'cal_report_sim': cal_report_sim,
        'APS': test_report_sim
    }

    # Store results
    results2[mat_file.parent.name] = report_data


    df = pd.DataFrame(report_data)
    df_stacked = df.stack() # Stack the DataFrame to create a MultiIndex
    df_stacked.index = df_stacked.index.set_names(['Metric', 'Report Type']) # Rename the index levels for clarity

    df_stacked

    results2[mat_file.parent.name] = df_stacked




results2


df = pd.concat(results2, axis=1)

results_df

# Concatenate horizontally
combined_df = pd.concat([df.T, results_df], axis=1)

combined_df.columns

df
