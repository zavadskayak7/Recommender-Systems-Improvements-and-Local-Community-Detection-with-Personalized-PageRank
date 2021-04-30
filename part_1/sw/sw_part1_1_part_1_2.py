###
### PART 1
###
import surprise as sp
import pprint as pp
import time
from multiprocessing import cpu_count

print(cpu_count()) # shows number of cpu-cores

file_name = './dataset/ratings.csv'

print("Loading Dataset...")
reader = sp.Reader(line_format='user item rating', sep=',', \
                   rating_scale=[1, 5], skip_lines=1)
data = sp.Dataset.load_from_file(file_name, reader=reader)
print("Done.")

# defining the number of folds = 5
print("Performing splits...")
kf = sp.model_selection.KFold(n_splits=5, random_state=0)
print("Done.")

###
### PART 1.1
###
'''
application of all algorithms for recommendation made available by 
“Surprise” libraries, according to their default configuration.
'''
algorithms = [sp.NormalPredictor(), sp.BaselineOnly(), sp.KNNBasic(),\
              sp.KNNWithMeans(), sp.KNNWithZScore(), sp.KNNBaseline(),\
              sp.SVD(), sp.SVDpp(), sp.NMF(), sp.SlopeOne(), sp.CoClustering()]
for elem in algorithms:
    start_time = time.time()
    algo = elem
    sp.model_selection.cross_validate(algo, data, measures=['RMSE'], \
                                      cv=kf, n_jobs = 2, verbose=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    print()


###
### PART 1.2
###
'''
Improvement of the quality of both KNNBaseline and SVD methods, 
by performing hyper-parameters tuning over 5-folds
Random-Search-Cross-Validation - KNN
Grid-Search-Cross-Validation - SVD
'''
###
### KNNBaseline
###
start_time = time.time()
grid_of_parameters_KNN  = {
   'k': [20, 40, 60, 80],  # The MAXIMUM number of neighbors to take into account for aggregation.
   'min_k': [1, 5],  # The minimum number of neighbors to take into account for aggregation.
   #
   'sim_options': {  # A dictionary of options for the the similarity measure...
       'user_based': [False, True],  # True ==> UserUser-CF, False ==> ItemItem-CF
       'name': ['pearson_baseline'],
        'shrinkage': [30, 60, 100]
   }
}
# Randomized-Search-Cross-Validation performing 5-Folds-Cross-Validation
rs_KNN = sp.model_selection.RandomizedSearchCV(sp.KNNBaseline,\
                                               param_distributions=grid_of_parameters_KNN,\
                                               n_iter=8, measures=['rmse'], cv=kf,\
                                               n_jobs=-1,joblib_verbose=1000)
rs_KNN.fit(data)

# best RMSE score
print("BEST_SCORE: " + str(rs_KNN.best_score['rmse']))
# combination of parameters that gave the best RMSE score
print("BEST_PARAMETERS: ")
pp.pprint(rs_KNN.best_params['rmse'])
print("--- %s seconds ---" % (time.time() - start_time))
print()

###
### SVD configuration
###
start_time = time.time()
grid_of_parameters_SVD = {'n_factors': [100, 200],\
                          'n_epochs':[10, 20, 50],'lr_all':[0.001, 0.005, 0.1],\
                          'reg_all':[0.0, 0.05, 0.1]}
#Grid-Search performing 5-Folds-Cross-Validation
gs_SVD = sp.model_selection.GridSearchCV(sp.SVD,\
                                         param_grid = grid_of_parameters_SVD,\
                                         measures=['rmse'],cv=kf,\
                                         n_jobs=-1,joblib_verbose=1000)

gs_SVD.fit(data) 
# best RMSE score
print("BEST_SCORE: " + str(gs_SVD.best_score['rmse']))
# combination of parameters that gave the best RMSE score
print("BEST_PARAMETERS: ")
pp.pprint(gs_SVD.best_params['rmse'])
print("--- %s seconds ---" % (time.time() - start_time))
print()