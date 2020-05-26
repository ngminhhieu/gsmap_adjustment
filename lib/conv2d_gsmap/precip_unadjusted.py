def cal_error_gauge_gsmap():
    dataset = 'data/conv2d_gsmap/npz/map_gauge_72_stations.npz'
    map_precip = np.load(dataset)['map_precip']
    gauge_precip = np.load(dataset)['gauge_precip']
    cal_error(gauge_precip[-354:, :], map_precip[-354:, :])
    mae_error_each_gauge = np.zeros(shape=(72,2))
    sum_map = 0
    sum_gauge = 0
    map_0 = 0
    gauge_0 = 0
    margin = np.zeros(shape=(72,1))
    for i in range(72):
        mae_error_each_gauge[i, :] = cal_error(gauge_precip[-354:, i], map_precip[-354:, i])
        gauge_0 = np.count_nonzero(gauge_precip[-354:, i] > 0)
        map_0 = np.count_nonzero(map_precip[-354:, i])
        margin[i, 0] = gauge_0 - map_0
        sum_map = sum_map + map_0
        sum_gauge = sum_gauge + gauge_0
    
    print(sum_gauge - sum_map)
    np.savetxt('data/conv2d_gsmap/mae.csv', mae_error_each_gauge, delimiter=',')
    np.savetxt('data/conv2d_gsmap/margin.csv', margin, delimiter=',')


def cal_error(test_arr, prediction_arr):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    with np.errstate(divide='ignore', invalid='ignore'):
        # cal mse
        error_mae = mean_absolute_error(test_arr, prediction_arr)

        # cal rmse
        error_mse = mean_squared_error(test_arr, prediction_arr)
        error_rmse = np.sqrt(error_mse)

        print("MAE: %.4f" % (error_mae))
        print("RMSE: %.4f" % (error_rmse))
        return [error_mae, error_rmse]

cal_error_gauge_gsmap()