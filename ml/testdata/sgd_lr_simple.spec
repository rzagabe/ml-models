name: "LinearRegression"
type: REGRESSION
[ml.LinearRegressionParameters.options]: {
    max_iter: 100
    min_iter: 5
    eps: 0.001
    gamma: 0.001
    optimization_parameters: {
        name: "StochasticGradientDescent"
        batch_size: 1
        loss_parameters: {
            name: "RidgeLoss"
            l2_regularization: 0.01
        }
    }
}
