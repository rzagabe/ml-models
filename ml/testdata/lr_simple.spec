name: "LinearRegression"
type: REGRESSION
[ml.LinearRegressionParameters.options]: {
    max_iter: 100
    min_iter: 5
    eps: 0.001
    gamma: 0.001
    optimization_parameters: {
        name: "GradientDescent"
        batch_size: 1
        loss_parameters: {
            name: "SquaredLoss"
        }
    }
}
