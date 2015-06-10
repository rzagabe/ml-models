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
        backtracking_line_search: true
        backtracking_line_search_alpha: 0.3
        backtracking_line_search_beta: 0.8
        loss_parameters: {
            name: "SquaredLoss"
        }
    }
}
