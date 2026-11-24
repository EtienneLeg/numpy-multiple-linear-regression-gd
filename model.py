"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """

    rng = np.random.default_rng(seed)
    indices = rng.permutation(X.shape[0])
    return X[indices], y[indices]
# TODO: Return (X, y) under one shared seeded row permutation
    pass

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    n=X.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    (X_train, y_train, X_val, y_val, X_test, y_test) = (X[:n_train],y[:n_train],X[n_train:n_train + n_val],y[n_train:n_train + n_val],X[n_train + n_val:],y[n_train + n_val:])
    return(X_train, y_train, X_val, y_val, X_test, y_test)
    pass

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    n=X.shape[1]
    mean = np.zeros(n)
    std = np.zeros(n)
    for x in range(n):
        mean[x] = np.mean(X[:,x])
        std[x] = np.std(X[:,x])
        if std[x]==0:
            std[x]=1
    return(mean,std)
    pass

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    return((X - mean)/std)

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    (n,d) = X.shape
    A=np.ones((n,d+1))
    A[:,1:]=X
    return(A)
    pass

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    pass
    X = standardize_features(X, mean, std)
    A = add_bias_column(X)
    return(A)

# Step 7 - predict_linear
def predict_linear(X, weights):
    target = X@weights
    return(target)
    # TODO: Return the predicted target vector from X and weights
    pass

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    err = np.mean((y_true - y_pred)**2)
    return(err)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    n = X.shape[0]
    return(2/n * np.transpose(X) @ (y_pred - y_true))
    pass

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    w = np.linalg.pinv(X) @ y
    return(w)
    pass

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    if seed == None:
        rng = np.random.default_rng(seed)
        w = rng.normal(loc=0.0, scale=0.01, size=(n_features,))
    else : 
        np.random.seed(seed)
        w = np.random.normal(loc=0.0, scale=0.01, size=(n_features,))
    return(w)
    pass

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    yest = predict_linear(X, weights)
    g = mse_gradient(X,y,yest)
    weights = weights - lr*g
    return(weights)

    # TODO: return the updated weight vector after one MSE gradient step
    pass

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    yesttrain = predict_linear(X_train,weights)
    train_loss = mse_loss(y_train,yesttrain)
    yestval = predict_linear(X_val,weights)
    val_loss = mse_loss(y_val,yestval)
    return((train_loss,val_loss)) 
    # TODO: return the pair (train_loss, val_loss) as MSE floats
    pass

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
    stop = False
    if val_loss >= best_val_loss:
        wait += 1
        if wait==patience:
            stop=True
            
    else :
        best_weights = weights
        best_val_loss = val_loss
        wait = 0
        
    
    return(best_val_loss,wait,best_weights,stop)

    pass

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.
    w0 = initialize_weights(n_features,seed)
    d = {"weights":w0,"best_weights":np.copy(w0),"best_val_loss":np.inf,"wait":0,"train_losses":[],"val_losses":[],"stopped" : False}
    return(d)
    pass

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    state["weights"] = gd_step(X_train, y_train, state["weights"], lr)
    
    train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, state["weights"])
    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)
    
    (
        state["best_val_loss"],
        state["wait"],
        state["best_weights"],
        state["stopped"],
    ) = update_early_stop_state(
        val_loss,
        state["best_val_loss"],
        state["wait"],
        state["weights"],
        state["best_weights"],
        patience,
    )
    
    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    # TODO: Train weights with full-batch GD for up to epochs, with early stopping.
    state = init_training_state(X_train.shape[1],seed)
    for _ in range(epochs):
        state = run_one_epoch(state,X_train,y_train,X_val,y_val,lr,patience)
        if state["stopped"]==True:
            break
    return(state["best_weights"],state["train_losses"],state["val_losses"])
    pass

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: Compute the mean absolute error between true targets and predictions
    s=0
    n = y_true.shape[0]
    for i in range(n):
        s += abs(y_true[i] - y_pred[i])
    return(s/n)
    pass

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    # TODO: Return the root mean squared error between y_true and y_pred.
    n = len(y_true)
    s=0
    for i in range(n):
        s+= (y_true[i]-y_pred[i])**2
    return((s/n)**(1/2))
    pass

# Step 20 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute the coefficient of determination R^2.
    s1 = 0
    s2 = 0
    n = len(y_true)
    m = np.mean(y_true)
    for i in range(n):
        s1 += (y_true[i] - y_pred[i])**2
        s2 += (y_true[i]-m)**2
    if s2 == 0:
        return(np.nan)
    return(1-s1/s2)
    pass

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, and R^2 into one metrics dictionary for test-set reporting.
    d={}
    d['mae'] = mean_absolute_error(y_true,y_pred)
    d['rmse'] = root_mean_squared_error(y_true,y_pred)
    d['r2'] = r_squared(y_true,y_pred)
    return(d)
    pass

# Step 22 - learning_curve_data
def learning_curve_data(train_losses, val_losses):
    # TODO: Return epoch indices and loss series for external plotting...
    epochs = [i+1 for i in range(len(train_losses))]
    train_lossesl = [float(i) for i in train_losses]
    val_lossesl = [float(i) for i in val_losses]
    return(epochs,train_lossesl, val_lossesl)
    pass

# Step 23 - weights_l2_distance
def weights_l2_distance(w_gd, w_closed):
    # TODO: Compute the L2 distance between two weight vectors
    n = len(w_gd)
    s=0
    for i in range(n):
        s+=(w_gd[i] - w_closed[i])**2
    return(s**(1/2))
    pass

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    # TODO: Build the initial LinearRegressionGD-style model dictionary...
    d={'learning_rate':learning_rate,'epochs':epochs,'patience':patience,'seed':seed,'weights':None,'normal_weights':None,'mean':None,'std':None,'train_losses':[],'val_losses':[]}
    return(d)
    pass

# Step 25 - fit_lr_model
def fit_lr_model(model, X_train, y_train, X_val, y_val):
    # TODO: Fit model with train stats, design matrices, GD, and normal eq
    model['mean'],model['std'] = compute_feature_stats(X_train)
    At = prepare_design_matrix(X_train,model['mean'],model['std'])
    Av = prepare_design_matrix(X_val,model['mean'],model['std'])
    model['normal_weights'] = normal_equation(At,y_train)
    model['weights'],model['train_losses'],model['val_losses'] = train_batch_gd(At,y_train,Av,y_val,model['learning_rate'],model["epochs"],model['patience'],model['seed']) 
    return(model)
    
    pass

# Step 26 - predict_lr_model
def predict_lr_model(model, X):
    # TODO: Return predicted targets for raw X using the fitted model.
    A = prepare_design_matrix(X,model["mean"],model["std"])
    target = predict_linear(A,model['weights'])
    return(target)
    pass

# Step 27 - score_lr_model
import numpy as np
def score_lr_model(model, X, y):
    # TODO: Predict on raw features and return MAE, RMSE, and R^2 metrics.
    target = predict_lr_model(model,X)
    d = evaluate_regression(y,target)
    return(d)
    pass

# Step 28 - compare_with_normal_equation
def compare_with_normal_equation(model):
    # TODO: Return the L2 distance between GD and normal-equation weights.
    return(weights_l2_distance(model['weights'],model['normal_weights']))
    pass

