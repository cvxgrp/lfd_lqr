# lfd_lqr

This repository hosts the code for running the experiments
described in our paper [Fitting a Linear Control Policy to Demonstrations with a Kalman Constraint](http://web.stanford.edu/~boyd/papers/lfd_lqr.html).

## Running the experiments

Create either a conda or virtualenv Python 3 environment.
Then run
```
pip install -r requirements.txt
```

Each experiment is in a self-contained Jupyter notebook.
Start a Jupyter notebook server with
```
jupyter notebook
```
and then run each of the individual notebooks: `small random.ipynb`, `flip.ipynb`, and `aircraft control.ipynb`.

## Methods

The methods used to carry out our algorithm is in the file `algorithms.py`.
The signature of the function is:
```
def _policy_fitting_with_a_kalman_constraint(L, r, xs, us_observed, A, B, P, Q, R, niter=50, rho=1):
    """
    Policy fitting with a Kalman constraint.

    Args:
        - L: function that takes in a cvxpy Variable
            and returns a cvxpy expression representing the objective.
        - r: function that takes in a cvxpy Variable
            and returns a cvxpy expression and a list of constraints
            representing the regularization function.
        - xs: N x n matrix of states.
        - us_observed: N x m matrix of inputs.
        - A: n x n dynamics matrix.
        - B: n x m dynamics matrix.
        - P: n x n PSD matrix, the initial PSD cost-to-go coefficient.
        - Q: n x n PSD matrix, the initial state cost coefficient.
        - R: n x n PD matrix, the initial input cost coefficient.
        - niter: int (optional). Number of iterations (default=50).
        - rho: double (optional). Penalty parameter (default=1).
    
    Returns:
        - K: m x n gain matrix found by policy fitting with a Kalman constraint. 
    """
```

One can call the above function multiple times with both zero and random restarts
and pick the control policy with the lowest loss by calling:
```

def policy_fitting_with_a_kalman_constraint(L, r, xs, us_observed, A, B, n_random=5, niter=50, rho=1):
    """
    Wrapper around _policy_fitting_with_a_kalman_constraint.
    """
```

## License
This repository carries an Apache 2.0 license.

## Citing
If you use our code for research, please cite our accompanying paper:
```
@article{palan2020fitting,
  author={Palan, M. and Barratt, S. and McCauley, A. and Sadigh, D. and Sindhwani, V. and Boyd, S.},
  title={Fitting a Linear Control Policy to Demonstrations with a Kalman Constraint},
  year={2020},
  howpublished={\texttt{http://web.stanford.edu/~boyd/papers/lfd_lqr.html}}
}
```
