## Hyperparameter Tuning

I started off by turning off the `regularization_strength` and `learning_rate_decay` to find an appropriate `learning_rate` range. 

With the following code:

```python
# hyperparameters
learning_rates = [1e-4]
learning_rate_decays = [1.0]
regularization_strengths = [0.0]
```
the loss barely budges so the learning rate is too small. I made it bigger. 

```python
# hyperparameter tuning
learning_rates = [1e-2]
learning_rate_decays = [1.0]
regularization_strengths = [0.0]
```

Terminal outputs `nan` and some other errors meaning our learning rate is too big now. 

Hence, I decided to set it to in the `1e-3` range. Now, I create a `random_search` function and try out values for `learning_rate_decays` and `reg_strength`.

For example:

```python
# bigger => more time
max_count = 20

for count in range(max_count):
    # randomly create params
    rs = 10**np.random.uniform(-4.0, -3.0)
    lrd = 10**np.random.uniform(-0.1, 0)
    # ...code...
```
