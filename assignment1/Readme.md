## Hyperparameter Tuning

Turn off regularization and learning rate decay to find appropriate learning rate range. 

Started with thw following:

```python
# hyperparameters
learning_rates = [1e-6]
learning_rate_decays = [1.0]
regularization_strengths = [0.0]
```
The loss does not decrease so the learning rate is too small. Make it bigger. 

```python
# hyperparameter tuning
learning_rates = [1e-1]
learning_rate_decays = [1.0]
regularization_strengths = [0.0]
```

We get `NaN` which means it is too big. We slowly decrease and find range to be between 1e-3 and 1e-4.


