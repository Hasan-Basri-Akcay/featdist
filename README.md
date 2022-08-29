# Don't Miss The Contents
Medium: https://medium.com/subscribe/@hasan.basri.akcay <br />
LinkedIn: https://www.linkedin.com/in/hasan-basri-akcay

# featdist
Featdist (Train Test Target Distribution) helps with feature understanding, calculating feature importances, feature comparison, feature debugging, and leakage detection

## Installation
```
pip install featdist
```

## Using featdist
Detailed [Medium post](https://medium.com/@hasan.basri.akcay) on using featdist.

Train test distribution is so crucial for machine learning. Train and test distribution should be similar for better ml models. Therefore, looking for distribution differences in exploratory data analysis is one of the most important steps of data science. The featdist library aims to get better and faster this process.

### For Numerical Features
```
from featdist import numerical_ttt_dist

df_stats = numerical_ttt_dist(train=X_train, test=X_test, features=num_features, target='target_reg')
df_stats
```
<img src="/outputs/numerical_ttt_train_test.png?raw=true"/>
<img src="/outputs/numerical_ttt_train_test_df.png?raw=true"/>

```
from featdist import numerical_ttt_dist

df_stats = numerical_ttt_dist(train=X_train, val=X_val, features=num_features, target='target_reg')
df_stats
```
<img src="/outputs/numerical_ttt_train_val.png?raw=true"/>
<img src="/outputs/numerical_ttt_train_val_df.png?raw=true"/>

### For Categorical Features
```
from featdist import categorical_ttt_dist

df_stats = categorical_ttt_dist(train=X_train, test=X_test, features=cat_features, target='target_reg')
df_stats
```
<img src="/outputs/categorical_ttt_train_test.png?raw=true"/>
<img src="/outputs/categorical_ttt_train_test_df.png?raw=true"/>

```
from featdist import categorical_ttt_dist

df_stats = categorical_ttt_dist(train=X_train, val=X_val, features=cat_features, target='target_reg')
df_stats
```
<img src="/outputs/categorical_ttt_train_val.png?raw=true"/>
<img src="/outputs/categorical_ttt_train_val_df.png?raw=true"/>
