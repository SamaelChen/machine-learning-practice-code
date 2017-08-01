```python
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15,
                                                min_samples_leaf=100,
                                                max_features=100,
                                                criterion='entropy',
                                                class_weight={0: 1,
                                                              1: 2.05}),
                         algorithm="SAMME",
                         n_estimators=100,
                         learning_rate=1)
```
<img src=1.png>

---

```python
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20,
                                                min_samples_leaf=100,
                                                max_features=100,
                                                criterion='entropy',
                                                class_weight={0: 1,
                                                              1: 2.05}),
                         algorithm="SAMME",
                         n_estimators=100,
                         learning_rate=1)
```
<img src=2.png>

---

```python
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=25,
                                                min_samples_leaf=100,
                                                max_features=100,
                                                criterion='entropy',
                                                class_weight={0: 1,
                                                              1: 2.05}),
                         algorithm="SAMME",
                         n_estimators=100,
                         learning_rate=1)
```
<img src=3.png>

---

```python
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=25,
                                                min_samples_leaf=100,
                                                max_features=100,
                                                criterion='gini',
                                                class_weight={0: 1,
                                                              1: 2.05}),
                         algorithm="SAMME",
                         n_estimators=100,
                         learning_rate=1)
```
<img src=4.png>

---

```python
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20,
                                                min_samples_leaf=100,
                                                max_features=100,
                                                criterion='gini',
                                                class_weight={0: 1,
                                                              1: 2.05}),
                         algorithm="SAMME",
                         n_estimators=100,
                         learning_rate=1)
```
<img src=5.png>

---

```python
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20,
                                                min_samples_leaf=100,
                                                max_features=150,
                                                criterion='gini',
                                                class_weight={0: 1,
                                                              1: 2.05}),
                         algorithm="SAMME",
                         n_estimators=100,
                         learning_rate=1)
```
<img src=6.png>

```python
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=25,
                                                min_samples_leaf=100,
                                                max_features=150,
                                                criterion='gini',
                                                class_weight={0: 1,
                                                              1: 2.05}),
                         algorithm="SAMME",
                         n_estimators=100,
                         learning_rate=1)
```
<img src=7.png>

```python
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30,
                                                min_samples_leaf=100,
                                                max_features=150,
                                                criterion='gini',
                                                class_weight={0: 1,
                                                              1: 2.05}),
                         algorithm="SAMME",
                         n_estimators=100,
                         learning_rate=1)
```
<img src=8.png>
