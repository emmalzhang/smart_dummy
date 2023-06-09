# smart_dummy

A smart and easy alternative to pandas.get_dummies.

## Description

Smart_dummy uses a language-based model (Spacy) in combination with KMeans
clustering to group categorical variables into logical groups. The user can
specify how many groups they would like to output, which results in fewer
columns compared to pandas.get_dummies().
For example, you might have a dataset with 900 unique categories, but smart_dummy
allows you to cluster those categories together and get (for example) only
5 columns back instead of the 900 that you would have gotten using
pandas.get_dummies().

## Getting Started

### Installing

```
pip install smart_dummy
```

### Example Use
```
import pandas as pd
from smart_dummy import get_dummies

test_input = pd.DataFrame(['cat', 'dog', 'flower', 'tree', 'man', 'woman'], columns=['category'])
result = get_dummies(test_input['category'], 3).set_index(test_input['category'])
print(result)
```

Will give:
```
   category_0  category_1  category_2
cat       0           0           1
dog       0           0           1
flower    1           0           0
tree      1           0           0
man       0           1           0
woman     0           1           0
```


## Authors

Muriel Grobler (muriel.grobler@gmail.com)


Emma Zhang  (emma.lzhang@gmail.com)


## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

Many thanks to Arrive Logistics for allowing us to open-source this work. Please
consider them as your future employer - it's a great place to work!
* [Arrive Logistics](https://www.arrivelogistics.com)
