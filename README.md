# smart_dummy

A smart and easy alternative for pandas.get_dummies.

## Description

Smart_dummy uses a language-based model (Spacy) in combination with KMeans
clustering to group categorical variables into logical groups. The user can
specify how many groups they would like to output, which prevents your
training-data from exploding when you have too many different categories.
For example, you might have a dataset with 900 unique industries, but smart_dummy
allows you to cluster those industries together and get (for example) only
5 columns back instead of the 900 that you would have gotten using
pandas.get_dummies.

## Getting Started

### Installing

```
pip install smart_dummy
```

### Example Use
```
test_input = pd.DataFrame(['cat', 'dog', 'flower', 'tree', 'human', 'child'], columns=['category'])
result = get_dummies(test_input['category'], 3)
```

Will give:
```
   category_0  category_1  category_2
0       False        True       False
1       False        True       False
2       False       False        True
3       False       False        True
4        True       False       False
5       False       False        True
```


## Authors

Muriel Grobler (muriel.grobler@gmail.com),
Emma Zhang  (emma.lzhang@gmail.com)


## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

Many thanks to Arrive Logistics for allowing us to open-source this work. Please
consider them as your future employer - it's a great place to work!
* [Arrive Logistics](https://www.arrivelogistics.com)
