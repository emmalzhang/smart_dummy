import pandas as pd
from smart_dummy import get_dummies

test_input = pd.DataFrame(['cat', 'dog', 'flower', 'tree', 'man', 'woman'] * 10000, columns=['category'])
print(get_dummies(test_input['category'], 3))