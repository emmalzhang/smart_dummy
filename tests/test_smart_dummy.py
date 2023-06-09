import pandas as pd
from smart_dummy import get_dummies

test_input = pd.DataFrame(['cat', 'dog', 'flower', 'tree', 'human', 'child'], columns=['category'])
print(get_dummies(test_input['category'], 3))