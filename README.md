# scikit-partial

Pipeline components that support partial_fit.

<img src="img.png" width=175 align="right">

# scikit-partial

> Pipeline components that support partial_fit.

The goal of **scikit-partial** is to offer a pipeline that can run
`partial_fit`. This allows of online learning on an entire pipeline.

## Installation

You can install everything with `pip`:

```
python -m pip install scikit-partial
```

## Usage 

Assuming that you use a stateless featurizer in your pipeline, such as [HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer) or language models from [whatlies](https://koaning.github.io/whatlies/api/language/universal_sentence/), you choose to pre-train your scikit-learn model beforehand and fine-tune it later using models that offer the `.partial_fit()`-api. If you're unfamiliar with this api, you might appreciate [this course on calmcode](https://calmcode.io/partial_fit/introduction.html).


```python
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

from skpartial.pipeline import make_partial_pipeline

url = "https://raw.githubusercontent.com/koaning/icepickle/main/datasets/imdb_subset.csv"
df = pd.read_csv(url)
X, y = list(df['text']), df['label']

# Construct a pipeline with components that are `.partial_fit()` compatible
pipe = make_partial_pipeline(HashingVectorizer(), SGDClassifier(loss="log"))

# Run the learning algorithm on batches of data
for i in range(10):
    # We could also do a whole bunch of data augmentation here!
    pipe.partial_fit(X, y)
```

When is this pattern useful? Let's consider spelling errors. Suppose that we'd like
our algorithm to be robust against typos. Then we can simulate typos on our `X` inside
of our learning loop. 
