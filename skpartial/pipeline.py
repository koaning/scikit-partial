from sklearn.pipeline import Pipeline, FeatureUnion, _name_estimators


class PartialFeatureUnion(FeatureUnion):
    """
    A `PartialFeatureUnion` is a `FeatureUnion` but able to `.partial_fit`.

    Arguments:
        transformer_list: a list of transformers to apply and concatenate

    Example:

    ```python
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import HashingVectorizer

    from skpartial.pipeline import PartialPipeline, PartialFeatureUnion

    pipe = PartialPipeline([
        ("feat", PartialFeatureUnion([
            ("hash1", HashingVectorizer()),
            ("hash2", HashingVectorizer(ngram_range=(1,2)))
        ])),
        ("clf", SGDClassifier())
    ])

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written"
    ]

    y = np.array([1, 1, 1, 0, 0, 0])

    for loop in range(3):
        pipe.partial_fit(X, y, classes=[0, 1])

    assert np.all(pipe.predict(X) == np.array([1, 1, 1, 0, 0, 0]))
    ```
    """

    def partial_fit(self, X, y=None, classes=None, **kwargs):
        """
        Fits the components, but allow for batches.
        """
        for name, step in self.transformer_list:
            if not hasattr(step, "partial_fit"):
                raise ValueError(
                    f"Step {name} is a {step} which does not have `.partial_fit` implemented."
                )
        for name, step in self.transformer_list:
            if hasattr(step, "predict"):
                step.partial_fit(X, y, classes=classes, **kwargs)
            else:
                step.partial_fit(X, y)
        return self


def make_partial_union(*transformer_list):
    """
    Utility function to generate a `PartialFeatureUnion`

    Arguments:
        transformer_list: a list of transformers to apply and concatenate

    Example:

    ```python
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import HashingVectorizer

    from skpartial.pipeline import make_partial_pipeline, make_partial_union

    pipe = make_partial_pipeline(
        make_partial_union(
            HashingVectorizer(),
            HashingVectorizer(ngram_range=(1,2))
        ),
        SGDClassifier()
    )

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written"
    ]

    y = np.array([1, 1, 1, 0, 0, 0])

    for loop in range(3):
        pipe.partial_fit(X, y, classes=[0, 1])

    assert np.all(pipe.predict(X) == np.array([1, 1, 1, 0, 0, 0]))
    ```
    """
    return PartialFeatureUnion(_name_estimators(transformer_list))


class PartialPipeline(Pipeline):
    """
    Utility function to generate a `PartialPipeline`

    Arguments:
        steps: a collection of text-transformers

    ```python
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import HashingVectorizer

    from skpartial.pipeline import PartialPipeline

    pipe = PartialPipeline([
        ("hash", HashingVectorizer()),
        ("clf", SGDClassifier())
    ])

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written"
    ]

    y = np.array([1, 1, 1, 0, 0, 0])

    for loop in range(3):
        pipe.partial_fit(X, y, classes=[0, 1])

    assert np.all(pipe.predict(X) == np.array([1, 1, 1, 0, 0, 0]))
    ```
    """

    def partial_fit(self, X, y=None, classes=None, **kwargs):
        """
        Fits the components, but allow for batches.
        """
        for name, step in self.steps:
            if not hasattr(step, "partial_fit"):
                raise ValueError(
                    f"Step {name} is a {step} which does not have `.partial_fit` implemented."
                )
        for name, step in self.steps:
            if hasattr(step, "predict"):
                step.partial_fit(X, y, classes=classes, **kwargs)
            else:
                step.partial_fit(X, y)
            if hasattr(step, "transform"):
                X = step.transform(X)
        return self


def make_partial_pipeline(*steps):
    """
    Utility function to generate a `PartialPipeline`

    Arguments:
        steps: a collection of text-transformers

    ```python
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import HashingVectorizer

    from skpartial.pipeline import make_partial_pipeline

    pipe = make_partial_pipeline(
        HashingVectorizer(),
        SGDClassifier()
    )

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written"
    ]

    y = np.array([1, 1, 1, 0, 0, 0])

    for loop in range(3):
        pipe.partial_fit(X, y, classes=[0, 1])

    assert np.all(pipe.predict(X) == np.array([1, 1, 1, 0, 0, 0]))
    ```
    """
    return PartialPipeline(_name_estimators(steps))
