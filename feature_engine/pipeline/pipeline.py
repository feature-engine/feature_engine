# Modified from scikit-learn's pipeline:
# https://github.com/scikit-learn/scikit-learn/blob/6eff1757e/sklearn/pipeline.py#L59

# Looked at imbalanced learn pipeline as template:
# https://github.com/scikit-learn-contrib/imbalanced-learn

from sklearn import pipeline
from sklearn.base import _fit_context, clone
from sklearn.pipeline import _final_estimator_has, _fit_transform_one
try:
    from sklearn.utils import _print_elapsed_time
except ImportError:
    from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils._metadata_requests import METHODS
from sklearn.utils._param_validation import HasMethods, Hidden
from sklearn.utils.metadata_routing import _routing_enabled, process_routing
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory

METHODS.append("transform_x_y")


def _fit_transform_x_y_one(
    transformer, X, y, message_clsname="", message=None, params=None
):
    with _print_elapsed_time(message_clsname, message):
        transformer.fit(X, y)
        Xt, yt = transformer.transform_x_y(X, y, **params.get("transform_x_y", {}))
        return Xt, yt, transformer


class Pipeline(pipeline.Pipeline):
    """
    A sequence of data transformers with an optional final predictor.

    `Pipeline` allows you to sequentially apply a list of transformers to
    preprocess the data and, if desired, conclude the sequence with a final
    `predictor` for predictive modeling.

    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement `fit` and `transform` methods.
    The final `estimator` only needs to implement `fit`.
    The transformers in the pipeline can be cached using ``memory`` argument.

    This pipeline allows intermediate transformers to remove rows from the
    dataset. It will automatically adjust the target variable to match the
    remaining observations.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters. For this, it
    enables setting parameters of the various steps using their names and the
    parameter name separated by a `'__'`, as in the example below. A step's
    estimator may be replaced entirely by setting the parameter with its name
    to another estimator, or a transformer removed by setting it to
    `'passthrough'` or `None`.

    More details in the :ref:`User Guide <pipeline>`.

    Parameters
    ----------
    steps : list of tuples
        List of (name of step, estimator) tuples that are to be chained in
        sequential order. To be compatible with the scikit-learn API, all steps
        must define `fit`. All non-last steps must also define `transform`. See
        :ref:`Combining Estimators <combining_estimators>` for more details.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the transformers
        before fitting. Therefore, the transformer instance given to the
        pipeline cannot be inspected directly. Use the attribute ``named_steps``
        or ``steps`` to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only exist if the last step of the pipeline is a
        classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying first estimator in `steps` exposes such an attribute
        when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.


    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from feature_engine.pipeline import Pipeline
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    >>> # The pipeline can be used as any other estimator
    >>> # and avoids leaking the test set into the train set
    >>> pipe.fit(X_train, y_train).score(X_test, y_test)
    0.88
    >>> # An estimator's parameter can be set using '__' syntax
    >>> pipe.set_params(svc__C=10).fit(X_train, y_train).score(X_test, y_test)
    0.76
    """

    # BaseEstimator interface
    _required_parameters = ["steps"]

    _parameter_constraints: dict = {
        "steps": [list, Hidden(tuple)],
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["boolean"],
    }

    def __init__(self, steps, *, memory=None, verbose=False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose

    def _fit(self, X, y=None, routed_params=None):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Set up the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        fit_transform_x_y_one_cached = memory.cache(_fit_transform_x_y_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)

            # Fit or load from cache the current transformer
            if hasattr(cloned_transformer, "transform_x_y"):
                X, y, fitted_transformer = fit_transform_x_y_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    message_clsname="Pipeline",
                    message=self._log_message(step_idx),
                    params=routed_params[name],
                )
            elif hasattr(cloned_transformer, "transform") or hasattr(
                cloned_transformer, "fit_transform"
            ):
                X, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    None,
                    message_clsname="Pipeline",
                    message=self._log_message(step_idx),
                    params=routed_params[name],
                )

            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X, y

    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **params):
        """Fit the model.

        Fit all the transformers one after the other and transform the data, then fit
        the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True` is set via
                :func:`~sklearn.set_config`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        self : Pipeline
            This estimator.
        """
        routed_params = self._check_method_params(method="fit", props=params)
        Xt, yt = self._fit(X, y, routed_params)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_params = routed_params[self.steps[-1][0]]
                self._final_estimator.fit(Xt, yt, **last_step_params["fit"])
        return self

    def _can_fit_transform(self):
        return (
            self._final_estimator == "passthrough"
            or hasattr(self._final_estimator, "transform")
            or hasattr(self._final_estimator, "fit_transform")
        )

    @available_if(_can_fit_transform)
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X, y=None, **params):
        """Fit the model and transform with the final transformer.

        Fit all the transformers one after the other and sequentially transform
        the data. Only valid if last step of the pipeline has method `transform`.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        routed_params = self._check_method_params(method="fit_transform", props=params)
        Xt, yt = self._fit(X, y, routed_params)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            last_step_params = routed_params[self.steps[-1][0]]
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(
                    Xt, yt, **last_step_params["fit_transform"]
                )
            else:
                return last_step.fit(Xt, y, **last_step_params["fit"]).transform(
                    Xt, **last_step_params["transform"]
                )

    @available_if(_final_estimator_has("fit_predict"))
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_predict(self, X, y=None, **params):
        """Transform the data, and apply `fit_predict` with the final estimator.

        Call `fit_transform` of each transformer in the pipeline. The
        transformed data are finally passed to the final estimator that calls
        `fit_predict` method. Only valid if the final estimator implements
        `fit_predict`.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters to the ``predict`` called at the end of all
                transformations in the pipeline.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionadded:: 0.20

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

            Note that while this may be used to return uncertainties from some
            models with ``return_std`` or ``return_cov``, uncertainties that are
            generated by the transformations in the pipeline are not propagated
            to the final estimator.

        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.
        """
        routed_params = self._check_method_params(method="fit_predict", props=params)
        Xt, yt = self._fit(X, y, routed_params)

        params_last_step = routed_params[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][1].fit_predict(
                Xt, yt, **params_last_step.get("fit_predict", {})
            )
        return y_pred

    def _can_transform_x_y(self):
        can_transform_x_y = any(
            [
                transformer
                for _, _, transformer in self._iter(
                    with_final=True, filter_passthrough=False
                )
                if hasattr(transformer, "transform_x_y")
            ]
        )
        last_step_is_transform = self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
        )
        return can_transform_x_y and last_step_is_transform

    @available_if(_can_transform_x_y)
    def transform_x_y(self, X, y, **params):
        """Fit the model and transform with the final estimator.

        Fit all the transformers one after the other and sequentially transform
        the data and the target. Only valid if the final estimator either implements
        `fit_transform` or `fit` and `transform`.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

            .. versionchanged:: 1.4
                Parameters are now passed to the ``transform`` method of the
                intermediate steps as well, if requested, and if
                `enable_metadata_routing=True`.

            See :ref:`Metadata Routing User Guide <metadata_routing>` for more
            details.

        Returns
        -------
        Xt : ndarray of shape (n_samples - n_rows, n_transformed_features)
            Transformed samples.

        yt : ndarray of length (n_samples - n_rows)
            Transformed target.
        """
        routed_params = super()._check_method_params(method="transform", props=params)

        Xt = X
        yt = y
        for _, name, transform in self._iter():
            if hasattr(transform, "transform_x_y"):
                Xt, yt = transform.transform_x_y(
                    Xt, yt, **routed_params[name].transform
                )
            else:
                Xt = transform.transform(Xt, **routed_params[name].transform)
        return Xt, yt

        Xt, yt = self._fit(X, y, routed_params)

    @available_if(pipeline._final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None, **params):
        """Transform the data, and apply `score` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score` method. Only valid if the final estimator implements `score`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        score : float
            Result of calling `score` on the final estimator.
        """
        Xt = X
        yt = y
        if not _routing_enabled():
            for _, name, transform in self._iter(with_final=False):
                if hasattr(transform, "transform_x_y"):
                    Xt, yt = transform.transform_x_y(Xt, yt)
                else:
                    Xt = transform.transform(Xt)
            score_params = {}
            if sample_weight is not None:
                score_params["sample_weight"] = sample_weight
            return self.steps[-1][1].score(Xt, yt, **score_params)

        # metadata routing is enabled.
        routed_params = process_routing(
            self, "score", sample_weight=sample_weight, **params
        )
        for _, name, transform in self._iter(with_final=False):
            if hasattr(transform, "transform_x_y"):
                Xt, yt = transform.transform_x_y(
                    Xt, yt, **routed_params[name].transform
                )
            else:
                Xt = transform.transform(Xt, **routed_params[name].transform)
        return self.steps[-1][1].score(Xt, yt, **routed_params[self.steps[-1][0]].score)


def make_pipeline(*steps, memory=None, verbose=False):
    """Construct a Pipeline from the given estimators.

    This is a shorthand for the `Pipeline` constructor; it does not
    require, and does not permit, naming the estimators. Instead, their names
    will be set to the lowercase of their types automatically.

    More details in the :ref:`User Guide <make_pipeline>`.

    Parameters
    ----------
    *steps : list of Estimator objects
        List of the scikit-learn estimators that are chained together.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the transformers
        before fitting. Therefore, the transformer instance given to the
        pipeline cannot be inspected directly. Use the attribute ``named_steps``
        or ``steps`` to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Returns
    -------
    p : Pipeline
        Returns a scikit-learn :class:`Pipeline` object.

    See Also
    --------
    Pipeline : Class for creating a pipeline of transforms with a final
        estimator.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> from feature_engine.pipeline import make_pipeline
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('gaussiannb', GaussianNB())])
    """
    return Pipeline(pipeline._name_estimators(steps), memory=memory, verbose=verbose)
