# Modified from scikit-learn's pipeline:
# https://github.com/scikit-learn/scikit-learn/blob/6eff1757e/sklearn/pipeline.py#L59

from sklearn.pipeline import Pipeline as pipe
from sklearn.pipeline import _final_estimator_has, _fit_transform_one

from sklearn.base import _fit_context, clone
from sklearn.utils import Bunch, _print_elapsed_time
from sklearn.utils.metaestimators import _BaseComposition, available_if
from sklearn.utils.validation import check_memory
from sklearn.utils.metadata_routing import (
    _routing_enabled,
    process_routing,
)
from sklearn.utils._metadata_requests import METHODS


class Pipeline(pipe):
    """
    A sequence of data transformers with an optional final predictor.

    `Pipeline` allows you to sequentially apply a list of transformers to
    preprocess the data and, if desired, conclude the sequence with a final
    :term:`predictor` for predictive modeling.

    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement `fit` and `transform` methods.
    The final :term:`estimator` only needs to implement `fit`.
    The transformers in the pipeline can be cached using ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters. For this, it
    enables setting parameters of the various steps using their names and the
    parameter name separated by a `'__'`, as in the example below. A step's
    estimator may be replaced entirely by setting the parameter with its name
    to another estimator, or a transformer removed by setting it to
    `'passthrough'` or `None`.

    For an example use case of `Pipeline` combined with
    :class:`~sklearn.model_selection.GridSearchCV`, refer to
    :ref:`sphx_glr_auto_examples_compose_plot_compare_reduction.py`. The
    example :ref:`sphx_glr_auto_examples_compose_plot_digits_pipe.py` shows how
    to grid search on a pipeline using `'__'` as a separator in the parameter names.

    Read more in the :ref:`User Guide <pipeline>`.

    .. versionadded:: 0.5

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

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    make_pipeline : Convenience function for simplified pipeline construction.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import Pipeline
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

    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps=steps, memory=memory, verbose=verbose)

    def _fit(self, X, y, routed_params=None):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

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
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                params=routed_params[name],
            )
            y = y.loc[X.index]
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

        Fit all the transformers one after the other and sequentially transform the
        data. Finally, fit the transformed data using the final estimator.

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
        self : object
            Pipeline with fitted steps.
        """
        routed_params = super()._check_method_params(method="fit", props=params)
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
        """Fit the model and transform with the final estimator.

        Fit all the transformers one after the other and sequentially transform
        the data. Only valid if the final estimator either implements
        `fit_transform` or `fit` and `transform`.

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
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        routed_params = super()._check_method_params(method="fit_transform", props=params)
        Xt, yt = self._fit(X, y, routed_params)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt, yt
            last_step_params = routed_params[self.steps[-1][0]]
            if hasattr(last_step, "fit_transform"):
                Xt = last_step.fit_transform(
                    Xt, yt, **last_step_params["fit_transform"]
                )
                yt = yt.loc[Xt.index]
                return Xt, yt
            else:
                Xt = last_step.fit(Xt, yt, **last_step_params["fit"]).transform(
                    Xt, **last_step_params["transform"]
                )
                yt = yt.loc[Xt.index]
                return Xt, yt

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
