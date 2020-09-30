import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.imputation import EndTailImputer


def test_EndTailImputer(dataframe_na):

    # test case 1: automatically find variables + gaussian limits + right tail
    imputer = EndTailImputer(distribution='gaussian', tail='right', fold=3, variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(58.94908118478389)
    ref_df['Marks'] = ref_df['Marks'].fillna(1.3244261503263175)

    # init params
    assert imputer.distribution == 'gaussian'
    assert imputer.tail == 'right'
    assert imputer.fold == 3
    assert imputer.variables == ['Age', 'Marks']
    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Age': 58.94908118478389, 'Marks': 1.3244261503263175}
    # transform params: indicated vars ==> no NA, not indicated vars with NA
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() == 0
    assert X_transformed[['City', 'Name']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 2: selected variables + IQR rule + right tail
    imputer = EndTailImputer(distribution='skewed', tail='right', fold=1.5, variables=['Age', 'Marks'])
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(65.5)
    ref_df['Marks'] = ref_df['Marks'].fillna(1.0625)
    # fit  and transform params
    assert imputer.imputer_dict_ == {'Age': 65.5, 'Marks': 1.0625}
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() == 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 3: selected variables + maximum value
    imputer = EndTailImputer(distribution='max', tail='right', fold=2, variables=['Age', 'Marks'])
    imputer.fit(dataframe_na)
    assert imputer.imputer_dict_ == {'Age': 82.0, 'Marks': 1.8}

    # test case 4: automatically select variables + gaussian limits + left tail
    imputer = EndTailImputer(distribution='gaussian', tail='left', fold=3)
    imputer.fit(dataframe_na)
    assert imputer.imputer_dict_ == {'Age': -1.520509756212462, 'Marks': 0.04224051634034898}

    # test case 5: IQR + left tail
    imputer = EndTailImputer(distribution='skewed', tail='left', fold=1.5, variables=['Age', 'Marks'])
    imputer.fit(dataframe_na)
    assert imputer.imputer_dict_ == {'Age': -6.5, 'Marks': 0.36249999999999993}

    with pytest.raises(ValueError):
        EndTailImputer(distribution='arbitrary')

    with pytest.raises(ValueError):
        EndTailImputer(tail='arbitrary')

    with pytest.raises(ValueError):
        EndTailImputer(fold=-1)

    with pytest.raises(NotFittedError):
        imputer = EndTailImputer()
        imputer.transform(dataframe_na)