from Imports import *

path = '../outputs/importances/'

def feature_importance(model, X_train, y_train, model_name, path=path):
    """
    Description:
        First try to compute feature importance using a Random Forest model.
    ----------------------------------------------------------------------
    Parameters:
        model (RandomForestClassifier): The Random Forest model to use for feature importance.
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
    -----------------------------------------------------------------------
    Returns:
        DataFrame: Feature importances sorted in descending order.
    """

    # Fit the model
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        f'Feature_{model_name}': X_train.columns,
        f'Importance_{model_name}': importances
    })

    # saving as json
    feature_importance_df.to_pickle(path + f'Feat_Import_{model_name}.pickle')

    return feature_importance_df



def test_function():
    print('This is a test function')
