import pandas as pd
import scipy.stats as stats
from scipy.stats import spearmanr
import numpy as np
import mlflow.sklearn


## Test between continuous variable and binary variable.
#   - Independance
#   - Linearity. The relationship between the continuous variable and the binary variable should be linear
#   - Normality. The continuous variable should follow a normal distribution within each level of the binary variable
#   - Homoscendasticity
def point_biserial_test(df, continuous_variable, binary_variable, i):
    correlation_coefficient, p_value = stats.pointbiserialr(df[continuous_variable], df[binary_variable])
    print("Point-Biserial Correlation coefficient:", correlation_coefficient)
    print("p-value:", p_value)

    mlflow.log_param(f"Continuous variable {i}", continuous_variable)
    mlflow.log_param("binary variable", binary_variable)

    mlflow.log_metric(f"Point-Biserial Correlation Coefficient  for {continuous_variable}", correlation_coefficient)
    #mlflow.log_metric(f"Point-Biserial p-value for {continuous_variable}", p_value)
    return correlation_coefficient, p_value


def ChiSquare(df, nominal_variable, binary_variable, i=1):
    contingency_table = pd.crosstab(df[binary_variable], df[nominal_variable])
    chi, p_value, _, _ = stats.chi2_contingency(contingency_table)
    n = np.sum(contingency_table.values)
    cramers_v = np.sqrt(chi / (n * (min(contingency_table.shape) - 1)))

    print("Chi-Square test statistic:", chi)
    print("cramer correlation:", cramers_v)
    print("p-value:", p_value)

    mlflow.log_param(f"Nominal variable {i}", nominal_variable)
    mlflow.log_param("binary variable", binary_variable)

    # Log Spearman correlation coefficient and p-value
    #mlflow.log_metric(f"Chi-Square test statistic for {nominal_variable}", chi)
    mlflow.log_metric(f"Crammer correlation for {nominal_variable}", cramers_v)
    #mlflow.log_metric(f"Chi-Square p-value for {nominal_variable}", p_value)


    return cramers_v, p_value


def SpearmanTest(df, continuous_variable, binary_variable, i=1):
    # Perform Spearman correlation
    correlation_coefficient, p_value = spearmanr(df[continuous_variable], df[binary_variable])

    mlflow.log_param(f"Continuous variable {i}", continuous_variable)
    mlflow.log_param("binary variable", binary_variable)

    # Log Spearman correlation coefficient and p-value
    mlflow.log_metric(f"Spearman Correlation Coefficient for {continuous_variable}", correlation_coefficient)
    #mlflow.log_metric(f"Spearman p-value for {continuous_variable}", p_value)

    print("Spearman Correlation Coefficient for {continuous_variable}:", correlation_coefficient)
    print("Spearman p-value for {continuous_variable}:", p_value)

    return correlation_coefficient, p_value
