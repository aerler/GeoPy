'''
Created on Mar 30, 2015

Modified statistical functions from scipy.stats

@author: Andre R. Erler
'''

import numpy as np
# imports from scipy's internal stats-helper module
from scipy.stats.stats import _chk_asarray, rankdata, distributions
from scipy.special import betainc

# helper function
def _sum_of_squares(x):
    return np.sum(x**2)

# helper function
def _betai(a, b, x):
    x = np.asarray(x)
    x = np.where(x < 1.0, x, 1.0)  # if x > 1 then return 1.0
    return betainc(a, b, x)

## Pearson's linear correlation coefficient
def pearsonr(x, y, dof=None):
    """
    Calculates a Pearson correlation coefficient and the p-value for testing
    non-correlation.

    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear
    relationship. Positive correlations imply that as x increases, so does
    y. Negative correlations imply that as x increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.
    
    This is a modified version that supports an optional argument to set the
    degrees of freedom (dof) manually.

    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input
    dof : int or None, optional
          Input

    Returns
    -------
    (Pearson's correlation coefficient,
     2-tailed p-value)

    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation

    """
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(_sum_of_squares(xm) * _sum_of_squares(ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)
    df = n-2 if dof is None else dof
    if abs(r) == 1.0:
        prob = 0.0
    else:
        t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
        prob = _betai(0.5*df, 0.5, df / (df + t_squared))
    return r, prob


## Spearman's rank correlation coefficient
def spearmanr(a, b=None, axis=0, dof=None):
    """
    Calculates a Spearman rank-order correlation coefficient and the p-value
    to test for non-correlation.

    The Spearman correlation is a nonparametric measure of the monotonicity
    of the relationship between two datasets. Unlike the Pearson correlation,
    the Spearman correlation does not assume that both datasets are normally
    distributed. Like other correlation coefficients, this one varies
    between -1 and +1 with 0 implying no correlation. Correlations of -1 or
    +1 imply an exact monotonic relationship. Positive correlations imply that
    as x increases, so does y. Negative correlations imply that as x
    increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Spearman correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.

    Parameters
    ----------
    a, b : 1D or 2D array_like, b is optional
        One or two 1-D or 2-D arrays containing multiple variables and
        observations. Each column of `a` and `b` represents a variable, and
        each row entry a single observation of those variables. See also
        `axis`. Both arrays need to have the same length in the `axis`
        dimension.
    axis : int or None, optional
        If axis=0 (default), then each column represents a variable, with
        observations in the rows. If axis=0, the relationship is transposed:
        each row represents a variable, while the columns contain observations.
        If axis=None, then both arrays will be raveled.
    dof : int or None, optional
        If dof=None (default), the degrees of freedom will be inferred from 
        the array length.

    Returns
    -------
    rho : float or ndarray (2-D square)
        Spearman correlation matrix or correlation coefficient (if only 2
        variables are given as parameters. Correlation matrix is square with
        length equal to total number of variables (columns or rows) in a and b
        combined.
    p-value : float
        The two-sided p-value for a hypothesis test whose null hypothesis is
        that two sets of data are uncorrelated, has same dimension as rho.

    Notes
    -----
    Changes in scipy 0.8.0: rewrite to add tie-handling, and axis.

    References
    ----------
    [CRCProbStat2000]_ Section  14.7

    .. [CRCProbStat2000] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.

    Examples
    --------
    >>> spearmanr([1,2,3,4,5],[5,6,7,8,7])
    (0.82078268166812329, 0.088587005313543798)
    >>> np.random.seed(1234321)
    >>> x2n=np.random.randn(100,2)
    >>> y2n=np.random.randn(100,2)
    >>> spearmanr(x2n)
    (0.059969996999699973, 0.55338590803773591)
    >>> spearmanr(x2n[:,0], x2n[:,1])
    (0.059969996999699973, 0.55338590803773591)
    >>> rho, pval = spearmanr(x2n,y2n)
    >>> rho
    array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],
           [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],
           [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],
           [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])
    >>> pval
    array([[ 0.        ,  0.55338591,  0.06435364,  0.53617935],
           [ 0.55338591,  0.        ,  0.27592895,  0.80234077],
           [ 0.06435364,  0.27592895,  0.        ,  0.73039992],
           [ 0.53617935,  0.80234077,  0.73039992,  0.        ]])
    >>> rho, pval = spearmanr(x2n.T, y2n.T, axis=1)
    >>> rho
    array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],
           [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],
           [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],
           [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])
    >>> spearmanr(x2n, y2n, axis=None)
    (0.10816770419260482, 0.1273562188027364)
    >>> spearmanr(x2n.ravel(), y2n.ravel())
    (0.10816770419260482, 0.1273562188027364)

    >>> xint = np.random.randint(10,size=(100,2))
    >>> spearmanr(xint)
    (0.052760927029710199, 0.60213045837062351)

    """
    a, axisout = _chk_asarray(a, axis)
    ar = np.apply_along_axis(rankdata,axisout,a)

    br = None
    if b is not None:
        b, axisout = _chk_asarray(b, axis)
        br = np.apply_along_axis(rankdata,axisout,b)
    n = a.shape[axisout] if dof is None else dof
    rs = np.corrcoef(ar,br,rowvar=axisout)

    olderr = np.seterr(divide='ignore')  # rs can have elements equal to 1
    try:
        t = rs * np.sqrt((n-2) / ((rs+1.0)*(1.0-rs)))
    finally:
        np.seterr(**olderr)
    prob = distributions.t.sf(np.abs(t),n-2)*2

    if rs.shape == (2,2):
        return rs[1,0], prob[1,0]
    else:
        return rs, prob


if __name__ == '__main__':
    pass
