from os import read
import numpy as np

def read_data(fname: str, tipo: type) -> np.ndarray:
    """
    Read data from a file and return a numpy array with the data
    :param fname: str
        name of the file with the data
    :param tipo: type
        type of the data
    :return: np.ndarray
        numpy array with the data
    :raise FileNotFoundError: if file does not exist
    :raise TypeError:  if data type is not the same as the one specified
    :raise ValueError: if data type is not the same as the one specified
    Examples
    --------
    >>> read_data('zonas.txt', int)
    array([[1, 1, 1, 1, 3, 3],
           [1, 1, 1, 1, 3, 1],
           [2, 2, 3, 3, 3, 4],
           [2, 2, 3, 3, 3, 4],
           [2, 2, 3, 3, 2, 2],
           [3, 3, 3, 3, 3, 2]])
    >>> read_data('valores.txt', float)
    array([[5., 3., 4., 4., 4., 2.],
           [2., 1., 4., 2., 6., 3.],
           [8., 4., 3., 5., 3., 1.],
           [4., 2., 4., 3., 2., 2.],
           [6., 3., 3., 7., 4., 2.],
           [5., 5., 2., 3., 1., 3.]])
   """
    array_file = np.loadtxt(fname, dtype = tipo)
    return array_file

def set_of_areas(zonas: np.ndarray)-> set[int]:
    """
    Returns the unique set of areas in the array
    :param zonas: np.ndarray
        array with the areas
    :return: set[int]  
        set of unique areas
    :raise TypeError: if elements type is not int

    Examples:
    --------
    >>> set_of_areas(np.arange(10).reshape(5, 2))
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  
    >>> set_of_areas(np.zeros(3, dtype=np.float_))
    Traceback (most recent call last):
        ...
    TypeError: The elements type must be int, not float64
    """
    if zonas.dtype != np.int_:
        raise TypeError('The elements type must be int, not {}'.format(zonas.dtype))
    return set(np.unique(zonas.flatten()))


def mean_areas(zones: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Returns the mean of the values of each area
    :param zones: np.ndarray
        array with the areas
    :param values: np.ndarray
        array with the values
    :return: np.ndarray
        array with the mean of the values of each area
    :raise ValueError: if shape of zones and values is not the same
    Examples:
    --------
    >>> mean_areas(np.array( [[1,2],[2,1]] ), np.array( [[11,330],[20, 118]] ) )
    array([[ 64.5, 175. ],
           [175. ,  64.5]])
    >>> mean_areas(np.array([2, 3, 4, 2, 3, 4], dtype=np.int_).reshape(3, 2), np.arange(6).reshape(2, 3))
    Traceback (most recent call last):
        ...
    ValueError: Shape of zones and values must be the same. Zones: (3, 2) != values: (2, 3)
    """
    if zones.shape != values.shape:
        raise ValueError('Shape of zones and values must be the same. Zones: {} != values: {}'.format(zones.shape, values.shape))
    else:
        distinct_zones = set_of_areas(zones)
        result = np.zeros(zones.shape)
        for zone in distinct_zones:
            idx = zones == zone
            result[idx] = round(values[idx].mean(), 1)
        return result


# ------------ test  --------#
import doctest

def test_doc()-> None:
    """
    The following instructions are to execute the tests of same functions
    If any test is fail, we will receive the notice when executing
    :return: None
    """
    doctest.run_docstring_examples(read_data, globals(),    verbose= False)  # vemos los resultados de los test que fallan
    doctest.run_docstring_examples(set_of_areas, globals(), verbose=False)  # vemos los resultados de los test que fallan
    doctest.run_docstring_examples(mean_areas, globals(),   verbose=False)  # vemos los resultados de los test que fallan


if __name__ == "__main__":
    test_doc()   # executing tests
