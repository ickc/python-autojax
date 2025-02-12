import functools

import pytest

def f1(x):
    return x*x

def f2(x):
    return x**2

@pytest.fixture
def setup_tilde_case():
    return ({"n":1},2,3)


def test_tilde_f1(setup_tilde_case):
    mydict, int1, int2 = setup_tilde_case

    assert f1(int1)==4

def test_tilde_f2(setup_tilde_case):
    mydict, int1, int2 = setup_tilde_case

    assert f2(int1)==4

@pytest.parameterize(
        'f,fout',
        [
            pytest.param(f1, 4, marks=pytest.mark.xfail, id="Test for f1"),
            pytest.param(f2, 4, marks=pytest.mark.unittest, id="Unittest for f2"),
            pytest.param(f2, 4, marks=pytest.mark.benchmark(group="tilde_benchmark"), id="f2 benchmark"),
        ]
)
def test_tilde_f(setup_tilde_case, f, fout):
    mydict, int1, int2 = setup_tilde_case

    assert f(int1)==fout

@pytest.parameterize('n1', [1,2,3])
@pytest.parameterize('n2', [10,20,30])
def test_x(n1, n2):
    out = n1 + n2
    assert out//10 == 10*n2
    assert out%10 == n1

@pytest.parameterize("n1", [1,2,3])
@pytest.parameterize(
        'n2',
        [
            pytest.param('n2', 10, marks=pytest.mark.benchmark("tilde_size_10")),
            pytest.param('n2', 20, marks=pytest.mark.benchmark("tilde_size_20")),
            pytest.param('n2', 30, marks=pytest.mark.benchmark("tilde_size_30")),
        ]
        )
@pytest.mark.benchmark("tilde")
def test_benchmark_x(n1, n2):
    pass

@pytest.parameterize(
    "n1,n2",
    list(map(
        lambda x: pytest.param(
            ("n1,n2",
            pytest.param(x, marks=pytest.mark.benchmark(f'tilde{x[1]}'))),
            functools.product([1,2,3], [10, 20, 30])
         )
    )))
