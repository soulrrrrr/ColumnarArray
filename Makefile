CXX=g++
INCLUDE = $(shell python3 -m pybind11 --includes)
CFLAGS = -O3 -Wall -shared -std=c++17 -fPIC
MKL_FLAGS = -I/usr/include/mkl -lblas
PYCONFIG = $(shell python3-config --includes)
SUFFIX = $(shell python3-config --extension-suffix)

.PHONY: all test clean

all: _col_array

_col_array: _col_array.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) $< -o $@$(SUFFIX) $(PYCONFIG) $(MKL_FLAGS)

test: all
	python3 -m pytest -v

clean:
	rm -rf __pycache__ .pytest_cache *.so