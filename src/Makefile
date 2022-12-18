CXX=g++
INCLUDE = $(shell python3 -m pybind11 --includes)
CFLAGS = -O3 -Wall -shared -std=c++17 -fPIC
MKL_FLAGS = -I/usr/include/mkl
PYCONFIG = $(shell python3-config --includes)
SUFFIX = $(shell python3-config --extension-suffix)
MKL_LIB_DIR=/usr/lib/x86_64-linux-gnu
MKL_LIBS=${MKL_LIB_DIR}/libmkl_def.so \
			${MKL_LIB_DIR}/libmkl_avx2.so \
			${MKL_LIB_DIR}/libmkl_core.so \
			${MKL_LIB_DIR}/libmkl_intel_lp64.so \
			${MKL_LIB_DIR}/libmkl_sequential.so \
			-lpthread -lm -ldl \
			-lblas -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

.PHONY: all test clean

all: _array

_array: _array.cpp

	$(CXX) $(CFLAGS) $(INCLUDE) $< -o $@$(SUFFIX) $(PYCONFIG) $(MKL_FLAGS) ${MKL_LIBS}

test: all
	python3 -m pytest -v

clean:
	rm -rf __pycache__ .pytest_cache *.so *.o