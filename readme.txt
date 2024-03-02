Project that implements a new methods from 
https://arxiv.org/abs/2402.12528
and test agains a convential MC

1. Provide correct links in CMakeLists.txt to Nlohmann (JSON), Eigen and AADC.
2. Parallelalized/NonParallelized version may be switched using pragma #define PARALLEL_IMPLEMENTATION in generic.cpp (parallel version is on by default)

To rebuild:

cd build
make

In the final version no rebuild will be required, all parameters will be able to be set from .json file

./Generic/par ../Generic/Examples/Heston.json 
