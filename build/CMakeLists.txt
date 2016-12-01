cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

set(source
  ../Config.h
  ../Config.cpp
  ../App.h
  ../App.cpp
  ../liblinear/blas/blas.h
  ../liblinear/blas/blasp.h
  ../liblinear/blas/daxpy.c
  ../liblinear/blas/ddot.c
  ../liblinear/blas/dnrm2.c
  ../liblinear/blas/dscal.c
  ../liblinear/tron.cpp
  ../liblinear/tron.h
  ../liblinear/linear.h
  ../liblinear/linear.cpp
  ../LBF.h
  ../LBF.cpp
  ../TrainModel.cpp
  ../TestModel.cpp
  ../Tree.h
  ../Tree.cpp
  ../RandomForest.cpp
  ../RandomForest.h
  ../LBFRegressor.h
  ../LBFRegressor.cpp)


#if you don't want to use openMP, you should comment this line
#FIND_PACKAGE( OpenMP REQUIRED)


if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

cmake_minimum_required(VERSION 2.8.3)
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
if(APPLE AND ${PYTHON_VERSION_MAJOR} EQUAL 3)
    FIND_PACKAGE(Boost COMPONENTS python3)
else()
    FIND_PACKAGE(Boost COMPONENTS python)
endif()

ENABLE_TESTING()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBS}) # Deprecated but so convenient!
PYTHON_ADD_MODULE(FaceAlign ${source})

#add_library(LBG ${source} ../main.cpp)
#add_executable(LBF.out ${source} ../main.cpp)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
#target_link_libraries(LBF.out ${OpenCV_LIBS})

set(CMAKE_BUILD_TYPE Debug)
