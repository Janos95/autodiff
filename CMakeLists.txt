cmake_minimum_required(VERSION 3.15)
project(autodiff)

set(CMAKE_CXX_STANDARD 17)
list(INSERT CMAKE_PREFIX_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/conan_modules)

find_package(TBB CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(Ipopt REQUIRED IMPORTED_TARGET ipopt)

add_library(autodiff
        autodiff.cpp
        autodiff.hpp)

target_link_libraries(autodiff
        PRIVATE
            TBB::TBB
            PkgConfig::Ipopt
        PUBLIC
            Eigen3::Eigen3
        )

add_executable(mle mle.cpp)
target_link_libraries(mle PRIVATE autodiff)