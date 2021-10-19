find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(LinearLayer
    #conv-field.cpp
    fc-field.cpp
    conv-new.cpp
    #elemwise-prod-field.cpp
    utils-HE.cpp
)

target_link_libraries(LinearLayer
    PUBLIC
    ${EMP-OT_LIBRARIES}
    SEAL::seal
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)
