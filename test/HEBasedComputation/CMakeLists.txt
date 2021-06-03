add_library(HE-CodeBase
utils.cpp
fclayer.cpp
convlayer.cpp
)

target_link_libraries( HE-CodeBase
${EMP-OT_LIBRARIES} SEAL::seal
)
