add_subdirectory(spatt)
add_subdirectory(spattplus)  # Temporarily disabled due to ygg library compilation issues
add_subdirectory(apps)

add_executable(buildspatt buildindex.cc)
target_link_libraries(buildspatt spatt)

add_executable(searchspatt searchindex.cc)
target_link_libraries(searchspatt spatt)

add_executable(buildspattplus buildspattplus.cc)
target_link_libraries(buildspattplus spattplus)

add_executable(searchspattplus searchspattplus.cc)
target_link_libraries(searchspattplus spattplus)
