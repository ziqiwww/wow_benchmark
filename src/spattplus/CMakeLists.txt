set(LIBS ${LIBS} fmt::fmt)

add_library(spattplus INTERFACE)
target_link_libraries(spattplus INTERFACE ${LIBS})