set(LIBS ${LIBS} fmt::fmt)

add_library(spatt INTERFACE)
target_link_libraries(spatt INTERFACE ${LIBS})