cmake_minimum_required(VERSION 3.16)
project(fcitx5-dbus-bridge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Fcitx5Core REQUIRED)
find_package(Fcitx5Utils REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(DBUS REQUIRED IMPORTED_TARGET dbus-1)

include(GNUInstallDirs)

add_library(dbusbridge MODULE src/dbusbridge.cpp)
target_link_libraries(dbusbridge PRIVATE Fcitx5::Core Fcitx5::Utils PkgConfig::DBUS)

# 让输出名为 libdbusbridge.so（与 Library=libdbusbridge 对应）
set_target_properties(dbusbridge PROPERTIES
    OUTPUT_NAME "dbusbridge"
    PREFIX "lib"
)

install(TARGETS dbusbridge
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/fcitx5
)

install(FILES dbusbridge.conf
    DESTINATION ${CMAKE_INSTALL_DATADIR}/fcitx5/addon
)
