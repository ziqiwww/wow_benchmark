#pragma once

#include <memory>
#include <filesystem>
#include <iostream>
#include <ctime>
#include "config.hh"
#include "fmt/format.h"

inline auto GetHourMinuteSecond() -> std::string
{
  time_t now = time(nullptr);
  struct tm tstruct
  {};
  char buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%X", &tstruct);
  return buf;
}

#define DISABLE_COPY_AND_ASSIGN(classname)          \
  classname(const classname &)            = delete; \
  classname &operator=(const classname &) = delete;

#define DISABLE_MOVE_AND_ASSIGN(classname)     \
  classname(classname &&)            = delete; \
  classname &operator=(classname &&) = delete;

#define DISABLE_COPY_MOVE_AND_ASSIGN(classname) \
  DISABLE_COPY_AND_ASSIGN(classname)            \
  DISABLE_MOVE_AND_ASSIGN(classname)

#define DEFINE_SHARED_PTR(type) using type##Sptr = std::shared_ptr<type>
#define DEFINE_UNIQUE_PTR(type) using type##Uptr = std::unique_ptr<type>

#define LOG(msg) \
  std::cout << fmt::format("\033[32m[{}]LOG <{}::{}>: {}\033[0m\n", GetHourMinuteSecond(), __func__, __LINE__, msg)

#ifdef DEBUG
#define LOG_DBG(msg) \
  std::cout << fmt::format("\033[34m[{}]DBG <{}::{}>: {}\033[0m\n", GetHourMinuteSecond(), __func__, __LINE__, msg)
#else
#define LOG_DBG(msg)
#endif

#ifdef DEBUG
#define ASSERT_MSG(condition, msg)                                     \
  if (!(condition)) {                                                  \
    LOG_DBG(fmt::format("Assertion failed: {}, {}", #condition, msg)); \
    std::abort();                                                      \
  }
#else
#define ASSERT_MSG(condition, msg)
#endif

#define DECLARE_ENUM(EnumName, ...) \
  enum EnumName                     \
  {                                 \
    ENUM_ENTITIES                   \
  };

// Helper macro to generate case statements
#define ENUM_TO_STRING_BODY(EnumName, ...)              \
  inline const char *EnumName##ToString(EnumName value) \
  {                                                     \
    switch (value) {                                    \
      ENUM_ENTITIES                                     \
      default: return "UNKNOWN";                        \
    }                                                   \
  }

#define STRING_TO_ENUM_BODY(EnumName, ...)                                \
  inline EnumName StringTo##EnumName(const std::string &value)            \
  {                                                                       \
    std::unordered_map<std::string, EnumName> enum_map = {ENUM_ENTITIES}; \
    if (enum_map.find(value) != enum_map.end()) {                         \
      return enum_map[value];                                             \
    }                                                                     \
    LOG(fmt::format("Unknown {} value: {}", #EnumName, value));           \
    std::abort();                                                         \
  }

#define ENUMENTRY(x) x,
#define ENUM2STRING(x) \
  case x: return #x;
#define STRING2ENUM(x) {#x, x},

#define Q(x) #x

#define PUSH_HEAP(vec, ...)      \
  vec.emplace_back(__VA_ARGS__); \
  std::push_heap(vec.begin(), vec.end())

#define POP_HEAP(vec)                    \
  std::pop_heap(vec.begin(), vec.end()); \
  vec.pop_back();

#define TOP_HEAP(vec) vec.front()

#define ABS(x) ((x) > 0 ? (x) : -(x))
