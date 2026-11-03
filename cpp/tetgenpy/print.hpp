#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tetgenpy {

template<typename T>
inline void SinglePrint(std::ostream& os, T&& t) {
  os << std::forward<T>(t) << " ";
}

inline void PrintToStream(std::ostream&) {}

template<typename T, typename... Ts>
inline void PrintToStream(std::ostream& os, T&& first, Ts&&... rest) {
  SinglePrint(os, std::forward<T>(first));
  PrintToStream(os, std::forward<Ts>(rest)...);
}

template<typename... Args>
inline void PrintInfo(Args&&... args) {
  std::cout << "TETGENPY INFO - ";
  PrintToStream(std::cout, std::forward<Args>(args)...);
  std::cout << "\n";
}

/// debug printer - first argument is bool, so <on, off> is switchable.
template<typename... Args>
inline void PrintDebug(bool on, Args&&... args) {
  if (on) {
    std::cout << "TETGENPY DEBUG - ";
    PrintToStream(std::cout, std::forward<Args>(args)...);
    std::cout << "\n";
  }
}

template<typename... Args>
inline void PrintWarning(Args&&... args) {
  std::cout << "TETGENPY WARNING - ";
  PrintToStream(std::cout, std::forward<Args>(args)...);
  std::cout << "\n";
}

template<typename... Args>
inline void PrintAndThrowError(Args&&... args) {
  std::ostringstream oss;
  PrintToStream(oss, std::forward<Args>(args)...);
  std::cout << "TETGENPY ERROR - " << oss.str() << "\n";
  throw std::runtime_error("TETGENPY ERROR: " + oss.str());
}

} // namespace tetgenpy
