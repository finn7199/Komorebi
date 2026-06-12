#pragma once

#include <string>
#include <deque>
#include <chrono>

namespace kmrb {

// Log levels matching the KMRB color palette
enum class LogLevel { Info, Ok, Warn, Error };

struct LogEntry {
    float timestamp;    // Seconds since app start
    LogLevel level;
    std::string message;
};

// Global log — call from anywhere, displayed in the Console panel.
// Lives in its own header so low-level modules (buffers, mesh, sim)
// don't have to depend on the UI module just to print a message.
class Log {
public:
    static void info(const std::string& msg);
    static void ok(const std::string& msg);
    static void warn(const std::string& msg);
    static void error(const std::string& msg);
    static void clear();

    static const std::deque<LogEntry>& getEntries();

private:
    static void add(LogLevel level, const std::string& msg);
    static std::deque<LogEntry> entries;
    static std::chrono::steady_clock::time_point startTime;
    static constexpr size_t MAX_ENTRIES = 500;
};

} // namespace kmrb
