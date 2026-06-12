#include "kmrb_log.hpp"

namespace kmrb {

std::deque<LogEntry> Log::entries;
std::chrono::steady_clock::time_point Log::startTime = std::chrono::steady_clock::now();

void Log::add(LogLevel level, const std::string& msg) {
    float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();
    entries.push_back({ t, level, msg });
    if (entries.size() > MAX_ENTRIES) entries.pop_front();
}

void Log::info(const std::string& msg)  { add(LogLevel::Info, msg); }
void Log::ok(const std::string& msg)    { add(LogLevel::Ok, msg); }
void Log::warn(const std::string& msg)  { add(LogLevel::Warn, msg); }
void Log::error(const std::string& msg) { add(LogLevel::Error, msg); }
void Log::clear()                       { entries.clear(); }
const std::deque<LogEntry>& Log::getEntries() { return entries; }

} // namespace kmrb
