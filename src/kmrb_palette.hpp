#pragma once

#include <cstdint>
#include <glm/glm.hpp>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// KMRB editor palette — warm forest theme with golden accents.
// The single place colors are defined: the UI converts with hex(), the
// renderer (gizmos, clear colors) with toVec4().
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace kmrb::palette {

// Backgrounds, darkest → lightest
inline constexpr uint32_t Base       = 0x0E0D0B;  // window bg, scrollbar track
inline constexpr uint32_t Panel      = 0x1A1714;  // child/popup/input bg
inline constexpr uint32_t Raised     = 0x252017;  // buttons, headers, tabs
inline constexpr uint32_t Hover      = 0x30291E;  // hovered variants
inline constexpr uint32_t Border     = 0x3D352A;  // borders, separators, active variants
inline constexpr uint32_t ViewportBg = 0x0A0A0F;  // 3D viewport clear color (slightly blue)

// Golden accents
inline constexpr uint32_t Gold          = 0xC8A44E;  // primary accent, selection, compute files
inline constexpr uint32_t GoldBright    = 0xE2C36B;  // active slider grab
inline constexpr uint32_t GoldDim       = 0x7A5F28;  // hovered separators/grips
inline constexpr uint32_t GoldFaint     = 0x4A3818;  // resting resize grip
inline constexpr uint32_t GoldSelection = 0x2E2210;  // text selection bg

// Text
inline constexpr uint32_t Text      = 0xE8DCC8;  // primary
inline constexpr uint32_t TextMuted = 0x8B7D6B;  // labels
inline constexpr uint32_t TextDim   = 0x5C5347;  // hints, disabled

// Status & file-type colors
inline constexpr uint32_t Green     = 0x7BA56E;  // ok / running / ready
inline constexpr uint32_t Red       = 0xD46B5A;  // errors, destructive actions
inline constexpr uint32_t RedDark   = 0xA84A3E;  // stop-recording button
inline constexpr uint32_t RedDarker = 0x8A3C32;  // stop-recording pressed
inline constexpr uint32_t Blue      = 0x5A9BD4;  // info, render shader files
inline constexpr uint32_t Cyan      = 0x5AAFCC;  // HDR files
inline constexpr uint32_t Tan       = 0xB8A47C;  // 3D model files

inline constexpr uint32_t Black = 0x000000;

// #RRGGBB → vec4 for renderer-side colors (gizmos, clear values)
inline glm::vec4 toVec4(uint32_t rgb, float a = 1.0f) {
    return glm::vec4(((rgb >> 16) & 0xFF) / 255.0f,
                     ((rgb >> 8)  & 0xFF) / 255.0f,
                     ((rgb)       & 0xFF) / 255.0f,
                     a);
}

} // namespace kmrb::palette
