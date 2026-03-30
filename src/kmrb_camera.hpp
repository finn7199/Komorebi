#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct GLFWwindow;

namespace kmrb {

// Unity-style fly camera:
//   Hold right-click + mouse = look around (pitch/yaw)
//   Hold right-click + WASD  = fly movement
//   Scroll = adjust fly speed
class Camera {
public:
    void init(const glm::vec3& position = glm::vec3(0, 2, 5),
              float pitch = -15.0f, float yaw = -90.0f);
    void update(GLFWwindow* window, float deltaTime);

    glm::mat4 getViewMatrix() const;
    glm::vec3 getForward() const;

    // World-space position and euler rotation (pitch, yaw, roll)
    glm::vec3 position = glm::vec3(0, 2, 5);
    float pitch = -15.0f;  // Degrees, up/down
    float yaw   = -90.0f;  // Degrees, left/right
    float roll  =  0.0f;   // Degrees, tilt (not used by mouse, editable in Inspector)

    float moveSpeed = 5.0f;
    float lookSensitivity = 0.15f;

    // Set by UI — only process input when viewport is hovered
    bool viewportHovered = false;
    // True during the frame when user is actively controlling
    bool isUserControlling = false;

private:
    double lastMouseX = 0.0, lastMouseY = 0.0;
    bool rightMouseWasDown = false;
};

} // namespace kmrb
