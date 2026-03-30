#include "kmrb_camera.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>

namespace kmrb {

void Camera::init(const glm::vec3& pos, float p, float y) {
    position = pos;
    pitch = p;
    yaw = y;
}

void Camera::update(GLFWwindow* window, float deltaTime) {
    isUserControlling = false;

    if (!viewportHovered) {
        rightMouseWasDown = false;
        return;
    }

    double mx, my;
    glfwGetCursorPos(window, &mx, &my);

    bool rightDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    // Right click just pressed — capture initial mouse position
    if (rightDown && !rightMouseWasDown) {
        lastMouseX = mx;
        lastMouseY = my;
    }
    rightMouseWasDown = rightDown;

    if (!rightDown) return;

    isUserControlling = true;

    // ── Look around (mouse delta → pitch/yaw) ──
    double dx = mx - lastMouseX;
    double dy = my - lastMouseY;
    lastMouseX = mx;
    lastMouseY = my;

    yaw   += static_cast<float>(dx) * lookSensitivity;
    pitch -= static_cast<float>(dy) * lookSensitivity;
    pitch = std::clamp(pitch, -89.0f, 89.0f);

    // ── WASD movement in camera-local space ──
    glm::vec3 forward = getForward();
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
    glm::vec3 up(0, 1, 0);

    float speed = moveSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 3.0f;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) position += forward * speed;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) position -= forward * speed;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) position -= right * speed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) position += right * speed;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) position += up * speed;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) position -= up * speed;
}

glm::vec3 Camera::getForward() const {
    float yawRad = glm::radians(yaw);
    float pitchRad = glm::radians(pitch);
    return glm::normalize(glm::vec3(
        cos(pitchRad) * cos(yawRad),
        sin(pitchRad),
        cos(pitchRad) * sin(yawRad)
    ));
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, position + getForward(), glm::vec3(0, 1, 0));
}

} // namespace kmrb
