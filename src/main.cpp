#include <iostream>
#include "kmrb_core.hpp"

int main() {
    kmrb::Core engine;
    try {
        engine.init();
        engine.run();
        engine.cleanup();
    }
    catch (const std::exception& e) {
        std::cerr << "KMRB Fatal: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
