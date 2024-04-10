#include "GravSim.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>

int main() {
    std::string filename = "GravSim_results.csv";
    std::ofstream file(filename);

    // Seed for random number generation
    std::random_device rd;
    std::mt19937 gen(rd());

    // Generate 1000 SpaceObjects with random x, y, and mass values
    std::vector<SpaceObject*> space_objects;
    for (int i = 0; i < 1000; ++i) {
        std::uniform_real_distribution<double> x_dist(0.0, 10000000.0);
        std::uniform_real_distribution<double> y_dist(0.0, 10000000.0);
        std::uniform_real_distribution<double> mass_dist(1000.0, 1000000000.0);

        space_objects.push_back(new SpaceObject(x_dist(gen), y_dist(gen), mass_dist(gen)));
    }

    for (double t = 0; t < 100; ++t) {
        auto start_time = std::chrono::high_resolution_clock::now();

        calculateAcceleration(space_objects, t);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Time taken by calculateAcceleration: " << duration.count() << " milliseconds" << std::endl;

        // Save the positions to the CSV file (replace this with your actual saving logic)
        file << "Time: " << t << "\n";
        for (SpaceObject* const& obj : space_objects) {
            file << "Object: x = " << obj->get_x() << ", y = " << obj->get_y() << "\n";
        }
        file << "\n";
    }

    // Don't forget to delete the dynamically allocated objects
    for (SpaceObject* obj : space_objects) {
        delete obj;
    }

    file.close();  // Close the file

    return 0;
}