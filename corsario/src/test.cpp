#include "Corsario.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {

    // Paths can be overridden via command-line args for convenience
    std::string dat_file     = (argc > 1) ? argv[1] : "/home/icantu24/corsika_data/DAT099999";
    std::string particles_csv = (argc > 2) ? argv[2] : "particles_ext.csv";
    std::string shower_csv    = (argc > 3) ? argv[3] : "shower.csv";

    try {   
        Corsario corsario(dat_file);

        std::cout << "Corsario v" << corsario.version() << std::endl;
        std::cout << "Showers found: " << corsario.n_showers() << std::endl;

        corsario.to_csv_ext(particles_csv);
        std::cout << "Particles CSV written: " << particles_csv << std::endl;

        corsario.shower_csv(shower_csv);
        std::cout << "Shower CSV written:    " << shower_csv << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "[Error] " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}