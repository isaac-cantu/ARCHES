#pragma once

#include <array>
#include <vector>
#include <string>
#include <cmath>

/* CSV básico
    particle_id
    px
    py
    pz
    x
    y
    time
    weight

    particle description encoded as:
    part. id×1000 + hadr. generation101 × 10 + no. of obs. level
*/


std::array<std::string,196>particles_name = []{
    std::array<std::string,196>arr{};
    arr[0] = "other";
    arr[1] = "γ";
    arr[2] = "e+"; // 9.11e-28 gramos
    arr[3] = "e−"; // 9.11e-28 gramos
    arr[4] = "none"; 
    arr[5] = "μ+"; //105.66 MeV/c²
    arr[6] = "μ−"; //105.66 MeV/c²
    arr[7] = "π◦"; //
    arr[8] = "π+"; //139.6 MeV/c² 
    arr[9] = "π-";

    arr[10] = "K◦";
    arr[11] = "K+";
    arr[12] = "K-";
    arr[13] = "n";
    arr[14] = "p";
    arr[15] = "p";
    arr[16] = "K◦";
    arr[17] = "η";
    arr[18] = "Λ";

   
    return arr;
}();

std::array<float,196>particles_mass = []{
    std::array<float,196>arr{};
    arr[0] = 0;
    arr[1] = 0;
    arr[2] = 0.000511; // MeV/c²
    arr[3] = 0.000511; // MeV/c²
    arr[4] = 0; 
    arr[5] = .105658; // MeV/c²
    arr[6] = .105658; //MeV/c²
    arr[7] = .134977; //
    arr[8] = .139570; //MeV/c² 
    arr[9] = .139570;
    arr[10] = .49761;
    arr[11] = .493677;
    arr[12] = .493677;
    arr[13] = .939565;
    arr[14] = .938272;
    arr[15] = .938272;
    arr[16] = .49761;
    arr[17] = .547862;
    arr[18] = 1.11568;


   
    return arr;
}();

class Particle{
    private:
        int particle_id;
        float px, py, pz; //GeV/c -> MeV
        float x, y; //cm
        float time; //nsec
        std::array<float, 3> momentum;

        std::string particle_type;
        int hadr_gen;
        int no_obs;
        float energy;
        float mass;
        int no_particle;


    public:
        Particle(int, float, float, float, float, float, float);

        float get_energy();

        std::string get_particle_type();

        int get_particle_no();

        int get_no_obs();

        int get_hadron_gen();

        float get_mass();

        std::array<float,3> get_momentum();

        float get_px();

        float get_py();

        float get_pz();

        int get_id();

        float get_x();

        float get_y();

        float get_time();

        std::string particle_data_csv();

        std::string particle_data_csv_ext();

};

Particle::Particle(int _particle_id, float _px, float _py, float _pz, float _x, float _y, float _time) : 
    particle_id(_particle_id), px(_px), py(_py), pz(_pz), x(_x), y(_y), time(_time) {
        momentum = {px, py, pz};
        no_particle = particle_id / 1000;
        hadr_gen = (particle_id % 1000) / 10;
        no_obs = particle_id % 10;

        if (no_particle >= 0 && no_particle < particles_name.size()){
            particle_type = particles_name[no_particle];
            mass = particles_mass[no_particle];
        }else{
            particle_type = particles_name[0];
            mass = particles_mass[0];
        }
        
        energy = sqrt(pow(px,2) + pow(py,2) + pow(pz,2) + pow(mass,2));
}


float Particle::get_energy(){
    return energy;
}

std::string Particle::get_particle_type(){
    return particle_type;
}

int Particle::get_particle_no(){
    return no_particle;
}

int Particle::get_no_obs(){
    return no_obs;
}

int Particle::get_hadron_gen(){
    return hadr_gen;
}

float Particle::get_mass(){
    return mass;
}

std::array<float,3> Particle::get_momentum(){
    return momentum;
}

float Particle::get_px(){
    return px;
}

float Particle::get_py(){
    return py;
}

float Particle::get_pz(){
    return pz;
}

int Particle::get_id(){
    return particle_id;
}

float Particle::get_x(){
    return x;
}

float Particle::get_y(){
    return y;
}

float Particle::get_time(){
    return time;
}


std::string Particle::particle_data_csv(){
    std::string coma = ",";
    std::string data = std::to_string(particle_id) + coma 
                        + std::to_string(px) + coma 
                        + std::to_string(py) + coma 
                        + std::to_string(pz) + coma 
                        + std::to_string(x) + coma
                        + std::to_string(y) + coma
                        + std::to_string(time);
    return data;
    
}


std::string Particle::particle_data_csv_ext(){
    std::string coma = ",";
    std::string data = particle_type + coma 
                        + std::to_string(no_particle) + coma 
                        + std::to_string(hadr_gen) + coma 
                        + std::to_string(no_obs) + coma 
                        + std::to_string(mass) + coma 
                        + std::to_string(energy) + coma 
                        + std::to_string(particle_id) + coma 
                        + std::to_string(px) + coma 
                        + std::to_string(py) + coma 
                        + std::to_string(pz) + coma 
                        + std::to_string(x) + coma
                        + std::to_string(y) + coma
                        + std::to_string(time);
    return data;
}