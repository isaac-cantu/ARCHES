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
    arr[2] = 0.000511; // GeV/c²
    arr[3] = 0.000511; // GeV/c²
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
    arr[19] = 1.18937;
    arr[20] = 1.192642;
    arr[21] = 1.197449;
    arr[22] = 1.31486;
    arr[23] = 1.32171;
    arr[24] = 1.67245;
    arr[25] = .939565;
    arr[26] = 1.115683;
    arr[27] = 1.18937;
    arr[28] = 1.192642;
    arr[29] = 1.197449;
    arr[30] = 1.31486;
    arr[31] = 1.32171;
    arr[32] = 1.67245;
    arr[33] = 0;
    arr[34] = 0;
    arr[35] = 0;
    arr[36] = 0;
    arr[37] = 0;
    arr[38] = 0;
    arr[39] = 0;
    arr[40] = 0;
    arr[41] = 0;
    arr[42] = 0;
    arr[43] = 0;
    arr[44] = 0;
    arr[45] = 0;
    arr[46] = 0;
    arr[47] = 0;
    arr[48] = .95778;
    arr[49] = 1.0194;
    arr[50] = .78265;
    arr[51] = .769;
    arr[52] = .7665;
    arr[53] = .7665;
    arr[54] = 1.2305;
    arr[55] = 1.2318;
    arr[56] = 1.2331;
    arr[57] = 1.2344;
    arr[58] = 1.2309;
    arr[59] = 1.2323;
    arr[60] = 1.2336;
    arr[61] = 1.2349;
    arr[62] = .89581;
    arr[63] = .89166;
    arr[64] = .89166;
    arr[65] = .89581;
    arr[66] = 0.;
    arr[67] = 0.;
    arr[68] = 0.;
    arr[69] = 0.;
    arr[116] = 1.8645;
    arr[117] = 1.8697;
    arr[118] = 1.8697;
    arr[119] = 1.8645;
    arr[120] = 1.9682;
    arr[121] = 1.9682;
    arr[122] = 2.9804;
    arr[123] = 2.0067;
    arr[124] = 2.0100;
    arr[125] = 2.0100;
    arr[126] = 2.0067;
    arr[127] = 2.1121;
    arr[128] = 2.1121;
    arr[130] = 3.096916;
    arr[131] = 1.77699;
    arr[132] = 1.77699;
    arr[133] = 0.;
    arr[134] = 0.;
    arr[137] = 2.28646;
    arr[138] = 2.4679;
    arr[139] = 2.4710;
    arr[140] = 2.45402;
    arr[141] = 2.4529;
    arr[142] = 2.45376;
    arr[143] = 2.5757;
    arr[144] = 2.5780;
    arr[145] = 2.6975;
    arr[149] = 2.28646;
    arr[150] = 2.4679;
    arr[151] = 2.4710;
    arr[152] = 2.45402;
    arr[153] = 2.4529;
    arr[154] = 2.45376;
    arr[155] = 2.5757;
    arr[156] = 2.5780;
    arr[157] = 2.6975;
    arr[161] = 2.5184;
    arr[162] = 2.5175;
    arr[163] = 2.5180;
    arr[171] = 2.5184;
    arr[172] = 2.5175;
    arr[173] = 2.5180;
    arr[176] = 5.27961;
    arr[177] = 5.27929;
    arr[178] = 5.27929;
    arr[179] = 5.27961;
    arr[180] = 5.36679;
    arr[181] = 5.36679;
    arr[182] = 6.2751;
    arr[183] = 6.2751;
    arr[184] = 5.61951;
    arr[185] = 5.8155;
    arr[186] = 5.8113;
    arr[187] = 5.7918;
    arr[188] = 5.7944;
    arr[189] = 6.0480;
    arr[190] = 5.61951;
    arr[191] = 5.8155;
    arr[192] = 5.8113;
    arr[193] = 5.7918;
    arr[194] = 5.7944;
    arr[195] = 6.0480;
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
        }else{
            particle_type = particles_name[0];
        }
        if (no_particle >= 0 && no_particle < particles_mass.size()){
            mass = particles_mass[no_particle];
        }else{
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