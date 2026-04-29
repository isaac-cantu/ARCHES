#pragma once

#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>

#include </home/icantu24/Documents/ARCHES/corsario/include/Particles.hpp>

struct particle_val{
    float particle_id;
    float px;
    float py;
    float pz;
    float x;
    float y; 
    float time;
};

class Shower{
    private:
        bool wparticles = false;

        int n_shower;
        int id_EVTH;
        int id_EVTE;
        std::string filename;

        std::vector<Particle> particles;

        float event_no;
        float particle_id;
        float total_energy;
        float starting_altitude;
        float no_first_target;
        float z_coordinate;
        float px_momentum;
        float py_momentum;
        float pz_momentum;
        float zenith_ang;
        float azimuth_ang;


    public:

        Shower(int _n_shower, int _EVTH, int _EVTE, std::string& _filename) : n_shower(_n_shower), id_EVTE(_EVTE), id_EVTH(_EVTH), filename(_filename){evth_data();};

        void add_particles();

        void add_particle(float, float, float, float, float, float, float);

        int n_particles();

        Particle particle(int);

        void evth_data();

        std::string get_evth_data();


         
        float get_event_no(){return event_no;};
        int get_particle_id(){return particle_id;};
        float get_total_energy(){return total_energy;};
        float get_starting_altitude(){return starting_altitude;};
        float get_no_first_target(){return no_first_target;};
        float get_z_coordinate(){return z_coordinate;};
        float get_px_momentum(){return px_momentum;};
        float get_py_momentum(){return py_momentum;};
        float get_pz_momentum(){return pz_momentum;};
        float get_zenith_ang(){return zenith_ang;};
        float get_azimuth_ang(){return azimuth_ang;};

        std::vector<float> get_energy();

        std::vector<std::string> get_particle_type();

        std::vector<int> get_particle_no();

        std::vector<float> get_mass();

        std::vector<std::array<float, 3>> get_momentum();

        std::vector<float> get_px();

        std::vector<float> get_py();

        std::vector<float> get_pz();

        std::vector<int> get_id();

        std::vector<float> get_x();

        std::vector<float> get_y();

        std::vector<float> get_time();


};
 
void Shower::add_particles(){

    if(!wparticles){
        particle_val values;
        float valor_desfase[2];
        int n = 0;
        int n_block = (id_EVTH-1)/273 + 1;
        int n_line = id_EVTH + 273;
        int desfase;
        // std::cout<<n_shower<<" ["<<id_EVTH<<", "<<id_EVTE<<"]: ";
        // std::cout<<n_block<<" | ";


        // float values[7]

        std::ifstream datFile(filename, std::ios::binary);

        bool cond = true;
        
        int cond_desfase = (n_line)%(273*21+2);
        if(cond_desfase==5734){
            n_line += 2; 
        }else{
            desfase = 0;
        }

        int position = sizeof(float)*(n_line); //(1+(id_EVTH)*273 + desfase);
        datFile.seekg(position, std::ios::beg);
        // std::cout<<position/4<<std::endl;
        // std::cout<<id_EVTH<<" "<<1+desfase<<std::endl;
        //std::cout<<(id_EVTE-(id_EVTH+273))%273<<std::endl;

        while(cond){
            datFile.read(reinterpret_cast<char*>(&values),sizeof(values));
            n_line++;
            // std::cout<<values.particle_id<<std::endl;
            if(values.particle_id == 0 || n_line >= id_EVTE-1){
                break; 
            }else{
                add_particle(values.particle_id, values.px, values.py, values.pz, values.x, values.y, values.time);
                // std::cout<<"ENTRO"<<std::endl;
                n++;
                n_line+=6;
            } 
            
            if(n%39==0){
                n_block += 1;
                // std::cout<<n_block<<" ";
                // desfase_n = (n_block/21)*2;   
                // std::cout<<n_shower<<" "<<id_EVTH<<" "<<n_block<<" "<<desfase<<" "<<desfase_n<<std::endl;
                // if(desfase < desfase_n){
                //     desfase = desfase_n;
                //     datFile.read(reinterpret_cast<char*>(&valor_desfase),sizeof(valor_desfase));
                //     n_line+=2;
                // }
                cond_desfase = (n_line)%(273*21+2);
                if(cond_desfase==5734){
                    datFile.read(reinterpret_cast<char*>(&valor_desfase),sizeof(valor_desfase));
                    n_line+=2;
                }else{
                    desfase = 0;
                }
            }
        } 
        wparticles = true;
        datFile.close();
        // std::cout<<n_block<<std::endl;
    }
}

void Shower::evth_data(){

    std::array<float,11> evth_floats;
    float float_size;
    std::ifstream datFile(filename, std::ios::binary);

    // int desfase = (((((id_EVTH-1)/273 + 1)-1)/21)*2)%273;
    
    int desfase_n = 0;
    //(id_EVTH-1)/273 + 1 = id_EVTH 
    // int position = sizeof(float)*(2+(((id_EVTH-1)/273 + 1)-1)*273 + desfase);
    int position = sizeof(float)*(id_EVTH+1);
    datFile.seekg(position, std::ios::beg);

    // std::cout<<n_shower<<" "<<id_EVTH<<" "<<2+desfase<<std::endl;

    datFile.read(reinterpret_cast<char*>(&evth_floats),sizeof(evth_floats));
    datFile.close();
    event_no = evth_floats[0];
    particle_id = evth_floats[1];
    total_energy = evth_floats[2];
    starting_altitude = evth_floats[3];
    no_first_target = evth_floats[4];
    z_coordinate = evth_floats[5];
    px_momentum = evth_floats[6];
    py_momentum = evth_floats[7];
    pz_momentum = evth_floats[8];
    zenith_ang = evth_floats[9];
    azimuth_ang = evth_floats[10];

}

std::string Shower::get_evth_data(){
    std::string coma = ",";
    std::string data = std::to_string(event_no) + coma 
                        + std::to_string(particle_id) + coma 
                        + std::to_string(total_energy) + coma 
                        + std::to_string(starting_altitude) + coma 
                        + std::to_string(no_first_target) + coma 
                        + std::to_string(z_coordinate) + coma 
                        + std::to_string(px_momentum) + coma 
                        + std::to_string(py_momentum) + coma 
                        + std::to_string(pz_momentum) + coma 
                        + std::to_string(zenith_ang) + coma 
                        + std::to_string(azimuth_ang);
    return data;
}

void Shower::add_particle(float particle_id, float px, float py, float pz, float x, float y, float time){
    Particle pa(particle_id, px, py, pz, x, y, time);
    particles.push_back(pa);
}


int Shower::n_particles(){
    add_particles();
    return particles.size();
}

Particle Shower::particle(int n){
    add_particles();
    return particles.at(n);
}


std::vector<float> Shower::get_energy(){
    add_particles();
    std::vector<float> energy_vector;

    for(int i = 0; i < particles.size(); i++){
        energy_vector.push_back(particles[i].get_energy());
    }
    return energy_vector;
}


std::vector<std::string> Shower::get_particle_type(){
    add_particles();
    std::vector<std::string> particle_type_vector;

    for(int i = 0; i < particles.size(); i++){
        particle_type_vector.push_back(particles[i].get_particle_type());
    }
    return particle_type_vector;
}

std::vector<int> Shower::get_particle_no(){
    add_particles();
    std::vector<int> particle_no_vector;

    for(int i = 0; i < particles.size(); i++){
        particle_no_vector.push_back(particles[i].get_particle_no());
    }
    return particle_no_vector;
}

std::vector<float> Shower::get_mass(){
    add_particles();
    std::vector<float> mass_vector;

    for(int i = 0; i < particles.size(); i++){
        mass_vector.push_back(particles[i].get_mass());
    }
    return mass_vector;
}

std::vector<std::array<float, 3>> Shower::get_momentum(){
    add_particles();
    std::vector<std::array<float, 3>> momentum_vector;

    for(int i = 0; i < particles.size(); i++){
        momentum_vector.push_back(particles[i].get_momentum());
    }
    return momentum_vector;
}

std::vector<float> Shower::get_px(){
    add_particles();
    std::vector<float> px_vector;

    for(int i = 0; i < particles.size(); i++){
        px_vector.push_back(particles[i].get_px());
    }
    return px_vector;
}

std::vector<float> Shower::get_py(){
    add_particles();
    std::vector<float> py_vector;

    for(int i = 0; i < particles.size(); i++){
        py_vector.push_back(particles[i].get_py());
    }
    return py_vector;
}

std::vector<float> Shower::get_pz(){
    add_particles();
    std::vector<float> pz_vector;

    for(int i = 0; i < particles.size(); i++){
        pz_vector.push_back(particles[i].get_pz());
    }
    return pz_vector;
}

std::vector<int> Shower::get_id(){
    add_particles();
    std::vector<int> id_vector;

    for(int i = 0; i < particles.size(); i++){
        id_vector.push_back(particles[i].get_id());
    }
    return id_vector;
}

std::vector<float> Shower::get_x(){
    add_particles();
    std::vector<float> x_vector;

    for(int i = 0; i < particles.size(); i++){
        x_vector.push_back(particles[i].get_x());
    }
    return x_vector;
}

std::vector<float> Shower::get_y(){
    add_particles();
    std::vector<float> y_vector;

    for(int i = 0; i < particles.size(); i++){
        y_vector.push_back(particles[i].get_y());
    }
    return y_vector;
}

std::vector<float> Shower::get_time(){
    add_particles();
    std::vector<float> time_vector;

    for(int i = 0; i < particles.size(); i++){
        time_vector.push_back(particles[i].get_time());
    }
    return time_vector;
}