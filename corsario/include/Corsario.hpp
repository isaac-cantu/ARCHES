#pragma once

#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>

#include "../include/Shower.hpp"

class Corsario{
    private:
    // Version
    std::string n_version = "Preeliminar"; // Version of the library

    // Archivo DAT
        std::string filename;

    // Variables de lectura
        char word_size[4];  // 4 bytes ASCII (words)
        float float_size;   // 4 bytes Float (data)

        int n_particles;    // Total number of particles

        int RUNH;   // Inicio de simulación
        int RUNE;   // Fin de simulación

    // Bloques
        int32_t blocks;             // Number of blocks
        const int8_t n_blocks = 21; // Number of default blocks

    // Showers
        std::vector<Shower> showers; // Vector of showers classes

    // Particles



    public:

        // Constructor - Open
        Corsario(const std::string& file_name);

        // Version of the library
        std::string version();

        // Information of the DAT file
        void info(std::string& csv_path);

        // Summary of the DAT file
        void summary(std::string& csv_path);

        // Number of showers
        int n_showers();

        // Output a Shower n -> shower(n)
        Shower shower(int);

        // DAT file to txt
        void to_txt(std::string& txt_path);

        // DAT file to csv
        void to_csv(std::string& csv_path);
        void to_csv_ext(std::string& csv_path);
        void shower_csv(std::string& csv_path);

        // Pruebas de clase
        void lectura_prueba();


        std::vector<float> get_energy();

        std::vector<std::string> get_particle_type();

        std::vector<int> get_particle_no(); 

        std::vector<float> get_mass();

        std::vector<std::array<float,3>> get_momentun();

        std::vector<float> get_px();

        std::vector<float> get_py();

        std::vector<float> get_pz();

        std::vector<int> get_id();

        std::vector<float> get_x();

        std::vector<float> get_y();

        std::vector<float> get_time();

};

// ==========================================
Corsario::Corsario(const std::string& _filename) : filename(_filename){


    // Lectura de binario DAT
    std::ifstream file(filename, std::ios::binary);
    if (!file){
        std::cerr << "File not found!" << std::endl;
        return;
    }

    // Lectura del total de bytes
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0,std::ios::beg);

    // Guardar indices
    int n = 1;
    int EVTH = -1;
    int EVTE = -1;

    long words = size / 4;
    blocks = 1;

    // Busqueda de indices para lazy iterator
    for(long i = 1; i < words; i += 273){
        // Inicio de la palabra 1 por cada bloque brincando 273 palabras de 4 bytes
        file.seekg(4*i, std::ios::beg);                     // acomodo de palabra inicial
        
        file.read(word_size,4);
        // file.read(reinterpret_cast<char*>(&word_size), sizeof(word_size));  // guardar en char[4]

        // std::cout<<word_size;

        if(std::memcmp(word_size, "RUNH", 4) == 0){
            RUNH = (i-1)/273 + 1;
        }

        else if(std::memcmp(word_size, "EVTH", 4) == 0){
            EVTH = i;//(i-1)/273 + 1;
            // std::cout<<EVTH<<std::endl;
        }

        else if(std::memcmp(word_size, "EVTE", 4) == 0){
            EVTE = i;//(i-1)/273 + 1;
            Shower sh(n,EVTH,EVTE, filename); 
            showers.push_back(sh);
            // std::cout<<n<<" "<<EVTH<<" "<<EVTE<<" "<<std::endl;
            n++;
        }

        // Si encuentra RUNE salir (Bloque de fin de simulación)
        else if(std::memcmp(word_size, "RUNE", 4) == 0){
            RUNE = (i-1)/273 + 1;
            break;
        }
        // std::cout<<n<<" "<<" "<<EVTH<<" "<<EVTE<<" "<<std::endl;
        if(blocks%21 == 0){
            i += 2;
            // file.read(word_size,4);
            // file.read(word_size,4);
        }

        blocks++;
        
    }
    

    file.close();
}


// ==========================================
std::string Corsario::version(){
    return n_version;
}

// ==========================================
int Corsario::n_showers(){
    return showers.size();
}


// ==========================================
void Corsario::info(std::string& csv_path){

}

// ==========================================
void Corsario::summary(std::string& csv_path){}

// ==========================================
void Corsario::to_txt(std::string& txt_path){}


// ==========================================
void Corsario::to_csv(std::string& csv_path){

    std::ofstream csvFIle(csv_path, std::ios::out);
    // if (!csvFIle.is_open()) {
    //     std::cerr << "[Error opening file](url)" << std::endl;
    //     return; // Exit if file cannot be opened
    // }

    csvFIle << "particle_description,px,py,pz,x,y,t" << std::endl;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        // std::cout<<i;
        for(int j = 0; j < showers[i].n_particles(); j++){
            Particle pa = showers[i].particle(j);
            csvFIle << pa.particle_data_csv() << std::endl;
            // std::cout << i << " " << pa.particle_data_csv() << std::endl;
        }
    }
    csvFIle.close();
}

void Corsario::shower_csv(std::string& csv_path){

    std::ofstream csvFIle(csv_path, std::ios::out);
    // if (!csvFIle.is_open()) {
    //     std::cerr << "[Error opening file](url)" << std::endl;
    //     return; // Exit if file cannot be opened
    // }

    csvFIle << "event_no,particle_id,total_energy,altitude,no_target,z,px,py,pz,zenith,azimuth" << std::endl;
    for(int i = 0; i < showers.size(); i++){
        csvFIle << showers[i].get_evth_data() << std::endl;
    }
    csvFIle.close();
}

void Corsario::to_csv_ext(std::string& csv_path){

    std::ofstream csvFIle(csv_path, std::ios::out);
    // if (!csvFIle.is_open()) {
    //     std::cerr << "[Error opening file](url)" << std::endl;
    //     return; // Exit if file cannot be opened
    // }

    csvFIle << "shower,particle_type,no_particle,hadr_gen,no_obs,mass,energy,particle_description,px,py,pz,x,y,t" << std::endl;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        // std::cout<<i;
        for(int j = 0; j < showers[i].n_particles(); j++){
            Particle pa = showers[i].particle(j);
            csvFIle <<showers[i].get_event_no()<<","<<pa.particle_data_csv_ext() << std::endl;
            // std::cout << i << " " << pa.particle_data_csv_ext() << std::endl;
        }
    } 
    csvFIle.close();
}

// ==========================================
Shower Corsario::shower(int num){
    return showers.at(num);
}


std::vector<float> Corsario::get_energy(){
    std::vector<float> energy_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            energy_vector.push_back(showers[i].particle(j).get_energy());
        }
    }
    return energy_vector;
}

std::vector<std::string> Corsario::get_particle_type(){
    std::vector<std::string> particle_type_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            particle_type_vector.push_back(showers[i].particle(j).get_particle_type());
        }
    }
    return particle_type_vector;
}

std::vector<int> Corsario::get_particle_no(){
    std::vector<int> particle_no_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            particle_no_vector.push_back(showers[i].particle(j).get_particle_no());
        }
    }
    return particle_no_vector;
}

std::vector<float> Corsario::get_mass(){
    std::vector<float> mass_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            mass_vector.push_back(showers[i].particle(j).get_mass());
        }
    }
    return mass_vector;
}

std::vector<std::array<float,3>> Corsario::get_momentun(){
    std::vector<std::array<float,3>> momentum_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            momentum_vector.push_back(showers[i].particle(j).get_momentum());
        }
    }
    return momentum_vector;
}

std::vector<float> Corsario::get_px(){
    std::vector<float> px_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            px_vector.push_back(showers[i].particle(j).get_px());
        }
    }
    return px_vector;
}

std::vector<float> Corsario::get_py(){
    std::vector<float> py_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            py_vector.push_back(showers[i].particle(j).get_py());
        }
    }
    return py_vector;
}

std::vector<float> Corsario::get_pz(){
    std::vector<float> pz_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            pz_vector.push_back(showers[i].particle(j).get_pz());
        }
    }
    return pz_vector;
}

std::vector<int> Corsario::get_id(){
    std::vector<int> id_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            id_vector.push_back(showers[i].particle(j).get_id());
        }
    }
    return id_vector;
}

std::vector<float> Corsario::get_x(){
    std::vector<float> x_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            x_vector.push_back(showers[i].particle(j).get_x());
        }
    }
    return x_vector;
}

std::vector<float> Corsario::get_y(){
        std::vector<float> y_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            y_vector.push_back(showers[i].particle(j).get_y());
        }
    }
    return y_vector;
}

std::vector<float> Corsario::get_time(){
    std::vector<float> time_vector;
    for(int i = 0; i < showers.size(); i++){
        showers[i].add_particles();
        for(int j = 0; j < showers[i].n_particles(); j++){
            time_vector.push_back(showers[i].particle(j).get_time());
        }
    }
    return time_vector;
}