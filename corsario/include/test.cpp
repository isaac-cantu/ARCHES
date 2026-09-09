#include </home/icantu24/Documents/ARCHES/corsario/include/Corsario.hpp>
#include </home/icantu24/Documents/ARCHES/corsario/include/Particles.hpp>

#include <iostream>
#include <cstring>

int main(){ 
 
    std::string filename = "/home/icantu24/corsika_data/DAT099999";
    std::string particles_csv = "/home/icantu24/Documents/ARCHES/data/processed/example/particles.csv"; 
    std::string shower_csv = "/home/icantu24/Documents/ARCHES/data/processed/example/shower.csv"; 
  
    Corsario corsario(filename);             
      
    corsario.to_csv_ext(particles_csv);    
    corsario.shower_csv(shower_csv);    

    std::cout<<"Completed!"<<std::endl;

    return 0;  
}                