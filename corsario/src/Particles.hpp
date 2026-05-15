#pragma once

#include <array>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <iomanip>

/*
 * CORSIKA particle encoding:
 *   particle_description = particle_id * 1000 + hadr_generation * 10 + obs_level
 *
 * Momenta in GeV/c, positions in cm, time in ns.
 * Energy computed via E² = p² + m²  (natural units, GeV).
 */

// ---------------------------------------------------------------------------
// Particle name & mass tables (indexed by CORSIKA particle number)
// ---------------------------------------------------------------------------
static const std::array<std::string, 196> PARTICLES_NAME = [] {
    std::array<std::string, 196> a{};
    a[0]  = "other";
    a[1]  = "gamma";
    a[2]  = "e+";
    a[3]  = "e-";
    a[4]  = "none";
    a[5]  = "mu+";
    a[6]  = "mu-";
    a[7]  = "pi0";
    a[8]  = "pi+";
    a[9]  = "pi-";
    a[10] = "K0L";
    a[11] = "K+";
    a[12] = "K-";
    a[13] = "n";
    a[14] = "p";
    a[15] = "pbar";
    a[16] = "K0S";
    a[17] = "eta";
    a[18] = "Lambda";
    a[19] = "Sigma+";
    a[20] = "Sigma0";
    a[21] = "Sigma-";
    a[22] = "Xi0";
    a[23] = "Xi-";
    a[24] = "Omega-";
    a[25] = "nbar";
    a[26] = "Lambdabar";
    a[27] = "Sigma-bar";
    a[28] = "Sigma0bar";
    a[29] = "Sigma+bar";
    a[30] = "Xi0bar";
    a[31] = "Xi+bar";
    a[32] = "Omega+bar";
    return a;
}();

static const std::array<float, 196> PARTICLES_MASS = [] {
    std::array<float, 196> a{};
    a[1]  = 0.f;
    a[2]  = 0.000511f;
    a[3]  = 0.000511f;
    a[5]  = 0.105658f;
    a[6]  = 0.105658f;
    a[7]  = 0.134977f;
    a[8]  = 0.139570f;
    a[9]  = 0.139570f;
    a[10] = 0.497611f;
    a[11] = 0.493677f;
    a[12] = 0.493677f;
    a[13] = 0.939565f;
    a[14] = 0.938272f;
    a[15] = 0.938272f;
    a[16] = 0.497611f;
    a[17] = 0.547862f;
    a[18] = 1.115683f;
    a[19] = 1.189370f;
    a[20] = 1.192642f;
    a[21] = 1.197449f;
    a[22] = 1.314860f;
    a[23] = 1.321710f;
    a[24] = 1.672450f;
    a[25] = 0.939565f;
    a[26] = 1.115683f;
    a[27] = 1.189370f;
    a[28] = 1.192642f;
    a[29] = 1.197449f;
    a[30] = 1.314860f;
    a[31] = 1.321710f;
    a[32] = 1.672450f;
    a[48] = 0.957780f;
    a[49] = 1.019400f;
    a[50] = 0.782650f;
    a[51] = 0.769000f;
    a[52] = 0.766500f;
    a[53] = 0.766500f;
    a[116] = 1.864500f;
    a[117] = 1.869700f;
    a[118] = 1.869700f;
    a[119] = 1.864500f;
    a[120] = 1.968200f;
    a[121] = 1.968200f;
    a[130] = 3.096916f;
    a[131] = 1.776990f;
    a[132] = 1.776990f;
    a[137] = 2.286460f;
    a[176] = 5.279610f;
    a[177] = 5.279290f;
    a[178] = 5.279290f;
    a[179] = 5.279610f;
    a[180] = 5.366790f;
    a[181] = 5.366790f;
    a[182] = 6.275100f;
    a[183] = 6.275100f;
    a[184] = 5.619510f;
    a[185] = 5.815500f;
    a[186] = 5.811300f;
    a[187] = 5.791800f;
    a[188] = 5.794400f;
    a[189] = 6.048000f;
    a[190] = 5.619510f;
    a[191] = 5.815500f;
    a[192] = 5.811300f;
    a[193] = 5.791800f;
    a[194] = 5.794400f;
    a[195] = 6.048000f;
    return a;
}();

// ---------------------------------------------------------------------------
// Particle class
// ---------------------------------------------------------------------------
class Particle {
public:
    // Constructor from raw CORSIKA particle description float + kinematics
    Particle(int description, float px, float py, float pz,
             float x, float y, float time);

    // Accessors
    int         get_id()            const { return particle_id; }
    int         get_particle_no()   const { return no_particle; }
    int         get_hadron_gen()    const { return hadr_gen; }
    int         get_no_obs()        const { return no_obs; }
    float       get_px()            const { return px; }
    float       get_py()            const { return py; }
    float       get_pz()            const { return pz; }
    float       get_x()             const { return x; }
    float       get_y()             const { return y; }
    float       get_time()          const { return time; }
    float       get_mass()          const { return mass; }
    float       get_energy()        const { return energy; }
    const std::string& get_particle_type() const { return particle_type; }
    std::array<float, 3> get_momentum()    const { return {px, py, pz}; }

    // CSV output helpers
    std::string particle_data_csv()     const;
    std::string particle_data_csv_ext() const;

private:
    int   particle_id;
    int   no_particle;
    int   hadr_gen;
    int   no_obs;
    float px, py, pz;
    float x, y;
    float time;
    float mass;
    float energy;
    std::string particle_type;
};

// ---------------------------------------------------------------------------
inline Particle::Particle(int description, float _px, float _py, float _pz,
                          float _x, float _y, float _time)
    : particle_id(description), px(_px), py(_py), pz(_pz),
      x(_x), y(_y), time(_time)
{
    // CORSIKA encoding: description = id*1000 + hadr_gen*10 + obs_level
    no_particle = particle_id / 1000;
    hadr_gen    = (particle_id % 1000) / 10;
    no_obs      = particle_id % 10;

    particle_type = (no_particle >= 0 && no_particle < (int)PARTICLES_NAME.size())
                    ? PARTICLES_NAME[no_particle] : PARTICLES_NAME[0];
    mass          = (no_particle >= 0 && no_particle < (int)PARTICLES_MASS.size())
                    ? PARTICLES_MASS[no_particle] : 0.f;

    energy = std::sqrt(px*px + py*py + pz*pz + mass*mass);
}

// ---------------------------------------------------------------------------
inline std::string Particle::particle_data_csv() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << particle_id << ','
        << px  << ',' << py  << ',' << pz << ','
        << x   << ',' << y   << ',' << time;
    return oss.str();
}

inline std::string Particle::particle_data_csv_ext() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << particle_type << ','
        << no_particle   << ','
        << hadr_gen      << ','
        << no_obs        << ','
        << mass          << ','
        << energy        << ','
        << particle_id   << ','
        << px  << ',' << py  << ',' << pz << ','
        << x   << ',' << y   << ',' << time;
    return oss.str();
}