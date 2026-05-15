#pragma once

#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>

#include "Shower.hpp"

/*
 * Corsario — CORSIKA DAT binary reader
 *
 * Design:
 *   - Single sequential pass to index EVTH/EVTE block positions.
 *   - Shower objects are lazy: particle data is read only when requested.
 *   - All physical word indices use long to handle files > 2 GB.
 *   - Paths are relative (no absolute hardcoded paths).
 *   - Public API is kept pybind11-friendly (no raw pointers, STL types only).
 *
 * CORSIKA DAT structure (no Fortran record markers, little-endian floats):
 *   Block        = 273 words × 4 bytes = 1092 bytes
 *   Super-block  = 21 blocks + 2-word Fortran padding = 5735 words
 *
 *   Block order inside a run:
 *     RUNH  (1 block)
 *     [ EVTH DATA... EVTE ] × N_showers
 *     RUNE  (1 block)
 */

class Corsario {
public:
    // Open and index a CORSIKA DAT file
    explicit Corsario(const std::string& filename);

    // Library version string
    std::string version() const { return n_version; }

    // Number of indexed showers
    int n_showers() const { return static_cast<int>(showers.size()); }

    // Access shower by index (0-based)
    Shower& shower(int n) { return showers.at(n); }
    const Shower& shower(int n) const { return showers.at(n); }

    // ----- CSV output -----

    // particles.csv: particle_id, px, py, pz, x, y, t
    void to_csv(const std::string& csv_path);

    // particles_ext.csv: shower, particle_type, no_particle, hadr_gen,
    //                    no_obs, mass, energy, particle_id, px, py, pz, x, y, t
    void to_csv_ext(const std::string& csv_path);

    // showers.csv: event_no, particle_id, total_energy, altitude,
    //              no_target, z, px, py, pz, zenith, azimuth
    void shower_csv(const std::string& csv_path);

    // ----- Bulk accessors across all showers -----
    std::vector<float>       get_energy();
    std::vector<std::string> get_particle_type();
    std::vector<int>         get_particle_no();
    std::vector<float>       get_mass();
    std::vector<std::array<float,3>> get_momentum();
    std::vector<float>       get_px();
    std::vector<float>       get_py();
    std::vector<float>       get_pz();
    std::vector<int>         get_id();
    std::vector<float>       get_x();
    std::vector<float>       get_y();
    std::vector<float>       get_time();

private:
    const std::string n_version = "1.0.0";
    std::string filename;
    std::vector<Shower> showers;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

inline Corsario::Corsario(const std::string& _filename) : filename(_filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Corsario: cannot open file: " + filename);
    }

    // Determine file size
    file.seekg(0, std::ios::end);
    const long file_size_bytes = static_cast<long>(file.tellg());
    file.seekg(0, std::ios::beg);
    const long total_words = file_size_bytes / 4;

    /*
     * CORSIKA DAT layout with Fortran unformatted binary (the common case):
     *
     *   Each "sub-block" = 1 record marker (4 bytes) + 273 floats (1092 bytes)
     *                    + 1 record marker (4 bytes) = 1100 bytes = 275 words
     *   BUT: CORSIKA groups 21 sub-blocks per physical record:
     *     1 record marker | 21 × 273 floats | 1 record marker
     *     = 4 + 21*1092 + 4 = 22936 bytes
     *
     *   In the *thinned* / no-marker variant (raw binary) there are no
     *   record markers and the layout is exactly 273 words per block with
     *   a 2-word pad every 21 blocks — as documented in the original code.
     *
     *   The original working code read word at index i (1-based, step 273),
     *   which corresponds to byte offset 4*i.  This means:
     *     Block 0 tag is at word index  1  (byte   4)
     *     Block 1 tag is at word index  274 (byte 1096)
     *     Block 2 tag is at word index  547 (byte 2188)
     *     ...
     *     After every 21 blocks (i % (273*21) == 0 adjusted) skip 2 words.
     *
     *   We replicate that exactly: `i` is the word index of the TAG word,
     *   step is 273, with a +2 skip every 21 blocks.
     *
     *   The physical word index stored in each Shower is this same `i`,
     *   so Shower::evth_data seeks to byte 4*(i+1) for the first data word,
     *   which is consistent.
     */

    char tag[4];
    long i          = 1;     // word index of first block tag (original logic)
    long blocks     = 1;     // 1-based block counter, matches original variable name
    int  shower_idx = 0;
    long evth_word  = -1;

    while (i < total_words) {
        file.seekg(i * 4, std::ios::beg);
        file.read(tag, 4);
        if (!file) break;

        if (std::memcmp(tag, "RUNH", 4) == 0) {
            // run header — nothing to store
        } else if (std::memcmp(tag, "EVTH", 4) == 0) {
            evth_word = i;
        } else if (std::memcmp(tag, "EVTE", 4) == 0 && evth_word >= 0) {
            showers.emplace_back(++shower_idx, evth_word, i, filename);
            evth_word = -1;
        } else if (std::memcmp(tag, "RUNE", 4) == 0) {
            break;
        }

        // Advance exactly as the original working code:
        //   i += 273
        //   if (blocks % 21 == 0) i += 2   ← check BEFORE blocks++
        //   blocks++
        i += 273;
        if (blocks % 21 == 0) i += 2;
        blocks++;
    }

    if (showers.empty()) {
        std::cerr << "[Corsario] Warning: no showers found in " << filename << std::endl;
    }
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

inline void Corsario::to_csv(const std::string& csv_path) {
    std::ofstream out(csv_path);
    if (!out) throw std::runtime_error("Corsario::to_csv: cannot open " + csv_path);

    out << "particle_id,px,py,pz,x,y,t\n";
    for (auto& sh : showers) {
        sh.add_particles();
        int n = sh.n_particles();
        for (int j = 0; j < n; ++j) {
            out << sh.particle(j).particle_data_csv() << '\n';
        }
    }
}

inline void Corsario::to_csv_ext(const std::string& csv_path) {
    std::ofstream out(csv_path);
    if (!out) throw std::runtime_error("Corsario::to_csv_ext: cannot open " + csv_path);

    out << "shower,particle_type,no_particle,hadr_gen,no_obs,mass,energy,"
           "particle_id,px,py,pz,x,y,t\n";
    for (auto& sh : showers) {
        sh.add_particles();
        int n   = sh.n_particles();
        int sno = sh.get_event_no();
        for (int j = 0; j < n; ++j) {
            out << sno << ',' << sh.particle(j).particle_data_csv_ext() << '\n';
        }
    }
}

inline void Corsario::shower_csv(const std::string& csv_path) {
    std::ofstream out(csv_path);
    if (!out) throw std::runtime_error("Corsario::shower_csv: cannot open " + csv_path);

    out << "event_no,particle_id,total_energy,altitude,no_target,"
           "z,px,py,pz,zenith,azimuth\n";
    for (auto& sh : showers) {
        out << sh.get_evth_data() << '\n';
    }
}

// ---------------------------------------------------------------------------
// Bulk accessors
// ---------------------------------------------------------------------------

inline std::vector<float> Corsario::get_energy() {
    std::vector<float> v;
    for (auto& sh : showers) {
        auto e = sh.get_energy();
        v.insert(v.end(), e.begin(), e.end());
    }
    return v;
}
inline std::vector<std::string> Corsario::get_particle_type() {
    std::vector<std::string> v;
    for (auto& sh : showers) {
        auto t = sh.get_particle_type();
        v.insert(v.end(), t.begin(), t.end());
    }
    return v;
}
inline std::vector<int> Corsario::get_particle_no() {
    std::vector<int> v;
    for (auto& sh : showers) {
        auto t = sh.get_particle_no();
        v.insert(v.end(), t.begin(), t.end());
    }
    return v;
}
inline std::vector<float> Corsario::get_mass() {
    std::vector<float> v;
    for (auto& sh : showers) {
        auto t = sh.get_mass();
        v.insert(v.end(), t.begin(), t.end());
    }
    return v;
}
inline std::vector<std::array<float,3>> Corsario::get_momentum() {
    std::vector<std::array<float,3>> v;
    for (auto& sh : showers) {
        auto t = sh.get_momentum();
        v.insert(v.end(), t.begin(), t.end());
    }
    return v;
}
inline std::vector<float> Corsario::get_px() {
    std::vector<float> v;
    for (auto& sh : showers) { auto t = sh.get_px(); v.insert(v.end(), t.begin(), t.end()); }
    return v;
}
inline std::vector<float> Corsario::get_py() {
    std::vector<float> v;
    for (auto& sh : showers) { auto t = sh.get_py(); v.insert(v.end(), t.begin(), t.end()); }
    return v;
}
inline std::vector<float> Corsario::get_pz() {
    std::vector<float> v;
    for (auto& sh : showers) { auto t = sh.get_pz(); v.insert(v.end(), t.begin(), t.end()); }
    return v;
}
inline std::vector<int> Corsario::get_id() {
    std::vector<int> v;
    for (auto& sh : showers) { auto t = sh.get_id(); v.insert(v.end(), t.begin(), t.end()); }
    return v;
}
inline std::vector<float> Corsario::get_x() {
    std::vector<float> v;
    for (auto& sh : showers) { auto t = sh.get_x(); v.insert(v.end(), t.begin(), t.end()); }
    return v;
}
inline std::vector<float> Corsario::get_y() {
    std::vector<float> v;
    for (auto& sh : showers) { auto t = sh.get_y(); v.insert(v.end(), t.begin(), t.end()); }
    return v;
}
inline std::vector<float> Corsario::get_time() {
    std::vector<float> v;
    for (auto& sh : showers) { auto t = sh.get_time(); v.insert(v.end(), t.begin(), t.end()); }
    return v;
}