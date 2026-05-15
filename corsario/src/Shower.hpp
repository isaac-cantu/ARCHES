#pragma once

#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include "Particles.hpp"

/*
 * CORSIKA DAT binary layout (Fortran unformatted, no record markers in
 * the version targeted here):
 *
 *   Each BLOCK  = 273 × 4-byte floats  = 1092 bytes
 *   Every 21 blocks a 2-float (8-byte) Fortran padding is inserted.
 *   Super-block = 21 blocks + 2 padding floats = 273*21 + 2 = 5735 words
 *
 *   Block types (identified by first 4-byte word read as ASCII):
 *     RUNH  – run header        (1 block)
 *     EVTH  – event/shower header
 *     EVTE  – event/shower end
 *     RUNE  – run end
 *
 *   Particle sub-block (inside a DATA block):
 *     Each block holds 39 particle records × 7 floats = 273 floats.
 *     Record layout: [particle_description, px, py, pz, x, y, time]
 *     A zero particle_description means the slot is empty (padding).
 */

// Word index → byte offset helper
static inline std::streampos word_offset(long word_index) {
    return static_cast<std::streampos>(word_index * 4);
}

// Given an absolute word index, compute how many 2-float Fortran padding
// words have been inserted before it.
// Super-block size (in words): 273*21 + 2 = 5735
// The padding is inserted AFTER every 21st block (after word 273*21 = 5733
// of the super-block, so it sits at positions 5733 and 5734 within each
// super-block of size 5735).
static inline long padding_before(long word_index) {
    // How many complete super-blocks precede this word?
    constexpr long SB = 273L * 21 + 2;   // 5735 words per super-block
    return (word_index / SB) * 2;
}

// Translate a "logical" word index (ignoring paddings) to a physical one.
static inline long logical_to_physical(long logical) {
    constexpr long BLOCK_WORDS  = 273L;
    constexpr long BLOCKS_PER_SB = 21L;
    constexpr long PAD_WORDS    = 2L;
    constexpr long SB_LOG       = BLOCK_WORDS * BLOCKS_PER_SB; // 5733

    long sb        = logical / SB_LOG;
    long remainder = logical % SB_LOG;
    return sb * (SB_LOG + PAD_WORDS) + remainder;
}

// ---------------------------------------------------------------------------
struct ParticleRecord {
    float particle_id;
    float px, py, pz;
    float x, y;
    float time;
};
static_assert(sizeof(ParticleRecord) == 28, "ParticleRecord must be 28 bytes");

// ---------------------------------------------------------------------------
class Shower {
public:
    Shower(int n_shower, long word_EVTH, long word_EVTE,
           const std::string& filename);

    // Lazy-load all particle records for this shower
    void add_particles();

    // Particle access
    int      n_particles();
    Particle particle(int n);

    // Shower-header accessors (always available after construction)
    float get_event_no()          const { return event_no; }
    int   get_particle_id()       const { return (int)particle_id_hdr; }
    float get_total_energy()      const { return total_energy; }
    float get_starting_altitude() const { return starting_altitude; }
    float get_no_first_target()   const { return no_first_target; }
    float get_z_coordinate()      const { return z_coordinate; }
    float get_px_momentum()       const { return px_momentum; }
    float get_py_momentum()       const { return py_momentum; }
    float get_pz_momentum()       const { return pz_momentum; }
    float get_zenith_ang()        const { return zenith_ang; }
    float get_azimuth_ang()       const { return azimuth_ang; }

    std::string get_evth_data() const;

    // Bulk accessors (trigger lazy load)
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
    void evth_data(std::ifstream& file);

    int         n_shower;
    long        word_EVTH;   // physical word index of the EVTH block start
    long        word_EVTE;   // physical word index of the EVTE block start
    std::string filename;

    bool particles_loaded = false;
    std::vector<Particle> particles;

    // EVTH fields
    float event_no         = 0;
    float particle_id_hdr  = 0;
    float total_energy     = 0;
    float starting_altitude= 0;
    float no_first_target  = 0;
    float z_coordinate     = 0;
    float px_momentum      = 0;
    float py_momentum      = 0;
    float pz_momentum      = 0;
    float zenith_ang       = 0;
    float azimuth_ang      = 0;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

inline Shower::Shower(int _n_shower, long _word_EVTH, long _word_EVTE,
                      const std::string& _filename)
    : n_shower(_n_shower), word_EVTH(_word_EVTH), word_EVTE(_word_EVTE),
      filename(_filename)
{
    // Read header immediately (file handle passed from Corsario constructor
    // would be ideal, but we open here to keep Shower self-contained)
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Shower: cannot open " + filename);
    evth_data(f);
}

// Read the 11 EVTH floats that follow the "EVTH" tag word.
inline void Shower::evth_data(std::ifstream& file) {
    // word_EVTH points to the "EVTH" word itself; data starts at word_EVTH+1
    file.seekg(word_offset(word_EVTH + 1));
    std::array<float, 11> buf{};
    file.read(reinterpret_cast<char*>(buf.data()), sizeof(buf));
    event_no          = buf[0];
    particle_id_hdr   = buf[1];
    total_energy      = buf[2];
    starting_altitude = buf[3];
    no_first_target   = buf[4];
    z_coordinate      = buf[5];
    px_momentum       = buf[6];
    py_momentum       = buf[7];
    pz_momentum       = buf[8];
    zenith_ang        = buf[9];
    azimuth_ang       = buf[10];
}

inline void Shower::add_particles() {
    if (particles_loaded) return;
    particles_loaded = true;

    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Shower::add_particles: cannot open " + filename);

    /*
     * The original working Corsario constructor walked the file as:
     *
     *   long i = 1;          // word index of current block tag
     *   long blocks = 1;
     *   for each block:
     *       read tag at word i
     *       i += 273
     *       if (blocks % 21 == 0) i += 2
     *       blocks++
     *
     * So after processing the block at word `i`, blocks is incremented.
     * Padding is added to `i` BEFORE blocks++ when blocks%21==0.
     *
     * We replay this (no file I/O) from word 1 to reach word_EVTH,
     * so we know the exact `blocks` value there.  Then we continue from
     * the first DATA block (word_EVTH + 273, plus any padding for EVTH's block).
     */

    // --- Replay index to find `blocks` value at word_EVTH ---
    long i      = 1;
    long blocks = 1;
    while (i < word_EVTH) {
        i += 273;
        if (blocks % 21 == 0) i += 2;
        blocks++;
    }
    // i == word_EVTH, blocks == the counter value for this block

    // --- Advance past the EVTH block (same step as original) ---
    i += 273;
    if (blocks % 21 == 0) i += 2;
    blocks++;
    // i now points to the first DATA block after EVTH

    // --- Read particle records slot by slot ---
    // Each block has 39 records × 7 floats = 273 words.
    // We read sequentially: for each slot we seek once (avoids accumulating
    // drift) and advance `i` by 7 after each slot.

    ParticleRecord rec{};
    int slot = 0;

    while (i < word_EVTE) {
        file.seekg(i * 4, std::ios::beg);
        file.read(reinterpret_cast<char*>(&rec), sizeof(rec));
        if (!file) break;

        int desc = static_cast<int>(rec.particle_id);
        if (desc != 0) {
            particles.emplace_back(desc, rec.px, rec.py, rec.pz,
                                   rec.x, rec.y, rec.time);
        }

        i    += 7;
        slot++;

        if (slot == 39) {
            // Completed one DATA block — apply padding check and advance
            // to next block tag, exactly as the original loop does
            if (blocks % 21 == 0) i += 2;
            blocks++;
            slot = 0;
        }
    }
}

inline std::string Shower::get_evth_data() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << event_no          << ','
        << particle_id_hdr   << ','
        << total_energy      << ','
        << starting_altitude << ','
        << no_first_target   << ','
        << z_coordinate      << ','
        << px_momentum       << ','
        << py_momentum       << ','
        << pz_momentum       << ','
        << zenith_ang        << ','
        << azimuth_ang;
    return oss.str();
}

inline int Shower::n_particles() {
    add_particles();
    return static_cast<int>(particles.size());
}

inline Particle Shower::particle(int n) {
    add_particles();
    return particles.at(n);
}

// Bulk accessors ----------------------------------------------------------
inline std::vector<float> Shower::get_energy() {
    add_particles();
    std::vector<float> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_energy());
    return v;
}
inline std::vector<std::string> Shower::get_particle_type() {
    add_particles();
    std::vector<std::string> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_particle_type());
    return v;
}
inline std::vector<int> Shower::get_particle_no() {
    add_particles();
    std::vector<int> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_particle_no());
    return v;
}
inline std::vector<float> Shower::get_mass() {
    add_particles();
    std::vector<float> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_mass());
    return v;
}
inline std::vector<std::array<float,3>> Shower::get_momentum() {
    add_particles();
    std::vector<std::array<float,3>> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_momentum());
    return v;
}
inline std::vector<float> Shower::get_px() {
    add_particles();
    std::vector<float> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_px());
    return v;
}
inline std::vector<float> Shower::get_py() {
    add_particles();
    std::vector<float> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_py());
    return v;
}
inline std::vector<float> Shower::get_pz() {
    add_particles();
    std::vector<float> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_pz());
    return v;
}
inline std::vector<int> Shower::get_id() {
    add_particles();
    std::vector<int> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_id());
    return v;
}
inline std::vector<float> Shower::get_x() {
    add_particles();
    std::vector<float> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_x());
    return v;
}
inline std::vector<float> Shower::get_y() {
    add_particles();
    std::vector<float> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_y());
    return v;
}
inline std::vector<float> Shower::get_time() {
    add_particles();
    std::vector<float> v; v.reserve(particles.size());
    for (auto& p : particles) v.push_back(p.get_time());
    return v;
}