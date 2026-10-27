// TailCompile v7: isolated XYZ AP graph export and native GP initialization.
// Pinned VTR 40f8d19708b0b60e2aa04452deb1cfed68d96b0b only.
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <unordered_map>
#include "logic_types.h"
#include "prepack.h"

[[maybe_unused]] static void tailcompile_v7_export(
    const char* prefix, const char* stage, const AtomNetlist& atoms,
    const APNetlist& ap, const Prepacker& pp, const PartialPlacement& placement) {
    if (!prefix || !*prefix) return;
    std::ofstream blocks(std::string(prefix) + ".blocks.tsv");
    std::ofstream nets(std::string(prefix) + ".nets.tsv");
    std::ofstream pins(std::string(prefix) + ".pins.tsv");
    std::ofstream meta(std::string(prefix) + ".meta.tsv");
    std::ofstream schema(std::string(prefix) + ".schema.json");
    if (!blocks || !nets || !pins || !meta || !schema)
        VPR_FATAL_ERROR(VPR_ERROR_AP, "TailCompile v7 cannot open AP export files");
    const auto& grid = g_vpr_ctx.device().grid;
    const auto& models = g_vpr_ctx.device().arch->models;
    blocks << std::setprecision(17);
    meta << grid.width() << '\t' << grid.height() << '\t' << grid.get_num_layers() << '\n';
    schema << "{\"schema\":\"tailcompile_v7_ap_graph\",\"stage\":\"" << stage
           << "\",\"coordinate_convention\":\"native_ap_tile_center\","
              "\"layer_count\":" << grid.get_num_layers() << ','
           << "\"raw_layer_to_discrete\":\"std_round_nonnegative\","
              "\"blocks_columns\":[\"ap_block\",\"atom\",\"model_id\",\"fixed\","
              "\"x\",\"y\",\"layer\",\"subtile\",\"model_name\",\"molecule\"],"
              "\"nets_format\":\"net_id followed by unique AP block IDs\","
              "\"pins_columns\":[\"net_id\",\"ap_block\",\"is_driver\"]}\n";
    for (APBlockId b : ap.blocks()) {
        for (PackMoleculeId m : ap.block_molecules(b)) {
            for (AtomBlockId a : pp.get_molecule(m).atom_block_ids) {
                if (!a.is_valid()) continue;
                blocks << size_t(b) << '\t' << atoms.block_name(a) << '\t'
                       << size_t(atoms.block_model(a)) << '\t'
                       << (ap.block_mobility(b) == APBlockMobility::FIXED) << '\t'
                       << placement.block_x_locs[b] << '\t' << placement.block_y_locs[b] << '\t'
                       << placement.block_layer_nums[b] << '\t' << placement.block_sub_tiles[b] << '\t'
                       << models.model_name(atoms.block_model(a)) << '\t' << size_t(m) << '\n';
            }
        }
    }
    for (APNetId n : ap.nets()) {
        if (ap.net_is_ignored(n)) continue;
        std::set<size_t> blocks_on_net;
        for (APPinId p : ap.net_pins(n)) {
            blocks_on_net.insert(size_t(ap.pin_block(p)));
            pins << size_t(n) << '\t' << size_t(ap.pin_block(p)) << '\t'
                 << (ap.pin_type(p) == PinType::DRIVER) << '\n';
        }
        if (blocks_on_net.size() < 2) continue;
        nets << size_t(n);
        for (size_t b : blocks_on_net) nets << '\t' << b;
        nets << '\n';
    }
    blocks.close(); nets.close(); pins.close(); meta.close(); schema.close();
    if (!blocks || !nets || !pins || !meta || !schema)
        VPR_FATAL_ERROR(VPR_ERROR_AP, "TailCompile v7 AP export write failed");
    VTR_LOG("TAILCOMPILE_V7_EXPORTED stage=%s prefix=%s blocks=%zu layers=%zu\n",
            stage, prefix, ap.blocks().size(), grid.get_num_layers());
}

[[maybe_unused]] static void tailcompile_v7_structure(
    const AtomNetlist& atoms, const APNetlist& ap, const Prepacker& pp) {
    if (std::getenv("TC_GP_REPLACE") || std::getenv("TC_GP_WORKER"))
        VPR_FATAL_ERROR(VPR_ERROR_AP, "v7 cannot mix legacy replacement/worker hooks");
    if (std::getenv("TC_GP_INIT") && g_vpr_ctx.atom().flat_placement_info().valid)
        VPR_FATAL_ERROR(VPR_ERROR_AP, "TC_GP_INIT must not be combined with --read_flat_place");
    const char* prefix = std::getenv("TC_GP_STRUCTURE");
    const char* only = std::getenv("TC_GP_STRUCTURE_ONLY");
    if (only && std::string(only) != "0" && (!prefix || !*prefix))
        VPR_FATAL_ERROR(VPR_ERROR_AP, "TC_GP_STRUCTURE_ONLY requires TC_GP_STRUCTURE");
    if (!prefix || !*prefix) return;
    PartialPlacement empty(ap);
    tailcompile_v7_export(prefix, "pre_gp_structure", atoms, ap, pp, empty);
    if (only && std::string(only) != "0") {
        VTR_LOG("TAILCOMPILE_V7_STRUCTURE_ONLY_COMPLETE no_gp_executed=1\n");
        std::fflush(nullptr);
        std::exit(EXIT_SUCCESS);
    }
}

// Complete moving APBlock atom rows, discrete initial layer, native continuous XY.
// Fixed atoms may be omitted; any provided row must respect every fixed axis.
[[maybe_unused]] static bool tailcompile_v7_load_initial(
    const AtomNetlist& atoms, const APNetlist& ap, const Prepacker& pp,
    PartialPlacement& placement) {
    const char* path = std::getenv("TC_GP_INIT");
    if (!path) return false;
    if (!*path) VPR_FATAL_ERROR(VPR_ERROR_AP, "TC_GP_INIT is empty");
    const auto& grid = g_vpr_ctx.device().grid;
    struct Loc { double x, y, z; int sub; };
    std::unordered_map<std::string, Loc> rows;
    std::ifstream input(path);
    if (!input) VPR_FATAL_ERROR(VPR_ERROR_AP, "Cannot read TC_GP_INIT file");
    std::string line, name, extra;
    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        if (line[line.find_first_not_of(" \t")] == '#') continue;
        std::istringstream row(line);
        Loc loc;
        if (!(row >> name >> loc.x >> loc.y >> loc.z >> loc.sub) || (row >> extra)
            || !std::isfinite(loc.x) || !std::isfinite(loc.y) || !std::isfinite(loc.z)
            || loc.x < 0 || loc.x >= grid.width() || loc.y < 0 || loc.y >= grid.height()
            || loc.z < 0 || loc.z >= grid.get_num_layers() || std::floor(loc.z) != loc.z
            || loc.sub < 0 || !rows.emplace(name, loc).second)
            VPR_FATAL_ERROR(VPR_ERROR_AP, "Invalid/duplicate/out-of-bounds TC_GP_INIT row");
    }
    size_t observed = 0, moving_blocks = 0, nonzero_layer_blocks = 0;
    double checksum = 0.;
    for (APBlockId b : ap.blocks()) {
        const bool fixed = ap.block_mobility(b) == APBlockMobility::FIXED;
        Loc chosen{};
        bool first = true;
        size_t seen = 0, total = 0;
        for (PackMoleculeId m : ap.block_molecules(b)) {
            for (AtomBlockId a : pp.get_molecule(m).atom_block_ids) {
                if (!a.is_valid()) continue;
                total++;
                auto found = rows.find(atoms.block_name(a));
                if (found == rows.end()) continue;
                const Loc& loc = found->second;
                if (!first && (loc.x != chosen.x || loc.y != chosen.y || loc.z != chosen.z || loc.sub != chosen.sub))
                    VPR_FATAL_ERROR(VPR_ERROR_AP, "TC_GP_INIT splits an AP molecule/RAM block");
                chosen = loc; first = false; seen++; observed++;
            }
        }
        if (total == 0 || (!fixed && seen != total))
            VPR_FATAL_ERROR(VPR_ERROR_AP, "TC_GP_INIT missing moving atoms/AP block");
        if (fixed) {
            if (seen) {
                const auto& fixed_loc = ap.block_loc(b);
                if ((fixed_loc.x != -1 && chosen.x != fixed_loc.x)
                    || (fixed_loc.y != -1 && chosen.y != fixed_loc.y)
                    || (fixed_loc.layer_num != -1 && chosen.z != fixed_loc.layer_num)
                    || (fixed_loc.sub_tile != -1 && chosen.sub != fixed_loc.sub_tile))
                    VPR_FATAL_ERROR(VPR_ERROR_AP, "TC_GP_INIT attempts to move a fixed AP block");
            }
            continue;
        }
        placement.block_x_locs[b] = chosen.x;
        placement.block_y_locs[b] = chosen.y;
        placement.block_layer_nums[b] = chosen.z;
        placement.block_sub_tiles[b] = chosen.sub;
        checksum += (size_t(b) + 1.) * (chosen.x + 3. * chosen.y + 17. * chosen.z + 0.125 * chosen.sub);
        moving_blocks++;
        nonzero_layer_blocks += chosen.z != 0.;
    }
    if (observed != rows.size())
        VPR_FATAL_ERROR(VPR_ERROR_AP, "TC_GP_INIT contains unknown or unused atoms");
    VTR_LOG("TAILCOMPILE_V7_GP_INIT_CONSUMED moving_blocks=%zu atoms=%zu nonzero_layer_blocks=%zu checksum=%.17g path=%s\n",
            moving_blocks, observed, nonzero_layer_blocks, checksum, path);
    tailcompile_v7_export(std::getenv("TC_GP_TRACE"), "injected_initial", atoms, ap, pp, placement);
    return true;
}
