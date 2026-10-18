// TailCompile experimental post-GP hook, pinned to VTR 40f8d197.
// Included in analytical_placement_flow.cpp; inactive without TC_GP_EXPORT.
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <set>

static void tailcompile_gp_hook(const AtomNetlist& atoms, const APNetlist& ap,
                               const Prepacker& pp, PartialPlacement& placement) {
    const char* prefix = std::getenv("TC_GP_EXPORT");
    if (!prefix) return;
    std::ofstream blocks(std::string(prefix) + ".blocks.tsv");
    std::ofstream nets(std::string(prefix) + ".nets.tsv");
    std::ofstream meta(std::string(prefix) + ".meta.tsv");
    if (!blocks || !nets || !meta)
        VPR_FATAL_ERROR(VPR_ERROR_AP, "TailCompile cannot open GP export files");
    blocks << std::setprecision(17);
    meta << g_vpr_ctx.device().grid.width() << '\t'
         << g_vpr_ctx.device().grid.height() << '\n';
    for (APBlockId b : ap.blocks()) {
        for (PackMoleculeId m : ap.block_molecules(b)) {
            for (AtomBlockId a : pp.get_molecule(m).atom_block_ids) {
                if (!a.is_valid()) continue;
                blocks << size_t(b) << '\t' << atoms.block_name(a) << '\t'
                       << size_t(atoms.block_model(a)) << '\t'
                       << (ap.block_mobility(b) == APBlockMobility::FIXED) << '\t'
                       << placement.block_x_locs[b] << '\t' << placement.block_y_locs[b] << '\t'
                       << placement.block_layer_nums[b] << '\t' << placement.block_sub_tiles[b] << '\n';
            }
        }
    }
    for (APNetId n : ap.nets()) {
        if (ap.net_is_ignored(n)) continue;
        std::set<size_t> pins;
        for (APPinId p : ap.net_pins(n)) pins.insert(size_t(ap.pin_block(p)));
        if (pins.size() < 2) continue;
        nets << size_t(n);
        for (size_t b : pins) nets << '\t' << b;
        nets << '\n';
    }
    blocks.close(); nets.close(); meta.close();
    VTR_LOG("TAILCOMPILE_GP_EXPORTED %s\n", prefix);
    // Optional synchronous CPU worker. All its time is inside this VPR trial.
    // The campaign supplies a fully shell-quoted command from trusted local paths.
    const char* worker = std::getenv("TC_GP_WORKER");
    if (worker && std::system(worker) != 0)
        VPR_FATAL_ERROR(VPR_ERROR_AP, "TailCompile post-GP worker failed");
    const char* replacement = std::getenv("TC_GP_REPLACE");
    if (!replacement) return;
    // Strict atom coverage, coherent AP blocks and immutable fixed locations.
    struct Loc { double x, y, z; int sub; };
    std::unordered_map<std::string, Loc> locations;
    std::ifstream input(replacement);
    if (!input) VPR_FATAL_ERROR(VPR_ERROR_AP, "TailCompile cannot read replacement");
    std::string line, name;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        Loc loc;
        if (!(row >> name >> loc.x >> loc.y >> loc.z >> loc.sub)
            || !std::isfinite(loc.x) || !std::isfinite(loc.y) || !std::isfinite(loc.z)
            || !locations.emplace(name, loc).second)
            VPR_FATAL_ERROR(VPR_ERROR_AP, "Invalid TailCompile replacement row");
    }
    size_t observed = 0;
    for (APBlockId b : ap.blocks()) {
        Loc chosen{};
        bool first = true;
        for (PackMoleculeId m : ap.block_molecules(b)) {
            for (AtomBlockId a : pp.get_molecule(m).atom_block_ids) {
                if (!a.is_valid()) continue;
                auto it = locations.find(atoms.block_name(a));
                if (it == locations.end()) VPR_FATAL_ERROR(VPR_ERROR_AP, "Missing replacement atom");
                const Loc& loc = it->second;
                if (!first && (loc.x != chosen.x || loc.y != chosen.y || loc.z != chosen.z || loc.sub != chosen.sub))
                    VPR_FATAL_ERROR(VPR_ERROR_AP, "Replacement splits an AP molecule/RAM group");
                chosen = loc; first = false; observed++;
            }
        }
        if (first) VPR_FATAL_ERROR(VPR_ERROR_AP, "Empty AP block");
        if (chosen.z != placement.block_layer_nums[b] || chosen.sub != placement.block_sub_tiles[b])
            VPR_FATAL_ERROR(VPR_ERROR_AP, "Layer/subtile changes are outside the 2D experiment");
        if (ap.block_mobility(b) == APBlockMobility::FIXED
            && (chosen.x != placement.block_x_locs[b] || chosen.y != placement.block_y_locs[b]))
            VPR_FATAL_ERROR(VPR_ERROR_AP, "Attempt to move fixed AP block");
        placement.block_x_locs[b] = chosen.x;
        placement.block_y_locs[b] = chosen.y;
    }
    if (observed != locations.size()) VPR_FATAL_ERROR(VPR_ERROR_AP, "Extra replacement atoms");
    VTR_LOG("TAILCOMPILE_GP_REPLACED %zu atoms\n", observed);
}
