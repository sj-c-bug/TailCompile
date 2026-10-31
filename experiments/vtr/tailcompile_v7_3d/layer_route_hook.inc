// First and latest completed routing iteration. RR IDs join the frozen pool map.
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

[[maybe_unused]] static void tailcompile_layer_route_usage(unsigned iteration) {
    const char* prefix = std::getenv("TC_GP_Z_USAGE");
    if (!prefix || !*prefix) return;
    const auto& graph = g_vpr_ctx.device().rr_graph;
    const auto& routing = g_vpr_ctx.routing();
    static std::vector<RRNodeId> nodes;
    if (iteration == 1) {
        nodes.clear();
        for (auto node : graph.nodes()) if (graph.node_type(node) == e_rr_type::CHANZ) nodes.push_back(node);
    }
    auto write = [&](const std::string& suffix) {
        std::ofstream out(std::string(prefix) + suffix);
        if (!out) VPR_FATAL_ERROR(VPR_ERROR_ROUTE, "Cannot write layer routing usage summary");
        out << "# iteration=" << iteration << " rr_id\tcapacity\toccupancy\n";
        for (auto node : nodes)
            out << size_t(node) << '\t' << graph.node_capacity(node) << '\t'
                << routing.rr_node_route_inf[node].occ() << '\n';
        out.close();
        if (!out) VPR_FATAL_ERROR(VPR_ERROR_ROUTE, "Layer routing usage summary write failed");
    };
    if (iteration == 1) write(".first.tsv");
    write(".last.tsv");
    VTR_LOG("TAILCOMPILE_V7_Z_USAGE iteration=%u independent_nodes=%zu\n", iteration, nodes.size());
}
