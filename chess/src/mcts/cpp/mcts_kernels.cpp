#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define MCTS_API extern "C" __declspec(dllexport)
#else
#define MCTS_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr std::int32_t kApiVersion = 6;

inline bool valid_size(std::int32_t size) noexcept {
    return size > 0;
}

template <typename T>
inline void reserve_geometric(std::vector<T>& values, const std::size_t additional) {
    const auto required = values.size() + additional;
    if (required <= values.capacity()) {
        return;
    }
    const auto current = values.capacity();
    const auto grown = current > 0
        ? current + std::max<std::size_t>(current / 2U, 1024U)
        : std::max<std::size_t>(required, 1024U);
    values.reserve(std::max(required, grown));
}

inline std::uint64_t rotate_left(const std::uint64_t value, const int shift) noexcept {
    return (value << shift) | (value >> (64 - shift));
}

inline std::uint64_t avalanche(std::uint64_t value) noexcept {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

inline std::uint64_t reverse_bits(std::uint64_t value) noexcept {
    value = ((value >> 1) & UINT64_C(0x5555555555555555))
        | ((value & UINT64_C(0x5555555555555555)) << 1);
    value = ((value >> 2) & UINT64_C(0x3333333333333333))
        | ((value & UINT64_C(0x3333333333333333)) << 2);
    value = ((value >> 4) & UINT64_C(0x0f0f0f0f0f0f0f0f))
        | ((value & UINT64_C(0x0f0f0f0f0f0f0f0f)) << 4);
    value = ((value >> 8) & UINT64_C(0x00ff00ff00ff00ff))
        | ((value & UINT64_C(0x00ff00ff00ff00ff)) << 8);
    value = ((value >> 16) & UINT64_C(0x0000ffff0000ffff))
        | ((value & UINT64_C(0x0000ffff0000ffff)) << 16);
    return (value >> 32) | (value << 32);
}

inline std::uint16_t float_to_half(const float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & UINT32_C(0x8000);
    const std::uint32_t mantissa = bits & UINT32_C(0x007fffff);
    const std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xff) - 127 + 15;
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        std::uint32_t shifted = (mantissa | UINT32_C(0x00800000)) >> (1 - exponent);
        const std::uint32_t rounding = shifted & UINT32_C(0x00001fff);
        shifted >>= 13;
        if (rounding > UINT32_C(0x00001000)
            || (rounding == UINT32_C(0x00001000) && (shifted & 1U))) {
            ++shifted;
        }
        return static_cast<std::uint16_t>(sign | shifted);
    }
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | UINT32_C(0x7c00));
    }
    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t rounding = mantissa & UINT32_C(0x00001fff);
    if (rounding > UINT32_C(0x00001000)
        || (rounding == UINT32_C(0x00001000) && (half_mantissa & 1U))) {
        ++half_mantissa;
        if (half_mantissa == UINT32_C(0x00000400)) {
            half_mantissa = 0;
            if (exponent + 1 >= 31) {
                return static_cast<std::uint16_t>(sign | UINT32_C(0x7c00));
            }
            return static_cast<std::uint16_t>(
                sign | (static_cast<std::uint32_t>(exponent + 1) << 10)
            );
        }
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10) | half_mantissa
    );
}

inline bool valid_az_policy_index(const std::int32_t index) noexcept {
    constexpr std::array<std::pair<std::int32_t, std::int32_t>, 8> queen_directions{{
        {1, 0}, {1, 1}, {0, 1}, {-1, 1},
        {-1, 0}, {-1, -1}, {0, -1}, {1, -1},
    }};
    constexpr std::array<std::pair<std::int32_t, std::int32_t>, 8> knight_deltas{{
        {2, 1}, {1, 2}, {-1, 2}, {-2, 1},
        {-2, -1}, {-1, -2}, {1, -2}, {2, -1},
    }};
    constexpr std::array<std::int32_t, 3> underpromotion_dc{{-1, 0, 1}};
    const std::int32_t from_square = index / 73;
    const std::int32_t plane = index % 73;
    const std::int32_t from_row = from_square / 8;
    const std::int32_t from_col = from_square % 8;
    std::int32_t to_row = from_row;
    std::int32_t to_col = from_col;
    if (plane < 56) {
        const auto direction = queen_directions[static_cast<std::size_t>(plane / 7)];
        const std::int32_t distance = (plane % 7) + 1;
        to_row += direction.first * distance;
        to_col += direction.second * distance;
    } else if (plane < 64) {
        const auto delta = knight_deltas[static_cast<std::size_t>(plane - 56)];
        to_row += delta.first;
        to_col += delta.second;
    } else {
        if (from_row != 6) {
            return false;
        }
        to_row += 1;
        to_col += underpromotion_dc[static_cast<std::size_t>((plane - 64) % 3)];
    }
    return to_row >= 0 && to_row < 8 && to_col >= 0 && to_col < 8;
}

const std::array<std::int16_t, 64 * 73>& az_to_policy_table() noexcept {
    static const auto table = [] {
        std::array<std::int16_t, 64 * 73> values{};
        values.fill(-1);
        std::int16_t compact_index = 0;
        for (std::int32_t az_index = 0; az_index < 64 * 73; ++az_index) {
            if (valid_az_policy_index(az_index)) {
                values[static_cast<std::size_t>(az_index)] = compact_index++;
            }
        }
        return values;
    }();
    return table;
}

inline std::int32_t move_hash_to_az_index(
    const std::int64_t move_hash,
    const bool black_to_move
) noexcept {
    std::int32_t from_square = static_cast<std::int32_t>((move_hash >> 16) & 0xff);
    std::int32_t to_square = static_cast<std::int32_t>((move_hash >> 8) & 0xff);
    const std::int32_t promotion = static_cast<std::int32_t>(move_hash & 0xff);
    if (black_to_move) {
        from_square ^= 63;
        to_square ^= 63;
    }
    const std::int32_t from_row = from_square / 8;
    const std::int32_t from_col = from_square % 8;
    const std::int32_t to_row = to_square / 8;
    const std::int32_t to_col = to_square % 8;
    const std::int32_t dr = to_row - from_row;
    const std::int32_t dc = to_col - from_col;

    std::int32_t plane = -1;
    if (promotion >= 4 && promotion <= 6) {
        if (dr != 1 || dc < -1 || dc > 1) {
            return -1;
        }
        plane = 64 + (promotion - 4) * 3 + (dc + 1);
    } else {
        constexpr std::array<std::pair<std::int32_t, std::int32_t>, 8> queen_directions{{
            {1, 0}, {1, 1}, {0, 1}, {-1, 1},
            {-1, 0}, {-1, -1}, {0, -1}, {1, -1},
        }};
        const std::int32_t abs_dr = std::abs(dr);
        const std::int32_t abs_dc = std::abs(dc);
        if ((dr == 0 || dc == 0 || abs_dr == abs_dc) && (dr != 0 || dc != 0)) {
            const std::int32_t distance = std::max(abs_dr, abs_dc);
            const std::int32_t step_row = dr == 0 ? 0 : (dr > 0 ? 1 : -1);
            const std::int32_t step_col = dc == 0 ? 0 : (dc > 0 ? 1 : -1);
            for (std::int32_t direction_idx = 0; direction_idx < 8; ++direction_idx) {
                if (queen_directions[static_cast<std::size_t>(direction_idx)]
                    == std::pair<std::int32_t, std::int32_t>{step_row, step_col}) {
                    plane = direction_idx * 7 + (distance - 1);
                    break;
                }
            }
        }
        if (plane < 0) {
            constexpr std::array<std::pair<std::int32_t, std::int32_t>, 8> knight_deltas{{
                {2, 1}, {1, 2}, {-1, 2}, {-2, 1},
                {-2, -1}, {-1, -2}, {1, -2}, {2, -1},
            }};
            for (std::int32_t knight_idx = 0; knight_idx < 8; ++knight_idx) {
                if (knight_deltas[static_cast<std::size_t>(knight_idx)]
                    == std::pair<std::int32_t, std::int32_t>{dr, dc}) {
                    plane = 56 + knight_idx;
                    break;
                }
            }
        }
    }
    return plane < 0 ? -1 : from_square * 73 + plane;
}

template <typename Output, typename Convert>
inline void encode_board_planes_batch(
    const std::uint64_t* piece_masks,
    const std::uint8_t* black_to_move,
    const std::uint64_t* castling_masks,
    const std::uint64_t* en_passant_masks,
    const float* halfmove_values,
    const float* fullmove_values,
    const std::int32_t batch_size,
    const std::int64_t output_stride,
    Output* output,
    Convert convert
) noexcept {
    constexpr std::int32_t planes = 16;
    constexpr std::int32_t squares = 64;
    const Output zero = convert(0.0F);
    const Output one = convert(1.0F);
    for (std::int32_t row = 0; row < batch_size; ++row) {
        Output* destination = output + static_cast<std::size_t>(row) * output_stride;
        std::fill(destination, destination + planes * squares, zero);
        const bool flip = black_to_move[row] != 0;
        for (std::int32_t plane = 0; plane < 12; ++plane) {
            const std::int32_t source_plane = flip
                ? (plane < 6 ? plane + 6 : plane - 6)
                : plane;
            std::uint64_t mask = piece_masks[
                static_cast<std::size_t>(row) * 12 + source_plane
            ];
            if (flip) {
                mask = reverse_bits(mask);
            }
            Output* plane_output = destination + plane * squares;
            while (mask != 0) {
                const auto square = static_cast<std::int32_t>(__builtin_ctzll(mask));
                plane_output[square] = one;
                mask &= mask - 1;
            }
        }
        for (const auto metadata : {
            std::pair<std::int32_t, std::uint64_t>{12, castling_masks[row]},
            std::pair<std::int32_t, std::uint64_t>{13, en_passant_masks[row]},
        }) {
            std::uint64_t mask = flip ? reverse_bits(metadata.second) : metadata.second;
            Output* plane_output = destination + metadata.first * squares;
            while (mask != 0) {
                const auto square = static_cast<std::int32_t>(__builtin_ctzll(mask));
                plane_output[square] = one;
                mask &= mask - 1;
            }
        }
        std::fill(
            destination + 14 * squares,
            destination + 15 * squares,
            convert(halfmove_values[row])
        );
        std::fill(
            destination + 15 * squares,
            destination + 16 * squares,
            convert(fullmove_values[row])
        );
    }
}

inline void hash_bytes(
    const std::uint8_t* data,
    const std::size_t size,
    std::uint64_t& first,
    std::uint64_t& second
) noexcept {
    constexpr std::uint64_t prime1 = UINT64_C(0x9e3779b185ebca87);
    constexpr std::uint64_t prime2 = UINT64_C(0xc2b2ae3d27d4eb4f);
    std::size_t offset = 0;
    while (offset + sizeof(std::uint64_t) <= size) {
        std::uint64_t word;
        std::memcpy(&word, data + offset, sizeof(word));
        first = rotate_left(first ^ (word * prime1), 29) * prime2;
        second = rotate_left(second + (word ^ prime2), 31) * prime1;
        offset += sizeof(word);
    }
    if (offset < size) {
        std::uint64_t tail = 0;
        std::memcpy(&tail, data + offset, size - offset);
        first = rotate_left(first ^ (tail * prime1), 29) * prime2;
        second = rotate_left(second + (tail ^ prime2), 31) * prime1;
    }
}

inline void apply_virtual_visit(
    const std::int32_t selected,
    std::int32_t* visit_counts,
    float* total_counts,
    std::int16_t* virtual_losses
) noexcept {
    const auto virtual_loss = static_cast<std::int16_t>(virtual_losses[selected] + 1);
    virtual_losses[selected] = virtual_loss;
    total_counts[selected] = static_cast<float>(visit_counts[selected] + virtual_loss);
}

struct NativeEdge {
    float prior = 0.0F;
    std::int32_t visits = 0;
    float value_sum = 0.0F;
    std::int16_t virtual_loss = 0;
    std::int32_t child = -1;
};

struct NativeNode {
    std::int32_t parent = -1;
    std::int32_t parent_edge = -1;
    std::int32_t edge_begin = -1;
    std::int32_t edge_count = 0;
    double raw_value = std::numeric_limits<double>::quiet_NaN();
    std::int32_t root_visits = 0;
    double root_value_sum = 0.0;
    std::int32_t root_virtual_loss = 0;
    bool expanded = false;
};

struct RootSelectionState {
    std::vector<double> initial_visits;
    std::vector<double> gumbel;
    std::vector<std::int32_t> sequence;
    double initial_total = 0.0;
    bool configured = false;
};

struct PendingSelection {
    std::int32_t path_begin = 0;
    std::int32_t path_count = 0;
    bool active = true;
};

struct NativeForest {
    double c_visit = 100.0;
    double c_scale = 0.10;
    double q_range_floor = 0.25;
    bool use_mixed_value = true;
    std::vector<NativeNode> nodes;
    std::vector<NativeEdge> edges;
    std::vector<RootSelectionState> root_states;
    std::vector<PendingSelection> selections;
    std::vector<std::int32_t> selection_paths;
    std::vector<double> completed_q;
    std::vector<double> improved_policy;
};

inline bool valid_node(const NativeForest& forest, const std::int32_t node_id) noexcept {
    return node_id >= 0 && static_cast<std::size_t>(node_id) < forest.nodes.size();
}

inline NativeEdge* node_edges(NativeForest& forest, const NativeNode& node) noexcept {
    if (node.edge_count <= 0 || node.edge_begin < 0) {
        return nullptr;
    }
    return forest.edges.data() + node.edge_begin;
}

inline const NativeEdge* node_edges(
    const NativeForest& forest,
    const NativeNode& node
) noexcept {
    if (node.edge_count <= 0 || node.edge_begin < 0) {
        return nullptr;
    }
    return forest.edges.data() + node.edge_begin;
}

inline double node_value_sum(const NativeForest& forest, const NativeNode& node) noexcept {
    if (node.parent >= 0 && valid_node(forest, node.parent)) {
        const auto& parent = forest.nodes[static_cast<std::size_t>(node.parent)];
        if (node.parent_edge >= 0 && node.parent_edge < parent.edge_count) {
            return static_cast<double>(
                forest.edges[static_cast<std::size_t>(parent.edge_begin + node.parent_edge)].value_sum
            );
        }
    }
    return node.root_value_sum;
}

inline std::int32_t node_visits(
    const NativeForest& forest,
    const NativeNode& node
) noexcept {
    if (node.parent >= 0 && valid_node(forest, node.parent)) {
        const auto& parent = forest.nodes[static_cast<std::size_t>(node.parent)];
        if (node.parent_edge >= 0 && node.parent_edge < parent.edge_count) {
            return forest.edges[
                static_cast<std::size_t>(parent.edge_begin + node.parent_edge)
            ].visits;
        }
    }
    return node.root_visits;
}

inline void compute_completed_q(
    NativeForest& forest,
    const std::int32_t node_id
) {
    const auto& node = forest.nodes[static_cast<std::size_t>(node_id)];
    const auto* edges = node_edges(forest, node);
    const auto count = node.edge_count;
    forest.completed_q.assign(static_cast<std::size_t>(count), 0.0);
    if (edges == nullptr || count <= 0) {
        return;
    }

    double raw_value = node.raw_value;
    if (!std::isfinite(raw_value)) {
        const auto visits = node_visits(forest, node);
        raw_value = visits > 0 ? node_value_sum(forest, node) / static_cast<double>(visits) : 0.0;
    }
    raw_value = std::clamp(raw_value, -1.0, 1.0);

    double visit_sum = 0.0;
    double visited_prior_sum = 0.0;
    double weighted_q_sum = 0.0;
    bool any_visited = false;
    const double tiny = std::numeric_limits<double>::min();
    for (std::int32_t index = 0; index < count; ++index) {
        const auto visits = edges[index].visits;
        if (visits > 0) {
            any_visited = true;
            const double q = -static_cast<double>(edges[index].value_sum) /
                             static_cast<double>(visits);
            forest.completed_q[static_cast<std::size_t>(index)] = q;
            visit_sum += static_cast<double>(visits);
            const double prior = std::max(static_cast<double>(edges[index].prior), tiny);
            visited_prior_sum += prior;
            weighted_q_sum += prior * q;
        }
    }

    double completion = raw_value;
    if (forest.use_mixed_value && any_visited && visit_sum > 0.0 && visited_prior_sum > 0.0) {
        const double weighted_q = weighted_q_sum / visited_prior_sum;
        completion = (raw_value + visit_sum * weighted_q) / (visit_sum + 1.0);
    }
    double q_min = std::numeric_limits<double>::infinity();
    double q_max = -std::numeric_limits<double>::infinity();
    for (std::int32_t index = 0; index < count; ++index) {
        auto& q = forest.completed_q[static_cast<std::size_t>(index)];
        if (edges[index].visits <= 0) {
            q = completion;
        }
        q_min = std::min(q_min, q);
        q_max = std::max(q_max, q);
    }
    const double range = q_max - q_min;
    if (range > 1.0e-8) {
        const double denominator = std::max(range, forest.q_range_floor);
        for (auto& q : forest.completed_q) {
            q = (q - q_min) / denominator;
        }
    } else {
        std::fill(forest.completed_q.begin(), forest.completed_q.end(), 0.0);
    }
}

inline std::int32_t select_interior(
    NativeForest& forest,
    const std::int32_t node_id
) {
    const auto& node = forest.nodes[static_cast<std::size_t>(node_id)];
    auto* edges = node_edges(forest, node);
    const auto count = node.edge_count;
    if (edges == nullptr || count <= 0) {
        return -1;
    }
    if (count == 1) {
        return 0;
    }
    compute_completed_q(forest, node_id);
    std::int32_t max_visits = 0;
    for (std::int32_t index = 0; index < count; ++index) {
        max_visits = std::max(max_visits, edges[index].visits);
    }
    const double scale = (forest.c_visit + static_cast<double>(max_visits)) * forest.c_scale;
    double maximum = -std::numeric_limits<double>::infinity();
    forest.improved_policy.assign(static_cast<std::size_t>(count), 0.0);
    for (std::int32_t index = 0; index < count; ++index) {
        const float q_score = static_cast<float>(
            forest.completed_q[static_cast<std::size_t>(index)] * scale
        );
        const double logit = std::log(std::max(
            static_cast<double>(edges[index].prior),
            std::numeric_limits<double>::min()
        )) + static_cast<double>(q_score);
        forest.improved_policy[static_cast<std::size_t>(index)] = logit;
        maximum = std::max(maximum, logit);
    }
    double probability_total = 0.0;
    double total_count_sum = 0.0;
    for (std::int32_t index = 0; index < count; ++index) {
        const double probability = std::exp(
            forest.improved_policy[static_cast<std::size_t>(index)] - maximum
        );
        forest.improved_policy[static_cast<std::size_t>(index)] = probability;
        probability_total += probability;
        total_count_sum += static_cast<double>(
            edges[index].visits + edges[index].virtual_loss
        );
    }
    std::int32_t selected = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::int32_t index = 0; index < count; ++index) {
        // Python's improved-policy cache is float32. Preserve its rounding
        // before the visit-share subtraction so native traversal makes the
        // same tie decisions as the existing search.
        const double probability = probability_total > 0.0
            ? static_cast<double>(static_cast<float>(
                forest.improved_policy[static_cast<std::size_t>(index)] / probability_total
            ))
            : static_cast<double>(edges[index].prior);
        const double count_share = static_cast<double>(
            edges[index].visits + edges[index].virtual_loss
        ) / (1.0 + total_count_sum);
        const double score = probability - count_share;
        if (score > best_score) {
            best_score = score;
            selected = index;
        }
    }
    return selected;
}

inline std::int32_t select_root(
    NativeForest& forest,
    const std::int32_t node_id,
    const RootSelectionState& state
) {
    const auto& node = forest.nodes[static_cast<std::size_t>(node_id)];
    auto* edges = node_edges(forest, node);
    const auto count = node.edge_count;
    if (edges == nullptr || count <= 0) {
        return -1;
    }
    if (count == 1) {
        return 0;
    }
    compute_completed_q(forest, node_id);
    double total_count_sum = 0.0;
    for (std::int32_t index = 0; index < count; ++index) {
        total_count_sum += static_cast<double>(
            edges[index].visits + edges[index].virtual_loss
        );
    }
    const auto simulation_index = std::max<std::int64_t>(
        0,
        static_cast<std::int64_t>(std::llround(total_count_sum - state.initial_total))
    );
    const double considered_visit = state.sequence.empty()
        ? 0.0
        : static_cast<double>(state.sequence[std::min<std::size_t>(
            static_cast<std::size_t>(simulation_index),
            state.sequence.size() - 1
        )]);

    double max_fresh = 0.0;
    double min_fresh = std::numeric_limits<double>::infinity();
    bool has_considered = false;
    for (std::int32_t index = 0; index < count; ++index) {
        const double initial = index < static_cast<std::int32_t>(state.initial_visits.size())
            ? state.initial_visits[static_cast<std::size_t>(index)]
            : 0.0;
        const double fresh = std::max(
            0.0,
            static_cast<double>(edges[index].visits + edges[index].virtual_loss) - initial
        );
        max_fresh = std::max(max_fresh, fresh);
        min_fresh = std::min(min_fresh, fresh);
        has_considered = has_considered || fresh == considered_visit;
    }
    const double eligible_visit = has_considered ? considered_visit : min_fresh;
    const double scale = (forest.c_visit + max_fresh) * forest.c_scale;
    double max_log_prior = -std::numeric_limits<double>::infinity();
    for (std::int32_t index = 0; index < count; ++index) {
        max_log_prior = std::max(
            max_log_prior,
            std::log(std::max(
                static_cast<double>(edges[index].prior),
                std::numeric_limits<double>::min()
            ))
        );
    }
    std::int32_t selected = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::int32_t index = 0; index < count; ++index) {
        const double initial = index < static_cast<std::int32_t>(state.initial_visits.size())
            ? state.initial_visits[static_cast<std::size_t>(index)]
            : 0.0;
        const double fresh = std::max(
            0.0,
            static_cast<double>(edges[index].visits + edges[index].virtual_loss) - initial
        );
        if (fresh != eligible_visit) {
            continue;
        }
        const float q_score = static_cast<float>(
            forest.completed_q[static_cast<std::size_t>(index)] * scale
        );
        const double gumbel = index < static_cast<std::int32_t>(state.gumbel.size())
            ? state.gumbel[static_cast<std::size_t>(index)]
            : 0.0;
        const double score = (
            std::log(std::max(
                static_cast<double>(edges[index].prior),
                std::numeric_limits<double>::min()
            )) - max_log_prior
        ) + gumbel + static_cast<double>(q_score);
        if (score > best_score) {
            best_score = score;
            selected = index;
        }
    }
    return selected;
}

inline std::int32_t ensure_child(
    NativeForest& forest,
    const std::int32_t parent_id,
    const std::int32_t local_edge
) {
    auto& parent = forest.nodes[static_cast<std::size_t>(parent_id)];
    auto& edge = forest.edges[
        static_cast<std::size_t>(parent.edge_begin + local_edge)
    ];
    if (edge.child >= 0) {
        return edge.child;
    }
    NativeNode child;
    child.parent = parent_id;
    child.parent_edge = local_edge;
    forest.nodes.push_back(child);
    forest.root_states.emplace_back();
    edge.child = static_cast<std::int32_t>(forest.nodes.size() - 1);
    return edge.child;
}

}  // namespace

MCTS_API std::int32_t mcts_api_version() noexcept {
    return kApiVersion;
}

MCTS_API std::int32_t mcts_encode_board_planes_f32(
    const std::uint64_t* piece_masks,
    const std::uint8_t* black_to_move,
    const std::uint64_t* castling_masks,
    const std::uint64_t* en_passant_masks,
    const float* halfmove_values,
    const float* fullmove_values,
    const std::int32_t batch_size,
    const std::int64_t output_stride,
    float* output
) noexcept {
    if (piece_masks == nullptr || black_to_move == nullptr
        || castling_masks == nullptr || en_passant_masks == nullptr
        || halfmove_values == nullptr || fullmove_values == nullptr
        || output == nullptr || batch_size <= 0 || output_stride < 16 * 64) {
        return 1;
    }
    encode_board_planes_batch(
        piece_masks,
        black_to_move,
        castling_masks,
        en_passant_masks,
        halfmove_values,
        fullmove_values,
        batch_size,
        output_stride,
        output,
        [](const float value) noexcept { return value; }
    );
    return 0;
}

MCTS_API std::int32_t mcts_encode_board_planes_f16(
    const std::uint64_t* piece_masks,
    const std::uint8_t* black_to_move,
    const std::uint64_t* castling_masks,
    const std::uint64_t* en_passant_masks,
    const float* halfmove_values,
    const float* fullmove_values,
    const std::int32_t batch_size,
    const std::int64_t output_stride,
    std::uint16_t* output
) noexcept {
    if (piece_masks == nullptr || black_to_move == nullptr
        || castling_masks == nullptr || en_passant_masks == nullptr
        || halfmove_values == nullptr || fullmove_values == nullptr
        || output == nullptr || batch_size <= 0 || output_stride < 16 * 64) {
        return 1;
    }
    encode_board_planes_batch(
        piece_masks,
        black_to_move,
        castling_masks,
        en_passant_masks,
        halfmove_values,
        fullmove_values,
        batch_size,
        output_stride,
        output,
        [](const float value) noexcept { return float_to_half(value); }
    );
    return 0;
}

MCTS_API std::int32_t mcts_encode_move_hashes(
    const std::int64_t* move_hashes,
    const std::int32_t* node_offsets,
    const std::uint8_t* black_to_move,
    const std::int32_t node_count,
    std::int32_t* output
) noexcept {
    if (move_hashes == nullptr || node_offsets == nullptr
        || black_to_move == nullptr || output == nullptr || node_count <= 0) {
        return 1;
    }
    const auto& policy_table = az_to_policy_table();
    for (std::int32_t node_idx = 0; node_idx < node_count; ++node_idx) {
        const std::int32_t begin = node_offsets[node_idx];
        const std::int32_t end = node_offsets[node_idx + 1];
        if (begin < 0 || end < begin) {
            return 2;
        }
        for (std::int32_t move_idx = begin; move_idx < end; ++move_idx) {
            const std::int32_t az_index = move_hash_to_az_index(
                move_hashes[move_idx],
                black_to_move[node_idx] != 0
            );
            if (az_index < 0 || az_index >= static_cast<std::int32_t>(policy_table.size())) {
                return 3;
            }
            const std::int32_t policy_index = policy_table[static_cast<std::size_t>(az_index)];
            if (policy_index < 0) {
                return 4;
            }
            output[move_idx] = policy_index;
        }
    }
    return 0;
}

MCTS_API void* mcts_forest_create(
    const double c_visit,
    const double c_scale,
    const double q_range_floor,
    const std::int32_t use_mixed_value,
    const std::int32_t reserve_nodes,
    const std::int32_t reserve_edges
) noexcept {
    try {
        auto* forest = new NativeForest();
        forest->c_visit = std::max(0.0, c_visit);
        forest->c_scale = std::max(0.0, c_scale);
        forest->q_range_floor = std::max(1.0e-8, q_range_floor);
        forest->use_mixed_value = use_mixed_value != 0;
        if (reserve_nodes > 0) {
            forest->nodes.reserve(static_cast<std::size_t>(reserve_nodes));
            forest->root_states.reserve(static_cast<std::size_t>(reserve_nodes));
        }
        if (reserve_edges > 0) {
            forest->edges.reserve(static_cast<std::size_t>(reserve_edges));
        }
        return forest;
    } catch (...) {
        return nullptr;
    }
}

MCTS_API void mcts_forest_destroy(void* context) noexcept {
    delete static_cast<NativeForest*>(context);
}

MCTS_API std::int32_t mcts_forest_add_node(
    void* context,
    const std::int32_t parent,
    const std::int32_t parent_edge,
    const std::int32_t expanded,
    const double raw_value,
    const std::int32_t root_visits,
    const double root_value_sum,
    const std::int32_t root_virtual_loss
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (forest == nullptr) {
        return -1;
    }
    try {
        NativeNode node;
        node.parent = parent;
        node.parent_edge = parent_edge;
        node.expanded = expanded != 0;
        node.raw_value = raw_value;
        node.root_visits = root_visits;
        node.root_value_sum = root_value_sum;
        node.root_virtual_loss = root_virtual_loss;
        forest->nodes.push_back(node);
        forest->root_states.emplace_back();
        return static_cast<std::int32_t>(forest->nodes.size() - 1);
    } catch (...) {
        return -2;
    }
}

MCTS_API std::int32_t mcts_forest_set_edges(
    void* context,
    const std::int32_t node_id,
    const float* priors,
    const std::int32_t* visits,
    const float* value_sums,
    const std::int16_t* virtual_losses,
    const std::int32_t* child_ids,
    const std::int32_t count
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (forest == nullptr || !valid_node(*forest, node_id) || count < 0) {
        return -1;
    }
    if (count > 0 && (
        priors == nullptr || visits == nullptr || value_sums == nullptr ||
        virtual_losses == nullptr || child_ids == nullptr
    )) {
        return -2;
    }
    try {
        auto& node = forest->nodes[static_cast<std::size_t>(node_id)];
        if (node.edge_begin >= 0) {
            return -3;
        }
        node.edge_begin = static_cast<std::int32_t>(forest->edges.size());
        node.edge_count = count;
        node.expanded = true;
        reserve_geometric(forest->edges, static_cast<std::size_t>(count));
        for (std::int32_t index = 0; index < count; ++index) {
            NativeEdge edge;
            edge.prior = priors[index];
            edge.visits = visits[index];
            edge.value_sum = value_sums[index];
            edge.virtual_loss = virtual_losses[index];
            edge.child = child_ids[index];
            forest->edges.push_back(edge);
        }
        return 0;
    } catch (...) {
        return -4;
    }
}

MCTS_API std::int32_t mcts_forest_configure_root(
    void* context,
    const std::int32_t node_id,
    const std::int32_t* initial_visits,
    const double* gumbel,
    const std::int32_t count,
    const std::int32_t* sequence,
    const std::int32_t sequence_count
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (
        forest == nullptr || !valid_node(*forest, node_id) || count < 0 ||
        sequence_count < 0 || (count > 0 && (initial_visits == nullptr || gumbel == nullptr)) ||
        (sequence_count > 0 && sequence == nullptr)
    ) {
        return -1;
    }
    if (forest->nodes[static_cast<std::size_t>(node_id)].edge_count != count) {
        return -2;
    }
    try {
        auto& state = forest->root_states[static_cast<std::size_t>(node_id)];
        state.initial_visits.resize(static_cast<std::size_t>(count));
        state.gumbel.clear();
        state.sequence.clear();
        if (count > 0) {
            state.gumbel.assign(gumbel, gumbel + count);
        }
        if (sequence_count > 0) {
            state.sequence.assign(sequence, sequence + sequence_count);
        }
        state.initial_total = 0.0;
        for (std::int32_t index = 0; index < count; ++index) {
            state.initial_visits[static_cast<std::size_t>(index)] =
                static_cast<double>(initial_visits[index]);
            state.initial_total += static_cast<double>(initial_visits[index]);
        }
        state.configured = true;
        return 0;
    } catch (...) {
        return -3;
    }
}

MCTS_API std::int32_t mcts_forest_reroot(
    void* context,
    const std::int32_t node_id,
    const double raw_value,
    const std::int32_t root_visits,
    const double root_value_sum,
    const std::int32_t root_virtual_loss
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (forest == nullptr || !valid_node(*forest, node_id)) {
        return -1;
    }
    auto& node = forest->nodes[static_cast<std::size_t>(node_id)];
    node.parent = -1;
    node.parent_edge = -1;
    node.raw_value = raw_value;
    node.root_visits = root_visits;
    node.root_value_sum = root_value_sum;
    node.root_virtual_loss = root_virtual_loss;
    forest->root_states[static_cast<std::size_t>(node_id)] = RootSelectionState{};
    return 0;
}

MCTS_API std::int32_t mcts_forest_select_batch(
    void* context,
    const std::int32_t* root_ids,
    const std::int32_t count,
    std::int32_t* selection_ids,
    std::int32_t* leaf_ids,
    std::int32_t* leaf_parent_ids,
    std::int32_t* leaf_parent_edges,
    std::int32_t* depths
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (
        forest == nullptr || count < 0 || (count > 0 && (
            root_ids == nullptr || selection_ids == nullptr || leaf_ids == nullptr ||
            leaf_parent_ids == nullptr || leaf_parent_edges == nullptr || depths == nullptr
        ))
    ) {
        return -1;
    }
    try {
        forest->selections.reserve(forest->selections.size() + static_cast<std::size_t>(count));
        forest->selection_paths.reserve(
            forest->selection_paths.size() + static_cast<std::size_t>(count) * 8U
        );
        for (std::int32_t slot = 0; slot < count; ++slot) {
            const auto root_id = root_ids[slot];
            if (!valid_node(*forest, root_id)) {
                return -2;
            }
            PendingSelection selection;
            selection.path_begin = static_cast<std::int32_t>(forest->selection_paths.size());
            forest->selection_paths.push_back(root_id);
            forest->nodes[static_cast<std::size_t>(root_id)].root_virtual_loss += 1;
            auto node_id = root_id;
            while (forest->nodes[static_cast<std::size_t>(node_id)].expanded) {
                const auto& node = forest->nodes[static_cast<std::size_t>(node_id)];
                if (node.edge_count <= 0) {
                    break;
                }
                const auto& root_state = forest->root_states[static_cast<std::size_t>(root_id)];
                const auto selected = (
                    node_id == root_id && root_state.configured
                ) ? select_root(*forest, node_id, root_state)
                  : select_interior(*forest, node_id);
                if (selected < 0 || selected >= node.edge_count) {
                    break;
                }
                auto& edge = forest->edges[
                    static_cast<std::size_t>(node.edge_begin + selected)
                ];
                edge.virtual_loss = static_cast<std::int16_t>(edge.virtual_loss + 1);
                const auto child_id = ensure_child(*forest, node_id, selected);
                forest->selection_paths.push_back(child_id);
                node_id = child_id;
            }
            selection.path_count = static_cast<std::int32_t>(
                forest->selection_paths.size()
            ) - selection.path_begin;
            const auto selection_id = static_cast<std::int32_t>(forest->selections.size());
            forest->selections.push_back(selection);
            const auto& leaf = forest->nodes[static_cast<std::size_t>(node_id)];
            selection_ids[slot] = selection_id;
            leaf_ids[slot] = node_id;
            leaf_parent_ids[slot] = leaf.parent;
            leaf_parent_edges[slot] = leaf.parent_edge;
            depths[slot] = forest->selections.back().path_count;
        }
        return 0;
    } catch (...) {
        return -3;
    }
}

MCTS_API std::int32_t mcts_forest_expand(
    void* context,
    const std::int32_t node_id,
    const float* priors,
    const std::int32_t count,
    const double raw_value
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (
        forest == nullptr || !valid_node(*forest, node_id) || count < 0 ||
        (count > 0 && priors == nullptr)
    ) {
        return -1;
    }
    try {
        auto& node = forest->nodes[static_cast<std::size_t>(node_id)];
        if (node.expanded) {
            return 1;
        }
        node.raw_value = raw_value;
        node.expanded = true;
        node.edge_begin = static_cast<std::int32_t>(forest->edges.size());
        node.edge_count = count;
        reserve_geometric(forest->edges, static_cast<std::size_t>(count));
        for (std::int32_t index = 0; index < count; ++index) {
            NativeEdge edge;
            edge.prior = priors[index];
            forest->edges.push_back(edge);
        }
        return 0;
    } catch (...) {
        return -2;
    }
}

MCTS_API std::int32_t mcts_forest_expand_batch(
    void* context,
    const std::int32_t* node_ids,
    const std::int32_t* prior_offsets,
    const float* priors,
    const double* raw_values,
    const std::int32_t count
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (
        forest == nullptr || count < 0 ||
        (count > 0 && (
            node_ids == nullptr || prior_offsets == nullptr || raw_values == nullptr
        ))
    ) {
        return -1;
    }
    if (count == 0) {
        return 0;
    }
    const auto total_priors = prior_offsets[count];
    if (prior_offsets[0] != 0 || total_priors < 0 || (total_priors > 0 && priors == nullptr)) {
        return -2;
    }
    for (std::int32_t slot = 0; slot < count; ++slot) {
        if (
            !valid_node(*forest, node_ids[slot]) ||
            prior_offsets[slot] < 0 ||
            prior_offsets[slot + 1] < prior_offsets[slot]
        ) {
            return -3;
        }
    }
    try {
        // One geometric reservation per inference batch avoids both repeated
        // allocator traffic and repeated copying of the growing edge pool.
        reserve_geometric(forest->edges, static_cast<std::size_t>(total_priors));
        std::int32_t expanded = 0;
        for (std::int32_t slot = 0; slot < count; ++slot) {
            auto& node = forest->nodes[static_cast<std::size_t>(node_ids[slot])];
            if (node.expanded) {
                continue;
            }
            const auto begin = prior_offsets[slot];
            const auto end = prior_offsets[slot + 1];
            node.raw_value = raw_values[slot];
            node.expanded = true;
            node.edge_begin = static_cast<std::int32_t>(forest->edges.size());
            node.edge_count = end - begin;
            for (auto index = begin; index < end; ++index) {
                NativeEdge edge;
                edge.prior = priors[index];
                forest->edges.push_back(edge);
            }
            ++expanded;
        }
        return expanded;
    } catch (...) {
        return -4;
    }
}

MCTS_API std::int32_t mcts_forest_backup_batch(
    void* context,
    const std::int32_t* selection_ids,
    const double* values,
    const std::int32_t count
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (
        forest == nullptr || count < 0 ||
        (count > 0 && (selection_ids == nullptr || values == nullptr))
    ) {
        return -1;
    }
    for (std::int32_t slot = 0; slot < count; ++slot) {
        const auto selection_id = selection_ids[slot];
        if (
            selection_id < 0 ||
            static_cast<std::size_t>(selection_id) >= forest->selections.size()
        ) {
            return -2;
        }
        auto& selection = forest->selections[static_cast<std::size_t>(selection_id)];
        if (
            !selection.active || selection.path_begin < 0 || selection.path_count <= 0 ||
            static_cast<std::size_t>(selection.path_begin + selection.path_count) >
                forest->selection_paths.size()
        ) {
            return -3;
        }
        double value = values[slot];
        for (std::int32_t offset = selection.path_count - 1; offset >= 0; --offset) {
            const auto node_id = forest->selection_paths[
                static_cast<std::size_t>(selection.path_begin + offset)
            ];
            auto& node = forest->nodes[static_cast<std::size_t>(node_id)];
            if (node.parent >= 0) {
                auto& parent = forest->nodes[static_cast<std::size_t>(node.parent)];
                auto& edge = forest->edges[
                    static_cast<std::size_t>(parent.edge_begin + node.parent_edge)
                ];
                edge.value_sum = static_cast<float>(
                    static_cast<double>(edge.value_sum) + value
                );
                edge.visits += 1;
                edge.virtual_loss = static_cast<std::int16_t>(edge.virtual_loss - 1);
            } else {
                node.root_value_sum += value;
                node.root_visits += 1;
                node.root_virtual_loss -= 1;
            }
            value = -value;
        }
        selection.active = false;
    }
    // Every selection batch is backed up before the next selection call.
    // Releasing these paths here keeps a persistent forest bounded.
    forest->selections.clear();
    forest->selection_paths.clear();
    return 0;
}

MCTS_API std::int32_t mcts_forest_export_node(
    void* context,
    const std::int32_t node_id,
    std::int32_t* root_visits,
    double* root_value_sum,
    std::int32_t* root_virtual_loss,
    std::int32_t* visits,
    float* value_sums,
    std::int16_t* virtual_losses,
    float* total_counts,
    const std::int32_t edge_capacity
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (
        forest == nullptr || !valid_node(*forest, node_id) ||
        root_visits == nullptr || root_value_sum == nullptr ||
        root_virtual_loss == nullptr
    ) {
        return -1;
    }
    const auto& node = forest->nodes[static_cast<std::size_t>(node_id)];
    if (edge_capacity < node.edge_count || (node.edge_count > 0 && (
        visits == nullptr || value_sums == nullptr || virtual_losses == nullptr ||
        total_counts == nullptr
    ))) {
        return -2;
    }
    *root_visits = node.root_visits;
    *root_value_sum = node.root_value_sum;
    *root_virtual_loss = node.root_virtual_loss;
    const auto* edges = node_edges(*forest, node);
    for (std::int32_t index = 0; index < node.edge_count; ++index) {
        visits[index] = edges[index].visits;
        value_sums[index] = edges[index].value_sum;
        virtual_losses[index] = edges[index].virtual_loss;
        total_counts[index] = static_cast<float>(
            edges[index].visits + edges[index].virtual_loss
        );
    }
    return node.edge_count;
}

MCTS_API std::int32_t mcts_forest_node_count(void* context) noexcept {
    const auto* forest = static_cast<const NativeForest*>(context);
    return forest == nullptr ? -1 : static_cast<std::int32_t>(forest->nodes.size());
}

MCTS_API std::int32_t mcts_forest_edge_count(void* context) noexcept {
    const auto* forest = static_cast<const NativeForest*>(context);
    return forest == nullptr ? -1 : static_cast<std::int32_t>(forest->edges.size());
}

MCTS_API std::int32_t mcts_forest_export_all(
    void* context,
    std::int32_t* node_edge_begins,
    std::int32_t* node_edge_counts,
    std::int32_t* root_visits,
    double* root_value_sums,
    std::int32_t* root_virtual_losses,
    std::int32_t* edge_visits,
    float* edge_value_sums,
    std::int16_t* edge_virtual_losses,
    float* edge_total_counts,
    const std::int32_t node_capacity,
    const std::int32_t edge_capacity
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (
        forest == nullptr ||
        node_capacity < static_cast<std::int32_t>(forest->nodes.size()) ||
        edge_capacity < static_cast<std::int32_t>(forest->edges.size()) ||
        node_edge_begins == nullptr || node_edge_counts == nullptr ||
        root_visits == nullptr || root_value_sums == nullptr ||
        root_virtual_losses == nullptr ||
        (!forest->edges.empty() && (
            edge_visits == nullptr || edge_value_sums == nullptr ||
            edge_virtual_losses == nullptr || edge_total_counts == nullptr
        ))
    ) {
        return -1;
    }
    for (std::size_t index = 0; index < forest->nodes.size(); ++index) {
        const auto& node = forest->nodes[index];
        node_edge_begins[index] = node.edge_begin;
        node_edge_counts[index] = node.edge_count;
        root_visits[index] = node.root_visits;
        root_value_sums[index] = node.root_value_sum;
        root_virtual_losses[index] = node.root_virtual_loss;
    }
    for (std::size_t index = 0; index < forest->edges.size(); ++index) {
        const auto& edge = forest->edges[index];
        edge_visits[index] = edge.visits;
        edge_value_sums[index] = edge.value_sum;
        edge_virtual_losses[index] = edge.virtual_loss;
        edge_total_counts[index] = static_cast<float>(
            edge.visits + edge.virtual_loss
        );
    }
    return 0;
}

MCTS_API std::int32_t mcts_forest_compact(
    void* context,
    const std::int32_t* root_ids,
    const std::int32_t root_count,
    std::int32_t* old_to_new,
    const std::int32_t old_capacity
) noexcept {
    auto* forest = static_cast<NativeForest*>(context);
    if (
        forest == nullptr || root_count < 0 ||
        (root_count > 0 && root_ids == nullptr) ||
        old_to_new == nullptr ||
        old_capacity < static_cast<std::int32_t>(forest->nodes.size())
    ) {
        return -1;
    }
    if (!forest->selections.empty() || !forest->selection_paths.empty()) {
        return -2;
    }

    try {
        const auto old_node_count = static_cast<std::int32_t>(forest->nodes.size());
        std::fill(old_to_new, old_to_new + old_node_count, -1);
        std::vector<std::uint8_t> is_root(
            static_cast<std::size_t>(old_node_count),
            static_cast<std::uint8_t>(0)
        );
        std::vector<std::int32_t> pending;
        std::vector<NativeNode> compact_nodes;
        std::vector<NativeEdge> compact_edges;
        std::vector<RootSelectionState> compact_root_states;
        std::vector<std::int32_t> compact_parents;
        std::vector<std::int32_t> compact_parent_edges;

        auto enqueue = [&](const std::int32_t old_id) -> std::int32_t {
            auto& mapped = old_to_new[old_id];
            if (mapped >= 0) {
                return mapped;
            }
            mapped = static_cast<std::int32_t>(compact_nodes.size());
            compact_nodes.emplace_back();
            compact_root_states.emplace_back();
            compact_parents.push_back(-1);
            compact_parent_edges.push_back(-1);
            pending.push_back(old_id);
            return mapped;
        };

        for (std::int32_t index = 0; index < root_count; ++index) {
            const auto root_id = root_ids[index];
            if (!valid_node(*forest, root_id)) {
                return -3;
            }
            is_root[static_cast<std::size_t>(root_id)] = 1;
            enqueue(root_id);
        }

        std::size_t cursor = 0;
        while (cursor < pending.size()) {
            const auto old_id = pending[cursor++];
            const auto new_id = old_to_new[old_id];
            const auto& old_node = forest->nodes[static_cast<std::size_t>(old_id)];
            NativeNode new_node = old_node;
            new_node.edge_begin = (
                old_node.edge_count > 0
                ? static_cast<std::int32_t>(compact_edges.size())
                : -1
            );
            new_node.parent = compact_parents[static_cast<std::size_t>(new_id)];
            new_node.parent_edge = compact_parent_edges[static_cast<std::size_t>(new_id)];
            if (is_root[static_cast<std::size_t>(old_id)] != 0) {
                new_node.parent = -1;
                new_node.parent_edge = -1;
            }
            compact_nodes[static_cast<std::size_t>(new_id)] = new_node;

            for (std::int32_t edge_index = 0; edge_index < old_node.edge_count; ++edge_index) {
                NativeEdge edge = forest->edges[
                    static_cast<std::size_t>(old_node.edge_begin + edge_index)
                ];
                if (edge.child >= 0 && valid_node(*forest, edge.child)) {
                    const auto old_child = edge.child;
                    const auto new_child = enqueue(old_child);
                    edge.child = new_child;
                    if (
                        is_root[static_cast<std::size_t>(old_child)] == 0 &&
                        compact_parents[static_cast<std::size_t>(new_child)] < 0
                    ) {
                        compact_parents[static_cast<std::size_t>(new_child)] = new_id;
                        compact_parent_edges[static_cast<std::size_t>(new_child)] = edge_index;
                    }
                } else {
                    edge.child = -1;
                }
                compact_edges.push_back(edge);
            }
        }

        // Parents may have been assigned after a child placeholder was queued.
        for (std::size_t new_id = 0; new_id < compact_nodes.size(); ++new_id) {
            auto& node = compact_nodes[new_id];
            const auto old_id = pending[new_id];
            if (is_root[static_cast<std::size_t>(old_id)] != 0) {
                node.parent = -1;
                node.parent_edge = -1;
            } else {
                node.parent = compact_parents[new_id];
                node.parent_edge = compact_parent_edges[new_id];
            }
        }

        forest->nodes.swap(compact_nodes);
        forest->edges.swap(compact_edges);
        forest->root_states.swap(compact_root_states);
        forest->selections.clear();
        forest->selection_paths.clear();
        return static_cast<std::int32_t>(forest->nodes.size());
    } catch (...) {
        return -4;
    }
}

MCTS_API std::int32_t mcts_hash_positions(
    const std::uint8_t* boards,
    const std::int64_t board_stride_bytes,
    const std::int64_t board_bytes_per_position,
    const std::int16_t* legal_indices,
    const std::int64_t legal_stride,
    const std::int16_t* legal_counts,
    const std::int32_t size,
    std::uint64_t* output
) noexcept {
    if (!valid_size(size) || boards == nullptr || legal_indices == nullptr ||
        legal_counts == nullptr || output == nullptr || board_stride_bytes <= 0 ||
        board_bytes_per_position <= 0 || legal_stride <= 0) {
        return -1;
    }

    for (std::int32_t index = 0; index < size; ++index) {
        std::uint64_t first = UINT64_C(0x243f6a8885a308d3) ^
                              static_cast<std::uint64_t>(board_bytes_per_position);
        std::uint64_t second = UINT64_C(0x13198a2e03707344);
        hash_bytes(
            boards + static_cast<std::int64_t>(index) * board_stride_bytes,
            static_cast<std::size_t>(board_bytes_per_position),
            first,
            second
        );

        const auto legal_count = std::max<std::int32_t>(0, legal_counts[index]);
        const auto* legal_row = legal_indices + static_cast<std::int64_t>(index) * legal_stride;
        hash_bytes(
            reinterpret_cast<const std::uint8_t*>(&legal_count),
            sizeof(legal_count),
            first,
            second
        );
        hash_bytes(
            reinterpret_cast<const std::uint8_t*>(legal_row),
            static_cast<std::size_t>(legal_count) * sizeof(std::int16_t),
            first,
            second
        );
        output[2 * index] = avalanche(first ^ static_cast<std::uint64_t>(legal_count));
        output[2 * index + 1] = avalanche(second ^ first ^ UINT64_C(0xa4093822299f31d0));
    }
    return 0;
}

MCTS_API std::int32_t mcts_completed_q(
    const std::int32_t* visit_counts,
    const float* value_sums,
    const float* priors,
    const std::int32_t size,
    double raw_value,
    const std::int32_t use_mixed_value,
    const double q_range_floor,
    double* output
) noexcept {
    if (!valid_size(size) || visit_counts == nullptr || value_sums == nullptr ||
        priors == nullptr || output == nullptr) {
        return -1;
    }

    raw_value = std::clamp(raw_value, -1.0, 1.0);
    double visit_sum = 0.0;
    double visited_prior_sum = 0.0;
    double weighted_q_sum = 0.0;
    bool any_visited = false;
    const double tiny = std::numeric_limits<double>::min();

    for (std::int32_t index = 0; index < size; ++index) {
        const auto visits = visit_counts[index];
        if (visits > 0) {
            any_visited = true;
            const double q = -static_cast<double>(value_sums[index]) /
                             static_cast<double>(visits);
            output[index] = q;
            visit_sum += static_cast<double>(visits);
            const double safe_prior = std::max(static_cast<double>(priors[index]), tiny);
            visited_prior_sum += safe_prior;
            weighted_q_sum += safe_prior * q;
        } else {
            output[index] = 0.0;
        }
    }

    double completion_value = raw_value;
    if (use_mixed_value != 0 && any_visited && visit_sum > 0.0 && visited_prior_sum > 0.0) {
        const double weighted_q = weighted_q_sum / visited_prior_sum;
        completion_value = (raw_value + visit_sum * weighted_q) / (visit_sum + 1.0);
    }

    double q_min = std::numeric_limits<double>::infinity();
    double q_max = -std::numeric_limits<double>::infinity();
    for (std::int32_t index = 0; index < size; ++index) {
        if (visit_counts[index] <= 0) {
            output[index] = completion_value;
        }
        q_min = std::min(q_min, output[index]);
        q_max = std::max(q_max, output[index]);
    }

    const double q_range = q_max - q_min;
    if (q_range > 1.0e-8) {
        const double denominator = std::max(q_range, q_range_floor);
        for (std::int32_t index = 0; index < size; ++index) {
            output[index] = (output[index] - q_min) / denominator;
        }
    } else {
        std::fill(output, output + size, 0.0);
    }
    return 0;
}

MCTS_API std::int32_t mcts_improved_policy(
    const double* log_priors,
    const float* priors,
    const double* completed_q,
    const std::int32_t size,
    const double max_visit,
    const double c_visit,
    const double c_scale,
    float* output
) noexcept {
    if (!valid_size(size) || log_priors == nullptr || priors == nullptr ||
        completed_q == nullptr || output == nullptr) {
        return -1;
    }

    const double scale = (c_visit + max_visit) * c_scale;
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int32_t index = 0; index < size; ++index) {
        // q_score_scratch is float32 in the Python implementation. Preserve
        // that rounding before adding it to the float64 policy logits.
        const float q_score = static_cast<float>(completed_q[index] * scale);
        const double logit = log_priors[index] + static_cast<double>(q_score);
        maximum = std::max(maximum, logit);
    }

    double total = 0.0;
    for (std::int32_t index = 0; index < size; ++index) {
        const float q_score = static_cast<float>(completed_q[index] * scale);
        const double probability = std::exp(
            log_priors[index] + static_cast<double>(q_score) - maximum
        );
        total += probability;
    }

    if (!(total > 0.0) || !std::isfinite(total)) {
        double prior_total = 0.0;
        for (std::int32_t index = 0; index < size; ++index) {
            prior_total += static_cast<double>(priors[index]);
        }
        if (prior_total > 0.0) {
            for (std::int32_t index = 0; index < size; ++index) {
                output[index] = static_cast<float>(static_cast<double>(priors[index]) / prior_total);
            }
        } else {
            const float uniform = 1.0F / static_cast<float>(size);
            std::fill(output, output + size, uniform);
        }
        return 0;
    }

    for (std::int32_t index = 0; index < size; ++index) {
        const float q_score = static_cast<float>(completed_q[index] * scale);
        const double probability = std::exp(
            log_priors[index] + static_cast<double>(q_score) - maximum
        );
        output[index] = static_cast<float>(probability / total);
    }
    return 0;
}

MCTS_API std::int32_t mcts_select_root(
    const double* log_priors,
    const double* gumbel,
    const double* completed_q,
    const double* initial_visits,
    std::int32_t* visit_counts,
    float* total_counts,
    std::int16_t* virtual_losses,
    const std::int32_t size,
    const double considered_visit,
    const double c_visit,
    const double c_scale,
    const std::int32_t apply_virtual_loss
) noexcept {
    if (!valid_size(size) || log_priors == nullptr || gumbel == nullptr ||
        completed_q == nullptr || initial_visits == nullptr || visit_counts == nullptr ||
        total_counts == nullptr || virtual_losses == nullptr) {
        return -1;
    }

    double maximum_log_prior = -std::numeric_limits<double>::infinity();
    double max_fresh = 0.0;
    double min_fresh = std::numeric_limits<double>::infinity();
    bool has_eligible = false;
    for (std::int32_t index = 0; index < size; ++index) {
        maximum_log_prior = std::max(maximum_log_prior, log_priors[index]);
        const double fresh = std::max(
            0.0,
            static_cast<double>(total_counts[index]) - initial_visits[index]
        );
        max_fresh = std::max(max_fresh, fresh);
        min_fresh = std::min(min_fresh, fresh);
        has_eligible = has_eligible || (fresh == considered_visit);
    }

    const double eligible_visit = has_eligible ? considered_visit : min_fresh;
    const double scale = (c_visit + max_fresh) * c_scale;
    std::int32_t selected = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::int32_t index = 0; index < size; ++index) {
        const double fresh = std::max(
            0.0,
            static_cast<double>(total_counts[index]) - initial_visits[index]
        );
        if (fresh != eligible_visit) {
            continue;
        }
        const float q_score = static_cast<float>(completed_q[index] * scale);
        const double score = (log_priors[index] - maximum_log_prior) + gumbel[index] +
                             static_cast<double>(q_score);
        if (score > best_score) {
            best_score = score;
            selected = index;
        }
    }
    if (apply_virtual_loss != 0) {
        apply_virtual_visit(selected, visit_counts, total_counts, virtual_losses);
    }
    return selected;
}

MCTS_API std::int32_t mcts_final_action(
    const double* log_priors,
    const double* gumbel,
    const double* completed_q,
    const double* initial_visits,
    const std::int32_t* visit_counts,
    const std::int32_t size,
    const double c_visit,
    const double c_scale
) noexcept {
    if (!valid_size(size) || log_priors == nullptr || gumbel == nullptr ||
        completed_q == nullptr || initial_visits == nullptr || visit_counts == nullptr) {
        return -1;
    }

    double maximum_log_prior = -std::numeric_limits<double>::infinity();
    double most_visited = 0.0;
    for (std::int32_t index = 0; index < size; ++index) {
        maximum_log_prior = std::max(maximum_log_prior, log_priors[index]);
        most_visited = std::max(
            most_visited,
            std::max(0.0, static_cast<double>(visit_counts[index]) - initial_visits[index])
        );
    }

    const double scale = (c_visit + most_visited) * c_scale;
    std::int32_t selected = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::int32_t index = 0; index < size; ++index) {
        const double fresh = std::max(
            0.0,
            static_cast<double>(visit_counts[index]) - initial_visits[index]
        );
        if (fresh != most_visited) {
            continue;
        }
        const float q_score = static_cast<float>(completed_q[index] * scale);
        const double score = (log_priors[index] - maximum_log_prior) + gumbel[index] +
                             static_cast<double>(q_score);
        if (score > best_score) {
            best_score = score;
            selected = index;
        }
    }
    return selected;
}

MCTS_API std::int32_t mcts_soften_policy(
    const float* probabilities,
    const std::int32_t size,
    const double temperature,
    float* output
) noexcept {
    if (!valid_size(size) || probabilities == nullptr || output == nullptr) {
        return -1;
    }
    const double tiny = std::numeric_limits<double>::min();
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int32_t index = 0; index < size; ++index) {
        const double logit = std::log(std::max(static_cast<double>(probabilities[index]), tiny)) /
                             temperature;
        maximum = std::max(maximum, logit);
    }
    double total = 0.0;
    for (std::int32_t index = 0; index < size; ++index) {
        const double logit = std::log(std::max(static_cast<double>(probabilities[index]), tiny)) /
                             temperature;
        total += std::exp(logit - maximum);
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        std::copy(probabilities, probabilities + size, output);
        return 0;
    }
    for (std::int32_t index = 0; index < size; ++index) {
        const double logit = std::log(std::max(static_cast<double>(probabilities[index]), tiny)) /
                             temperature;
        output[index] = static_cast<float>(std::exp(logit - maximum) / total);
    }
    return 0;
}
