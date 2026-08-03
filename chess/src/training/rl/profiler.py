"""Console reporting for RL self-play profiling."""


def print_selfplay_profiler(selfplay_profile, selfplay_time):
    wall_selfplay_time = max(1e-8, float(selfplay_time))
    summed_worker_reference_time = float(
        selfplay_profile.get("mcts_search_many_time", 0.0) or 0.0
    )
    summed_worker_reference_time = max(1e-8, float(summed_worker_reference_time))

    search_many_time = max(0.0, float(selfplay_profile.get("mcts_search_many_time", 0.0) or 0.0))
    batch_expand_time = max(0.0, float(selfplay_profile.get("mcts_batch_expand_eval_time", 0.0) or 0.0))
    board_to_tensor_time = max(0.0, float(selfplay_profile.get("mcts_board_to_tensor_time", 0.0) or 0.0))
    nn_inference_time = max(0.0, float(selfplay_profile.get("mcts_nn_inference_time", 0.0) or 0.0))
    nn_calls = int(selfplay_profile.get("mcts_nn_inference_calls", 0) or 0)
    nn_batch_items = int(selfplay_profile.get("mcts_nn_inference_batch_items", 0) or 0)
    nn_legal_move_items = int(selfplay_profile.get("mcts_nn_legal_move_items", 0) or 0)
    nn_h2d_time = max(0.0, float(selfplay_profile.get("mcts_nn_h2d_time", 0.0) or 0.0))
    nn_gpu_forward_time = max(0.0, float(selfplay_profile.get("mcts_nn_gpu_forward_time", 0.0) or 0.0))
    nn_gpu_postprocess_time = max(0.0, float(selfplay_profile.get("mcts_nn_gpu_postprocess_time", 0.0) or 0.0))
    nn_d2h_time = max(0.0, float(selfplay_profile.get("mcts_nn_d2h_time", 0.0) or 0.0))
    search_root_setup_time = max(0.0, float(selfplay_profile.get("mcts_search_root_setup_time", 0.0) or 0.0))
    search_selection_time = max(0.0, float(selfplay_profile.get("mcts_search_selection_time", 0.0) or 0.0))
    search_backprop_time = max(0.0, float(selfplay_profile.get("mcts_search_backprop_time", 0.0) or 0.0))
    search_metadata_time = max(0.0, float(selfplay_profile.get("mcts_search_metadata_time", 0.0) or 0.0))
    board_materialize_time = max(0.0, float(selfplay_profile.get("mcts_board_materialize_time", 0.0) or 0.0))
    terminal_checks_time = max(0.0, float(selfplay_profile.get("mcts_terminal_checks_time", 0.0) or 0.0))
    batch_dedup_terminal_time = max(0.0, float(selfplay_profile.get("mcts_batch_expand_dedup_terminal_time", 0.0) or 0.0))
    batch_legal_moves_time = max(0.0, float(selfplay_profile.get("mcts_batch_expand_legal_moves_time", 0.0) or 0.0))
    batch_move_index_time = max(0.0, float(selfplay_profile.get("mcts_batch_expand_move_index_time", 0.0) or 0.0))
    batch_history_time = max(0.0, float(selfplay_profile.get("mcts_batch_expand_history_time", 0.0) or 0.0))
    batch_input_pack_time = max(0.0, float(selfplay_profile.get("mcts_batch_expand_input_pack_time", 0.0) or 0.0))
    batch_legal_index_pack_time = max(0.0, float(selfplay_profile.get("mcts_batch_expand_legal_index_pack_time", 0.0) or 0.0))
    batch_cpu_policy_time = max(0.0, float(selfplay_profile.get("mcts_batch_expand_cpu_policy_time", 0.0) or 0.0))
    batch_value_fanout_time = max(0.0, float(selfplay_profile.get("mcts_batch_expand_value_fanout_time", 0.0) or 0.0))
    policy_target_build_time = max(0.0, float(selfplay_profile.get("policy_target_build_time", 0.0) or 0.0))
    policy_target_postgame_time = max(0.0, float(selfplay_profile.get("policy_target_postgame_time", 0.0) or 0.0))
    move_selection_time = max(0.0, float(selfplay_profile.get("move_selection_time", 0.0) or 0.0))
    adjudication_time = max(0.0, float(selfplay_profile.get("adjudication_time", 0.0) or 0.0))
    syzygy_time = max(0.0, float(selfplay_profile.get("syzygy_time", 0.0) or 0.0))
    central_requests = int(selfplay_profile.get("mcts_central_inference_requests", 0) or 0)
    central_batch_items = int(selfplay_profile.get("mcts_central_inference_server_batch_items", 0) or 0)
    central_remote_wait_time = max(
        0.0,
        float(selfplay_profile.get("mcts_central_inference_remote_wait_time", 0.0) or 0.0),
    )
    central_request_put_time = max(
        0.0,
        float(selfplay_profile.get("mcts_central_inference_request_put_time", 0.0) or 0.0),
    )
    central_server_queue_wait_time = max(
        0.0,
        float(selfplay_profile.get("mcts_central_inference_server_queue_wait_time", 0.0) or 0.0),
    )
    central_descriptor_queue_wait_time = max(
        0.0,
        float(
            selfplay_profile.get(
                "mcts_central_inference_server_descriptor_queue_wait_time",
                0.0,
            ) or 0.0
        ),
    )
    central_batch_coalesce_wait_time = max(
        0.0,
        float(
            selfplay_profile.get(
                "mcts_central_inference_server_batch_coalesce_wait_time",
                0.0,
            ) or 0.0
        ),
    )
    central_server_forward_time = max(
        0.0,
        float(selfplay_profile.get("mcts_central_inference_server_forward_time", 0.0) or 0.0),
    )
    central_server_concat_time = max(
        0.0,
        float(selfplay_profile.get("mcts_central_inference_server_concat_time", 0.0) or 0.0),
    )
    central_server_h2d_time = max(
        0.0,
        float(selfplay_profile.get("mcts_central_inference_server_h2d_time", 0.0) or 0.0),
    )
    central_server_d2h_time = max(
        0.0,
        float(selfplay_profile.get("mcts_central_inference_server_d2h_time", 0.0) or 0.0),
    )
    central_server_total_time = max(
        0.0,
        float(selfplay_profile.get("mcts_central_inference_server_total_time", 0.0) or 0.0),
    )

    batch_expand_capped = min(batch_expand_time, search_many_time)
    nn_inference_capped = min(nn_inference_time, batch_expand_capped)
    board_to_tensor_capped = min(board_to_tensor_time, batch_expand_capped)
    batch_known_time = (
        nn_inference_capped
        + board_to_tensor_capped
        + batch_dedup_terminal_time
        + board_materialize_time
        + terminal_checks_time
        + batch_legal_moves_time
        + batch_move_index_time
        + batch_history_time
        + batch_input_pack_time
        + batch_legal_index_pack_time
        + batch_cpu_policy_time
        + batch_value_fanout_time
    )
    batch_other_time = max(0.0, batch_expand_capped - batch_known_time)
    search_known_outside_expand_time = (
        search_root_setup_time
        + search_selection_time
        + search_backprop_time
        + search_metadata_time
    )
    search_other_time = max(0.0, search_many_time - batch_expand_capped - search_known_outside_expand_time)
    worker_nn_wait_share_pct_display = 0.0
    if search_many_time > 0.0:
        worker_nn_wait_share_pct_display = 100.0 * (nn_inference_capped / search_many_time)
    worker_nn_wait_share_pct_display = max(
        0.0,
        min(100.0, float(worker_nn_wait_share_pct_display)),
    )
    average_batch_size_display = float(nn_batch_items / nn_calls) if nn_calls > 0 else 0.0
    average_legal_moves_display = float(nn_legal_move_items / nn_batch_items) if nn_batch_items > 0 else 0.0
    inference_per_batch_ms_display = (
        1000.0 * float(nn_inference_capped) / float(nn_calls) if nn_calls > 0 else 0.0
    )
    inference_per_position_ms_display = (
        1000.0 * float(nn_inference_capped) / float(nn_batch_items) if nn_batch_items > 0 else 0.0
    )

    h2d_raw = max(0.0, float(nn_h2d_time))
    gpu_forward_raw = max(0.0, float(nn_gpu_forward_time))
    gpu_postprocess_raw = max(0.0, float(nn_gpu_postprocess_time))
    d2h_raw = max(0.0, float(nn_d2h_time))
    gpu_stage_total = h2d_raw + gpu_forward_raw + gpu_postprocess_raw + d2h_raw
    transfer_total_raw = h2d_raw + d2h_raw

    def _pct_of_gpu_stages(value):
        return 100.0 * float(value) / max(1e-8, gpu_stage_total)

    gpu_bottleneck = "mixed"
    if gpu_stage_total > 0.0:
        if gpu_forward_raw >= max(transfer_total_raw * 1.25, gpu_stage_total * 0.55):
            gpu_bottleneck = "compute-bound"
        elif transfer_total_raw >= max(gpu_forward_raw * 0.95, gpu_stage_total * 0.45):
            gpu_bottleneck = "transfer-bound"
        elif gpu_postprocess_raw >= max(gpu_stage_total * 0.20, transfer_total_raw * 0.85):
            gpu_bottleneck = "postprocess-bound"

    def _pct_of_search(value):
        return 100.0 * float(value) / max(1e-8, search_many_time)

    def _pct_of_batch_expand(value):
        return 100.0 * float(value) / max(1e-8, batch_expand_capped)

    top_level_components = [
        ("search_many", search_many_time),
        ("policy_target_build", policy_target_build_time),
        ("policy_target_postgame", policy_target_postgame_time),
        ("move_selection", move_selection_time),
        ("adjudication", adjudication_time),
        ("syzygy", syzygy_time),
    ]
    total_selfplay_sum_time = sum(float(value) for _, value in top_level_components)

    def _pct_of_selfplay_total(value):
        return 100.0 * float(value) / max(1e-8, total_selfplay_sum_time)

    def _print_total_line(label, value):
        print(f"   - {label:<22} {float(value):7.2f}s ({_pct_of_selfplay_total(value):5.1f}% selfplay_total)")

    def _print_search_line(label, value):
        print(f"     - {label:<20} {float(value):7.2f}s ({_pct_of_search(value):5.1f}% search_many)")

    def _print_batch_expand_line(label, value):
        print(f"       - {label:<18} {float(value):7.2f}s ({_pct_of_batch_expand(value):5.1f}% _batch_expand_eval)")

    summed_vs_wall_ratio = summed_worker_reference_time / wall_selfplay_time
    print("Self-play profiler (sumowany czas workerow):")
    print(
        "   "
        f"wall_clock={wall_selfplay_time:.2f}s, "
        f"reference_sum={summed_worker_reference_time:.2f}s, "
        f"sum/wall={summed_vs_wall_ratio:.2f}x"
    )
    print("")
    print(f"   Sekcja A: Self-play total ({total_selfplay_sum_time:.2f}s, 100.0%):")
    _print_total_line("search_many", search_many_time)
    _print_search_line("root_setup", search_root_setup_time)
    _print_search_line("selection", search_selection_time)
    _print_search_line("_batch_expand_eval", batch_expand_capped)
    _print_batch_expand_line("dedup_terminal", batch_dedup_terminal_time)
    _print_batch_expand_line("board_materialize", board_materialize_time)
    _print_batch_expand_line("terminal_checks", terminal_checks_time)
    _print_batch_expand_line("legal_moves", batch_legal_moves_time)
    _print_batch_expand_line("move_index", batch_move_index_time)
    _print_batch_expand_line("history_fetch", batch_history_time)
    _print_batch_expand_line("input_pack", batch_input_pack_time)
    _print_batch_expand_line("nn_inference", nn_inference_capped)
    _print_batch_expand_line("board_to_tensor", board_to_tensor_capped)
    _print_batch_expand_line("legal_index_pack", batch_legal_index_pack_time)
    _print_batch_expand_line("cpu_policy_expand", batch_cpu_policy_time)
    _print_batch_expand_line("value_fanout", batch_value_fanout_time)
    _print_batch_expand_line("batch_expand_other", batch_other_time)
    _print_search_line("backprop", search_backprop_time)
    _print_search_line("metadata", search_metadata_time)
    _print_search_line("search_other", search_other_time)
    _print_total_line("policy_target_build", policy_target_build_time)
    _print_total_line("policy_target_postgame", policy_target_postgame_time)
    _print_total_line("move_selection", move_selection_time)
    _print_total_line("adjudication", adjudication_time)
    _print_total_line("syzygy", syzygy_time)
    print("")
    print(f"   Sekcja B: GPU (raw_stage_sum={gpu_stage_total:.2f}s, inference_wall={nn_inference_capped:.2f}s):")
    print(
        f"   {'worker NN wait share':<24} {worker_nn_wait_share_pct_display:7.2f}% "
        "(not hardware GPU use)"
    )
    print(f"   {'average_batch_size':<24} {average_batch_size_display:7.2f} pos/batch")
    print(f"   {'average_legal_moves':<24} {average_legal_moves_display:7.2f} legal/pos")
    print(f"   {'h2d_transfer':<24} {h2d_raw:7.2f}s ({_pct_of_gpu_stages(h2d_raw):5.1f}% gpu_stages)")
    print(f"   {'gpu_forward':<24} {gpu_forward_raw:7.2f}s ({_pct_of_gpu_stages(gpu_forward_raw):5.1f}% gpu_stages)")
    print(f"   {'gpu_postprocess':<24} {gpu_postprocess_raw:7.2f}s ({_pct_of_gpu_stages(gpu_postprocess_raw):5.1f}% gpu_stages)")
    print(f"   {'d2h_transfer':<24} {d2h_raw:7.2f}s ({_pct_of_gpu_stages(d2h_raw):5.1f}% gpu_stages)")
    print(f"   {'inference_per_batch':<24} {inference_per_batch_ms_display:7.2f} ms/batch")
    print(f"   {'inference_per_position':<24} {inference_per_position_ms_display:7.2f} ms/pos")
    print(f"   {'gpu_bottleneck':<24} {gpu_bottleneck}")
    queue_wait_total_s = max(0.0, float(selfplay_profile.get('queue_wait_total_s', 0.0) or 0.0))
    print("")
    print(f"   Sekcja C: Queue/IPC ({queue_wait_total_s:.2f}s lacznego czekania):")
    print(f"   {'queue_wait_time_ms':<24} {float(selfplay_profile.get('queue_wait_time_ms', 0.0) or 0.0):7.2f} ms/event")
    print(f"   {'queue_wait_events':<24} {int(selfplay_profile.get('queue_wait_events', 0) or 0)}")
    if central_requests > 0:
        central_avg_batch = float(central_batch_items / central_requests)
        central_remote_ms = 1000.0 * central_remote_wait_time / max(1, central_requests)
        central_put_ms = 1000.0 * central_request_put_time / max(1, central_requests)
        central_queue_ms = 1000.0 * central_server_queue_wait_time / max(1, central_requests)
        central_descriptor_queue_ms = (
            1000.0 * central_descriptor_queue_wait_time / max(1, central_requests)
        )
        central_batch_coalesce_ms = (
            1000.0 * central_batch_coalesce_wait_time / max(1, central_requests)
        )
        central_concat_ms = 1000.0 * central_server_concat_time / max(1, central_requests)
        central_h2d_ms = 1000.0 * central_server_h2d_time / max(1, central_requests)
        central_forward_ms = 1000.0 * central_server_forward_time / max(1, central_requests)
        central_d2h_ms = 1000.0 * central_server_d2h_time / max(1, central_requests)
        central_total_ms = 1000.0 * central_server_total_time / max(1, central_requests)
        central_queue_share = 100.0 * central_server_queue_wait_time / max(1e-8, central_remote_wait_time)
        central_forward_share = 100.0 * central_server_forward_time / max(1e-8, central_remote_wait_time)
        print("")
        print("   Sekcja D: Central inference server (srednie na request):")
        print(f"   {'central_requests':<32} {central_requests}")
        print(f"   {'central_avg_batch':<32} {central_avg_batch:7.2f} pos/batch")
        print(f"   {'worker_remote_wait':<32} {central_remote_ms:7.2f} ms/request")
        print(f"   {'worker_queue_put':<32} {central_put_ms:7.2f} ms/request")
        print(f"   {'server_queue_total':<32} {central_queue_ms:7.2f} ms/request ({central_queue_share:5.1f}% remote_wait)")
        print(f"   {'descriptor_queue_backlog':<32} {central_descriptor_queue_ms:7.2f} ms/request")
        print(f"   {'batch_coalesce_wait':<32} {central_batch_coalesce_ms:7.2f} ms/request")
        print(f"   {'server_concat_ipc':<32} {central_concat_ms:7.2f} ms/request")
        print(f"   {'server_h2d':<32} {central_h2d_ms:7.2f} ms/request")
        print(f"   {'server_forward':<32} {central_forward_ms:7.2f} ms/request ({central_forward_share:5.1f}% remote_wait)")
        print(f"   {'server_d2h':<32} {central_d2h_ms:7.2f} ms/request")
        print(f"   {'server_total':<32} {central_total_ms:7.2f} ms/request")
