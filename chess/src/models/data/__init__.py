from src.models.data.registry import (
    INPUT_ENCODERS,
    IL_DATA_PIPELINES,
    POLICY_CODECS,
    get_il_data_pipeline,
    get_il_data_pipeline_for_config,
    get_input_encoder,
    get_policy_codec,
    validate_data_contract,
)


def process_pgn_files(pgn_files, config):
    pipeline = get_il_data_pipeline_for_config(config)
    return pipeline.process_pgn_files(pgn_files, config)


def create_dataloaders(metadata, config):
    pipeline = get_il_data_pipeline_for_config(config)
    return pipeline.create_dataloaders(metadata, config)

__all__ = [
    "INPUT_ENCODERS",
    "IL_DATA_PIPELINES",
    "POLICY_CODECS",
    "get_input_encoder",
    "get_policy_codec",
    "get_il_data_pipeline",
    "get_il_data_pipeline_for_config",
    "validate_data_contract",
    "create_dataloaders",
    "process_pgn_files",
]
