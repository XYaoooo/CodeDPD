export TOKENIZERS_PARALLELISM=True
accelerate launch \
        --config_file "./config/deepspeed_zero2.yaml" \
    inference.py