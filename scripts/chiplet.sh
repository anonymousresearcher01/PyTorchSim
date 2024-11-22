#!/bin/bash

if [ -z "$TORCHSIM_DIR" ]; then
    echo "Error: TORCHSIM_DIR environment variable is not set."
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 GEMM_PATH [ATTRIBUTE_FILE]"
    echo "  GEMM_PATH: Path to the gemm directory (e.g., ../../gemmx1024x1024x1024)"
    echo "  ATTRIBUTE_FILE: Optional path to the attribute file"
    exit 1
fi

GEMM_PATH="$1"
SIMULATOR_PATH="$TORCHSIM_DIR/PyTorchSimBackend/build/bin/Simulator"
GEMM_DIR_NAME=$(basename "$GEMM_PATH")
echo "GEMM Directory Name: $GEMM_DIR_NAME"

CONFIG_LIST=(
    "$TORCHSIM_DIR/PyTorchSimBackend/configs/systolic_ws_128x128_c2_simple_noc_tpuv2.json"
    "$TORCHSIM_DIR/PyTorchSimBackend/configs/systolic_ws_128x128_c2_booksim_tpuv2.json"
    "$TORCHSIM_DIR/PyTorchSimBackend/configs/systolic_ws_128x128_c2_chiplet_tpuv2.json"
    "$TORCHSIM_DIR/PyTorchSimBackend/configs/systolic_ws_128x128_c2_chiplet_tpuv2_xnuma.json"
)
shift
for ATTRIBUTE in "$@"; do
    ATTRIBUTE_FILE="$GEMM_PATH/attribute/$ATTRIBUTE"
    if [ ! -f "$ATTRIBUTE_FILE" ]; then
        echo "Error: Attribute file '$ATTRIBUTE_FILE' does not exist."
        exit 1
    fi
    ATTRIBUTE_FILES+=("$ATTRIBUTE_FILE")
done
MODELS_LIST="$GEMM_PATH/tile_graph.onnx"
ATTRIBUTE_PATH="$GEMM_PATH/attribute"

for CONFIG in "${CONFIG_LIST[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG" .json)

    for ATTRIBUTE_FILE in "${ATTRIBUTE_FILES[@]}"; do
        ATTRIBUTE_NAME=$(basename "$ATTRIBUTE_FILE")

        RESULTS_DIR="./results/$GEMM_DIR_NAME/$ATTRIBUTE_NAME"
        mkdir -p "$RESULTS_DIR"
        OUTPUT_FILE="$RESULTS_DIR/${CONFIG_NAME}_result.txt"

        # Run Simulator
        echo "$SIMULATOR_PATH" --config "$CONFIG" --models_list "$MODELS_LIST" --attributes_list "$ATTRIBUTE_PATH/$ATTRIBUTE_NAME"
        "$SIMULATOR_PATH" --config "$CONFIG" --models_list "$MODELS_LIST" --log_level trace --attributes_list "$ATTRIBUTE_PATH/$ATTRIBUTE_NAME" > "$OUTPUT_FILE" &

        echo "===== Simulation for $CONFIG completed. Results saved to $OUTPUT_FILE ====="
    done
done

wait