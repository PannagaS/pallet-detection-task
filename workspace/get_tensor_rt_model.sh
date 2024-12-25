#!/bin/bash
source /opt/ros/humble/setup.bash
# Path to the YAML configuration file
CONFIG_FILE="/home/ws/model_config.yaml"


echo "=======================CREATING TENSOR RT FILE============================"
# Prompt the user for the set_half flag
echo "Would you like to use FP16 precision (set_half)?"
read -p "Enter 'yes' or 'no': " user_input
echo "Would you like to apply quantization (int8)?"
read -p "Enter 'yes' or 'no': " user_input_quant
# Determine the value for set_half based on user input
if [[ "$user_input" == "yes" || "$user_input" == "y" ]]; then
    SET_HALF="True"
elif [[ "$user_input" == "no" || "$user_input" == "n" ]]; then
    SET_HALF="False"
else
    echo "Invalid input. Exiting."
    exit 1
fi

if [[ "$user_input_quant" == "yes" || "$user_input_quant" == "y" ]]; then
    SET_INT_8="True"
elif [[ "$user_input_quant" == "no" || "$user_input_quant" == "n" ]]; then
    SET_INT_8="False"
else
    echo "Invalid input. Exiting."
    exit 1
fi

# Update the set_half flag in the YAML file
if command -v yq &> /dev/null; then
    yq eval ".set_half = \"$SET_HALF\"" -i "$CONFIG_FILE"
    echo "Updated set_half to \"$SET_HALF\" in $CONFIG_FILE"
    yq eval ".set_int8 = \"$SET_INT_8\"" -i "$CONFIG_FILE"
    echo "Updated set_int8 to  \"$SET_INT_8\"" in "$CONFIG_FILE"
else
    echo "Error: 'yq' is not installed. Please install it to use this script."
    exit 1
fi

# Parse the updated YAML configuration
PYTHON_SCRIPT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['python_script'])")
MODELS_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['models_dir'])")
SET_HALF=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['set_half'])")
SET_INT_8=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['set_int8'])")

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script $PYTHON_SCRIPT not found!"
    exit 1
fi

# Run the Python script with the arguments
echo "Running $PYTHON_SCRIPT with models_dir=$MODELS_DIR, set_half=$SET_HALF, and set_int8=$SET_INT_8 ..."
python3 "$PYTHON_SCRIPT" "$MODELS_DIR" "$SET_HALF" "$SET_INT_8"

 