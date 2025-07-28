sudo apt upgrade
sudo apt install uv

uv venv
source .venv/bin/activate

uv pip install -e .

source .venv/bin/activate

echo "PROJECT_ROOT=\"$(pwd)\"" > .env
