conda init zsh
conda init bash 
eval "$(conda shell.bash hook)"
conda activate jupyter
pip install -r requirements.txt
pip install gradio