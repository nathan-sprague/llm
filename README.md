# LLM
Transfomer model. Trained on various datasets. Most recently this: https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/data/CC-MAIN-2013-20
Download 57M parameter model here: https://drive.google.com/file/d/1wuHtlXhQ8JmVbugbtAqp0BZxfkqsgzTF/view?usp=sharing


## Install

```sh

# Create a new virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS and Linux
# venv\Scripts\activate    # On Windows

# Install dependencies from requirements.txt
pip install -r requirements.txt


```


## Run

```sh
python3 run_transformer_token.py -i "The quick brown fox"
```

You'll get output that looks like

```txt
using cpu
The quick brown fox showed the same reaction as almost any other walking cat (Dig:????????) On his next step
```

## View the Model

```sh
pip install netron
```

```sh
netron llm_57m_gpt2.pt
```

and open a browser to [localhost:8080](http://localhost:8080) to see the weights

![Model weights](./llm_57m_gpt2.pt.png)


