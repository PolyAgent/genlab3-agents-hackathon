# VC Pilot

[Production app](https://vc-pilot.streamlit.app)
[Dev release](https://vc-pilot-dev.streamlit.app)

### Prerequisites

sync submodules

```
git submodule update --init --recursive
```

create virual environemnt and activate it

```
python3 -m venv .venv && source .venv/bin/activate
```

install python dependencies 

```
pip install -r requirments.txt
```

### Setting up streamlit 

You will need to provide `.streamlit/secrets.toml`. You can use provided example

```
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

and fill out the ENV values.

## Running the app

```
.venv/bin/streamlit run anthropic-app.py
```
