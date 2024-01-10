# MixtralLLM

Example of how to deploy the Mixtral LLM serverlessly using Modal

## Quick start

Clone the repository, and set your current working directory to the root of the repository:

```shell
git clone https://github.com/FreddyWordingham/MixtralLLM.git
cd MixtralLLM
```

Install the required dependencies:

```shell
poetry install
```

Test run the example web `scraper` function:

```shell
poetry run modal run scrape.py --url https://example.com/
```
