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

Test run the example web `scraper` function over a comma separated list of urls:

```shell
poetry run modal run scrape.py --urls https://example.com,https://github.com
```

Or run the `Mixtral` LLM:

````shell
poetry run modal run -q mixtral.py
``

> Note! We use the `-q` flag to stream the output out as it is generated.


## Deployment

Deploying the `scraper` function will run it on a schedule, scraping the supplied urls:

```shell
poetry run modal deploy scrape.py --urls https://example.com,https://github.com
````
