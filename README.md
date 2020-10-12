# Supplementary material for the "Logging in Reinforcement Learning frameworks" post on Neptune Blog

Read the blog post first here _(link to be added)_. In each directory, you'll find the logger code with the run example. Remember to install each framework following up-to-date instructions on its website. Moreover, remember to set the environment variable `NEPTUNE_API_TOKEN` to a valid API token, [read more here](https://docs.neptune.ai/security-and-privacy/api-tokens/how-to-find-and-set-neptune-api-token.html). Below are extra instructions concerning the example for each framework.

## Acme

Run `python run_d4pg.py --help` to show available (and required) parameters.

## RLlib

Note that this RLlib example will run a grid of three experiments with different learning rates. This is very useful for the hyper-parameter search.

Replace `<namespace/project_name>` with a valid qualified name of your project (this is most probably: your user name / a project name e.g. `jankowalski/sandbox`).

## Spinning Up

Remember that in the SpinUp case you have to replace the `logx.py` in the Spinning Up code with the one supplied in this repository.

Replace `<namespace/project_name>` with a valid qualified name of your project (this is most probably: your user name / a project name e.g. `jankowalski/sandbox`).

## Stable Baselines

Replace `<namespace/project_name>` with a valid qualified name of your project (this is most probably: your user name / a project name e.g. `jankowalski/sandbox`).
