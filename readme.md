# LLParison
This is my WIP master's diploma project. It automatically tracks popular LLMs, and periodically conducts experiments on them with different tasks with different configurations, like measuring their accuracy in reading comprehension tests. The results are then provided on a Django website

# Access-restricted datasets
This project uses the following access-restricted datasets, which you will need to obtain before running the project properly:
- "bot_or_not.json" is a preprocessed dataset used for bot detection based on post history, raw data was obtained from the following link: https://zenodo.org/record/3692340

# Temporary notes for me:
- Docker deployments are currently broken and do not support new file structure. Fix them by:
- - setting the context to the root directory
- - copying only necessary files for the container
- - editing deployment.yaml to launch the appropriate file
- - Todo list for fixing containers:
- - - frontend container
- - - tracking container
- - - experiment running container
