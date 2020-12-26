# poetry_generator

Generates poetry of famous poets by beginning (space= ' ' can be used too, as an easy beginning)

Now the application can generate textes that rely on poems of:
* Shakespeare
* Pushkin


How to run:

1. Clone the repository
2. Create conda env:

    ENV_NAME=poetry_generator
    conda create -y -n $ENV_NAME python=3.7
    conda activate $ENV_NAME
    pip install -r requirements.txt
    ipython kernel install --user --name=$ENV_NAME

3. Run next code in console:

	export FLASK_APP=poetry_generator.py
	flask run

4. Open local URL
5. Enjoy! :)
