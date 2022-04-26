# Setup
Run:
```bash
conda create --channel conda-forge --name simulation 'python=3.8' bokeh mkl mkl-service numpy pandas phantomjs scikit-learn scipy tensorflow
conda activate simulation
pip install -e .
```

# Running Simulations
Run:
```bash
bokeh serve simu/simulation/simulate_imdb_perceptron.py
```

Then open the browser to the address the above command tells you.
It should be something like: [http://localhost:5006/simulate_imdb_perceptron](http://localhost:5006/simulate_imdb_perceptron).

[keras-imdb]: https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
#   s i m u l a t i o n  
 