# Yule-Simon
Markov Chain Monte Carlo sampler for inference of latent mean and volatility in financial time series. Generative model is based on a latent Yule-Simon process which controls regime switching and models power-law distributed volatility clustering with long term memory effects. Model parameters are automatically learned from the data including the number of volatility clusters. Each volatility clustering is modeled as conditionally Gaussian, meaning that standardized log-returns will follow the unit Normal distribution as shown below:

![MSFT Demo](demos/MSFT.gif)

Getting Started:

Clone or download repository and start [here](https://github.com/ahensley3/Yule-Simon/tree/master/applications/0_quick_start)