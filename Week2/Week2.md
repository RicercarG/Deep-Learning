# Lecture
* Alf:
	* Slides: [[02P - PyTorch training.pdf]]
	* Notebooks:
		* https://github.com/Atcold/NYU-DLSP20/blob/master/02-space_stretching.ipynb
		* https://github.com/Atcold/NYU-DLSP20/blob/master/04-spiral_classification.ipynb

* Yann:
	* [[001-intro.pdf]]
	* [[002-architectures.pdf]]

# Notes

* Five Steps Training:
```python
for (x, y) in dataset:

y_tilde = model(x) # generate a prediction
L = F = C(y_tilde, y) # compute the loss
optimiser.zero_grad() # zero gradient params
L.backward() # compute & accumulate gradient params
optimizer.step() # step in towards - gradient params
				 # logging
```