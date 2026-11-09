# Chemical Grokking SandBox
- This Repo documents My work to try and induce some form of chemical superstructure within a basic Transformer model
- It is Primarily Inspired by Nanda et al. (see nanda_original_reproduction) and Liu et al. (see omnigrok_original_reproduction)
- The core idea was to see if a very simple mathematical description of certain chemical structures, like Hückewl matrix descriptions of conjugated chemicals, could generate a large dataset and 'grok' / generalize in a way similar to that detailed on Nanda et al.
- Omnigrok's own data for possible grokking on the the QM9 dataset was also key evidence that this idea might be able to work
## Key Problems
- The key problem identified here is that Nanda et al. had integers as input and output, that made the space easily discrete. Furthermore, there where only 52 possible frequencies to scan over for his FFT analysis for 'reverse-engineeering' his proposed clock algorithm
- The QM9 dataset by Liu et al. was continuous and even more mathematically ambiguous than the Hückel matrices which I have attempted, as the inputs were SMILEs of various chemicals and their polarizablitiy as the output
- I have reproduced the Omnigrok code and basic interpretations have only revealed PCA3 mapping as an identification of Nitrile groups or not.
## Needed Extensions
- I believe to fulfill the mechanistic interpretability goals of this project, a full reverse-engineering process must take place on a transformer trained to take any matrix and guess it's eigenvalues.
- This process has many chemically relevant functions regarding chemical structure, and could perhaps tell us something about how a model might be able to guess chemically relevant phenomena might emerge from simple mathematical training
## Key limitations
- Even if this project were to have worked, I do not believe this 'bottom-up' research will actually reveal the true 'platonic' chemical structure that I hypothesize forms within the weights of a big enough transformer model exposed to chemistry
- A promising direction is using the weight modification study with Astra to a ChemBERT, or other chemical foundation model and seeing if Astra can recognize the structure of the weights and help us see if chemical structure is generalized in a scieintifically relevant way.


