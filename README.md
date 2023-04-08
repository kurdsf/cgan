## cgan - a simple GAN written in C     
![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/kurdsf/cgan?style=plastic)
#### ATTENTION: This project is currently WIP.

### Requirements
This project uses the GNU Scientific Library (GSL). 
Installation instructions can be found [here](https://www.gnu.org/software/gsl/#downloading).
This project requires make and zip.

### Quick Start
```bash 
        git clone https://github.com/kurdsf/cgan.git 
        && cd cgan 
        && make unzip
        && make 
```

### Note about testing the GAN
The test on the GAN do not seem to work 100% of the time.
This is mainly due to the random number generator being initialized with time(NULL),
thus the initial weights are different each time you run the program.
If you encounter a failure, try again, and / or 
tweak the parameters in gan.h and / or nn.h.



The goal of the project is to generate MNIST Digits with 
a [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network).

The MNIST Dataset we use is taken from [here](https://pjreddie.com/projects/mnist-in-csv/).












