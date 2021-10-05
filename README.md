# SIMC
This repo contains implementation of our scheme "SIMC: ML Inference Secure Against Malicious Clients at Semi-Honest Cost". The repository is built on \[[emp-toolkit/emp-sh2pc](https://github.com/emp-toolkit/emp-sh2pc)\].    

# Installation
1. Follow the installation steps of \[[emp-toolkit/emp-sh2pc](https://github.com/emp-toolkit/emp-sh2pc)\].
2. Clone this repo in the same parent directory as emp-sh2pc repo.

#Compilation
1. In the parent directory, go to emp-tool and run the following for multi-threading support:
```
cmake . -DTHREADING=ON
make -j
sudo make install
```
2. Do the same for emp-ot repository.
3. Finally, do the same in our (maxi) repository.

## Run
Run the following test files:
1. Run 'bin/test_msi_relu_final' to benchmark non-linear layers.
2. Run 'bin/test_msi_linearlayer' to benchmark fully-connected linear layer.
3. Run 'bin/test_msi_convlayer' to benchmark convolution linear layer.
4. Run 'bin/test_msi_average' to benchmark average pool layer.

## Contact
For any queries, kindly contact akashshah08@outlook.com.
