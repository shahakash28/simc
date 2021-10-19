# SIMC
This repo contains implementation of our scheme "SIMC: ML Inference Secure Against Malicious Clients at Semi-Honest Cost". The repository is built on \[[emp-toolkit/emp-sh2pc](https://github.com/emp-toolkit/emp-sh2pc)\].    

# Installation
1. Follow the installation steps of \[[emp-toolkit/emp-sh2pc](https://github.com/emp-toolkit/emp-sh2pc)\].
2. Clone this repo in the same parent directory as emp-sh2pc repo.
3. Install SEAL 3.64 
   a. Clone \[[SEAL](https://github.com/microsoft/SEAL.git)\] repo.
   b. Execute 
   ```
   cd SEAL
   git checkout 3.6.4
   mkdir build && cd build
   cmake ..
   make -j
   sudo make install
   ```

#Compilation
1. In the parent directory, go to emp-tool and run the following for multi-threading support:
```
cmake . -DTHREADING=ON
make -j
sudo make install
```
2. Do the same for emp-ot repository.
3. Finally, do the same in our (simc) repository.

## Run
Run the following test files:
Fully-connected Layer: In one terminal run `bin/test_msi_linearlayer 1 0.0.0.0 <port_no> 44 <neural_network>` and in other terminal run `bin/test_msi_linearlayer 1 <server_ip_address> <port_no> 44 <neural_network>`. 
\item Convolution Layer: In one terminal run `bin/test_msi_convlayer 1 0.0.0.0 <port_no> 44 <neural_network>` and in other terminal run `bin/test_msi_convlayer 1 <server_ip_address> <port_no> 44 <neural_network>`.
\item Non-Linear Layer (ReLU): In one terminal run `bin/test_msi_relu_final 1 0.0.0.0 <port_no> 44 <neural_network> 0 0 <num_threads>` and in other terminal run `bin/test_msi_relu_final 2 <server_ip_address> <port_no> 44 <neural_network> 0 0 <num_threads>'`.
\item Average Pool Layer: In one terminal run `bin/test_msi_average 1 0.0.0.0 <port_no> 44 <neural_network>` and in other terminal run `bin/test_msi_average 1 <server_ip_address> <port_no> 44 <neural_network>`.

Here, the first parameters 1 and 2 denote the ID of the participating party. <server_ip_address> denotes the ip address of the server machine and set <neural_network>=1 for MNIST and <neural_network>=2 for CIFAR-10.

## Contact
For any queries, kindly contact akashshah08@outlook.com.
