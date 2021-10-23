# SIMC
This repo contains implementation of our scheme "SIMC: ML Inference Secure Against Malicious Clients at Semi-Honest Cost". The repository is built on \[[emp-toolkit/emp-sh2pc](https://github.com/emp-toolkit/emp-sh2pc)\].    

Disclaimer: This repository is a proof-of-concept prototype. 

TODO: Update the licensing information.

# Installation
1. Create parent directory `msi-code`
   ```
   mkdir msi-code && cd msi-code
   ```
2. To install Eigen3 do:
   ```
   sudo apt-get update -y
   sudo apt-get install -y libeigen3-dev
   ```
3. Follow the installation steps of \[[emp-toolkit/emp-sh2pc](https://github.com/emp-toolkit/emp-sh2pc)\].
4. Clone this repo in the parent directory `msi-code`.
5. Install SEAL 3.64 
   a. Clone \[[SEAL](https://github.com/microsoft/SEAL.git)\] repo in the parent directory `msi-code`.
   b. Execute 
   ```
   cd SEAL
   git checkout 3.6.4
   mkdir build && cd build
   cmake ..
   make -j
   sudo make install
   ```

# Compilation
1. In `msi-code`, go to `emp-tool` and do `git checkout df363bf30b56c48a12c352845efa3a4d8f75b388`.
2. Next, go to `emp-ot` in `msi-code` and do `git checkout 3b21d6314cb1e7d8dbb9bb1f1ed80261738e4f4c`.
3. For multi-threading support, go to `emp-tool` and run the following:
```
cmake . -DTHREADING=ON
make -j
sudo make install
```
2. Do the same for emp-ot repository.
3. Finally, do the same in our (simc) repository.

## Run Neuralnet Benchmarks
Run the following test files from path `msi-code/simc`:
1. Fully-connected Layer: In one terminal run `bin/test_msi_linearlayer 1 0.0.0.0 <port_no> 44 <neural_network>` and in other terminal run `bin/test_msi_linearlayer 2 <server_ip_address> <port_no> 44 <neural_network>`. 

2. Convolution Layer: In one terminal run `bin/test_msi_convlayer 1 0.0.0.0 <port_no> 44 <neural_network>` and in other terminal run `bin/test_msi_convlayer 2 <server_ip_address> <port_no> 44 <neural_network>`.

3. Non-Linear Layer (ReLU): In one terminal run `bin/test_msi_relu_final 1 0.0.0.0 <port_no> 44 <neural_network> 0 0 <num_threads>` and in other terminal run `bin/test_msi_relu_final 2 <server_ip_address> <port_no> 44 <neural_network> 0 0 <num_threads>'`.

4. Average Pool Layer: In one terminal run `bin/test_msi_average 1 0.0.0.0 <port_no> 44 <neural_network>` and in other terminal run `bin/test_msi_average 2 <server_ip_address> <port_no> 44 <neural_network>`.

Here, the first parameters 1 and 2 denote the ID of the participating party. <server_ip_address> denotes the ip address of the server machine and set <neural_network>=1 for MNIST and <neural_network>=2 for CIFAR-10.

Examples:
```
Fully connected Layer:
Terminal 1: bin/test_msi_linearlayer 1 0.0.0.0 31000 44 1
Terminal 2: bin/test_msi_linearlayer 2 <server_ip_address> 31000 44 1

Convolution Layer:
Terminal 1: bin/test_msi_convlayer 1 0.0.0.0 31000 44 1
Terminal 2: bin/test_msi_convlayer 2 <server_ip_address> 31000 44 1

Non-Linear Layer (ReLU):
Terminal 1: bin/test_msi_relu_final 1 0.0.0.0 31000 44 1 0 0 8
Terminal 2: bin/test_msi_relu_final 2 <server_ip_address> 31000 44 1 0 0 8

Average Pool Layer:
Terminal 1: bin/test_msi_average 1 0.0.0.0 31000 44 1
Terminal 2: bin/test_msi_average 2 <server_ip_address> 31000 44 1
```

## Run Neuralnet Micro-benchmarks
```
Terminal 1: bin/test_msi_microbenchmark 1 0.0.0.0 31000 44 <benchmark_choice> <num_relus> <#threads> 
Terminal 2: bin/test_msi_microbenchmark 2 <server_ip_address> 31000 44 <benchmark_choice> <num_relus> <#threads> 
```
Input Parameters:
1. <server_ip_address>: IP Address of Server.
2. <benchmark_choice>: 0 - ReLU6, 1 - ReLU.
3. <num_relus>: Number of ReLUs
4. <#threads>: Number of threads

If <num_relus> <=2, set <#threads>=1,

else if <num_relus> <=4, set <#threads>=2,

else if <num_relus> <=16, set <#threads>=4,

else if <num_relus> >16, set <#threads>=8.

Example:
```
Terminal 1: bin/test_msi_microbenchmark 1 0.0.0.0 31000 44 0 16384 8 
Terminal 2: bin/test_msi_microbenchmark 2 <server_ip_address> 31000 44 0 16384 8 
```

## Contact
For any queries, kindly contact akashshah08@outlook.com.
