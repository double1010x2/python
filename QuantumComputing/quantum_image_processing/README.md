# **Quantum Image process(QIMP)**
 ## Introduce:  
*  **Goal**
   * Encode MNIST image by 2D image to another latent space by QIMP in order to reduce feature dimension from 784 to 384  
   * Use latent space to classify the MNIST data
   * Without using quantum-computing software in the webside in order to forbid error correct on quantum process
*  **Algorithm**
   * NEQR (Novel Enhanced Quantum Representation)
*  **Data**  
   * MNIST
*  **Parameter**
   * 10 qubits to describe image localtion
   * 4  qubits to describe image value
   * total qubits 10+3=13
   * epochs = 40
   * lr     = 0.005
   * batch_size = 64 -> 128  
*  **Results**
   * Accuracy of 0: 1.000000
   * Accuracy of 1: 1.000000
   * Accuracy of 2: 0.979592
   * Accuracy of 3: 0.973684
   * Accuracy of 4: 0.975000
   * Accuracy of 5: 0.914286
   * Accuracy of 6: 0.966667
   * Accuracy of 7: 0.976744
   * Accuracy of 8: 0.974359
   * Accuracy of 9: 0.978261
*  **Reference**
   [NEQR paper](https://doi.org/10.1007/s11128-013-0567-z)
   [MNIST pytorch example](https://clay-atlas.com/blog/2019/10/19/pytorch-教學-getting-start-訓練分類器-mnist/)
   [MLP-Mixer](https://arxiv.org/abs/2105.01601)
## Describe
*  **NEAR simple example**
   - ![NEQR example](NEQR_example.png) 
*  **Mearsurement(with only numpy)**
   - ![measurement](network_structure.png) 
*  **Training results**
   - ![loss(MSE)_to_epoch](loss_epoch.png)
*  **Prediction results(on 10000 images)**
   - ![precision_for_each_number](precision_mnist.png)
  

