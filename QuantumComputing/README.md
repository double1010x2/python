# **Quantum Image process(QIMP)**
 ## Introduce:  
*  **Goal**
   * Encode MNIST image by 2D image to another latent space by QIMP in order to reach dimension reduction   
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
   * epochs = 20
   * lr     = 0.002
   * batch_size = randomly switch between 64 and 32  
*  **Results**
   * Accuracy of 0: 0.735294
   * Accuracy of 1: 0.978261
   * Accuracy of 2: 0.816327
   * Accuracy of 3: 0.868421
   * Accuracy of 4: 0.625000
   * Accuracy of 5: 0.485714
   * Accuracy of 6: 0.766667
   * Accuracy of 7: 0.697674
   * Accuracy of 8: 0.538462
   * Accuracy of 9: 0.347826 
*  **Reference**
   - NEQR paper: https://doi.org/10.1007/s11128-013-0567-z
   - MNIST pytorch example: https://clay-atlas.com/blog/2019/10/19/pytorch-教學-getting-start-訓練分類器-mnist/
## Describe
*  **NEAR simple example**
   *![NEQR example]() 
*  **Mearsurement(with only numpy)**
   *![measurement]()  
  

