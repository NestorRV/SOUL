# SOUL
### Scala Oversampling and Undersampling Library

Included algorithms for undersampling:

* **Balance Cascade.** Original paper: "Exploratory Undersampling for Class-Imbalance Learning" by Xu-Ying Liu, Jianxin Wu and Zhi-Hua Zhou.

* **Class Purity Maximization algorithm.** Original paper: "An Unsupervised Learning Approach to Resolving the soul.data Imbalanced Issue in Supervised Learning Problems in Functional Genomics" by Kihoon Yoon and Stephen Kwek.

* **ClusterOSS.** Original paper: "ClusterOSS: a new undersampling method for soul learning." by Victor H Barella, Eduardo P Costa and André C. P. L. F. Carvalho.

* **Condensed Nearest Neighbor decision rule.** Original paper: "The Condensed Nearest Neighbor Rule" by P. Hart.

* **Easy Ensemble.** Original paper: "Exploratory Undersampling for Class-Imbalance Learning" by Xu-Ying Liu, Jianxin Wu and Zhi-Hua Zhou.

* **Edited Nearest Neighbour rule.** Original paper: "Asymptotic Properties of Nearest Neighbor Rules Using Edited soul.data" by Dennis L. Wilson.

* **Evolutionary Undersampling.** Original paper: "Evolutionary Under-Sampling for Classification with Imbalanced soul.data Sets: Proposals and Taxonomy" by Salvador Garcia and Francisco Herrera.

* **Instance Hardness Threshold.** Original paper: "An Empirical Study of Instance Hardness" by Michael R. Smith, Tony Martinez and Christophe Giraud-Carrier.

* **Iterative Instance Adjustment for Imbalanced Domains.** Original paper: "Addressing soul classification with instance generation techniques: IPADE-ID" by Victoria López, Isaac Triguero, Cristóbal J. Carmona, Salvador García and Francisco Herrera.

* **NearMiss.** Original paper: "kNN Approach to Unbalanced soul.data Distribution: A Case Study involving Information Extraction" by Jianping Zhang and Inderjeet Mani.

* **Neighbourhood Cleaning Rule.** Original paper: "Improving Identification of Difficult Small Classes by Balancing Class Distribution" by J. Laurikkala.

* **One-Side Selection.** Original paper: "Addressing the Curse of Imbalanced Training Sets: One-Side Selection" by Miroslav Kubat and Stan Matwin.

* **Random Undersampling.**

* **Tomek Link.** Original paper: "Two Modifications of CNN" by Ivan Tomek.

* **Undersampling Based on Clustering.** Original paper: "Under-Sampling Approaches for Improving Prediction of the Minority Class in an Imbalanced Dataset" by Show-Jane Yen and Yue-Shi Lee.

Included algorithms for oversampling:

* **Adasyn.** Original paper: "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning" by Haibo He, Yang Bai, Edwardo A. Garcia, and Shutao Li.

* **Adoms.** Original paper: "The Generation Mechanism of Synthetic Minority Class Examples" by Sheng TANG and Si-ping CHEN.

* **Borderline-SMOTE.** Original paper: "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning." by Hui Han, Wen-Yuan Wang, and Bing-Huan Mao.

* **DBSMOTE.** Original paper: "DBSMOTE: Density-Based Synthetic Minority Over-sampling TEchnique" by Chumphol Bunkhumpornpat,  Krung Sinapiromsaran and Chidchanok Lursinsap.

* **MDO.** Original paper: "To combat multi-class soul problems by means of over-sampling and boosting techniques" by Lida Adbi and Sattar Hashemi.

* **MWMOTE.** Original paper: "MWMOTE—Majority Weighted Minority Oversampling Technique for Imbalanced Data Set Learning" by Sukarna Barua, Md. Monirul Islam, Xin Yao, Fellow, IEEE, and Kazuyuki Muras.

* **SafeLevel-SMOTE.** Original paper: "Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-Sampling TEchnique for Handling the Class Imbalanced Problem" by Chumphol Bunkhumpornpat, Krung Sinapiromsaran, and Chidchanok Lursinsap.

* **SMOTE.** Original paper: "SMOTE: Synthetic Minority Over-sampling Technique" by Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall and W. Philip Kegelmeyer.

* **SMOTE + ENN and SMOTE + TL.** Original paper: "A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data" by Gustavo E. A. P. A. Batista, Ronaldo C. Prati and Maria Carolina Monard.

* **SMOTE-RSB.** Original paper: "kNN Approach to Unbalanced Data Distribution: SMOTE-RSB: a hybrid preprocessing approach based on oversampling and undersampling for high soul data-sets using SMOTE and rough sets theory" by Enislay Ramentol, Yailé Caballero, Rafael Bello and Francisco Herrera.

* **Spider2.** Original paper: "Learning from Imbalanced Data in Presence of Noisy and Borderline Examples" by Krystyna Napiera la, Jerzy Stefanowski and Szymon Wilk.

### How-to use it

To read a data file you only need to do this:

```scala
import soul.io.Reader

val reader = new Reader

/* Read a csv file or any delimited text file */
val csvData: Data = reader.readDelimitedText(file = pathToFile)
/* Read a WEKA arff file */
val arffData: Data = reader.readArff(file = pathToFile)
```

Now we're going to run an undersampling algorithm:

```scala
import soul.algorithm.undersampling.NCL

val nclCSV = new NCL(csvData)
val resultCSV: Data = nclCSV.sample(file = Option("mylogCSV.log"))

val nclARFF = new NCL(arffData)
val resultARFF: Data = nclARFF.sample(file = Option("mylogARFF.log"))
```

In this example we've used an undersampling algorithm but it's the same for an oversampling one. All the algorithm's parameters have default values so you don't need to specify any of them.

Finally, we only need to save the result to a file: 

```scala
import soul.io.Writer

val writer: Writer = new Writer

writer.writeDelimitedText(file = "resultCSV.csv", data = resultCSV)
writer.writeArff(file = "resultARFF.arff", data = resultARFF)
```