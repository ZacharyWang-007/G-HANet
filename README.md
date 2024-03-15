# Pytorch implementation of 'Histo-Genomic Knowledge Distillation For Cancer Prognosis From Histopathology Whole Slide Images'

 ## Requirements
 ### Installation
Please refer to [Patch-GCN](https://github.com/mahmoodlab/Patch-GCN) 

 ### Dataset Preparation
 Please download the official [TCGA datasets](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) of BRCA, BLCA, GBMLGG, LUAD, and UCEC. 
 For more details on pre-processing, please refer to [CLAM](https://github.com/mahmoodlab/CLAM) and [Patch-GCN](https://github.com/mahmoodlab/Patch-GCN). 
 
 ## Model training and testing
 before training and testing, please update the configs. Generally, we train the model with one 24 GB memory GPU. You can adjust the 'num_instances_maximum' to sample the number of instances in accordance with your GPU power.  
 Testing will be performed after each training epoch, and the last model will be employed for the final evaluation. 
 
 ~~~~~~~~~~~~~~~~~~
   e.g., python main.py
 ~~~~~~~~~~~~~~~~~~


## Contact
If you have any questions, please don't hesitate to contact us. E-mail: [zhikang.wang@monash.edu](zhikang.wang@monash.edu) 

## Acknowledgement 
This work was built upon the [Patch-GCN](https://github.com/mahmoodlab/Patch-GCN) and [CLAM](https://github.com/mahmoodlab/CLAM).

