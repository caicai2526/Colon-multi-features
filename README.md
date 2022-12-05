# Colon-multi-features
#### CRC Prognostic analysis

# Introduction
> Colorectal cancer (CRC) is a malignant tumor within digestive tract with both high incidence rate and and mortality. Early detection and intervention could improve patient clinical outcome and survival. This study computationally investigate a set of prognostic tissue and celluer features from diagnostic tissue slide. With the combination of clinical prognostic variable, the pathological image features could predict the prognosis in CRC patients.In this paper, we analyze microenvironmental features, deep network features, and clinical features in colorectal cancer pathological images. And through the fusion of multiple modal features, the prognosis and survival of patients are predicted. the model has obtained relatively good prediction results.

The code in this progject will reproduce the results in our paper submitted to Bioinformatics, "Prognostic analysis based on multi-feature calculation and clinical information fusion of colorectal cancer whole slide pathological image". Code is written in matlab and python.

Some mat file we will upload [Baidu Cloud](https://pan.baidu.com/s/1sBv8m233kp0tz2ngcac0RA), pwd: x0jk 

# Outline
* Dataset
* Methods
* Experiment setup
* Contact information

# Dataset
> there are a total of 660 WSI colorectal cancer pathology images in the dataset, and the dataset comes from two units.
> * Nanfang Hospital of Southern Medical University (SMU) 
> * Guangdong Provincial People’s Hospital (GD).

# Methods
> ### Prognostic analysis process based on WSI of Colorectal Cancer Histopathology Images
![main process](https://github.com/caicai2526/Colon-multi-features/blob/main/Fig/fig1.jpg)
  > 1. Using [Deeptissue Net](https://github.com/caicai2526) to segment different tissue region
![Deeptissue Net](https://github.com/caicai2526/Colon-multi-features/blob/main/Fig/fig2.jpg)
  > 2. Using Deeptissue Net to extract deep features and fusion
![extract deep feature](https://github.com/caicai2526/Colon-multi-features/blob/main/Fig/fig3.jpg)
  > 3. Pathomic feature extraction based on cancerous regions
![extract Pathomic feature](https://github.com/caicai2526/Colon-multi-features/blob/main/Fig/fig4.jpg)
  > 4. Prognostic survival analysis with fusion of clinical features and image features
  ![ Prognostic survival analysis result](https://github.com/caicai2526/Colon-multi-features/blob/main/Fig/fig6.jpg)
  
# Experiment setup
* Tissue ratio Feature and Deep Feature Extraction Based on DeepTissue Net
* Pathomic feature extraction based on cancerous regions
* Feature extraction based on clinical information
* Prognostic survival analysis for each group of features individually
* Prognostic survival analysis after different feature combinations
* Prognostic survival analysis with fusion of clinical features and image features

# Contact information 
If you have any questions, feel free to contact me. Chengfei Cai, Nanjing University of Information Science and Technology, Nanjing, China. Email: chengfeicai@nuist.edu.cn


