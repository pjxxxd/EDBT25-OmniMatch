# EDBT25 - OmniMatch <br />

Source codes for EDBT25 - OmniMatch: Overcoming the Cold-Start Problem in Cross-Domain Recommendations using Auxiliary Reviews <br />

## Getting Started <br />

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. <br />

### Installation <br />

1. **Download the datasets (Amazon Reviews 2014):** <br />
   wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz <br />
   wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz <br />
   wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz <br />

2. **Download the fastText word embedding:** <br />
   wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip <br />

3. **Organize the data:** <br />
   Unzip the fastText word embedding and move the downloaded dataset files into the 'data' folder in your project directory. <br />

### Runing the Project <br />

1. **Preprocess the Data:** <br />
   python preprocess_data.py <br />
   
3. **Train the Model:** <br />
   python train_model.py <br />




