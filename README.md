1. Download dataset (Amazon Reviews 2014) from: <br />
   wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz <br />
   wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz <br />
   wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz <br />
2. Move the downloaded data in to the 'data' folder <br />
3. run: python preprocess_data.py <br />
4. run: python train_model.py <br />
