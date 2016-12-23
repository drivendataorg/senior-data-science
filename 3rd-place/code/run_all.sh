cd src

echo "generating features...."
python feature_extraction_v3.py
python feature_extraction_v5.py    
python feature_extraction_v7.py
python feature_extraction_v8.py    
python feature_extraction_v9.py
python feature_extraction_v11.py
python feature_extraction_v17.py   

echo "training L1 models...."
python knn_v5a.py  
python rf_v13a.py                 
python xgb_v5.py                                   
python xgb_v8a.py
python xgb_v10a.py                 
python xgb_v12a.py
python xgb_v14a.py                 
python xgb_v15a.py
python xgb_v16a.py                 
python xgb_v18a.py
python xgb_v19a.py                 
python xgb_v20a.py
python xgb_v21a.py                 
python xgb_v22a.py
python xgb_v24a.py                 
python xgb_v25a.py
python nn_v20.py  
python gen_esb_data_v20.py

echo "training L2 models..."
python sgd_esb_v20.py              
python xgb_esb_v3.py               
python xgb_esb_v4.py
python xgb_esb_v5.py               
python xgb_esb_v7.py
python xgb_esb_v25.py
python nn_esb_v20.py

echo "generating final submission..."
python blend.py

echo "Done! Please find the final submission ens13.csv in sub folder."
