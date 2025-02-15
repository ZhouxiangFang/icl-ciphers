# icl-ciphers

This is the repo for ICL CIPHERS. 

The `dataset` folder contains examples files for each dataset used in the paper.

To install the required dependencies

````
pip install -r requirements.txt
````

To generate the token frequencies using wikipedia corpus, which will be used in zipfian shuffling

````
python generate_tf.py --model {model} --hf_token {hf_token}
````

To accelerate the priority sampling of demonstrations, we first create mappings between all the tokens and demonstrations.

````
python gen_demo_dict.py --dataset {dataset} --model {model} --hf_token {hf_token}
````

To run ICL CIPHERS, run `icl_all.py`.