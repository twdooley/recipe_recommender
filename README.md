# "I'll Bake What She's Having!": Using Computer Vision and NLP to Recommend Recipes from BonAppetit.com
## Timothy Dooley, PhD
## Metis Project 5 (Capstone)
----------------------------------
## Objective:

To provide accurate recipe recommendations from BonAppetit.com based on user images and/or user text input. 

## Contents
* `web_apps/ba_app.py`, "I'll Bake What She's Having!": A computer vision image retrieval model which compares a user uploaded image to ~5k Bon Appetit images to provide a number of recommendations based on k-Nearest Neighbors.
* `ba_build_model.py`, Builds the aforementioned Computer Vision model. 
* `web_apps/ingred_app.py`, 'Bon Appetit Ingredient Genie': An NLP model that uses cosine similarity between user text input and the corpus of Bon Appetit recipes to suggest a number of recipes (based on a user provided threshold). This model works on distance and is not a string matching model. This was in order to provide 'close' responses in concept if not a direct checklist of available ingredients. 
* `ingred_app_builder.ipynb` Builds the aforementioned NLP model.
* `ba_scrape.ipynb` notebook shows the scraping workflow. Users can follow the same workflow to scrape for themselves. Note: this project scraped only until Dec. 2, 2020 
* `w_urls.csv` and `web_apps/super.csv` are two resulting `.csv` files with scraped BA data. Super is a merged file for use in web apps. **NB** I do not own the rights to these recipes, they are the IP of Bon Appetit and Conde Nast. I provide this as the result of my student project. 
* `ba_presentation.pdf` is a pdf of the presentation I gave on this project. It has further visual explanations. 


## Methods:
I scraped ~5,000 recipes and images from around an 8 year period as found freely on BonAppetit.com
<br><br>
*Topic Modeling* Non-Negative Matrix Factorization was used to provide useful topic models in creating the applications. These topics served as filters to allow the user to add specificity to their search. Such unsupervised topic modeling was necessary given the poor quality of BA provided topics which are often subjective, particular to their magazine, and inconsistent. <br><br>
*Computer Vision Image Retrieval* After retrieving weights/features from my corpus of ~5,000 BA images on a ResNet50 model, I binarized these features to use in a k-Nearest Neighbors model. A user can then submit a photograph which is processed with ResNet50, compared to the weights for distance in k-NN and a result of the top neighbors is produced with images and links to BA. <br><br>
*NLP Recommender* Using a vectorized dataframe of all BA recipe ingredients, user text input is similarly vectorized and its distance to other recipes is calculated. Within a given cosine similarity score threshold, results are produced for the user with images and links to BA.<br><br>


