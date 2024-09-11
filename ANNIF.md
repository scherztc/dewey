# DEWEY can leverage ANNIF to classify subject terms from a FUll-TEXT document using ANNIF.

## URLS
   https://liberquarterly.eu/article/view/10732

## Interfaces
-   CLI 
-   GUI
-   API

## Prerequisites
-   SUBJECT_FILE (TSV SKOS/RDF)
-   TRAINING_DATA
-   VOCAB_ID : 
    -  https://skos.um.es/unescothes/
    -  https://id.loc.gov/download/
    -  https://github.com/NatLibFi/Annif/wiki/Subject-vocabulary-formats
-   PROJECT_ID

## ANNIF Instructions

1. Git Clone : git@github.com:NatLibFi/Annif.git

Install pipx and Poetry if you don't have them. First pipx:
python3 -m pip install --user pipx
python3 -m pipx ensurepath

1. Open a new shell, and then install Poetry:
pipx install poetry

1. Create a virtual environment and install dependencies:
poetry install --all-extras

1. Enter the virtual environment:
poetry shell

1. You will also need NLTK data files:
python -m nltk.downloader punkt

1. Copy projects.cfg.dist to projects.cfg

1. Load Subject Vocabulary

1. git clone : git@github.com:NatLibFi/Annif-corpora.git
annif load-vocab yso /path/to/Annif-corpora/vocab/yso-skos.ttl

1. We will train the model using the the English language training data generated from Finna.fi metadata:
annif train tfidf-en /path/to/Annif-corpora/training/yso-finna-en.tsv.gz

Other Models to consider MLLM, Hogwarts/Fast, Languages, NNEnsemble, Custom, DVC, OmiKuji

1. Start up the application:
annif run

1. Using Docker
docker run -v /Users/scherztc/Workspaces/dewey/annif-docker:/annif-projects -u $(id -u):$(id -g) -it quay.io/natlibfi/annif bash


1. Preprocessing allows us to clean unstructured text
  1.  Remove OCR noise (works that appear frequently, such as definite and indefinite articles.
  2.  Tokenization (sequencing characters and breaking them up into words, phrases, sentences)
  3.  Create stopwords.txt (This includes prepositions, articles, and conjunctions, like “a,” “an,” “the,” “but,” and so on.) 
  4.  Filtering (Token.filter)
  5.  Lemmatization (Token.lemma : morphological analysis and grouping like words together i.e. verb tenses)     
  6.  Stemming (Token.stem : Reduces inflected words to their base or root form)

1. Test Document (GUI)
- Visit localhost:5000
- Copy/Paste Text
- Select TF-IDF English
- Run

1. or CLI

- cat document.txt | annif suggest tfidf-en

1. or CLI with folder of documents

- annif eval tfidf-en /path/to/documents/

1. Explore API
The Swagger UI documentation for the REST API is at http://localhost:5000/v1/ui/

# Links

1. https://annif.org/ 
1. https://liberquarterly.eu/article/view/10732

# References

1.  PubMED : https://pubmed.ncbi.nlm.nih.gov/30153250/
1.  Other :https://www.aiforlibrarians.com/ai-cases/
1.  Science Direct : Weed Collections : https://www.sciencedirect.com/science/article/pii/S0099133317304160?via%3Dihub
1.  Apache Mahout : https://mahout.apache.org//
1.  Spark MLlib Apache : https://spark.apache.org/mllib/
1.  Stanford : https://library.stanford.edu/blogs/stanford-libraries-blog/2022/07/working-students-library-collections-data
1.  M-Files. Smart subjects provide tag suggestions based on document content
1.  Magellan's AI capabilities include speech and text analytics from contextual hypothesis and meaning deduction.
1.  AWS Innovate: Data and AI/ML Edition
1.  Data Science : https://www.dataplusscience.com/GenerativeAI.html
1.  AI Got Talent : https://dataplusscience.com/files/UCCBAGenAI20240206.pdf
1.  https://www.sciencedirect.com/science/article/pii/S0099133317304160?via%3Dihub
1.  https://library.stanford.edu/blogs/stanford-libraries-blog/2022/07/working-students-library-collections-data

# Regulation

MEPs substantially amended the list to include bans on intrusive and discriminatory uses of AI systems such as:


1.  “Real-time” remote biometric identification systems in publicly accessible spaces;
1.  “Post” remote biometric identification systems, with the only exception of law enforcement for the prosecution of serious crimes and only after judicial authorization;
1.  Biometric categorisation systems using sensitive characteristics (e.g. gender, race, ethnicity, citizenship status, religion, political orientation);
1.  Predictive policing systems (based on profiling, location or past criminal behaviour);
1.  Emotion recognition systems in law enforcement, border management, workplace, and educational institutions; and
1.  Indiscriminate scraping of biometric data from social media or CCTV footage to create facial recognition databases (violating human rights and right to privacy).

# AI Feedback Loop

-  Programming : Algorithm + Input => Answers
-  Supervised Learning :  Answer + Input = > Algorithm
-  Feedback Loop : Re-perturbed feeds classification and classification feed perturbed
-  Decision Tree : mathematically produced else
-  Neural Network : binary tree diagrams
-  Natural Language Processing (NLP_ :  the identification of patterns in spoken or written text.
    -  Read Understand Derive meaning from Human Languages
    -  Lanaguage Structures
    -  INteract Transfer Data
    -  Feed Document -> encode -> segmentation into sentences by punctuations.  words in the sentences into constiutainet words into tokens.  tokenize.   remove no$
    -  ALgorithm
    -  Explain (skip, skipping skipped _ same stemming.
    -  limitization lemmatizaition
    -  verbs particle - speech tagging
    -  Pop culture references movies places news locations- named entitity tagging

>    Segmentation into sentences and store them.
>    Tokenizing into words and store them.
>    Remove Stop Words (Non Essential Words.)  like are, and, the from sentence segments.
>    Stemming treats skip, skipping, skipped as the same words.
>    Lemmetization Am Are Is for all genders ich bin, du bist, er/sie/es ist base word (lemma)= be
>    Named Entity Tagging of Proper Nouns
>    Sentiment and Speech with naive bayes

-  Reinforcment Learning : data - > model
-  Cumulative Selection : Building off the last step.  Not starting over everyt ime
-  Semi-Supervisied Machine Learning : Supervised means there is some human involvement in setting up the tool,
-  Inductive Reasoning
   -   The corpus-based approach using a training set, as described above, uses the process of inductive reasoning. This is the kind of thinking that states ‘the sun rose yesterday, the sun rose today, so the chances are the sun will rise tomorrow’. Now, philosophers will argue that inductive reasoning is not scientific. Just because the sun rose yesterday does not mean the sun will rise tomorrow. 

# AI Terms
   - A Corpus : All text documents in Scholar
   - A Training Set :  is a subset of the corpus, which has been tagged in some way to identify the characteristic you are looking for
        Common Crawl 
        RefinedWeb
        The Pile
        C4
        Starcoder
        BookCorpus o ROOTS
        Wikipedia o Red Pajama

   - A Test Set : collection of documents to be used for trialling the algorithm, to see how successfully it carries out the operation.
       - Example : Modified National Institute of Standards and Technology (MNIST) database of handwritten numbers,10
   - An Algorithm : The ‘algorithm’ is simply the tool that looks at each item in the corpus and enables a decision to be made. An algorithm may be as simple (and frequently is as simple) as matching a pattern. selecting and applying an algorithm or method

# AI Process Cycle
   1. Identify
   1. Explore / Analyze / Encode (Change Everything into a number)
   1. Model
   1. Predict
       1. Clarity (Makes Sense)
       1. Original (Novelty)
       1. Useful
   1. Feedback


# Tennis dataset (Machine Learning)

This is a famous dataset used to demonstrate how Machine Learning and Decision Making work.

When do you play tennis? (Class, Outlook, Temp, Windy)

- label (y) = things you want to predict
    - play / no play

- features = 
    - outlook, temp, windy

- values (x) = { Sunny, Low, Yes }

- yak shaving (Remove Turtle Cloud, Kindof, 28%)

## Model  

>                 Outlook
>             Sunny  Overcast   Rainy
>           Temp    True          Windy
>         low high             False  True
>        True  False

How do you know how well it is doing?

> Train        - > Test
> Use ~ 80%        Use ~ 20%


> Train    -> Dev   -> Test
> Use ~ 80    Use ~10   Use~10


## The Confusion Matrix

-  values (x) = Truth {true, false}
-  values (y) = Guess {positive, negative}

>  true positive      false positive

>  false negative     true negative

-  precision  = tp / tp + tp
-  accuracy = tp + tn / tp + tn + fp + fn
-  recall  = tp / tp + fn


Resources
https://www.youtube.com/watch?v=awGJkRe9m50
